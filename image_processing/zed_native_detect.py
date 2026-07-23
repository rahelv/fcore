"""
ZED-native person detection + cropping (ZED X).

Uses ZED SDK's built-in Object Detection module to find people and crop
them from the left image. For each detected person you get a 2D bounding box
-> crop for the CLIP classifier.

    ZED grab -> retrieve_objects -> crop each person

Requirements:
 - pyzed

usage:
    python zed_native_detect.py --save-crops ./crops_native
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import pyzed.sl as sl


@dataclass
class ZedDetection:
    box: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in the left image
    score: float                      # ZED confidence, 0-100
    crop: Image.Image                 # RGB crop


class ZedPersonDetector:
    def __init__(
        self,
        resolution: "sl.RESOLUTION" = None,
        depth_mode: "sl.DEPTH_MODE" = None,
        detection_model: "sl.OBJECT_DETECTION_MODEL" = None,
        conf: float = 40,   # ZED confidence threshold, 0-100 (NOT 0-1)
        pad: float = 0.08,  # expand each box by this fraction
    ):
        self.pad = pad

        self.zed = sl.Camera()

        init = sl.InitParameters()
        init.camera_resolution = resolution or sl.RESOLUTION.HD1080
        # Object detection needs depth. NEURAL is most accurate; PERFORMANCE is faster.
        init.depth_mode = depth_mode or sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER

        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        od_params = sl.ObjectDetectionParameters()
        od_params.enable_tracking = False
        # NOTE: SDK 4.x uses MULTI_CLASS_BOX_FAST / _MEDIUM / _ACCURATE.
        # Older SDKs use MULTI_CLASS_BOX. Adjust if you get an AttributeError.
        od_params.detection_model = detection_model or sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

        od_status = self.zed.enable_object_detection(od_params)
        if od_status != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError(f"enable_object_detection failed: {od_status}")

        # Runtime params: confidence + restrict to the PERSON class only.
        self.od_runtime = sl.ObjectDetectionRuntimeParameters()
        self.od_runtime.detection_confidence_threshold = conf
        self.od_runtime.object_class_filter = [sl.OBJECT_CLASS.PERSON]

        self._image = sl.Mat()
        self._objects = sl.Objects()
        self._runtime = sl.RuntimeParameters()

    @staticmethod
    def _mat_to_rgb(mat: "sl.Mat") -> np.ndarray:
        bgra = mat.get_data()                               # H x W x 4, BGRA
        return np.ascontiguousarray(bgra[:, :, [2, 1, 0]])  # -> RGB, copy

    def _pad_box(self, x1, y1, x2, y2, w, h):
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * self.pad, bh * self.pad
        return (
            max(0, int(x1 - dx)),
            max(0, int(y1 - dy)),
            min(w, int(x2 + dx)),
            min(h, int(y2 + dy)),
        )

    def detect_frame(self, warmup: int = 8) -> List[ZedDetection]:
        # settle auto-exposure
        for _ in range(max(0, warmup)):
            self.zed.grab(self._runtime)

        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab() failed")

        self.zed.retrieve_image(self._image, sl.VIEW.LEFT)
        self.zed.retrieve_objects(self._objects, self.od_runtime)

        rgb = self._mat_to_rgb(self._image)
        H, W = rgb.shape[:2]
        full = Image.fromarray(rgb, mode="RGB")

        detections: List[ZedDetection] = []
        for obj in self._objects.object_list:
            bb = np.asarray(obj.bounding_box_2d)     # 4 points (x, y)
            if bb.size == 0:
                continue                             # no valid 2D box this frame
            x1, y1 = bb[:, 0].min(), bb[:, 1].min()
            x2, y2 = bb[:, 0].max(), bb[:, 1].max()
            box = self._pad_box(x1, y1, x2, y2, W, H)
            crop = full.crop(box)

            detections.append(
                ZedDetection(box=box, score=float(obj.confidence), crop=crop)
            )
        return detections

    def close(self):
        self.zed.disable_object_detection()
        self.zed.close()


def main():
    ap = argparse.ArgumentParser(description="ZED-native person detection + crop.")
    ap.add_argument("--conf", type=float, default=40, help="confidence threshold, 0-100")
    ap.add_argument("--pad", type=float, default=0.08, help="box padding fraction")
    ap.add_argument("--save-crops", default=None, help="dir to write crops (optional)")
    args = ap.parse_args()

    detector = ZedPersonDetector(conf=args.conf, pad=args.pad)
    try:
        detections = detector.detect_frame()

        print(f"Found {len(detections)} person(s)")

        save_dir = Path(args.save_crops) if args.save_crops else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for i, det in enumerate(detections):
            print(f"  [{i}] box={det.box} conf={det.score:.0f}")
            if save_dir:
                det.crop.save(save_dir / f"zed_person_{i}.jpg")

        if save_dir:
            print(f"Crops written to {save_dir}/")
    finally:
        detector.close()


if __name__ == "__main__":
    main()

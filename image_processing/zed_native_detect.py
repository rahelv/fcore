"""
ZED-native person detection + cropping (ZED X).

Instead of YOLO, this uses the ZED SDK's built-in Object Detection module
(MULTI_CLASS_BOX). For each detected person you get:
  - a 2D bounding box  -> crop for your CLIP step
  - a 2D mask          -> optional background-removed crop
  - a 3D position (m)  -> where the person is in space
  - a persistent ID    -> if tracking is enabled

    ZED grab -> retrieve_objects -> crop each person -> (later) CLIP

Requires the ZED SDK Python API (pyzed) and a ZED that supports Object
Detection (ZED Mini / 2i / X / X Mini / X Nano). Depth must be enabled.

usage:
    python zed_native_detect.py --save-crops ./crops_native --masked
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import pyzed.sl as sl


@dataclass
class ZedDetection:
    box: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in the left image
    score: float                      # ZED confidence, 0-100
    crop: Image.Image                 # plain RGB crop
    position: Tuple[float, float, float]  # 3D (x, y, z) in meters, camera frame
    object_id: int                    # tracking ID (-1 if tracking disabled)
    masked_crop: Optional[Image.Image] = None  # background removed, if requested


class ZedPersonDetector:
    def __init__(
        self,
        resolution: "sl.RESOLUTION" = None,
        depth_mode: "sl.DEPTH_MODE" = None,
        detection_model: "sl.OBJECT_DETECTION_MODEL" = None,
        conf: float = 40,               # ZED confidence threshold, 0-100 (NOT 0-1)
        pad: float = 0.08,              # expand each box by this fraction
        enable_segmentation: bool = True,   # needed for masked crops
        enable_tracking: bool = False,      # True -> persistent IDs (needs pos. tracking)
    ):
        self.pad = pad
        self.enable_tracking = enable_tracking

        self.zed = sl.Camera()

        init = sl.InitParameters()
        init.camera_resolution = resolution or sl.RESOLUTION.HD1080
        # Object detection needs depth. NEURAL is most accurate; PERFORMANCE is faster.
        init.depth_mode = depth_mode or sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER

        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        # Positional tracking is required only if we want tracking IDs.
        if enable_tracking:
            pt_status = self.zed.enable_positional_tracking(
                sl.PositionalTrackingParameters()
            )
            if pt_status != sl.ERROR_CODE.SUCCESS:
                self.zed.close()
                raise RuntimeError(f"enable_positional_tracking failed: {pt_status}")

        od_params = sl.ObjectDetectionParameters()
        od_params.enable_tracking = enable_tracking
        od_params.enable_segmentation = enable_segmentation
        # NOTE: on SDK 4.x this enum may be MULTI_CLASS_BOX_MEDIUM /
        # MULTI_CLASS_BOX_FAST / MULTI_CLASS_BOX_ACCURATE. Older SDKs use
        # MULTI_CLASS_BOX. Adjust if you get an AttributeError here.
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
        bgra = mat.get_data()                       # H x W x 4, BGRA
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

        rgb = self._mat_to_rgb(self._image)         # full left frame
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

            masked = None
            if getattr(obj, "mask", None) is not None and obj.mask.is_init():
                masked = self._apply_mask(rgb, obj, box)

            pos = tuple(float(v) for v in obj.position) if obj.position is not None else (0.0, 0.0, 0.0)
            detections.append(
                ZedDetection(
                    box=box,
                    score=float(obj.confidence),
                    crop=crop,
                    position=pos,
                    object_id=int(obj.id),
                    masked_crop=masked,
                )
            )
        return detections

    @staticmethod
    def _apply_mask(rgb: np.ndarray, obj, box) -> Optional[Image.Image]:
        """Composite the person over white using the ZED 2D mask.

        The mask is defined within the object's *unpadded* 2D bbox. We paste it
        back over the padded crop region so it lines up.
        """
        m = obj.mask.get_data()          # (bh, bw) uint8, 255 = object
        if m is None or m.size == 0:
            return None

        # unpadded bbox top-left (mask origin)
        bb = np.asarray(obj.bounding_box_2d)
        ox, oy = int(bb[:, 0].min()), int(bb[:, 1].min())
        bh, bw = m.shape[:2]

        x1, y1, x2, y2 = box
        region = rgb[y1:y2, x1:x2].copy()
        out = np.full_like(region, 255)  # white background

        # place mask into the padded region coordinate space
        my1, mx1 = oy - y1, ox - x1
        my2, mx2 = my1 + bh, mx1 + bw
        # clip to region bounds
        ry1, rx1 = max(0, my1), max(0, mx1)
        ry2, rx2 = min(region.shape[0], my2), min(region.shape[1], mx2)
        if ry2 <= ry1 or rx2 <= rx1:
            return None
        sub = (m[ry1 - my1:ry2 - my1, rx1 - mx1:rx2 - mx1] > 127)
        out[ry1:ry2, rx1:rx2][sub] = region[ry1:ry2, rx1:rx2][sub]
        return Image.fromarray(out, mode="RGB")

    def close(self):
        self.zed.disable_object_detection()
        if self.enable_tracking:
            self.zed.disable_positional_tracking()
        self.zed.close()


def main():
    ap = argparse.ArgumentParser(description="ZED-native person detection + crop.")
    ap.add_argument("--conf", type=float, default=40, help="confidence threshold, 0-100")
    ap.add_argument("--pad", type=float, default=0.08, help="box padding fraction")
    ap.add_argument("--masked", action="store_true", help="also save background-removed crops")
    ap.add_argument("--track", action="store_true", help="enable tracking IDs (adds pos. tracking)")
    ap.add_argument("--save-frame", default=None, help="also save the raw ZED frame here")
    ap.add_argument("--save-crops", default=None, help="dir to write crops (optional)")
    args = ap.parse_args()

    detector = ZedPersonDetector(
        conf=args.conf,
        pad=args.pad,
        enable_segmentation=args.masked,
        enable_tracking=args.track,
    )
    try:
        detections = detector.detect_frame()
    finally:
        pass  # keep camera until after we've used the crops below

    print(f"Found {len(detections)} person(s)")

    save_dir = Path(args.save_crops) if args.save_crops else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, det in enumerate(detections):
        x, y, z = det.position
        print(f"  [{i}] id={det.object_id} box={det.box} conf={det.score:.0f} "
              f"pos=({x:.2f}, {y:.2f}, {z:.2f})m")
        if save_dir:
            det.crop.save(save_dir / f"zed_person_{i}.jpg")
            if det.masked_crop is not None:
                det.masked_crop.save(save_dir / f"zed_person_{i}_masked.jpg")

    detector.close()
    if save_dir:
        print(f"Crops written to {save_dir}/")


if __name__ == "__main__":
    main()

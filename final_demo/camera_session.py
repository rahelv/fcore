"""
camera_session.py
=================

A *persistent* ZED camera session for the live GUI.

`CameraSession` keeps the camera + object-detection module alive and exposes
a single `grab()` that returns the current frame plus the people in it.

Each grab() returns:
    (rgb_frame, detections)
      rgb_frame  : np.ndarray  (H, W, 3) uint8, RGB      -- for display
      detections : list[Detection]                        -- boxes + crops
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

import pyzed.sl as sl

@dataclass
class Detection:
    """One detected person in the current frame."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in the left image
    score: float                     # ZED person confidence, 0-100
    crop: Image.Image                # RGB crop, fed to the CLIP classifier

class CameraSession:
    def __init__(
        self,
        resolution: "sl.RESOLUTION" = None,
        depth_mode: "sl.DEPTH_MODE" = None,
        detection_model: "sl.OBJECT_DETECTION_MODEL" = None,
        conf: float = 40,   # ZED confidence threshold, 0-100
        pad: float = 0.08,  # grow each box by this fraction before cropping
        warmup: int = 8,    # frames discarded once, so auto-exposure settles
    ):
        self.pad = pad

        # --- open the camera (depth ON, required by object detection) --------
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = resolution or sl.RESOLUTION.HD1080
        init.camera_fps = 30
        init.depth_mode = (
            depth_mode
            if depth_mode is not None
            else sl.DEPTH_MODE.NEURAL_LIGHT
        ) 
        init.depth_stabilization = 0
        
        init.coordinate_units = sl.UNIT.METER
        status = self.zed.open(init)
        
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        # --- enable the built-in person detector -----------------------------
        od_params = sl.ObjectDetectionParameters()
        od_params.enable_tracking = False
        # SDK 4.x: MULTI_CLASS_BOX_FAST / _MEDIUM / _ACCURATE (older: MULTI_CLASS_BOX)
        od_params.detection_model = (
            detection_model or sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
        )
        od_status = self.zed.enable_object_detection(od_params)
        if od_status != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError(f"enable_object_detection failed: {od_status}")

        # confidence + PERSON-only filter, applied every grab
        self.od_runtime = sl.ObjectDetectionRuntimeParameters()
        self.od_runtime.detection_confidence_threshold = conf
        self.od_runtime.object_class_filter = [sl.OBJECT_CLASS.PERSON]

        # reusable buffers (avoid re-allocating each frame)
        self._image = sl.Mat()
        self._objects = sl.Objects()
        self._runtime = sl.RuntimeParameters()

        # settle auto-exposure ONCE (not every frame, unlike the one-shot script)
        for _ in range(max(0, warmup)):
            self.zed.grab(self._runtime)

    # ------------------------------------------------------------------ utils
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

    def set_confidence(self, conf: float):
        """Change the detection threshold live (0-100)."""
        self.od_runtime.detection_confidence_threshold = conf

    # ------------------------------------------------------------------ grab
    def grab(self) -> Tuple[np.ndarray, List[Detection]]:
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab() failed")

        self.zed.retrieve_image(self._image, sl.VIEW.LEFT)
        self.zed.retrieve_objects(self._objects, self.od_runtime)

        rgb = self._mat_to_rgb(self._image)
        H, W = rgb.shape[:2]
        full = Image.fromarray(rgb, mode="RGB")

        detections: List[Detection] = []
        for obj in self._objects.object_list:
            bb = np.asarray(obj.bounding_box_2d)
            if bb.size == 0:
                continue
            x1, y1 = bb[:, 0].min(), bb[:, 1].min()
            x2, y2 = bb[:, 0].max(), bb[:, 1].max()
            box = self._pad_box(x1, y1, x2, y2, W, H)
            detections.append(
                Detection(box=box, score=float(obj.confidence), crop=full.crop(box))
            )
        return rgb, detections

    def close(self):
        self.zed.disable_object_detection()
        self.zed.close()

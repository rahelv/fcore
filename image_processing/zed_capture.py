"""
Grab a single RGB frame from a ZED camera and hand it to the existing
YOLO person-crop pipeline.

    ZED left image  ->  PIL.Image (RGB)  ->  PersonCropper.crop()

Requires the ZED SDK Python API (pyzed). Install per Stereolabs docs:
    https://docs.stereolabs.com/docs/development/api-languages/python/

usage:
    python zed_capture.py --save-crops ./crops
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import pyzed.sl as sl

from costume_pipeline import PersonCropper


def grab_frame(
    resolution: "sl.RESOLUTION" = None,
    warmup: int = 8,
) -> Image.Image:
    """
    Open the ZED, grab one left-eye RGB frame, return it as a PIL RGB image.

    `warmup` grabs and discards a few frames first so auto-exposure /
    white-balance settle — a cold first frame is often dark or washed out.
    """
    zed = sl.Camera()

    init = sl.InitParameters()
    init.camera_resolution = resolution or sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NONE  # we only need the color image here

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    try:
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()

        # let the sensor settle
        for _ in range(max(0, warmup)):
            zed.grab(runtime)

        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab() failed")

        zed.retrieve_image(mat, sl.VIEW.LEFT)   # left eye, aligned to depth
        bgra = mat.get_data()                   # H x W x 4, BGRA, uint8

        # drop alpha and reorder BGRA -> RGB (fancy indexing copies, so this
        # stays valid after the camera closes)
        rgb = np.ascontiguousarray(bgra[:, :, [2, 1, 0]])
        return Image.fromarray(rgb, mode="RGB")
    finally:
        zed.close()


def main():
    ap = argparse.ArgumentParser(description="Grab one ZED frame, run person cropping.")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO weights")
    ap.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    ap.add_argument("--pad", type=float, default=0.08, help="box padding fraction")
    ap.add_argument("--save-frame", default=None, help="also save the raw ZED frame here")
    ap.add_argument("--save-crops", default=None, help="dir to write crops (optional)")
    args = ap.parse_args()

    frame = grab_frame()
    print(f"Grabbed frame: {frame.size[0]}x{frame.size[1]}")

    if args.save_frame:
        frame.save(args.save_frame)
        print(f"Frame saved to {args.save_frame}")

    cropper = PersonCropper(weights=args.weights, conf=args.conf, pad=args.pad)
    detections = cropper.crop(frame)
    print(f"Found {len(detections)} person(s)")

    save_dir = Path(args.save_crops) if args.save_crops else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, det in enumerate(detections):
        print(f"  [{i}] box={det.box} det_conf={det.score:.2f}")
        if save_dir:
            det.crop.save(save_dir / f"zed_person_{i}.jpg")

    if save_dir:
        print(f"Crops written to {save_dir}/")


if __name__ == "__main__":
    main()

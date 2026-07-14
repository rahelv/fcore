"""
Costume recognition pipeline — STEP 1: detect + crop people.

image  ->  YOLO person detection  ->  crop each person

usage:
    python costume_pipeline_step1_crop.py path/to/image.jpg --save-crops crops/
    python costume_pipeline_step1_crop.py path/to/folder/  --save-crops crops/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

# Data structures
@dataclass
class Detection:
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords
    score: float                     # YOLO detection confidence
    crop: Image.Image                # cropped PIL image

class PersonCropper:
   # Detects people with YOLO and returns padded crops.

    def __init__(
        self,
        weights: str = "yolov8n.pt",   # TODO: try yolov8m/l.pt
        conf: float = 0.35,            # min detection confidence
        person_class_id: int = 0,      # 0 = 'person' in the COCO classes YOLO ships with
        pad: float = 0.08,             # padding around box
        device: Optional[str] = None,  # 'cuda', 'cpu', or None to auto-pick
    ):
        from ultralytics import YOLO  # imported lazily so the file loads without it

        self.model = YOLO(weights)
        self.conf = conf
        self.person_class_id = person_class_id
        self.pad = pad
        self.device = device

    def _pad_box(self, x1, y1, x2, y2, w, h):
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * self.pad, bh * self.pad
        return (
            max(0, int(x1 - dx)),
            max(0, int(y1 - dy)),
            min(w, int(x2 + dx)),
            min(h, int(y2 + dy)),
        )

    def crop(self, image: Image.Image) -> List[Detection]:
        w, h = image.size
        results = self.model.predict(
            source=image,
            conf=self.conf,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                score = float(b.conf[0])
                box = self._pad_box(x1, y1, x2, y2, w, h)
                crop = image.crop(box)
                detections.append(Detection(box=box, score=score, crop=crop))
        return detections


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def gather_images(path: Path) -> List[Path]:
    """Return a sorted list of image files from a file or directory path."""
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [path]

def main():
    ap = argparse.ArgumentParser(description="Step 1: detect people and crop them.")
    ap.add_argument("input", help="path to an image OR a folder of images")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO weights")
    ap.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    ap.add_argument("--pad", type=float, default=0.08, help="box padding fraction")
    ap.add_argument("--save-crops", default=None, help="dir to write crops (optional)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Path does not exist: {input_path}")

    images = gather_images(input_path)
    if not images:
        raise SystemExit(f"No images found in {input_path}")

    cropper = PersonCropper(weights=args.weights, conf=args.conf, pad=args.pad)

    save_dir = Path(args.save_crops) if args.save_crops else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        detections = cropper.crop(image)
        total += len(detections)
        print(f"{img_path.name}: found {len(detections)} person(s)")

        for i, det in enumerate(detections):
            print(f"  [{i}] box={det.box} det_conf={det.score:.2f}")
            if save_dir:
                # prefix with source stem so crops from different images don't collide
                det.crop.save(save_dir / f"{img_path.stem}_person_{i}.jpg")

    print(f"\nProcessed {len(images)} image(s), {total} crop(s) total.")
    if save_dir:
        print(f"Crops written to {save_dir}/")


if __name__ == "__main__":
    main()

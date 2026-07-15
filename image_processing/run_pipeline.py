"""
End-to-end costume recognition on a single ZED frame.

    ZED grab -> detect + crop each person -> OpenCLIP -> "who is this?"

Combines:
  - ZedPersonDetector  (zed_native_detect.py)   -> person crops
  - ClipCostumeClassifier (clip_classifier.py)  -> costume label per crop

Model / checkpoint / labels are hardcoded in clip_classifier.py.

usage:
    python3 run_pipeline.py --save-crops ./crops_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zed_native_detect import ZedPersonDetector
from clip_classifier import ClipCostumeClassifier, MODEL_ID, CHECKPOINT, LABELS


def main():
    ap = argparse.ArgumentParser(description="ZED frame -> person crop -> costume label.")
    ap.add_argument("--conf", type=float, default=40, help="ZED person confidence, 0-100")
    ap.add_argument("--pad", type=float, default=0.08, help="box padding fraction")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="force CPU for the classifier")
    ap.add_argument("--save-crops", default=None, help="dir to write annotated crops")
    args = ap.parse_args()

    # 1) load the classifier once (loads the model + precomputes text features)
    print(f"Model: {MODEL_ID}\nCheckpoint: {CHECKPOINT}\nLabels: {len(LABELS)}")
    clf = ClipCostumeClassifier(device="cpu" if args.cpu else None)

    # 2) grab one frame and detect + crop people
    detector = ZedPersonDetector(conf=args.conf, pad=args.pad)
    try:
        detections = detector.detect_frame()
    finally:
        detector.close()

    print(f"\nFound {len(detections)} person(s)\n")
    if not detections:
        return

    # 3) classify all crops in one batch
    crops = [det.crop for det in detections]
    all_preds = clf.classify_batch(crops, k=args.topk)

    save_dir = Path(args.save_crops) if args.save_crops else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, (det, preds) in enumerate(zip(detections, all_preds)):
        top = preds[0]
        alts = ", ".join(f"{d['label']} {d['score']}%" for d in preds[1:])
        print(f"Person {i} (det_conf={det.score:.0f})")
        print(f"    -> {top['label']}  ({top['score']}%)")
        if alts:
            print(f"       also: {alts}")
        if save_dir:
            fname = f"person_{i}_{top['label'].replace(' ', '_')}.jpg"
            det.crop.save(save_dir / fname)

    if save_dir:
        print(f"\nCrops written to {save_dir}/")


if __name__ == "__main__":
    main()

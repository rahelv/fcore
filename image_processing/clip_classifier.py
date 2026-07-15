"""
OpenCLIP costume classifier for cropped person images.

Mirrors the zero-shot setup in openCLIP/finetuning/00_evaluate_baseline.py so
predictions match your evaluation exactly:

    caption list -> text features        (encode_text, L2-normalized)
    person crop  -> image feature        (encode_image, L2-normalized)
    label = argmax( image_feat @ text_feats.T )   (cosine similarity)

Config (MODEL_ID, CHECKPOINT, LABELS) is hardcoded below -- none of it is
passed on the command line. Labels are the source of truth in labels.json and
loaded at import.

usage:
    # classify a folder of already-saved crops:
    python clip_classifier.py --crops-dir crops [--topk 3] [--cpu]

    # or import and call inside your own detect->crop pipeline:
    from clip_classifier import ClipCostumeClassifier
    clf = ClipCostumeClassifier()          # uses MODEL_ID / CHECKPOINT / LABELS
    preds = clf.classify(pil_crop)         # [{"label": ..., "score": ...}, ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageFile

from open_clip import create_model_from_pretrained, get_tokenizer

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── config: edit these ──────────────────────────────────────────────────────
MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP2-256"   # the OpenCLIP model
CHECKPOINT = "40c_epoch8.pt"                     # the finetuned weights

# Labels are hardcoded via labels.json (loaded at import). Edit labels.json to
# change the class set; keep it in sync with the model you load.
LABELS_PATH = Path(__file__).with_name("labels.json")
LABELS: List[str] = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ClipCostumeClassifier:
    def __init__(
        self,
        labels: List[str] = LABELS,
        model_id: str = MODEL_ID,
        checkpoint: str | None = CHECKPOINT,
        device: str | None = None,
    ):
        if not labels:
            raise ValueError("labels is empty -- check labels.json")
        self.labels = list(labels)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # same loader as 00_evaluate_baseline.py
        self.model, self.preprocess = create_model_from_pretrained(model_id)
        self.tokenizer = get_tokenizer(model_id)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            cleaned = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")

        self.model = self.model.to(self.device).eval()

        # precompute text features once (they never change)
        self.text_features = self._encode_texts(self.labels)

    @torch.no_grad()
    def _encode_texts(self, captions: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(captions).to(self.device)
        with torch.amp.autocast(device_type=self.device.type,
                                enabled=self.device.type == "cuda"):
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        batch = batch.to(self.device)
        with torch.amp.autocast(device_type=self.device.type,
                                enabled=self.device.type == "cuda"):
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def classify(self, image: Image.Image, k: int = 3) -> List[dict]:
        """Top-k predictions for a single crop."""
        return self.classify_batch([image], k=k)[0]

    @torch.no_grad()
    def classify_batch(self, images: List[Image.Image], k: int = 3) -> List[List[dict]]:
        """Top-k predictions for a list of crops (batched, faster)."""
        if not images:
            return []
        img_feats = self._encode_images(images)          # (N, D)
        sims = img_feats @ self.text_features.T          # (N, C) cosine sims
        k = min(k, len(self.labels))
        results: List[List[dict]] = []
        for row in sims:
            # softmax over sims -> relative score (SigLIP sims aren't calibrated
            # probabilities; this is for display/ranking only)
            probs = torch.softmax(row.float(), dim=-1)
            top_p, top_i = torch.topk(probs, k)
            results.append([
                {"label": self.labels[i], "score": round(float(p) * 100, 1)}
                for p, i in zip(top_p, top_i)
            ])
        return results


def main():
    ap = argparse.ArgumentParser(description="Classify person crops with the OpenCLIP model.")
    ap.add_argument("--crops-dir", required=True, help="folder of crop images to classify")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    print(f"Model: {MODEL_ID}\nCheckpoint: {CHECKPOINT}\nLabels: {len(LABELS)}")

    clf = ClipCostumeClassifier(device="cpu" if args.cpu else None)

    crop_dir = Path(args.crops_dir)
    crop_paths = sorted(p for p in crop_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not crop_paths:
        raise SystemExit(f"No crop images found in {crop_dir}")

    images = [Image.open(p) for p in crop_paths]
    all_preds = clf.classify_batch(images, k=args.topk)

    for path, preds in zip(crop_paths, all_preds):
        top = preds[0]
        rest = ", ".join(f"{d['label']} {d['score']}%" for d in preds[1:])
        print(f"{path.name:40s} -> {top['label']} ({top['score']}%)   [{rest}]")


if __name__ == "__main__":
    main()

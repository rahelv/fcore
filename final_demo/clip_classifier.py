"""
OpenCLIP costume classifier for cropped person images.

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
import os
from pathlib import Path
from typing import List
# offline mode 
_BUNDLED_CACHE = Path(__file__).with_name("hf_cache")
if _BUNDLED_CACHE.is_dir():
    os.environ.setdefault("HF_HOME", str(_BUNDLED_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# ─────────────────────────────────────────────────────────────────────────────

import torch  # noqa: E402
from PIL import Image, ImageFile  # noqa: E402

from open_clip import create_model_from_pretrained, get_tokenizer  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── config: TODO: edit if needed ─────────────────────────────────────────────
MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP2-256"   # OpenCLIP model
CHECKPOINT = str(Path(__file__).with_name("40c_epoch8.pt"))  # finetuned weights

# Labels are hardcoded via labels.json (loaded at import). Edit labels.json to
# change the class set; keep it in sync with the model you load.
LABELS_PATH = Path(__file__).with_name("labels.json")
LABELS: List[str] = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _cache_error(model_id: str, e: Exception) -> str:
    return (
        f"Could not load '{model_id}' from the local Hugging Face cache "
        f"(HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}, "
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}).\n"
        "Run  python fetch_model.py  once on a machine with internet to "
        "populate it, then this will work offline.\n"
        f"Original error: {e}"
    )


def _tokenizer_from_local_snapshot(model_id: str):
    """Build open_clip's HFTokenizer from the cached snapshot directory.

    Same arguments open_clip would have passed -- context_length and
    tokenizer_kwargs are read from open_clip_config.json rather than
    hardcoded, since a wrong context_length changes padding and would shift
    predictions silently.
    """
    from huggingface_hub import snapshot_download
    from open_clip.tokenizer import HFTokenizer

    repo_id = model_id.removeprefix("hf-hub:")
    snap = Path(snapshot_download(repo_id, local_files_only=True))

    cfg = json.loads((snap / "open_clip_config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("model_cfg", {}).get("text_cfg", {})
    context_length = text_cfg.get("context_length", 64)
    tokenizer_kwargs = dict(text_cfg.get("tokenizer_kwargs", {}))

    print(f"[tokenizer] offline: loading from {snap.name} "
          f"(context_length={context_length}, {tokenizer_kwargs})")
    return HFTokenizer(str(snap), context_length=context_length, **tokenizer_kwargs)


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

        try:
            self.model, self.preprocess = create_model_from_pretrained(model_id)
        except OSError as e:
            raise RuntimeError(_cache_error(model_id, e)) from e

        try:
            self.tokenizer = get_tokenizer(model_id)
        except OSError:
            self.tokenizer = _tokenizer_from_local_snapshot(model_id)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("state_dict", ckpt)
            cleaned = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")

        self.model = self.model.to(self.device).eval()

        # Learned temperature. 
        if hasattr(self.model, "logit_scale"):
            self.logit_scale = float(self.model.logit_scale.exp().item())
        else:
            self.logit_scale = 100.0

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
            # scale by the learned temperature BEFORE softmax, otherwise the
            # tightly-clustered cosine sims give a near-uniform distribution.
            probs = torch.softmax(row.float() * self.logit_scale, dim=-1)
            top_p, top_i = torch.topk(probs, k)
            results.append([
                {
                    "label": self.labels[i],
                    "score": round(float(p) * 100, 1),      # softmax %, 0-100
                    "sim": round(float(row[i]), 3),          # raw cosine similarity
                }
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

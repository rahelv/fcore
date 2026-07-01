"""
01_evaluate_image_search_512.py — Image Search Evaluation (512-dim, xlm-roberta-base-ViT-B-32)
===============================================================================================
For every image in VAL_DIR (ImageFolder layout):
  1. Embed the image with CLIP
  2. Compare against all label text-embeddings stored in the DB
  3. Record whether the ground-truth label is in top-1 / top-5
  4. Record what it was mistaken as when top-1 is wrong

Usage
-----
    python 01_evaluate_image_search_512.py
    python 01_evaluate_image_search_512.py --cpu
    python 01_evaluate_image_search_512.py --batch-size 64
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import psycopg2
import torch
from dotenv import load_dotenv
from PIL import Image, ImageFile
from torchvision import datasets
from tqdm import tqdm

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
VAL_DIR      = Path("/data/val")
MODEL_NAME   = "xlm-roberta-base-ViT-B-32"
PRETRAINED   = "laion5b_s13b_b90k"
OUTPUT_PATH  = Path("/app/results_01_512.json")

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",           "172.21.0.2"),
    "port":     os.getenv("DB_PORT",           "5432"),
    "dbname":   os.getenv("POSTGRES_FIRST_DB", "fcore_db_512"),
    "user":     os.getenv("POSTGRES_USER",     "your_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────
def load_clip(device: torch.device):
    print("Loading CLIP classification_model …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device).eval()
    print("CLIP ready.")
    return model, preprocess


@torch.no_grad()
def embed_images_batch(model, preprocess, pil_images: list, device: torch.device) -> np.ndarray:
    """Embed a batch of PIL images → [N, 512] normalised float32."""
    tensors  = [preprocess(img) for img in pil_images]
    batch    = torch.stack(tensors).to(device)
    features = model.encode_image(batch)
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# DB — fetch all label embeddings once
# ──────────────────────────────────────────────────────────────────────────────
def fetch_label_embeddings() -> tuple[list[str], np.ndarray]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT l.label, e.vector
                FROM   labels l
                JOIN   embeddings e ON l.embedding_id = e.id
                ORDER  BY l.label
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No label embeddings found in the database.")

    labels = [row[0] for row in rows]
    matrix = np.array([json.loads(row[1]) for row in rows], dtype=np.float32)

    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-8, None)

    print(f"Loaded {len(labels)} label embeddings from DB.")
    return labels, matrix


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    model, preprocess = load_clip(device)

    val_ds       = datasets.ImageFolder(VAL_DIR)
    idx_to_class = {v: k for k, v in val_ds.class_to_idx.items()}
    print(f"Val set: {len(val_ds)} images across {len(val_ds.classes)} classes")

    db_labels, label_matrix = fetch_label_embeddings()

    top1_correct = 0
    top5_correct = 0
    total        = 0

    per_class = defaultdict(lambda: {
        "top1_correct": 0,
        "top5_correct": 0,
        "total": 0,
        "confused_as": defaultdict(int),
    })
    errors = []

    batch_paths  = []
    batch_labels = []

    def process_batch(paths, true_labels):
        nonlocal top1_correct, top5_correct, total

        pil_images, valid_labels = [], []
        for path, lbl in zip(paths, true_labels):
            try:
                pil_images.append(Image.open(path).convert("RGB"))
                valid_labels.append(lbl)
            except Exception as e:
                errors.append((str(path), str(e)))

        if not pil_images:
            return

        img_vecs = embed_images_batch(model, preprocess, pil_images, device)
        sims     = img_vecs @ label_matrix.T  # [B, N_labels]

        for i, true_label in enumerate(valid_labels):
            top5_idx   = np.argsort(sims[i])[::-1][:5]
            top5_preds = [db_labels[j] for j in top5_idx]
            top1_pred  = top5_preds[0]

            hit1 = top1_pred  == true_label
            hit5 = true_label in top5_preds

            top1_correct += int(hit1)
            top5_correct += int(hit5)
            total        += 1

            per_class[true_label]["total"]        += 1
            per_class[true_label]["top1_correct"] += int(hit1)
            per_class[true_label]["top5_correct"] += int(hit5)

            if not hit1:
                per_class[true_label]["confused_as"][top1_pred] += 1

    print("\nRunning evaluation …\n")
    for img_path, label_idx in tqdm(val_ds.imgs, desc="Evaluating", unit="img"):
        true_label = idx_to_class[label_idx]
        batch_paths.append(img_path)
        batch_labels.append(true_label)

        if len(batch_paths) == args.batch_size:
            process_batch(batch_paths, batch_labels)
            batch_paths.clear()
            batch_labels.clear()

    if batch_paths:
        process_batch(batch_paths, batch_labels)

    # ── PRINT ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  EVALUATION RESULTS  (512-dim · xlm-roberta-base-ViT-B-32)")
    print("═" * 60)
    print(f"  Model      : {MODEL_NAME} / {PRETRAINED}")
    print(f"  Val dir    : {VAL_DIR}")
    print(f"  Total imgs : {total}")
    print(f"  Top-1 acc  : {top1_correct/total*100:.2f}%  ({top1_correct}/{total})")
    print(f"  Top-5 acc  : {top5_correct/total*100:.2f}%  ({top5_correct}/{total})")
    print("═" * 60)

    sorted_classes = sorted(
        per_class.keys(),
        key=lambda c: per_class[c]["top1_correct"] / max(per_class[c]["total"], 1),
        reverse=True,
    )

    print(f"\n{'Class':<40} {'Top-1':>12}  {'Top-5':>12}  {'N':>5}")
    print(f"{'-'*40} {'-'*12}  {'-'*12}  {'-'*5}")

    for cls in sorted_classes:
        d  = per_class[cls]
        n  = d["total"]
        t1 = d["top1_correct"]
        t5 = d["top5_correct"]
        print(f"{cls:<40} {t1:>4}/{n:<4} {t1/n*100:>5.1f}%  {t5:>4}/{n:<4} {t5/n*100:>5.1f}%  {n:>5}")

        confused = sorted(d["confused_as"].items(), key=lambda x: x[1], reverse=True)
        if confused:
            confusion_str = ", ".join(f"{lbl} ({cnt}x)" for lbl, cnt in confused)
            print(f"  {'↳ mistaken as:':<18} {confusion_str}")

    if errors:
        print(f"\nSkipped {len(errors)} images due to load errors.")

    # ── SAVE JSON ────────────────────────────────────────────────────────────
    output = {
        "classification_model": f"{MODEL_NAME} / {PRETRAINED}",
        "val_dir": str(VAL_DIR),
        "total_images": total,
        "overall": {
            "top1_accuracy": round(top1_correct / total * 100, 2),
            "top1_correct": top1_correct,
            "top5_accuracy": round(top5_correct / total * 100, 2),
            "top5_correct": top5_correct,
            "total": total,
        },
        "per_class": {
            cls: {
                "total": per_class[cls]["total"],
                "top1_correct": per_class[cls]["top1_correct"],
                "top1_accuracy": round(per_class[cls]["top1_correct"] / per_class[cls]["total"] * 100, 2),
                "top5_correct": per_class[cls]["top5_correct"],
                "top5_accuracy": round(per_class[cls]["top5_correct"] / per_class[cls]["total"] * 100, 2),
                "confused_as": dict(
                    sorted(per_class[cls]["confused_as"].items(), key=lambda x: x[1], reverse=True)
                ),
            }
            for cls in sorted_classes
        },
        "errors": errors,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Copy out with:")
    print(f"  docker cp openclip-search_gui-1:/app/results_01_512.json /home/ubuntu/results_01_512.json")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu",        action="store_true", help="Force CPU")
    p.add_argument("--batch-size", type=int, default=64, help="Images per batch (default 64)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

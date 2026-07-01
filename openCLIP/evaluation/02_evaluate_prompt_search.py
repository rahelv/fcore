"""
004_evaluate.py — OpenCLIP Text Search Evaluation
==================================================
For every label in the DB:
  1. Embed the label name as text with CLIP
  2. Search against all image embeddings in the DB (cosine similarity)
  3. Check how many of the top-K results belong to the correct class

Metrics per class:
  - P@K  (Precision at K)  — correct / K, shown as fraction + %
  - R@K  (Recall at K)     — correct / total images of that class
  - AP   (Average Precision) — area under precision-recall curve
  - mAP  (mean AP across all classes) — headline number
  - Top false positives — which wrong classes appear most in top results

Usage
-----
    python 004_evaluate.py
    python 004_evaluate.py --cpu
    python 004_evaluate.py --top-k 5 10 20
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from open_clip import create_model_from_pretrained, get_tokenizer
from tqdm import tqdm

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
HF_MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP2-256"
OUTPUT_PATH = Path("/app/results_004.json")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",           "172.22.0.2"),
    "port":     os.getenv("DB_PORT",           "5432"),
    "dbname":   os.getenv("POSTGRES_DB",       "your_db"),
    "user":     os.getenv("POSTGRES_USER",     "your_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────
def load_clip(device: torch.device):
    print("Loading CLIP classification_model …")
    model, preprocess = create_model_from_pretrained(HF_MODEL_ID)
    model = model.to(device).eval()
    tokenizer = get_tokenizer(HF_MODEL_ID)
    print("CLIP ready.")
    return model, tokenizer


@torch.no_grad()
def embed_text(model, tokenizer, text: str, device: torch.device) -> np.ndarray:
    tokens   = tokenizer(text).to(device)
    features = model.encode_text(tokens)       # [1, 768]
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()    # [768]


# ──────────────────────────────────────────────────────────────────────────────
# DB
# ──────────────────────────────────────────────────────────────────────────────
def fetch_image_embeddings() -> tuple[list[str], list[str], np.ndarray]:
    """
    Returns:
        filepaths  — list of image filepaths
        labels     — list of label strings (one per image)
        matrix     — float32 [N_images, 768] l2-normalised
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.filepath, l.label, e.vector
                FROM   images i
                JOIN   embeddings e ON e.id = i.embedding_id
                LEFT JOIN labels l  ON l.id = i.label_id
                ORDER  BY i.id
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No image embeddings found in the database.")

    filepaths = [row[0] for row in rows]
    labels    = [row[1] for row in rows]
    matrix    = np.array([json.loads(row[2]) for row in rows], dtype=np.float32)

    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-8, None)

    print(f"Loaded {len(filepaths)} image embeddings from DB.")
    return filepaths, labels, matrix


def fetch_label_names() -> list[str]:
    """Fetch all unique label names from the DB."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT label FROM labels ORDER BY label")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────
def average_precision(relevant: np.ndarray) -> float:
    """
    Compute AP given a boolean array of hits in ranked order.
    relevant[i] = True if the i-th result is the correct class.
    """
    hits       = 0
    ap         = 0.0
    n_relevant = relevant.sum()
    if n_relevant == 0:
        return 0.0
    for i, hit in enumerate(relevant):
        if hit:
            hits += 1
            ap   += hits / (i + 1)
    return ap / n_relevant


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    model, tokenizer = load_clip(device)

    # Fetch all image embeddings + their labels from DB
    filepaths, img_labels, img_matrix = fetch_image_embeddings()
    img_labels_arr = np.array(img_labels)

    # Fetch all label names
    label_names = fetch_label_names()
    print(f"Evaluating {len(label_names)} labels\n")

    # Count total images per class (in DB)
    class_total = defaultdict(int)
    for lbl in img_labels:
        class_total[lbl] += 1

    ks = args.top_k  # e.g. [5, 10, 20]

    per_class = {}
    ap_scores = []

    for label in tqdm(label_names, desc="Text search eval", unit="label"):
        # Embed the label name as text
        text_vec = embed_text(model, tokenizer, label, device)  # [768]

        # Cosine similarity against all image embeddings
        sims = img_matrix @ text_vec  # [N_images]

        # Rank all images by similarity (descending)
        ranked_idx    = np.argsort(sims)[::-1]
        ranked_labels = img_labels_arr[ranked_idx]

        # Boolean relevance array
        relevant = ranked_labels == label

        # AP
        ap = average_precision(relevant)
        ap_scores.append(ap)

        n_class = class_total[label]

        # P@K and R@K for each K
        precision_at_k = {}
        recall_at_k    = {}
        false_positives_at_k = {}

        max_k = max(ks)
        top_k_labels = ranked_labels[:max_k]

        for k in ks:
            top_k    = ranked_labels[:k]
            n_correct = int((top_k == label).sum())
            precision_at_k[k] = {
                "correct": n_correct,
                "k": k,
                "fraction": f"{n_correct}/{k}",
                "percent": round(n_correct / k * 100, 1),
            }
            recall_at_k[k] = {
                "correct": n_correct,
                "total_in_db": n_class,
                "fraction": f"{n_correct}/{n_class}",
                "percent": round(n_correct / max(n_class, 1) * 100, 1),
            }

        # False positives in top max_k results
        fp_counts = defaultdict(int)
        for lbl in top_k_labels:
            if lbl != label:
                fp_counts[lbl] += 1
        top_fp = sorted(fp_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        per_class[label] = {
            "total_in_db": n_class,
            "ap": round(ap, 4),
            "precision_at_k": {str(k): v for k, v in precision_at_k.items()},
            "recall_at_k":    {str(k): v for k, v in recall_at_k.items()},
            "top_false_positives": [
                {"label": lbl, "count": cnt} for lbl, cnt in top_fp
            ],
        }

    mAP = float(np.mean(ap_scores))

    # ── PRINT ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  TEXT SEARCH EVALUATION RESULTS")
    print("═" * 70)
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  Labels: {len(label_names)}   Images in DB: {len(filepaths)}")
    print(f"  mAP   : {mAP*100:.2f}%")
    print("═" * 70)

    # header
    k_headers = "  ".join(f"P@{k:<6}  R@{k:<6}" for k in ks)
    print(f"\n{'Label':<35} {'AP':>6}  {k_headers}")
    print(f"{'-'*35} {'-'*6}  " + "  ".join([f"{'-'*8}  {'-'*8}" for _ in ks]))

    sorted_labels = sorted(per_class.keys(), key=lambda l: per_class[l]["ap"], reverse=True)

    for label in sorted_labels:
        d   = per_class[label]
        ap  = d["ap"]
        pk  = "  ".join(
            f"{d['precision_at_k'][str(k)]['fraction']:>8}  {d['recall_at_k'][str(k)]['fraction']:>8}"
            for k in ks
        )
        print(f"{label:<35} {ap*100:>5.1f}%  {pk}")

        fp = d["top_false_positives"]
        if fp:
            fp_str = ", ".join(f"{x['label']} ({x['count']}x)" for x in fp)
            print(f"  {'↳ false positives:':<20} {fp_str}")

    # ── SAVE JSON ────────────────────────────────────────────────────────────
    output = {
        "classification_model": HF_MODEL_ID,
        "top_k_values": ks,
        "total_images_in_db": len(filepaths),
        "total_labels": len(label_names),
        "mAP": round(mAP * 100, 2),
        "per_class": {lbl: per_class[lbl] for lbl in sorted_labels},
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"\nCopy out with:")
    print(f"  docker cp openclip-search_gui-1:/app/results_004.json /home/ubuntu/results_004.json")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu",    action="store_true", help="Force CPU")
    p.add_argument("--top-k",  type=int, nargs="+", default=[5, 10, 20],
                   help="K values for P@K and R@K (default: 5 10 20)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

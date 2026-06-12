"""
02_evaluate_prompt_search_512.py — Text Search Evaluation (512-dim, xlm-roberta-base-ViT-B-32)
===============================================================================================
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
    python 02_evaluate_prompt_search_512.py
    python 02_evaluate_prompt_search_512.py --cpu
    python 02_evaluate_prompt_search_512.py --top-k 5 10 20
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
from tqdm import tqdm

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "xlm-roberta-base-ViT-B-32"
PRETRAINED  = "laion5b_s13b_b90k"
OUTPUT_PATH = Path("/app/results_02_512.json")

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
    print("Loading CLIP model …")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model     = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    print("CLIP ready.")
    return model, tokenizer


@torch.no_grad()
def embed_text(model, tokenizer, text: str, device: torch.device) -> np.ndarray:
    tokens   = tokenizer(text).to(device)
    features = model.encode_text(tokens)       # [1, 512]
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()    # [512]


# ──────────────────────────────────────────────────────────────────────────────
# DB
# ──────────────────────────────────────────────────────────────────────────────
def fetch_image_embeddings() -> tuple[list[str], list[str], np.ndarray]:
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

    filepaths, img_labels, img_matrix = fetch_image_embeddings()
    img_labels_arr = np.array(img_labels)

    label_names = fetch_label_names()
    print(f"Evaluating {len(label_names)} labels\n")

    class_total = defaultdict(int)
    for lbl in img_labels:
        class_total[lbl] += 1

    ks = args.top_k

    per_class = {}
    ap_scores = []

    for label in tqdm(label_names, desc="Text search eval", unit="label"):
        text_vec = embed_text(model, tokenizer, label, device)  # [512]

        sims = img_matrix @ text_vec  # [N_images]

        ranked_idx    = np.argsort(sims)[::-1]
        ranked_labels = img_labels_arr[ranked_idx]

        relevant = ranked_labels == label

        ap = average_precision(relevant)
        ap_scores.append(ap)

        n_class  = class_total[label]
        max_k    = max(ks)
        top_k_labels = ranked_labels[:max_k]

        precision_at_k = {}
        recall_at_k    = {}

        for k in ks:
            top_k     = ranked_labels[:k]
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
    print("  TEXT SEARCH EVALUATION RESULTS  (512-dim · xlm-roberta-base-ViT-B-32)")
    print("═" * 70)
    print(f"  Model : {MODEL_NAME} / {PRETRAINED}")
    print(f"  Labels: {len(label_names)}   Images in DB: {len(filepaths)}")
    print(f"  mAP   : {mAP*100:.2f}%")
    print("═" * 70)

    k_headers = "  ".join(f"P@{k:<6}  R@{k:<6}" for k in ks)
    print(f"\n{'Label':<35} {'AP':>6}  {k_headers}")
    print(f"{'-'*35} {'-'*6}  " + "  ".join([f"{'-'*8}  {'-'*8}" for _ in ks]))

    sorted_labels = sorted(per_class.keys(), key=lambda l: per_class[l]["ap"], reverse=True)

    for label in sorted_labels:
        d  = per_class[label]
        ap = d["ap"]
        pk = "  ".join(
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
        "model": f"{MODEL_NAME} / {PRETRAINED}",
        "top_k_values": ks,
        "total_images_in_db": len(filepaths),
        "total_labels": len(label_names),
        "mAP": round(mAP * 100, 2),
        "per_class": {lbl: per_class[lbl] for lbl in sorted_labels},
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Copy out with:")
    print(f"  docker cp openclip-search_gui-1:/app/results_02_512.json /home/ubuntu/results_02_512.json")


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

"""
Cosplay Image Search — Web App
===============================
Tab 1 — Text Search:  type a query → find similar images from DB
Tab 2 — Image Search: upload or pick a random val image → top-5 closest labels from pgvector DB

Usage
-----
    python app.py          # runs on port 5000
    python app.py --port 8080
    python app.py --cpu
"""

import argparse
import base64
import io
import os
import random
from pathlib import Path

import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from torchvision import datasets, transforms

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
VAL_DIR = Path("/data/val")
HF_MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP2-256"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "172.22.0.2"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "your_db"),
    "user": os.getenv("POSTGRES_USER", "your_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
}

app = Flask(__name__, static_folder="static")


# ──────────────────────────────────────────────────────────────────────────────
# CLIP
# ──────────────────────────────────────────────────────────────────────────────
def load_clip(device: torch.device):
    print("Loading CLIP model …")
    clip_model, preprocess = create_model_from_pretrained(HF_MODEL_ID)
    clip_model = clip_model.to(device).eval()
    tokenizer = get_tokenizer(HF_MODEL_ID)
    print("CLIP ready.")
    return clip_model, preprocess, tokenizer


@torch.no_grad()
def embed_text_clip(text: str) -> np.ndarray:
    tokenizer = app.config["clip_tokenizer"]
    clip_model = app.config["clip_model"]
    device = app.config["device"]
    tokens = tokenizer(text).to(device)
    features = clip_model.encode_text(tokens)  # [1, 768]
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()  # [768]


@torch.no_grad()
def embed_image_clip(pil_img: Image.Image) -> np.ndarray:
    preprocess = app.config["clip_preprocess"]
    clip_model = app.config["clip_model"]
    device = app.config["device"]
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    features = clip_model.encode_image(tensor)  # [1, 768]
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()  # [768]


# ──────────────────────────────────────────────────────────────────────────────
# PGVECTOR SEARCH
# ──────────────────────────────────────────────────────────────────────────────
def search_images(vec: np.ndarray, top_k: int) -> list[dict]:
    """Text query → find similar images (used by text search tab)."""
    vec_str = str(vec.tolist())
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.filepath,
                       l.label,
                       1 - (e.vector <=> %s::vector) AS similarity
                FROM   images i
                JOIN   embeddings e ON e.id = i.embedding_id
                LEFT JOIN labels l  ON l.id = i.label_id
                ORDER  BY e.vector <=> %s::vector
                LIMIT  %s
                """,
                (vec_str, vec_str, top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for filepath, label, similarity in rows:
        try:
            with open(filepath, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            ext = Path(filepath).suffix.lower().lstrip(".")
            mime = {
                "jpg": "jpeg",
                "jpeg": "jpeg",
                "png": "png",
                "webp": "webp",
                "bmp": "bmp",
            }.get(ext, "jpeg")
            results.append(
                {
                    "filepath": filepath,
                    "label": label or "unknown",
                    "similarity": round(float(similarity), 4),
                    "image": f"data:image/{mime};base64,{img_b64}",
                }
            )
        except Exception as e:
            print(f"Could not load image {filepath}: {e}")
    return results


def search_labels(vec: np.ndarray, top_k: int = 5) -> list[dict]:
    """Image query → find closest label embeddings (used by image search tab)."""
    vec_list = vec.tolist()
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT l.label,
                       1 - (e.vector <=> %s::vector) AS score
                FROM   labels l
                JOIN   embeddings e ON l.embedding_id = e.id
                ORDER  BY e.vector <=> %s::vector
                LIMIT  %s
                """,
                (vec_list, vec_list, top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"label": row[0], "score": round(float(row[1]) * 100, 1)} for row in rows]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image, max_size=500) -> str:
    img = img.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def random_val_image() -> tuple[Image.Image, str]:
    val_ds = app.config["val_ds"]
    idx_to_class_ds = {v: k for k, v in val_ds.class_to_idx.items()}
    idx = random.randint(0, len(val_ds) - 1)
    _, label_idx = val_ds[idx]
    true_label = idx_to_class_ds[label_idx]
    pil_img = Image.open(Path(val_ds.imgs[idx][0])).convert("RGB")
    return pil_img, true_label


def run_image_search(pil_img: Image.Image, true_label=None) -> dict:
    vec = embed_image_clip(pil_img)
    results = search_labels(vec, top_k=5)
    return {
        "image": pil_to_b64(pil_img),
        "true_label": true_label,
        "results": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# — Text Search tab —
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 20))

    if not query:
        return jsonify({"error": "Empty query"}), 400

    vec = embed_text_clip(query)
    results = search_images(vec, top_k)
    return jsonify(results)


# — Image Search tab —
@app.route("/api/search/random")
def search_random():
    pil_img, true_label = random_val_image()
    return jsonify(run_image_search(pil_img, true_label=true_label))


@app.route("/api/search/upload", methods=["POST"])
def search_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    pil_img = Image.open(request.files["file"].stream).convert("RGB")
    return jsonify(run_image_search(pil_img, true_label=None))


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    clip_model, clip_preprocess, clip_tokenizer = load_clip(device)

    val_ds = datasets.ImageFolder(VAL_DIR, transform=transforms.ToTensor())
    print(f"Val set: {len(val_ds)} images across {len(val_ds.classes)} classes\n")

    app.config.update(
        val_ds=val_ds,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        device=device,
    )

    print(f"Open http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()

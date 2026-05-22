from pathlib import Path
import os
import base64

import numpy as np
import torch
import open_clip
import psycopg2
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "172.21.0.2"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "your_db"),
    "user": os.getenv("POSTGRES_USER", "your_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
}

# ── CLIP model ────────────────────────────────────────────────────────────────
model_name = "xlm-roberta-base-ViT-B-32"
pretrained = "laion5b_s13b_b90k"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=model_name,
    pretrained=pretrained,
)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer(model_name)


def embed_text(text: str) -> np.ndarray:
    tokens = tokenizer(text).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()


def get_db():
    return psycopg2.connect(**DB_CONFIG)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 20))

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Embed the query text
    vec = embed_text(query).tolist()

    conn = get_db()
    cur = conn.cursor()

    # pgvector cosine similarity search — joins back to get filepath + label
    cur.execute(
        """
        SELECT
            i.filepath,
            l.label,
            1 - (e.vector <=> %s::vector) AS similarity
        FROM images i
        JOIN embeddings e ON e.id = i.embedding_id
        LEFT JOIN labels l ON l.id = i.label_id
        ORDER BY e.vector <=> %s::vector
        LIMIT %s
    """,
        (str(vec), str(vec), top_k),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for filepath, label, similarity in rows:
        # Read image from disk and base64-encode it for the frontend
        try:
            with open(filepath, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = Path(filepath).suffix.lower().lstrip(".")
            mime_map = {
                "jpg": "jpeg",
                "jpeg": "jpeg",
                "png": "png",
                "webp": "webp",
                "bmp": "bmp",
            }
            mime = mime_map.get(ext, "jpeg")
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

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

from pathlib import Path
from queue import Queue
from threading import Thread
import os
import itertools

import numpy as np
import torch
import psycopg2
from PIL import Image, ImageFile
from open_clip import create_model_from_pretrained, get_tokenizer

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ

# PIL safety settings
Image.MAX_IMAGE_PIXELS = None  # allow large images without warning
ImageFile.LOAD_TRUNCATED_IMAGES = True  # handle truncated files gracefully

DATASET_ROOT = Path("/home/ubuntu/data/robust_dataset")
IGNORE_FOLDERS = {"originals"}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 128
NUM_WORKERS = 4

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "172.21.0.2"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "your_db"),
    "user": os.getenv("POSTGRES_USER", "your_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
}

# ViT-L-16-SigLIP2-256: produces 1024-dim vectors
# requires: open-clip-torch >= 2.31.0, timm >= 1.0.15
HF_MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP2-256"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = create_model_from_pretrained(HF_MODEL_ID)
model = model.to(device).eval()
tokenizer = get_tokenizer(HF_MODEL_ID)


def embed_text(text: str) -> np.ndarray:
    tokens = tokenizer(text).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)  # [1, 1024]
        features /= features.norm(dim=-1, keepdim=True)  # l2 normalization
    return features.cpu().numpy().flatten()  # [1024]


def embed_image(image) -> np.ndarray:
    img = (
        preprocess(
            image
        )  # resizes, center-crops and normalizes PIL image → [3, 256, 256]
        .unsqueeze(0)  # add batch dimension → [1, 3, 256, 256]
        .to(device)
    )
    with torch.no_grad():
        features = model.encode_image(img)  # [1, 1024]
        features /= features.norm(dim=-1, keepdim=True)  # l2 normalization
    return features.cpu().numpy().flatten()  # [1024]


def embed_images_batch(pil_images: list) -> np.ndarray:
    """Embed a batch of PIL images, return array of shape [N, 1024]."""
    tensors = [preprocess(img) for img in pil_images]
    batch = torch.stack(tensors).to(device)  # [N, 3, 256, 256]
    with torch.no_grad():
        features = model.encode_image(batch)  # [N, 1024]
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()  # [N, 1024]


# -- DATABASE FUNCTIONS --


def insert_embedding(cur, vec: np.ndarray) -> int:
    cur.execute(
        "INSERT INTO embeddings (vector) VALUES (%s) RETURNING id", (vec.tolist(),)
    )
    return cur.fetchone()[0]


def insert_label(cur, label: str, embedding_id: int) -> int:
    cur.execute(
        "INSERT INTO labels (label, embedding_id) VALUES (%s, %s) RETURNING id",
        (label, embedding_id),
    )
    return cur.fetchone()[0]


def insert_image(cur, filepath: str, label_id: int, embedding_id: int) -> None:
    cur.execute(
        "INSERT INTO images (filepath, label_id, embedding_id) VALUES (%s, %s, %s)",
        (filepath, label_id, embedding_id),
    )


# 1. IMAGES (PATHS)
matched = [
    path
    for path in itertools.chain.from_iterable(
        DATASET_ROOT.glob(f"**/*{ext}") for ext in SUPPORTED_EXTS
    )
    if not IGNORE_FOLDERS.intersection(path.parts)  # skip ignored folders
]
print(f"Found {len(matched)} images")


def load_image(path: Path):
    try:
        img = Image.open(path)
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA").convert("RGB")  # fix palette transparency warning
        else:
            img = img.convert("RGB")
        return path, img
    except Exception as e:
        print(f"  Skipping {path}: {e}")
        return path, None


def loader_worker(matched, queue, batch_size):
    batch = []
    for path in matched:
        result = load_image(path)
        if result[1] is not None:
            batch.append(result)
        if len(batch) == batch_size:
            queue.put(batch)
            batch = []  # frees RAM before loading next batch
    if batch:
        queue.put(batch)
    queue.put(None)  # signals done


queue = Queue(maxsize=4)  # max 4 batches in RAM at any time
Thread(target=loader_worker, args=(matched, queue, BATCH_SIZE), daemon=True).start()

# 2. LABELS
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

label_ids: dict[str, int] = {}  # folder_name -> label_id

for folder in DATASET_ROOT.iterdir():
    if not folder.is_dir() or folder.name in IGNORE_FOLDERS:
        continue

    vec = embed_text(folder.name)
    emb_id = insert_embedding(cur, vec)
    lbl_id = insert_label(cur, folder.name, emb_id)
    label_ids[folder.name] = lbl_id
    print(f"  Label '{folder.name}' → label_id={lbl_id}")

conn.commit()
print(f"Inserted {len(label_ids)} labels")

# 3. IMAGE EMBEDDINGS
batch_num = 0
while True:
    batch = queue.get()
    if batch is None:
        break

    paths, imgs = zip(*batch)

    # Embed the whole batch in one forward pass
    vecs = embed_images_batch(list(imgs))  # [N, 1024]

    for path, vec in zip(paths, vecs):
        label_name = path.parent.name
        label_id = label_ids.get(label_name)

        if label_id is None:
            print(
                f"  Warning: no label found for folder '{label_name}', skipping {path}"
            )
            continue

        emb_id = insert_embedding(cur, vec)
        insert_image(cur, str(path), label_id, emb_id)

    conn.commit()
    batch_num += 1
    print(
        f"  Inserted batch {batch_num} "
        f"({min(batch_num * BATCH_SIZE, len(matched))}/{len(matched)})"
    )

cur.close()
conn.close()
print("finito.")

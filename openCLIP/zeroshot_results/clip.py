"""
CLIP Image-Text Cosine Similarity
==================================
Uses open_clip to:
  1. Encode images → embedding vectors
  2. Encode text descriptions → embedding vectors
  3. Compute cosine similarity between each image & text pair

Usage:
  python3 clip.py                          # runs built-in demo
  python3 clip.py --images a.jpg b.png \
                  --texts "a cat" "a dog" # custom inputs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    """Download (once) and return the CLIP model, transforms, and tokenizer."""
    print(f"Loading CLIP model: {model_name} (pretrained={pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  → running on {device.upper()}\n")
    return model, preprocess, tokenizer, device


def embed_images(image_paths: list[str], model, preprocess, device) -> np.ndarray:
    """Return L2-normalised image embeddings, shape (N, D)."""
    images = torch.stack(
        [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    ).to(device)
    with torch.no_grad(), torch.autocast(device_type=device):
        feats = model.encode_image(images)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def embed_texts(texts: list[str], model, tokenizer, device) -> np.ndarray:
    """Return L2-normalised text embeddings, shape (M, D)."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad(), torch.autocast(device_type=device):
        feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every row of a and every row of b.
    Both inputs must already be L2-normalised → dot product = cosine sim.
    Returns shape (len(a), len(b)).
    """
    return a @ b.T


def print_similarity_table(image_paths, texts, sim_matrix):
    img_names = [Path(p).name for p in image_paths]
    col_w = max(len(t) for t in texts) + 2
    img_w = max(len(n) for n in img_names) + 2

    header = f"{'':>{img_w}}" + "".join(f"{t:>{col_w}}" for t in texts)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(img_names):
        row = f"{name:>{img_w}}"
        for j in range(len(texts)):
            score = sim_matrix[i, j]
            row += f"{score:>{col_w}.4f}"
        print(row)
    print()


def best_matches(image_paths, texts, sim_matrix):
    img_names = [Path(p).name for p in image_paths]
    print("Best text match per image:")
    for i, name in enumerate(img_names):
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = sim_matrix[i, best_j]
        print(f'  {name:30s} → "{texts[best_j]}"  (score: {best_score:.4f})')
    print()
    print("Best image match per text:")
    for j, text in enumerate(texts):
        best_i = int(np.argmax(sim_matrix[:, j]))
        best_score = sim_matrix[best_i, j]
        print(f'  "{text:30s}" → {img_names[best_i]}  (score: {best_score:.4f})')


# ── demo (no real images needed) ─────────────────────────────────────────────


def run_demo(model, preprocess, tokenizer, device):
    """
    Demo with synthetic solid-colour images and text prompts.
    Useful for verifying the pipeline without any dataset.
    """
    print("═" * 60)
    print("DEMO MODE  (synthetic images)")
    print("═" * 60 + "\n")

    import tempfile, os

    tmp_dir = tempfile.mkdtemp()

    colours = {
        "red_square.png": (220, 50, 50),
        "blue_square.png": (50, 80, 220),
        "green_square.png": (50, 180, 80),
    }
    paths = []
    for fname, rgb in colours.items():
        img = Image.new("RGB", (224, 224), color=rgb)
        path = os.path.join(tmp_dir, fname)
        img.save(path)
        paths.append(path)

    texts = [
        "a red square",
        "a blue square",
        "a green square",
        "a photo of a cat",
    ]

    img_embs = embed_images(paths, model, preprocess, device)
    txt_embs = embed_texts(texts, model, tokenizer, device)
    sim = cosine_similarity_matrix(img_embs, txt_embs)

    print("Cosine Similarity Matrix  (image rows × text columns)\n")
    print_similarity_table(paths, texts, sim)
    best_matches(paths, texts, sim)

    for p in paths:
        os.remove(p)
    os.rmdir(tmp_dir)


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="CLIP image-text cosine similarity")
    parser.add_argument("--images", nargs="+", help="Paths to image files")
    parser.add_argument("--texts", nargs="+", help="Text descriptions (quoted)")
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name")
    parser.add_argument(
        "--pretrained", default="openai", help="open_clip pretrained weights"
    )
    args = parser.parse_args()

    model, preprocess, tokenizer, device = load_clip(args.model, args.pretrained)

    if args.images and args.texts:
        missing = [p for p in args.images if not Path(p).exists()]
        if missing:
            sys.exit(f"Image(s) not found: {missing}")

        print("Embedding images …")
        img_embs = embed_images(args.images, model, preprocess, device)
        print("Embedding texts  …\n")
        txt_embs = embed_texts(args.texts, model, tokenizer, device)
        sim = cosine_similarity_matrix(img_embs, txt_embs)

        print("Cosine Similarity Matrix  (image rows × text columns)\n")
        print_similarity_table(args.images, args.texts, sim)
        best_matches(args.images, args.texts, sim)
    else:
        run_demo(model, preprocess, tokenizer, device)


if __name__ == "__main__":
    main()

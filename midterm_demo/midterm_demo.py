"""
Cosplay Costume Recognition — Demo Web App
==========================================
Single-image demo: upload your own image OR pick one randomly from the val set.
Both models predict on the same image side by side.

Usage
-----
    pip install flask torch torchvision Pillow

    python app.py \
        --model_a /home/ubuntu/data/models/resilient_sweep_32.pt \
        --model_b /home/ubuntu/data/models/resilient_sweep_32_pretrained.pt \
        --val_dir /home/ubuntu/data/robust_dataset_split/val

    Then open: http://localhost:5000
"""

import argparse
import base64
import io
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ──────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

app = Flask(__name__, static_folder="static")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: Path, device: torch.device):
    ckpt         = torch.load(checkpoint_path, map_location=device)
    num_classes  = ckpt["num_classes"]
    class_to_idx = ckpt["class_to_idx"]
    dropout_p    = ckpt.get("dropout_p", 0.0)

    idx_to_class = [None] * num_classes
    for name, idx in class_to_idx.items():
        idx_to_class[idx] = name

    model = resnet18(weights=None)
    in_features = model.fc.in_features
    if dropout_p > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, idx_to_class


@torch.no_grad()
def predict_topk(model, pil_img: Image.Image, idx_to_class: list, device, k=3):
    tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    k      = min(k, len(idx_to_class))
    top_p, top_i = torch.topk(probs, k)
    return [
        {"label": idx_to_class[i.item()], "confidence": round(p.item() * 100, 1)}
        for p, i in zip(top_p, top_i)
    ]


def pil_to_b64(img: Image.Image, max_size=500) -> str:
    img = img.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def run_both_models(pil_img: Image.Image, true_label=None):
    model_a   = app.config["model_a"]
    model_b   = app.config["model_b"]
    idx_to_a  = app.config["idx_to_class_a"]
    idx_to_b  = app.config["idx_to_class_b"]
    device    = app.config["device"]

    return {
        "image":      pil_to_b64(pil_img),
        "true_label": true_label,           # None for uploaded images
        "preds_a":    predict_topk(model_a, pil_img, idx_to_a, device),
        "preds_b":    predict_topk(model_b, pil_img, idx_to_b, device),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/random")
def random_val():
    """Pick one random image from the val set."""
    val_ds          = app.config["val_ds"]
    idx_to_class_ds = {v: k for k, v in val_ds.class_to_idx.items()}
    idx             = random.randint(0, len(val_ds) - 1)
    tensor, label_idx = val_ds[idx]
    true_label      = idx_to_class_ds[label_idx]
    pil_img         = Image.open(Path(val_ds.imgs[idx][0])).convert("RGB")
    return jsonify(run_both_models(pil_img, true_label=true_label))


@app.route("/api/upload", methods=["POST"])
def upload():
    """Run inference on a user-uploaded image."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file    = request.files["file"]
    pil_img = Image.open(file.stream).convert("RGB")
    return jsonify(run_both_models(pil_img, true_label=None))


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", type=Path, required=True)
    p.add_argument("--model_b", type=Path, required=True)
    p.add_argument("--val_dir", type=Path, required=True)
    p.add_argument("--port",    type=int,  default=5000)
    p.add_argument("--cpu",     action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print("Loading Model 32 …")
    model_a, idx_to_class_a = load_model(args.model_a, device)

    print("Loading Model 32 (Pretrained) …")
    model_b, idx_to_class_b = load_model(args.model_b, device)

    val_ds = datasets.ImageFolder(args.val_dir, transform=EVAL_TRANSFORM)
    print(f"Val set: {len(val_ds)} images across {len(val_ds.classes)} classes\n")

    app.config.update(
        val_ds         = val_ds,
        model_a        = model_a,
        model_b        = model_b,
        idx_to_class_a = idx_to_class_a,
        idx_to_class_b = idx_to_class_b,
        device         = device,
    )

    print(f"  Open http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()

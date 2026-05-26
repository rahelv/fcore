"""
Cosplay Costume Recognition — Model Comparison Demo
====================================================
Randomly samples images from the val set and shows side-by-side what
Model 32 and Model 32 (Pretrained Weights) predict for each image.

Usage
-----
    python compare_models.py \
        --model_a  /home/ubuntu/data/models/resilient_sweep_32.pt \
        --model_b  /home/ubuntu/data/models/resilient_sweep_32_pretrained.pt \
        --val_dir  /home/ubuntu/data/robust_dataset_split/val \
        --n        5 \
        --save     comparison.png

    # with a fixed random seed for reproducibility
    python compare_models.py ... --seed 42
"""

import argparse
import random
import textwrap
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LABEL_A = "Model 32"
LABEL_B = "Model 32\n(Pretrained)"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
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


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_topk(model, tensor: torch.Tensor, idx_to_class: list, device, k=3):
    tensor = tensor.to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    k      = min(k, len(idx_to_class))
    top_p, top_i = torch.topk(probs, k)
    return [(idx_to_class[i.item()], p.item()) for p, i in zip(top_p, top_i)]


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────
def plot_comparison(samples, save_path: Path | None):
    """
    samples: list of dicts with keys:
        image      : PIL.Image
        true_label : str
        preds_a    : [(class, conf), ...]
        preds_b    : [(class, conf), ...]
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — pip install matplotlib")
        return

    n        = len(samples)
    # each sample gets a row of 3 columns: image | bars_a | bars_b
    fig_h    = 3.2 * n
    fig, axes = plt.subplots(
        n, 3,
        figsize=(13, fig_h),
        gridspec_kw={"width_ratios": [1, 1.4, 1.4]},
    )
    if n == 1:
        axes = [axes]   # keep consistent indexing

    fig.patch.set_facecolor("#0d0d0d")

    # column headers
    col_titles = ["Image (Ground Truth)", LABEL_A, LABEL_B]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(
            title.replace("\n", " "), color="white",
            fontsize=11, fontweight="bold", pad=8,
        )

    for row, sample in enumerate(samples):
        ax_img, ax_a, ax_b = axes[row]

        # ── image ────────────────────────────────────────────────────────────
        ax_img.imshow(sample["image"])
        ax_img.set_facecolor("#0d0d0d")
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#333")
        label_short = textwrap.shorten(sample["true_label"], width=22, placeholder="…")
        ax_img.set_xlabel(
            f"✦ {label_short}", color="#facc15",
            fontsize=9, labelpad=6,
        )

        # ── bar chart helper ──────────────────────────────────────────────────
        def draw_bars(ax, preds, true_label):
            ax.set_facecolor("#1a1a1a")
            classes = [p[0] for p in preds]
            confs   = [p[1] for p in preds]
            y_pos   = list(range(len(classes)))

            colors = []
            for cls, conf in preds:
                if cls == true_label:
                    colors.append("#22c55e")   # green  = correct
                elif y_pos.index(preds.index((cls, conf))) == 0:
                    colors.append("#f97316")   # orange = top-1 wrong
                else:
                    colors.append("#4b5563")   # grey   = other

            bars = ax.barh(y_pos, confs, color=colors, height=0.55, zorder=2)
            ax.set_xlim(0, 1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [textwrap.shorten(c, width=20, placeholder="…") for c in classes],
                color="white", fontsize=8,
            )
            ax.invert_yaxis()
            ax.tick_params(colors="#aaa", labelsize=8)
            ax.set_xlabel("Confidence", color="#777", fontsize=8, labelpad=4)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.grid(axis="x", color="#2a2a2a", linewidth=0.5, zorder=0)

            for bar, conf in zip(bars, confs):
                ax.text(
                    min(conf + 0.02, 0.95),
                    bar.get_y() + bar.get_height() / 2,
                    f"{conf*100:.1f}%", va="center",
                    color="white", fontsize=8,
                )

        draw_bars(ax_a, sample["preds_a"], sample["true_label"])
        draw_bars(ax_b, sample["preds_b"], sample["true_label"])

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#22c55e", label="Correct class"),
        Patch(facecolor="#f97316", label="Top-1 (wrong)"),
        Patch(facecolor="#4b5563", label="Other"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, frameon=False,
        labelcolor="white", fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Model Comparison — Cosplay Costume Recognition",
        color="white", fontsize=14, fontweight="bold", y=1.005,
    )
    plt.tight_layout(h_pad=2.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n  ✓ Saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Side-by-side comparison of two cosplay recognition models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_a", type=Path, required=True,
                   help="Checkpoint for Model 32  (resilient_sweep_32.pt)")
    p.add_argument("--model_b", type=Path, required=True,
                   help="Checkpoint for Model 32 Pretrained  (resilient_sweep_32_pretrained.pt)")
    p.add_argument("--val_dir", type=Path, required=True,
                   help="Path to the val/ split folder")
    p.add_argument("--n",       type=int,  default=5,
                   help="Number of random images to sample (default: 5)")
    p.add_argument("--topk",    type=int,  default=3,
                   help="Top-k predictions per model (default: 3)")
    p.add_argument("--save",    type=Path, default=None,
                   help="Save the figure to this path instead of showing it")
    p.add_argument("--seed",    type=int,  default=None,
                   help="Random seed for reproducible sampling")
    p.add_argument("--cpu",     action="store_true",
                   help="Force CPU even if a GPU is available")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if args.seed is not None:
        random.seed(args.seed)

    # ── load both models ──────────────────────────────────────────────────────
    print(f"\nLoading {LABEL_A} …")
    model_a, idx_to_class_a = load_model(args.model_a, device)

    print(f"Loading {LABEL_B.replace(chr(10), ' ')} …")
    model_b, idx_to_class_b = load_model(args.model_b, device)

    # ── sample images from val set ────────────────────────────────────────────
    # Use ImageFolder just to get paths + true labels
    val_ds   = datasets.ImageFolder(args.val_dir, transform=EVAL_TRANSFORM)
    all_idxs = list(range(len(val_ds)))
    chosen   = random.sample(all_idxs, min(args.n, len(all_idxs)))

    idx_to_class_ds = {v: k for k, v in val_ds.class_to_idx.items()}

    print(f"\nSampled {len(chosen)} images from {args.val_dir}\n")
    print(f"{'─'*70}")
    print(f"  {'True label':<28}  {LABEL_A:<25}  {LABEL_B.replace(chr(10),' ')}")
    print(f"{'─'*70}")

    samples = []
    for idx in chosen:
        tensor, label_idx = val_ds[idx]
        tensor      = tensor.unsqueeze(0)
        true_label  = idx_to_class_ds[label_idx]

        # load original PIL image for display
        img_path = Path(val_ds.imgs[idx][0])
        pil_img  = Image.open(img_path).convert("RGB")

        preds_a = predict_topk(model_a, tensor, idx_to_class_a, device, args.topk)
        preds_b = predict_topk(model_b, tensor, idx_to_class_b, device, args.topk)

        top1_a = preds_a[0][0]
        top1_b = preds_b[0][0]
        mark_a = "✓" if top1_a == true_label else "✗"
        mark_b = "✓" if top1_b == true_label else "✗"

        print(
            f"  {true_label:<28}  "
            f"{mark_a} {top1_a:<23}  "
            f"{mark_b} {top1_b}"
        )

        samples.append({
            "image":      pil_img,
            "true_label": true_label,
            "preds_a":    preds_a,
            "preds_b":    preds_b,
        })

    print(f"{'─'*70}\n")

    # ── visualise ─────────────────────────────────────────────────────────────
    plot_comparison(samples, args.save)


if __name__ == "__main__":
    main()

"""
Zero-Shot OpenCLIP Evaluation for Cosplay Costume Recognition
=============================================================
Tries multiple prompt strategies and reports accuracy per class.

Usage:
    pip install open_clip_torch torch torchvision scikit-learn tqdm matplotlib

    python zeroshot_eval.py --data_dir /path/to/your/dataset
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import open_clip
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Labels ────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Aayla Secura from Star Wars",
    "Ariel as Human from The Little Mermaid",
    "Ariel as Mermaid from The Little Mermaid",
    "Batman from DC Comics",
    "D.Va from Overwatch",
    "Deadpool from Marvel",
    "Frieren from Frieren",
    "Furina from Genshin Impact",
    "Harley Quinn from DC Comics",
    "Hatsune Miku from Vocaloid",
    "Hu Tao from Genshin Impact",
    "Inosuke Hashibira from Demon Slayer",
    "Jinx from League of Legends",
    "Kafka from Honkai Star Rail",
    "Lucy MacLean from Fallout",
    "Raiden Shogun from Genshin Impact",
    "Sailor Moon from Sailor Moon",
    "Spider-Man from Marvel",
    "Superman from DC Comics",
    "Thor from Marvel",
    "Wolverine from Marvel",
]

# ── Prompt strategies to compare ──────────────────────────────────────────────

PROMPT_STRATEGIES = {

    "bare_label": [
        "{}"
    ],

    "cosplay_single": [
        "a cosplay of {}"
    ],

    "ensemble_5": [
        "a cosplay of {}",
        "a person cosplaying as {}",
        "a photo of someone dressed as {}",
        "a cosplayer wearing a {} costume",
        "a photo of {} cosplay",
    ],

    "ensemble_detailed": [
        "a cosplay of {}",
        "a person cosplaying as {}",
        "a photo of someone dressed as {}",
        "a cosplayer wearing a {} costume",
        "a realistic photo of a {} cosplay at a convention",
        "a full body photo of a {} cosplayer",
        "a person in a handmade {} costume",
        "a detailed {} cosplay photo",
    ],
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class CosplayDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def load_dataset(data_dir: str):
    data_dir = Path(data_dir)
    class_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.lower() != "originals"
    ])

    paths, int_labels = [], []
    found_classes = []

    for idx, cls_dir in enumerate(class_dirs):
        images = (
            list(cls_dir.glob("*.jpg")) +
            list(cls_dir.glob("*.jpeg")) +
            list(cls_dir.glob("*.png")) +
            list(cls_dir.glob("*.webp"))
        )
        if not images:
            continue
        found_classes.append(cls_dir.name)
        for img_path in images:
            paths.append(str(img_path))
            int_labels.append(idx)

    print(f"\nLoaded {len(paths)} images across {len(found_classes)} classes")
    for i, cls in enumerate(found_classes):
        count = int_labels.count(i)
        print(f"  {cls}: {count} images")

    return paths, int_labels, found_classes


# ── Core zero-shot logic ───────────────────────────────────────────────────────

@torch.no_grad()
def build_text_embeddings(model, tokenizer, class_names, prompts, device):
    """Average text embeddings across all prompt templates for each class."""
    all_embeddings = []
    for label in class_names:
        filled = [p.format(label) for p in prompts]
        tokens = tokenizer(filled).to(device)
        embs = model.encode_text(tokens)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        avg = embs.mean(dim=0)
        avg = avg / avg.norm()
        all_embeddings.append(avg)
    return torch.stack(all_embeddings)  # (num_classes, embed_dim)


@torch.no_grad()
def run_zero_shot(model, tokenizer, dataloader, class_names, prompts, device):
    text_embs = build_text_embeddings(model, tokenizer, class_names, prompts, device)

    all_preds, all_labels, all_scores = [], [], []

    for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
        images = images.to(device)
        img_embs = model.encode_image(images)
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

        sims = img_embs.float() @ text_embs.float().T  # (batch, num_classes)
        preds = sims.argmax(dim=-1).cpu().numpy()
        scores = sims.max(dim=-1).values.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_scores.extend(scores)

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return accuracy, all_preds, all_labels, all_scores


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_strategy_comparison(results: dict, save_path: str):
    strategies = list(results.keys())
    accuracies = [results[s]["accuracy"] for s in strategies]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(strategies, accuracies, color="#4f86c6", edgecolor="white", height=0.5)
    ax.axvline(x=0.83, color="#e05c5c", linestyle="--", linewidth=1.5, label="Your baseline (0.83)")
    ax.set_xlabel("Accuracy")
    ax.set_title("Zero-Shot Prompt Strategy Comparison")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend()

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_per_class_accuracy(preds, labels, class_names, strategy_name, save_path):
    short_names = [c.split(" from ")[0] for c in class_names]
    per_class_acc = []
    for i in range(len(class_names)):
        mask = np.array(labels) == i
        if mask.sum() == 0:
            per_class_acc.append(0)
        else:
            per_class_acc.append(np.mean(np.array(preds)[mask] == i))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4f86c6" if a >= 0.7 else "#e8a838" if a >= 0.4 else "#e05c5c"
              for a in per_class_acc]
    bars = ax.barh(short_names, per_class_acc, color=colors)
    ax.axvline(x=np.mean(per_class_acc), color="gray", linestyle="--",
               linewidth=1, label=f"Mean: {np.mean(per_class_acc):.1%}")
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Per-Class Accuracy — {strategy_name}")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(preds, labels, class_names, strategy_name, save_path):
    short = [c.split(" from ")[0] for c in class_names]
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {strategy_name}")

    # Annotate cells
    for i in range(len(short)):
        for j in range(len(short)):
            val = cm_norm[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root (one subfolder per class)")
    parser.add_argument("--model", type=str, default="ViT-L-14",
                        help="OpenCLIP model name (default: ViT-L-14)")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k",
                        help="Pretrained weights tag")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./zeroshot_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {args.model} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    # Load dataset
    paths, int_labels, class_names = load_dataset(args.data_dir)
    dataset = CosplayDataset(paths, int_labels, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Run all strategies
    results = {}
    best_strategy, best_acc = None, 0

    for strategy_name, prompts in PROMPT_STRATEGIES.items():
        print(f"\n── Strategy: {strategy_name} ({len(prompts)} prompt(s)) ──")
        acc, preds, labels, scores = run_zero_shot(
            model, tokenizer, dataloader, class_names, prompts, device
        )
        results[strategy_name] = {
            "accuracy": acc,
            "preds": preds,
            "labels": labels,
        }
        print(f"  Accuracy: {acc:.4f} ({acc:.1%})")

        if acc > best_acc:
            best_acc, best_strategy = acc, strategy_name

    # Summary table
    print("\n" + "═" * 50)
    print(f"{'Strategy':<25} {'Accuracy':>10}")
    print("─" * 50)
    for name, r in results.items():
        marker = " ← best" if name == best_strategy else ""
        print(f"{name:<25} {r['accuracy']:>9.1%}{marker}")
    print(f"{'Your baseline':<25} {'83.0%':>10}")
    print("═" * 50)

    # Plots
    print("\nGenerating plots...")
    plot_strategy_comparison(results, str(output_dir / "strategy_comparison.png"))

    # Detailed plots for best strategy
    best = results[best_strategy]
    print(f"\nDetailed analysis for best strategy: {best_strategy}")
    plot_per_class_accuracy(
        best["preds"], best["labels"], class_names, best_strategy,
        str(output_dir / "per_class_accuracy.png")
    )
    plot_confusion_matrix(
        best["preds"], best["labels"], class_names, best_strategy,
        str(output_dir / "confusion_matrix.png")
    )

    # Full classification report for best strategy
    print(f"\nClassification report ({best_strategy}):")
    short_names = [c.split(" from ")[0] for c in class_names]
    print(classification_report(best["labels"], best["preds"], target_names=short_names))


if __name__ == "__main__":
    main()

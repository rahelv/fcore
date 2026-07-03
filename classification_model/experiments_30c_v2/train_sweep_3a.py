"""
train_sweep_3a.py — Phase 1: Core Training Dynamics (30 classes)

Goal
────
Find the optimal learning rate, weight decay, batch size, and crop scale
BEFORE introducing any regularisation or augmentation tuning.  Every run
uses the same minimal, fixed augmentation so that differences in val/acc
reflect optimiser configuration only — not regularisation strength.

What is swept (4 parameters)
─────────────────────────────
  lr             most impactful HP; log-uniform to search across scales
  weight_decay   interacts tightly with lr in AdamW; 1e-3 is restored
                 (most common in S1 top-15; mistakenly dropped in S2)
  batch_size     S2 showed BS=32 in 15/15 top runs, but this is likely
                 confounded: high-regularisation configs happened to use
                 BS=64 and performed worse due to epoch-count bias.
                 Phase 1 (no regularisation) gives a clean test.
  crop_scale_min S2 showed 0.80 dominant, but again mixed with
                 regularisation configs. Confirm with clean runs.
                 0.50 dropped — clearly worse in S2.

What is fixed
─────────────
  Augmentation:  minimal standard only.
    RandomResizedCrop (scale swept), RandomHorizontalFlip, mild fixed
    ColorJitter (brightness/contrast 0.2, saturation 0.1, hue 0.05).
    ── excluded because they are regularisation techniques ──
    RandomRotation, RandomErasing, strong jitter, RandomGrayscale,
    GaussianBlur, RandomPerspective  →  all deferred to Phase 2.

  Regularisation: none.
    dropout = 0.0, label_smoothing = 0.0

  Optimiser: AdamW (best in both sweeps, not re-tested)
  Scheduler: ReduceLROnPlateau (patience 5, factor 0.5)

Discrete search space: 3 (wd) × 2 (bs) × 2 (crop) = 12 combos + continuous lr
→ 40 Bayesian runs gives comfortable coverage.

Usage
─────
    CUDA_VISIBLE_DEVICES=0 python3 train_sweep_3a.py
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import wandb

# ─────────────────────────────────────────────
# FIXED CONSTANTS
# ─────────────────────────────────────────────
DATA_DIR    = Path("/home/ubuntu/data/robust_dataset_split_30c_v2")
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
TEST_DIR    = DATA_DIR / "test"
LABELS_PATH = DATA_DIR / "labels.json"

SAVE_DIR    = Path("/home/ubuntu/data/models")
IMAGE_SIZE  = 224
NUM_WORKERS = min(4, os.cpu_count() or 1)
SEED        = 42
MAX_EPOCHS  = 80
MIN_EPOCHS  = 20
PATIENCE    = 10   # epochs without val_acc improvement → stop

PROJECT = "fcore_hyperparam_sweep_3a"
ENTITY  = "rahelvalerie-universit-t-basel"

# ─────────────────────────────────────────────
# SWEEP CONFIGURATION
# ─────────────────────────────────────────────
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val/accuracy",
        "goal": "maximize",
    },
    "parameters": {

        # ── optimiser ──────────────────────────────────────────────────────
        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-4,
            "max": 3e-3,
            # Tightened from S2 [5e-4, 8e-3]:
            #   top S2 runs clustered below 2e-3; autumn-sweep-3 crashed at
            #   lr=6.4e-3, so the upper end is removed.
        },
        "weight_decay": {
            "values": [1e-5, 1e-4, 1e-3],
            # 1e-3 restored: most common value in S1 top-15.
            # S2 only tested {1e-4, 1e-5}, biasing results toward low WD.
        },
        "batch_size": {
            "values": [32, 64],
        },

        # ── preprocessing ──────────────────────────────────────────────────
        "crop_scale_min": {
            "values": [0.65, 0.80],
            # 0.50 dropped (clearly dominated by 0.80 in S2).
            # 0.65 kept to confirm the 0.80 signal is real, not correlated noise.
        },
    },
}


# ─────────────────────────────────────────────
# BUILD TRANSFORMS
# ─────────────────────────────────────────────
def build_transforms(cfg):
    """
    Minimal, fixed augmentation for Phase 1.

    Included — standard preprocessing, not regularisation:
      convert("RGB")       ensures palette/transparency images don't crash
      RandomResizedCrop    scale=(crop_scale_min, 1.0), swept
      RandomHorizontalFlip always on (13/15 top S1 runs)
      ColorJitter (mild)   brightness/contrast 0.2, saturation 0.1, hue 0.05

    Excluded — regularisation techniques, deferred to Phase 2:
      RandomErasing, strong ColorJitter, RandomRotation, RandomGrayscale,
      GaussianBlur, RandomPerspective
    """
    train_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(cfg.crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tf, eval_tf


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def build_loaders(train_tf, eval_tf, batch_size):
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    assert (
        train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx
    ), "Class mappings differ between splits!"

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes, device):
    """ResNet-18 from scratch.  No dropout in Phase 1."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# ─────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss    = 0.0
    running_correct = 0
    total           = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping — prevents NaN loss crashes (cf. autumn-sweep-3
                # in S2 which crashed at lr=6.4e-3 without clipping).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            preds            = outputs.argmax(dim=1)
            batch_size       = labels.size(0)
            running_loss    += loss.item() * batch_size
            running_correct += (preds == labels).sum().item()
            total           += batch_size

    epoch_loss = running_loss    / total if total > 0 else 0.0
    epoch_acc  = running_correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


# ─────────────────────────────────────────────
# PER-CLASS EVALUATION
# ─────────────────────────────────────────────
def evaluate_per_class(model, loader, idx_to_class, num_classes, device):
    model.eval()
    correct_per_class = [0] * num_classes
    total_per_class   = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds  = model(images).argmax(dim=1)

            for label, pred in zip(labels, preds):
                i = label.item()
                total_per_class[i]   += 1
                correct_per_class[i] += int(label == pred)

    return {
        idx_to_class[i]: {
            "correct":  correct_per_class[i],
            "total":    total_per_class[i],
            "accuracy": correct_per_class[i] / total_per_class[i]
                        if total_per_class[i] > 0 else 0.0,
        }
        for i in range(num_classes)
    }


# ─────────────────────────────────────────────
# TRAIN — called once per sweep run
# ─────────────────────────────────────────────
def train():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init() as run:
        cfg = run.config

        train_tf, eval_tf = build_transforms(cfg)
        train_loader, val_loader, test_loader, class_to_idx = build_loaders(
            train_tf, eval_tf, cfg.batch_size
        )
        num_classes  = len(class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

        model     = build_model(num_classes, device)
        criterion = nn.CrossEntropyLoss()   # no label smoothing in Phase 1

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        run.watch(model, log_freq=50)

        # Log all fixed choices so they appear in the W&B run table
        wandb.config.update({
            "optimizer":       "adamw",
            "scheduler":       "plateau",
            "dropout_p":       0.0,
            "label_smoothing": 0.0,
            "aug_hflip":       True,
            "aug_jitter":      "mild_fixed",
            "aug_rotation":    False,
            "aug_erasing":     False,
            "aug_grayscale":   False,
            "aug_blur":        False,
            "aug_perspective": False,
            "phase":           "1_core_dynamics",
        }, allow_val_change=True)

        best_val_acc          = float("-inf")
        epochs_no_improvement = 0
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SAVE_DIR / f"{run.id}_best.pt"

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, None, device
            )

            scheduler.step(val_loss)

            train_val_gap = train_acc - val_acc
            wandb.log({
                "epoch":          epoch,
                "train/loss":     train_loss,
                "train/accuracy": train_acc,
                "val/loss":       val_loss,
                "val/accuracy":   val_acc,
                "train_val_gap":  train_val_gap,
                "lr":             optimizer.param_groups[0]["lr"],
            })

            print(
                f"[{run.name}] Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"train={train_acc:.3f}  val={val_acc:.3f}  "
                f"gap={train_val_gap:.3f}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_acc > best_val_acc:
                best_val_acc          = val_acc
                epochs_no_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_to_idx":     class_to_idx,
                        "num_classes":      num_classes,
                        "image_size":       IMAGE_SIZE,
                        "dropout_p":        0.0,
                        "sweep":            "3a",
                        "phase":            "1_core_dynamics",
                    },
                    save_path,
                )
            else:
                epochs_no_improvement += 1

            if epoch >= MIN_EPOCHS:
                if train_acc >= 1.0:
                    print("Train accuracy 100% — stopping early (perfect overfit).")
                    break
                if epochs_no_improvement >= PATIENCE:
                    print(f"No val improvement for {PATIENCE} epochs — stopping.")
                    break

        # ── final test evaluation ──────────────────────────────────────────
        ckpt       = torch.load(save_path, map_location=device)
        best_model = build_model(num_classes, device)
        best_model.load_state_dict(ckpt["model_state_dict"])

        test_loss, test_acc = run_epoch(
            best_model, test_loader, criterion, None, device
        )
        per_class = evaluate_per_class(
            best_model, test_loader, idx_to_class, num_classes, device
        )

        wandb.log({
            "test/loss":     test_loss,
            "test/accuracy": test_acc,
            "best_val_acc":  best_val_acc,
            **{
                f"test_per_class/{cls}": stats["accuracy"]
                for cls, stats in per_class.items()
            },
        })

        print(f"\n── Best val: {best_val_acc:.4f}  |  Test: {test_acc:.4f} ──")
        for cls, stats in per_class.items():
            print(
                f"  {cls}: {stats['correct']}/{stats['total']}"
                f"  acc={stats['accuracy']:.3f}"
            )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project=PROJECT, entity=ENTITY)
    wandb.agent(sweep_id, function=train, count=40)


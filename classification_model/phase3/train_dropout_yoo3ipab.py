"""
train_dropout_experiment.py — Same config as a promising run, but with dropout

Reads hyperparameters from an existing checkpoint (so you don't have to
retype them), then trains from scratch with one or more dropout values added.
The checkpoint weights are NOT loaded — only the config is reused.

Usage
─────
  Fill in CHECKPOINT_PATH below, then run on 2 GPUs in parallel:
    CUDA_VISIBLE_DEVICES=0 python3 train_dropout_yoo3ipab.py --worker 0
    CUDA_VISIBLE_DEVICES=1 python3 train_dropout_yoo3ipab.py --worker 1
  Worker 0 → dropout=0.0 (original config), worker 1 → dropout=0.2
"""

import argparse
import os
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import wandb

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────
CHECKPOINT_PATH = "/home/ubuntu/data/models/fcore_hyperparam_sweep_3b_cutmix/fcore_hyperparam_sweep_3b_cutmix/yoo3ipab_best.pt"
DROPOUT_VALUES  = [0.0, 0.2]   # 0.0 = exact original config, 0.2 = dropout added

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MAX_EPOCHS  = 200
MIN_EPOCHS  = 60
PATIENCE    = 30

DATA_DIR    = Path("/home/ubuntu/data/robust_dataset_split_30c_v2")
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
TEST_DIR    = DATA_DIR / "test"

SAVE_DIR    = Path("/home/ubuntu/data/models")
PROJECT     = "fcore_hyperparam_sweep_3b_cutmix"
ENTITY      = "rahelvalerie-universit-t-basel"
IMAGE_SIZE  = 224
NUM_WORKERS = min(4, os.cpu_count() or 1)
SEED        = 42

# ─────────────────────────────────────────────
# BUILD TRANSFORMS
# ─────────────────────────────────────────────
def build_transforms(cfg):
    jitter_params = {
        "mild":   dict(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        "strong": dict(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    }[cfg.jitter_strength]

    pre_tensor = [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(**jitter_params),
    ]
    if cfg.rotation_degrees > 0:
        pre_tensor.append(transforms.RandomRotation(degrees=cfg.rotation_degrees))

    to_tensor = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    post_tensor = []
    if cfg.aug_erasing:
        post_tensor.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)))

    train_tf = transforms.Compose(pre_tensor + to_tensor + post_tensor)
    eval_tf  = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def build_loaders(train_tf, eval_tf):
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    assert (
        train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx
    ), "Class mappings differ between splits!"

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=NUM_WORKERS, pin_memory=pin)
    return (
        DataLoader(train_ds, batch_size=32, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=32, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=32, shuffle=False, **kw),
        train_ds.class_to_idx,
    )


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(dropout_p, num_classes, device):
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    if dropout_p > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


# ─────────────────────────────────────────────
# CUTMIX
# ─────────────────────────────────────────────
def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_batch(images, labels, alpha):
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(W, H, lam)
    images[:, :, bby1:bby2, bbx1:bbx2] = images[idx, :, bby1:bby2, bbx1:bbx2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return images, labels, labels[idx], lam


# ─────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, cutmix_alpha=0.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, running_correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train and cutmix_alpha > 0.0:
                images, la, lb, lam = cutmix_batch(images, labels, cutmix_alpha)
                out  = model(images)
                loss = lam * criterion(out, la) + (1.0 - lam) * criterion(out, lb)
            else:
                out  = model(images)
                loss = criterion(out, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            bs               = labels.size(0)
            running_loss    += loss.item() * bs
            running_correct += (out.argmax(1) == labels).sum().item()
            total           += bs

    return running_loss / total, running_correct / total


# ─────────────────────────────────────────────
# PER-CLASS EVALUATION
# ─────────────────────────────────────────────
def evaluate_per_class(model, loader, idx_to_class, num_classes, device):
    model.eval()
    correct = [0] * num_classes
    totals  = [0] * num_classes
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds  = model(images).argmax(1)
            for lbl, pred in zip(labels, preds):
                i = lbl.item()
                totals[i]  += 1
                correct[i] += int(lbl == pred)
    return {
        idx_to_class[i]: {
            "correct":  correct[i],
            "total":    totals[i],
            "accuracy": correct[i] / totals[i] if totals[i] > 0 else 0.0,
        }
        for i in range(num_classes)
    }

# ─────────────────────────────────────────────
# RUN ONE EXPERIMENT
# ─────────────────────────────────────────────
def run_experiment(cfg, dropout_p, original_run_id, class_to_idx, device):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*60}")
    print(f"dropout={dropout_p}  (config from {original_run_id}, training from scratch)")
    print(f"{'='*60}")

    train_tf, eval_tf = build_transforms(cfg)
    train_loader, val_loader, test_loader, class_to_idx = build_loaders(train_tf, eval_tf)
    num_classes  = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model     = build_model(dropout_p, num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    project_dir = SAVE_DIR / PROJECT
    project_dir.mkdir(parents=True, exist_ok=True)
    save_path = project_dir / f"{original_run_id}_dropout{int(dropout_p*100)}_scratch_best.pt"

    with wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=f"{original_run_id}_dropout{int(dropout_p*100)}",
        config={
            "original_run_id":  original_run_id,
            "phase":            "dropout_experiment",
            "from_scratch":     True,
            "dropout_p":        dropout_p,
            "lr":               cfg.lr,
            "weight_decay":     cfg.weight_decay,
            "label_smoothing":  cfg.label_smoothing,
            "jitter_strength":  cfg.jitter_strength,
            "rotation_degrees": cfg.rotation_degrees,
            "aug_erasing":      cfg.aug_erasing,
            "cutmix_alpha":     cfg.cutmix_alpha,
            "crop_scale_min":   0.65,
            "batch_size":       32,
            "max_epochs":       MAX_EPOCHS,
        },
    ):
        best_val_acc          = float("-inf")
        epochs_no_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device,
                cutmix_alpha=cfg.cutmix_alpha,
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, None, device,
            )
            scheduler.step(val_loss)

            wandb.log({
                "epoch":          epoch,
                "train/loss":     train_loss,
                "train/accuracy": train_acc,
                "val/loss":       val_loss,
                "val/accuracy":   val_acc,
                "train_val_gap":  train_acc - val_acc,
                "lr":             optimizer.param_groups[0]["lr"],
            })

            print(
                f"  Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"train={train_acc:.3f}  val={val_acc:.3f}  "
                f"gap={train_acc - val_acc:.3f}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_acc > best_val_acc:
                best_val_acc          = val_acc
                epochs_no_improvement = 0
                torch.save(
                    {
                        "model_state_dict":     model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch":                epoch,
                        "best_val_acc":         best_val_acc,
                        "class_to_idx":         class_to_idx,
                        "num_classes":          num_classes,
                        "image_size":           IMAGE_SIZE,
                        "phase":                "dropout_experiment",
                        "original_run_id":      original_run_id,
                        "dropout_p":            dropout_p,
                        "lr_initial":           cfg.lr,
                        "weight_decay":         cfg.weight_decay,
                        "label_smoothing":      cfg.label_smoothing,
                        "jitter_strength":      cfg.jitter_strength,
                        "rotation_degrees":     cfg.rotation_degrees,
                        "aug_erasing":          cfg.aug_erasing,
                        "cutmix_alpha":         cfg.cutmix_alpha,
                        "crop_scale_min":       0.65,
                        "batch_size":           32,
                    },
                    save_path,
                )
                print(f"  ✓ New best: {best_val_acc:.4f}")
            else:
                epochs_no_improvement += 1

            if epoch >= MIN_EPOCHS:
                if train_acc >= 1.0:
                    print("  Train 100% — stopping (perfect overfit).")
                    break
                if epochs_no_improvement >= PATIENCE:
                    print(f"  No improvement for {PATIENCE} epochs — stopping.")
                    break

        # ── test eval ────────────────────────────────────────────────
        ckpt_best  = torch.load(save_path, map_location=device)
        best_model = build_model(dropout_p, num_classes, device)
        best_model.load_state_dict(ckpt_best["model_state_dict"])

        test_loss, test_acc = run_epoch(best_model, test_loader, criterion, None, device)
        per_class = evaluate_per_class(best_model, test_loader, idx_to_class, num_classes, device)

        wandb.log({
            "test/loss":     test_loss,
            "test/accuracy": test_acc,
            "best_val_acc":  best_val_acc,
            **{f"test_per_class/{cls}": s["accuracy"] for cls, s in per_class.items()},
        })

        print(f"\n── dropout={dropout_p}  best val: {best_val_acc:.4f}  |  test: {test_acc:.4f} ──")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker", type=int, choices=[0, 1], required=True,
        help="0 → dropout=0.0 (original config)   |   1 → dropout=0.2",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Worker: {args.worker}")

    # Read config from checkpoint (weights ignored)
    ckpt_path       = Path(CHECKPOINT_PATH)
    original_run_id = ckpt_path.stem.replace("_best", "")
    ckpt            = torch.load(ckpt_path, map_location="cpu")

    cfg = types.SimpleNamespace(
        lr               = ckpt["lr_initial"],
        weight_decay     = ckpt["weight_decay"],
        label_smoothing  = ckpt["label_smoothing"],
        jitter_strength  = ckpt["jitter_strength"],
        rotation_degrees = ckpt["rotation_degrees"],
        aug_erasing      = ckpt["aug_erasing"],
        cutmix_alpha     = ckpt.get("cutmix_alpha", 0.0),
    )

    print(f"Config from: {original_run_id}")
    print(f"  lr={cfg.lr:.2e}  wd={cfg.weight_decay}  ls={cfg.label_smoothing}")
    print(f"  jitter={cfg.jitter_strength}  rotation={cfg.rotation_degrees}  "
          f"erasing={cfg.aug_erasing}  cutmix_alpha={cfg.cutmix_alpha}")

    dp = DROPOUT_VALUES[args.worker]
    print(f"  dropout={dp}")

    run_experiment(cfg, dp, original_run_id, ckpt["class_to_idx"], device)

    print("\nDone.")

"""
train_sweep_3b.py — Phase 2: Regularisation & Augmentation (30 classes)

Goal
────
Find the optimal regularisation and augmentation configuration using the
optimiser settings locked in from Phase 1.  Runs are longer (120 epochs,
patience 15) so that regularisation effects have time to materialise —
the main failure mode of the previous sweeps was cutting regularised runs
short before they could show their benefit.

What is fixed (from Phase 1 results)
──────────────────────────────────────
  crop_scale_min = 0.65   Phase 1: 11/15 top runs; clear signal in clean conditions
  batch_size     = 32     Phase 1: slight edge (8/15); smaller BS adds gradient noise
  aug_hflip      = True   consistent signal across all sweeps
  aug_grayscale  = False  colour is discriminative for costumes
  aug_blur       = False  negative signal in sweep 1
  aug_perspective= False  no signal across sweeps
  optimizer      = AdamW  dominant in sweep 1 top-15; not re-tested
  scheduler      = ReduceLROnPlateau (patience 5, factor 0.5)

What is swept (7 parameters)
──────────────────────────────
  lr               Phase 1 top-15 clustered [3e-4, 1.7e-3]; upper bound
                   extended slightly to 3e-3 — regularisation allows higher LR.
  weight_decay     1e-5 dropped (weak in Phase 1: 3/15); 1e-4 and 1e-3 tied (6/15 each).
  dropout_p        0.0 kept — lets search confirm whether dropout helps on top
                   of other regularisation.  Previous sweeps had epoch-count bias
                   against dropout; longer runs here give it a fair chance.
  label_smoothing  0.1 dominated sweep 1 (14/15 top runs).  Range extended to
                   0.20 since longer runs may benefit from stronger smoothing.
  jitter_strength  "mild" = Phase 1 fixed values (b/c 0.2, sat 0.1, hue 0.05).
                   "strong" = b/c 0.4, sat 0.3, hue 0.05 — hue kept small because
                   colour is discriminative for costume recognition.
  rotation_degrees 0° included to let search confirm rotation helps.
                   20° included despite costume orientation concern — Bayesian
                   search will reject it if it hurts.
  aug_erasing      Disabled in sweep 2 (never fairly tested on 30 classes).
                   Slight positive signal in sweep 1.

Discrete search space:
  2 (wd) × 3 (dropout) × 4 (ls) × 2 (jitter) × 3 (rotation) × 2 (erasing) = 288 combos
  + continuous lr  →  50 Bayesian runs (25 per agent) covers this well.

Usage
─────
    python3 create_sweep.py          # run once — prints sweep ID
    CUDA_VISIBLE_DEVICES=0 python3 run_agent.py <sweep_id> 25
    CUDA_VISIBLE_DEVICES=1 python3 run_agent.py <sweep_id> 25
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
DATA_DIR    = Path("/home/ubuntu/data/robust_dataset_split_2")
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
TEST_DIR    = DATA_DIR / "test"
LABELS_PATH = DATA_DIR / "labels.json"

SAVE_DIR    = Path("/home/ubuntu/data/models")
IMAGE_SIZE  = 224
NUM_WORKERS = min(4, os.cpu_count() or 1)
SEED        = 42
MAX_EPOCHS  = 120   # longer than Phase 1 — regularisation needs time to show benefit
MIN_EPOCHS  = 40    # don't early-stop before this
PATIENCE    = 15    # increased from 10 — regularised runs converge more slowly

PROJECT = "fcore_hyperparam_sweep_3b"
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
            # Phase 1 top-15 clustered [3e-4, 1.7e-3], mean ~7e-4.
            # Upper bound extended slightly vs Phase 1 because regularisation
            # allows a higher effective LR.
        },
        "weight_decay": {
            "values": [1e-4, 1e-3],
            # 1e-5 dropped — only 3/15 top runs in Phase 1.
            # 1e-4 and 1e-3 tied at 6/15 each; both kept.
        },

        # ── regularisation ─────────────────────────────────────────────────
        "dropout_p": {
            "values": [0.0, 0.2, 0.4],
            # 0.0 included: confirms whether dropout adds value on top of
            # label smoothing + WD + augmentation.
            # Previous sweeps were biased against dropout due to short runs;
            # 120 epochs gives it a fair evaluation.
        },
        "label_smoothing": {
            "values": [0.05, 0.10, 0.15, 0.20],
            # 0.1 dominated sweep 1 (14/15 top runs).
            # 0.20 added: longer runs may benefit from stronger smoothing.
        },

        # ── augmentation ───────────────────────────────────────────────────
        "jitter_strength": {
            "values": ["mild", "strong"],
            # mild:   brightness/contrast 0.2, saturation 0.1, hue 0.05
            # strong: brightness/contrast 0.4, saturation 0.3, hue 0.05
            # Hue kept small in both — colour is discriminative for costumes.
        },
        "rotation_degrees": {
            "values": [0, 10, 20],
            # 0 included to confirm rotation is beneficial.
            # 20° may be too aggressive for upright costumes; Bayesian search
            # will reject it if it hurts.
        },
        "aug_erasing": {
            "values": [True, False],
            # Never fairly tested on 30 classes (disabled in sweep 2).
            # Slight positive signal in sweep 1.
        },
    },
}


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
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),  # fixed from Phase 1
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(**jitter_params),
    ]

    if cfg.rotation_degrees > 0:
        pre_tensor.append(transforms.RandomRotation(degrees=cfg.rotation_degrees))

    to_tensor = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    post_tensor = []
    if cfg.aug_erasing:
        post_tensor.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)))

    train_tf = transforms.Compose(pre_tensor + to_tensor + post_tensor)

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
def build_loaders(train_tf, eval_tf):
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    assert (
        train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx
    ), "Class mappings differ between splits!"

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,         # batch size fixed from Phase 1
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(cfg, num_classes, device):
    model = resnet18(weights=None)
    in_features = model.fc.in_features

    if cfg.dropout_p > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=cfg.dropout_p),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

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
            train_tf, eval_tf
        )
        num_classes  = len(class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

        model     = build_model(cfg, num_classes, device)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        run.watch(model, log_freq=50)

        # Log fixed choices so they appear in the W&B run table
        wandb.config.update({
            "optimizer":       "adamw",
            "scheduler":       "plateau",
            "batch_size":      32,
            "crop_scale_min":  0.65,
            "aug_hflip":       True,
            "aug_grayscale":   False,
            "aug_blur":        False,
            "aug_perspective": False,
            "phase":           "2_regularisation",
        }, allow_val_change=True)

        best_val_acc          = float("-inf")
        best_epoch            = 0
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
                best_epoch            = epoch
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
                        "sweep":                "3b",
                        "phase":                "2_regularisation",
                        # all HPs — so you know exactly what to reproduce
                        "lr":                   cfg.lr,
                        "weight_decay":         cfg.weight_decay,
                        "dropout_p":            cfg.dropout_p,
                        "label_smoothing":      cfg.label_smoothing,
                        "jitter_strength":      cfg.jitter_strength,
                        "rotation_degrees":     cfg.rotation_degrees,
                        "aug_erasing":          cfg.aug_erasing,
                        "crop_scale_min":       0.65,
                        "batch_size":           32,
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

        print(f"\nBest val: {best_val_acc:.4f} at epoch {best_epoch}")

        # ── final test evaluation ──────────────────────────────────────────
        ckpt       = torch.load(save_path, map_location=device)
        best_model = build_model(cfg, num_classes, device)
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
            "best_epoch":    best_epoch,
            **{
                f"test_per_class/{cls}": stats["accuracy"]
                for cls, stats in per_class.items()
            },
        })

        print(f"── Best val: {best_val_acc:.4f}  |  Test: {test_acc:.4f} ──")
        for cls, stats in per_class.items():
            print(
                f"  {cls}: {stats['correct']}/{stats['total']}"
                f"  acc={stats['accuracy']:.3f}"
            )

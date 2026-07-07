"""
train_phase3.py — All Phase 3 long-training runs from scratch

Seven configurations from the best Phase 2 sweep runs, each trained for
200 epochs from scratch and logged to the fcore_phase_3 W&B project.
Hyperparameters are loaded from the Phase 2 checkpoint files (weights NOT loaded)
so exact values are used. Ablation overrides (dropout, cutmix_alpha) are applied
on top of the checkpoint config.

Workers
───────
  0  devoted-sweep-19    (s8i1hx30)            CutMix α=0.6, dropout=0.2
  1  glad-sweep-11       (mlx6xapm)            CutMix α=0.4, dropout=0.2
  2  treasured-sweep-22  (yoo3ipab)            CutMix α=0.6, dropout=0.0
  3  soft-sweep-27       (cnid1bsb)            CutMix α=0.6, dropout=0.0, ls=0.15
  4  yoo3ipab+dropout    (7583pogc)            CutMix α=0.6, dropout=0.2  [treasured-22 + dropout ablation]
  5  dainty-sweep-41     (5jl5y046)            no CutMix   [best non-CutMix run]
  6  dainty-sweep-41+CM  (5jl5y046_cutmix06)  CutMix α=0.6 added         [CutMix ablation on dainty]

Usage
─────
  CUDA_VISIBLE_DEVICES=0 python3 train_phase3.py --worker 0
  CUDA_VISIBLE_DEVICES=1 python3 train_phase3.py --worker 1
"""

import argparse
import os
import random
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.transforms import v2
import wandb

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

SAVE_DIR    = Path("/home/ubuntu/data/models/fcore_phase_3")
PROJECT     = "fcore_phase_3"
ENTITY      = "rahelvalerie-universit-t-basel"
IMAGE_SIZE  = 224
NUM_WORKERS = min(4, os.cpu_count() or 1)
SEED        = 42

# ─────────────────────────────────────────────
# PHASE 3 CONFIGURATIONS
# ─────────────────────────────────────────────
# Checkpoint paths — configs are loaded from here at runtime.
# Only weights are NOT loaded; everything else (lr, wd, augmentation, etc.) is read
# from the checkpoint so we use the exact values stored on the server.
# cutmix_alpha_override: set when the checkpoint has no cutmix key (non-CutMix sweep)
#                        or when deliberately changing alpha (dainty ablation).
# dropout_p_override:    set when deliberately changing dropout (dropout ablation).
CKPT_BASE_CM  = "/home/ubuntu/data/models/fcore_hyperparam_sweep_3b_cutmix/fcore_hyperparam_sweep_3b_cutmix"
CKPT_BASE_NCM = "/home/ubuntu/data/models/fcore_hyperparam_sweep_3b"

CONFIGS = [
    # ── Worker 0 ─────────────────────────────────────────────────────────
    # devoted-sweep-19 (s8i1hx30) — Phase 2 CutMix best val (81.9%)
    dict(
        run_name             = "s8i1hx30_scratch200",
        original_id          = "s8i1hx30",
        sweep_name           = "devoted-sweep-19",
        checkpoint_path      = f"{CKPT_BASE_CM}/s8i1hx30_best.pt",
        cutmix_alpha_override= None,   # use value from checkpoint
        dropout_p_override   = None,
    ),
    # ── Worker 1 ─────────────────────────────────────────────────────────
    # glad-sweep-11 (mlx6xapm) — Phase 2 CutMix rank 2 (81.1% val)
    dict(
        run_name             = "mlx6xapm_scratch200",
        original_id          = "mlx6xapm",
        sweep_name           = "glad-sweep-11",
        checkpoint_path      = f"{CKPT_BASE_CM}/mlx6xapm_best.pt",
        cutmix_alpha_override= None,
        dropout_p_override   = None,
    ),
    # ── Worker 2 ─────────────────────────────────────────────────────────
    # treasured-sweep-22 (yoo3ipab) — best test acc in CutMix sweep (79.4%)
    dict(
        run_name             = "yoo3ipab_scratch200",
        original_id          = "yoo3ipab",
        sweep_name           = "treasured-sweep-22",
        checkpoint_path      = f"{CKPT_BASE_CM}/yoo3ipab_best.pt",
        cutmix_alpha_override= None,
        dropout_p_override   = None,
    ),
    # ── Worker 3 ─────────────────────────────────────────────────────────
    # soft-sweep-27 (cnid1bsb) — rank 4, ls=0.15 variant
    dict(
        run_name             = "cnid1bsb_scratch200",
        original_id          = "cnid1bsb",
        sweep_name           = "soft-sweep-27",
        checkpoint_path      = f"{CKPT_BASE_CM}/cnid1bsb_best.pt",
        cutmix_alpha_override= None,
        dropout_p_override   = None,
    ),
    # ── Worker 4 ─────────────────────────────────────────────────────────
    # treasured-sweep-22 + dropout=0.2 ablation
    # Loads yoo3ipab config but overrides dropout_p → 0.2
    dict(
        run_name             = "yoo3ipab_dropout20_scratch200",
        original_id          = "yoo3ipab_dropout20",
        sweep_name           = "treasured-sweep-22 + dropout=0.2",
        checkpoint_path      = f"{CKPT_BASE_CM}/yoo3ipab_best.pt",
        cutmix_alpha_override= None,
        dropout_p_override   = 0.2,   # ablation: add dropout to yoo3ipab config
    ),
    # ── Worker 5 ─────────────────────────────────────────────────────────
    # dainty-sweep-41 (5jl5y046) — best non-CutMix run, no CutMix
    dict(
        run_name             = "5jl5y046_scratch200",
        original_id          = "5jl5y046",
        sweep_name           = "dainty-sweep-41",
        checkpoint_path      = f"{CKPT_BASE_NCM}/5jl5y046_best.pt",
        cutmix_alpha_override= 0.0,   # non-CutMix sweep: checkpoint has no cutmix key
        dropout_p_override   = None,
    ),
    # ── Worker 6 ─────────────────────────────────────────────────────────
    # dainty-sweep-41 + CutMix α=0.6 — ablation: does CutMix help this config?
    dict(
        run_name             = "5jl5y046_cutmix06_scratch200",
        original_id          = "5jl5y046_cutmix06",
        sweep_name           = "dainty-sweep-41 + CutMix α=0.6",
        checkpoint_path      = f"{CKPT_BASE_NCM}/5jl5y046_best.pt",
        cutmix_alpha_override= 0.6,   # add CutMix that wasn't in original config
        dropout_p_override   = None,
    ),
]


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
def _worker_init_fn(worker_id):
    """Seed each DataLoader worker independently so augmentation is reproducible."""
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_loaders(train_tf, eval_tf):
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    assert (
        train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx
    ), "Class mappings differ between splits!"

    pin = torch.cuda.is_available()
    g   = torch.Generator()
    g.manual_seed(SEED)
    kw  = dict(num_workers=NUM_WORKERS, pin_memory=pin, worker_init_fn=_worker_init_fn)
    return (
        DataLoader(train_ds, batch_size=32, shuffle=True,  generator=g, **kw),
        DataLoader(val_ds,   batch_size=32, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=32, shuffle=False, **kw),
        train_ds.class_to_idx,
    )


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
def run_epoch(model, loader, criterion, optimizer, device, cutmix=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, running_correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train and cutmix is not None:
                images, labels = cutmix(images, labels)   # labels → soft one-hot [B, C]
                outputs = model(images)
                loss    = criterion(outputs, labels)       # CE handles soft targets natively
                correct = (outputs.argmax(1) == labels.argmax(1)).sum().item()
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)
                correct = (outputs.argmax(1) == labels).sum().item()

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            bs               = labels.shape[0]
            running_loss    += loss.item() * bs
            running_correct += correct
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
# LOAD CONFIG FROM CHECKPOINT
# ─────────────────────────────────────────────
def load_cfg(cfg_dict):
    """
    Load hyperparameters from the Phase 2 checkpoint.
    Weights are NOT loaded — only config keys are used.
    Any _override keys in cfg_dict take precedence over the checkpoint values.
    """
    ckpt_path = Path(cfg_dict["checkpoint_path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Check CKPT_BASE_CM / CKPT_BASE_NCM paths at the top of the script."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Read all hyperparams from the checkpoint
    lr               = ckpt["lr_initial"]
    weight_decay     = ckpt["weight_decay"]
    dropout_p        = ckpt.get("dropout_p", 0.0)
    label_smoothing  = ckpt["label_smoothing"]
    cutmix_alpha     = ckpt.get("cutmix_alpha", 0.0)
    jitter_strength  = ckpt["jitter_strength"]
    rotation_degrees = ckpt["rotation_degrees"]
    aug_erasing      = ckpt["aug_erasing"]

    # Apply overrides (ablations / non-CutMix sweep has no cutmix key)
    if cfg_dict["cutmix_alpha_override"] is not None:
        cutmix_alpha = cfg_dict["cutmix_alpha_override"]
    if cfg_dict["dropout_p_override"] is not None:
        dropout_p = cfg_dict["dropout_p_override"]

    return types.SimpleNamespace(
        run_name        = cfg_dict["run_name"],
        original_id     = cfg_dict["original_id"],
        sweep_name      = cfg_dict["sweep_name"],
        lr              = lr,
        weight_decay    = weight_decay,
        dropout_p       = dropout_p,
        label_smoothing = label_smoothing,
        cutmix_alpha    = cutmix_alpha,
        jitter_strength = jitter_strength,
        rotation_degrees= rotation_degrees,
        aug_erasing     = aug_erasing,
    )


# ─────────────────────────────────────────────
# TRAIN ONE CONFIG
# ─────────────────────────────────────────────
def train(cfg_dict, device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    cfg = load_cfg(cfg_dict)

    print(f"\n{'='*60}")
    print(f"  Worker config: {cfg.sweep_name} ({cfg.original_id})")
    print(f"  lr={cfg.lr:.4e}  wd={cfg.weight_decay}  dropout={cfg.dropout_p}")
    print(f"  ls={cfg.label_smoothing}  cutmix_alpha={cfg.cutmix_alpha}")
    print(f"  jitter={cfg.jitter_strength}  rot={cfg.rotation_degrees}°  erasing={cfg.aug_erasing}")
    print(f"{'='*60}\n")

    train_tf, eval_tf = build_transforms(cfg)
    train_loader, val_loader, test_loader, class_to_idx = build_loaders(
        train_tf, eval_tf
    )
    num_classes  = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    cutmix = (
        v2.CutMix(num_classes=num_classes, alpha=cfg.cutmix_alpha)
        if cfg.cutmix_alpha > 0 else None
    )

    model     = build_model(cfg, num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / f"{cfg.run_name}_best.pt"

    with wandb.init(
        project = PROJECT,
        entity  = ENTITY,
        name    = cfg.run_name,
        config  = {
            "original_id":      cfg.original_id,
            "sweep_name":       cfg.sweep_name,
            "phase":            "3_scratch_200",
            "from_scratch":     True,
            "lr":               cfg.lr,
            "weight_decay":     cfg.weight_decay,
            "dropout_p":        cfg.dropout_p,
            "label_smoothing":  cfg.label_smoothing,
            "cutmix_alpha":     cfg.cutmix_alpha,
            "jitter_strength":  cfg.jitter_strength,
            "rotation_degrees": cfg.rotation_degrees,
            "aug_erasing":      cfg.aug_erasing,
            "crop_scale_min":   0.65,
            "batch_size":       32,
            "max_epochs":       MAX_EPOCHS,
            "min_epochs":       MIN_EPOCHS,
            "patience":         PATIENCE,
            "seed":             SEED,
        },
    ):
        best_val_acc          = float("-inf")
        epochs_no_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, cutmix=cutmix
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, None, device, cutmix=None
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
                        "phase":                "3_scratch_200",
                        "original_id":          cfg.original_id,
                        "sweep_name":           cfg.sweep_name,
                        "lr_initial":           cfg.lr,
                        "weight_decay":         cfg.weight_decay,
                        "dropout_p":            cfg.dropout_p,
                        "label_smoothing":      cfg.label_smoothing,
                        "cutmix_alpha":         cfg.cutmix_alpha,
                        "jitter_strength":      cfg.jitter_strength,
                        "rotation_degrees":     cfg.rotation_degrees,
                        "aug_erasing":          cfg.aug_erasing,
                        "crop_scale_min":       0.65,
                        "batch_size":           32,
                        "seed":                 SEED,
                        "min_epochs":           MIN_EPOCHS,
                        "patience":             PATIENCE,
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

        # ── test evaluation on best checkpoint ────────────────────────────
        ckpt_best  = torch.load(save_path, map_location=device)
        best_model = build_model(cfg, num_classes, device)
        best_model.load_state_dict(ckpt_best["model_state_dict"])

        test_loss, test_acc = run_epoch(best_model, test_loader, criterion, None, device)
        per_class = evaluate_per_class(
            best_model, test_loader, idx_to_class, num_classes, device
        )

        wandb.log({
            "test/loss":     test_loss,
            "test/accuracy": test_acc,
            "best_val_acc":  best_val_acc,
            **{f"test_per_class/{cls}": s["accuracy"] for cls, s in per_class.items()},
        })

        print(f"\n── {cfg.run_name}  best val: {best_val_acc:.4f}  |  test: {test_acc:.4f} ──")
        for cls, stats in per_class.items():
            print(f"  {cls}: {stats['correct']}/{stats['total']}  acc={stats['accuracy']:.3f}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker", type=int, required=True,
        choices=list(range(len(CONFIGS))),
        help=(
            "Which config to run:\n"
            + "\n".join(
                f"  {i} — {c['sweep_name']} ({c['original_id']})"
                for i, c in enumerate(CONFIGS)
            )
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Worker: {args.worker}")

    train(CONFIGS[args.worker], device)

    print("\nDone.")

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
# FIXED CONSTANTS (never swept)
# ─────────────────────────────────────────────
DATA_DIR    = Path("/home/ubuntu/data/robust_dataset_split")
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
TEST_DIR    = DATA_DIR / "test"
LABELS_PATH = DATA_DIR / "labels.json"

SAVE_DIR    = Path("/home/ubuntu/data/models")
IMAGE_SIZE  = 224
NUM_WORKERS = min(4, os.cpu_count() or 1)
SEED        = 42
MAX_EPOCHS  = 80  # hard ceiling per run
MIN_EPOCHS  = 20  # don't early-stop before this
PATIENCE    = 10  # epochs without val_acc improvement → stop

PROJECT = "fcore_21c_hyperparameter_sweep"
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

        # ── optimiser ──────────────────────────
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-2,
        },
        "weight_decay": {
            "values": [1e-3, 1e-4, 1e-5],
        },
        "batch_size": {
            "values": [32, 64],
        },
        "optimizer": {
            "values": ["adamw", "sgd"],
        },

        # ── loss ───────────────────────────────
        "label_smoothing": {
            "values": [0.0, 0.05, 0.1],
        },

        # ── regularisation ─────────────────────
        "dropout_p": {
            "values": [0.0, 0.2, 0.4],
        },

        # ── scheduler ──────────────────────────
        "scheduler": {
            "values": ["step", "cosine", "plateau"],
        },

        # ── augmentation: geometric ────────────
        "aug_hflip": {
            "values": [True, False],
        },
        "aug_rotation": {
            "values": [True, False],
        },
        "aug_perspective": {
            "values": [True, False],
        },

        # ── augmentation: color ────────────────
        "color_jitter_strength": {
            "values": ["none", "mild", "strong"],
        },
        "aug_grayscale": {
            "values": [True, False],
        },
        "aug_blur": {
            "values": [True, False],
        },

        # ── augmentation: occlusion ────────────
        "aug_erasing": {
            "values": [True, False],
        },
    },
}


# ─────────────────────────────────────────────
# BUILD TRANSFORMS
# ─────────────────────────────────────────────
def build_transforms(cfg):
    pre_tensor = [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
    ]

    if cfg.aug_hflip:
        pre_tensor.append(transforms.RandomHorizontalFlip())

    if cfg.aug_rotation:
        pre_tensor.append(transforms.RandomRotation(degrees=10))

    if cfg.aug_perspective:
        pre_tensor.append(
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
        )

    if cfg.aug_grayscale:
        pre_tensor.append(transforms.RandomGrayscale(p=0.08))

    if cfg.aug_blur:
        pre_tensor.append(
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        )

    if cfg.color_jitter_strength == "mild":
        pre_tensor.append(
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
            )
        )
    elif cfg.color_jitter_strength == "strong":
        pre_tensor.append(
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )
        )

    to_tensor = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    # RandomErasing must come after ToTensor — it operates on tensors
    post_tensor = []
    if cfg.aug_erasing:
        post_tensor.append(
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
        )

    train_tf = transforms.Compose(pre_tensor + to_tensor + post_tensor)

    eval_tf = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_tf, eval_tf


# ─────────────────────────────────────────────
# BUILD DATA LOADERS
# ─────────────────────────────────────────────
def build_loaders(train_tf, eval_tf, batch_size):
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    assert (
        train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx
    ), "Class mappings differ between train/val/test splits!"

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
# BUILD MODEL
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
# BUILD OPTIMIZER
# ─────────────────────────────────────────────
def build_optimizer(cfg, model):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr,
            momentum=0.9, weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


# ─────────────────────────────────────────────
# BUILD SCHEDULER
# ─────────────────────────────────────────────
def build_scheduler(cfg, optimizer):
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    elif cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS
        )
    elif cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


# ─────────────────────────────────────────────
# RUN ONE EPOCH
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
# TRAIN FUNCTION — called once per sweep run
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

        model     = build_model(cfg, num_classes, device)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimizer = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg, optimizer)

        run.watch(model, log_freq=50)

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

            if cfg.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            wandb.log({
                "epoch":          epoch,
                "train/loss":     train_loss,
                "train/accuracy": train_acc,
                "val/loss":       val_loss,
                "val/accuracy":   val_acc,
                "lr":             optimizer.param_groups[0]["lr"],
            })

            print(
                f"[{run.name}] Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}"
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
                        "dropout_p":        cfg.dropout_p,
                    },
                    save_path,
                )
            else:
                epochs_no_improvement += 1

            if epoch >= MIN_EPOCHS:
                if train_acc >= 1.0:
                    print("Train accuracy 100% — stopping early (overfitting).")
                    break
                if epochs_no_improvement >= PATIENCE:
                    print(f"No val improvement for {PATIENCE} epochs — stopping early.")
                    break

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
            **{
                f"test_per_class/{cls}": stats["accuracy"]
                for cls, stats in per_class.items()
            },
        })

        print(f"\n── Test accuracy: {test_acc:.4f} ──")
        for cls, stats in per_class.items():
            print(
                f"  {cls}: {stats['correct']}/{stats['total']}"
                f"  acc={stats['accuracy']:.3f}"
            )

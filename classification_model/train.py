"""
TODO: update usage script (using wandb CLI)

"""

import json
import os
import random
from pathlib import Path

import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.transforms import v2
import wandb

# FIXED CONSTANTS
IMAGE_SIZE  = 224 # ImageNet standard
NUM_WORKERS = min(4, os.cpu_count() or 1)
SEED        = 57 # grothendiecks prime


# BUILD TRANSFORMS
def build_transforms(config):
    jitter_params = {
        "mild":   dict(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        "strong": dict(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    }[config.jitter_strength]

    pre_tensor = [ 
        transforms.Lambda(lambda img: img.convert("RGB")), 
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(**jitter_params), # TODO: add no jitter to sweep 
    ]

    if config.rotation_degrees > 0:
        pre_tensor.append(transforms.RandomRotation(degrees=config.rotation_degrees)) 

    to_tensor = [ 
        transforms.ToTensor(),
        transforms.Normalize( 
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ), # ImageNet standard 
    ]

    post_tensor = []
    if config.aug_erasing:
        post_tensor.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)))

    train_transforms = transforms.Compose(pre_tensor + to_tensor + post_tensor) 

    eval_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ), # ImageNet standard
    ])

    return train_transforms, eval_transforms

# DATA LOADERS
def build_loaders(config, train_transforms, eval_transforms):
    data_dir = Path(config.data_dir) # load data dir from config
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transforms) # ImageFolder returns (filepath, class_idx) tuples
    val_dataset   = datasets.ImageFolder(data_dir / "val",   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(data_dir / "test",  transform=eval_transforms)

    assert ( # all splits must have identical class-index mappings 
        train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx
    ), "Class mappings differ between splits!" 

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

# MODEL
def build_model(config, num_classes, device):
    model = resnet18(weights=None) # TODO: experiment with pretrained weights
    in_features = model.fc.in_features

    if config.dropout_p > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=config.dropout_p),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model.to(device)

def train_epoch(model, loader, criterion, optimizer, device, cutmix=None):
    model.train()

    running_loss    = 0.0
    running_correct = 0
    total           = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if cutmix is not None:
            images, labels = cutmix(images, labels)  # labels → soft one-hot [B, C] batch size, classes matrix [32, 30]

        outputs = model(images) # logits [batch size, 30]
        loss    = criterion(outputs, labels) # CE handles soft targets natively

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        preds = outputs.argmax(dim=1) # find index of largest score (=predicted class)
        hard_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels   # cutmix labels have probability mix of labels, argmax recovers biggest one
        correct = (preds == hard_labels).sum().item()  # counts true predictions (compare to true labels)

        batch_size       = labels.shape[0]
        running_loss    += loss.item() * batch_size
        running_correct += correct
        total           += batch_size

    epoch_loss = running_loss    / total if total > 0 else 0.0
    epoch_acc  = running_correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def eval_epoch(model, loader, criterion, device):
    model.eval()

    running_loss    = 0.0
    running_correct = 0
    total           = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()

            batch_size       = labels.shape[0]
            running_loss    += loss.item() * batch_size
            running_correct += correct
            total           += batch_size

    epoch_loss = running_loss    / total if total > 0 else 0.0
    epoch_acc  = running_correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

# PER-CLASS EVALUATION
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

# TRAIN — called once per sweep run
def train():
    # set all seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init() as run:
        run.define_metric("val/accuracy", summary="max")
        config = run.config

        max_epochs = config.max_epochs
        min_epochs = config.min_epochs
        patience = config.patience

        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{run.id}_best.pt"

        train_transforms, eval_transforms = build_transforms(config)
        train_loader, val_loader, test_loader, class_to_idx = build_loaders(
            config, train_transforms, eval_transforms
        )
        num_classes  = len(class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        with open(save_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

        alpha = config.get("cutmix_alpha", 0.0)
        cutmix = v2.CutMix(num_classes=num_classes, alpha=alpha) if alpha > 0 else None

        model = build_model(config, num_classes, device)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # run.watch(model, log_freq=50)

        best_val_acc          = float("-inf")
        best_epoch            = 0
        epochs_no_improvement = 0

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device,cutmix=cutmix)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

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
                f"[{run.name}] Epoch {epoch:03d}/{max_epochs} | "
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
                        **dict(config),
                    },
                    save_path,
                )
            else:
                epochs_no_improvement += 1

            if epoch >= min_epochs:
                if train_acc >= 0.995: # TODO: attention to this, might be too aggressive ..
                    print("Train accuracy almost 100 — stopping early (overfit).")
                    break
                if epochs_no_improvement >= patience:
                    print(f"No val improvement for {patience} epochs — stopping.")
                    break

        print(f"\nBest val: {best_val_acc:.4f} at epoch {best_epoch}")

        # ── final test evaluation ──────────────────────────────────────────
        ckpt       = torch.load(save_path, map_location=device)
        best_model = build_model(config, num_classes, device)
        best_model.load_state_dict(ckpt["model_state_dict"])

        test_loss, test_acc = eval_epoch(best_model, test_loader, criterion, device)
        per_class = evaluate_per_class(best_model, test_loader, idx_to_class, num_classes, device)

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

if __name__ == "__main__":
    train()

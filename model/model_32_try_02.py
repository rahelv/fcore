import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import wandb

import os
from pathlib import Path

RUN = "32_tryout_2"

DATA_DIR = Path("/home/ubuntu/data/robust_dataset_split_2")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 200
MIN_EPOCHS = 80
PATIENCE_AFTER_MIN_EPOCHS = 10
LR = 0.00045327
DROPOUT_P = 0.2
WEIGHT_DECAY = 0.01
NUM_WORKERS = min(4, os.cpu_count() or 1)

SAVE_PATH = "/home/ubuntu/data/models/32_tryout_2.pt"
LABELS_PATH = DATA_DIR / "labels.json"

SEED = 42

config = {
    "run": RUN,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "min_epochs": MIN_EPOCHS,
    "patience_after_min_epochs": PATIENCE_AFTER_MIN_EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "optimizer": "adamw",
    "scheduler": "plateau",
    "dropout_p": 0.2,
    "label_smoothing": 0.1,
    "aug_blur": True,
    "aug_erasing": False,
    "aug_grayscale": True,
    "aug_hflip": True,
    "aug_perspective": False,
    "aug_rotation": True,
    "color_jitter_strength": "strong",
    "model": "resnet18",
}

# ── TRANSFORMS ────────────────────────────────────────────

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        # transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ]
)

eval_transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── DATASETS ──────────────────────────────────────────────

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)

assert (
    train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx
), "Class mappings differ between train/val/test folders."

class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

print(f"Number of classes: {num_classes}")
for cls_name, idx in class_to_idx.items():
    print(f"  {idx}: {cls_name}")

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

# ── DATA LOADERS ──────────────────────────────────────────

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# ── MODEL ─────────────────────────────────────────────────

model = resnet18(weights=None)
# dropout_p=0 for this run, so just a plain linear head
model.fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_P), nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

# ── LOSS / OPTIMIZER / SCHEDULER ──────────────────────────

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

dataset_info = {
    "num_classes": num_classes,
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "test_size": len(test_dataset),
}

# ── HELPER FUNCTIONS ──────────────────────────────────────


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, running_correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_per_class(model, loader, num_classes):
    model.eval()
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1)

            for label, pred in zip(labels, preds):
                i = label.item()
                total_per_class[i] += 1
                correct_per_class[i] += int(label == pred)

    return {
        idx_to_class[i]: {
            "correct": correct_per_class[i],
            "total": total_per_class[i],
            "accuracy": correct_per_class[i] / total_per_class[i]
            if total_per_class[i] > 0
            else 0.0,
        }
        for i in range(num_classes)
    }


# ── TRAINING LOOP ─────────────────────────────────────────

best_val_acc = float("-inf")
epochs_without_improvement = 0

run = wandb.init(
    project="costume_recognition_model",
    config={**config, **dataset_info},
    name=RUN,
)
run.watch(model)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion)

    scheduler.step(val_loss)  # ReduceLROnPlateau needs the metric

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "image_size": IMAGE_SIZE,
            },
            SAVE_PATH,
        )
    else:
        epochs_without_improvement += 1

    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )

# ── TEST EVALUATION ───────────────────────────────────────

checkpoint = torch.load(SAVE_PATH, map_location=device)
best_model = resnet18(weights=None)
best_model.fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_P),
    nn.Linear(best_model.fc.in_features, checkpoint["num_classes"]),
)
best_model.load_state_dict(checkpoint["model_state_dict"])
best_model = best_model.to(device)

test_loss, test_acc = run_epoch(best_model, test_loader, criterion)
test_per_class_acc = evaluate_per_class(best_model, test_loader, num_classes)

wandb.log(
    {
        "test/loss": test_loss,
        "test/accuracy": test_acc,
        **{
            f"test_per_class/{cls_name}": stats["accuracy"]
            for cls_name, stats in test_per_class_acc.items()
        },
    }
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print("\nPer-class test accuracy:")
for cls_name, stats in test_per_class_acc.items():
    print(
        f"  {cls_name}: {stats['correct']}/{stats['total']}  accuracy={stats['accuracy']:.4f}"
    )

wandb.save(SAVE_PATH)
wandb.finish()

# This is the second try of training a model, using dataset with 9 labels and approx. 100 images per label
# 1. 15 epochs
# 2. 30 epochs
# 3. add

import json
import os
from pathlib import Path

import torch
from torch.cuda import is_available
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import wandb

DATA_DIR = Path("/home/ubuntu/data/robust_dataset_split")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMAGE_SIZE = 224  # TODO: explain / text why this image size
BATCH_SIZE = 32  # TODO:
EPOCHS = 30  # TODO:
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = min(
    4, os.cpu_count() if os.cpu_count() is not None else 0
)  # TODO: change

USE_PRETRAINED = False  # TODO: remove everywhere
SAVE_PATH = "/home/ubuntu/data/models/0421_model.pt"
LABELS_PATH = DATA_DIR / "labels.json"  # TODO: create

SEED = 42  # for reproducibility

# WANDB CONFIG

config = {
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "model": "resnet18",
    "pretrained": USE_PRETRAINED,
}

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TRANSFORMS

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),  # resize and crop
        transforms.RandomHorizontalFlip(),  # spiegeln
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # TODO: play around with values !
    ]
)

eval_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # no randomness for test / val
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # TODO: keep same as in training set
    ]
)

# DATASETS

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)

# Make sure class mapping is consistent
assert (
    train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx
), "Class mappings differ between train/val/test folders."

class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

dataset_info = {
    "num_classes": num_classes,
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "test_size": len(test_dataset),
}

print(f"Number of classes: {num_classes}")
print("Classes:")
for cls_name, idx in class_to_idx.items():
    print(f"  {idx}: {cls_name}")

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

# DATA LOADERS

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

# MODEL

model = resnet18(weights=None)  # TODO: use pretrained ??
model.fc = nn.Linear(
    model.fc.in_features, num_classes
)  # output is the amount of labels ...
model = model.to(device)

# LOSS / OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)  # update the weights after backpropagation

# Optional scheduler, changs learning rate during training TODO: try out
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# HELPER FUNCTIONS


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:  # backpropagation and optimizer step
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

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for label, pred in zip(labels, preds):
                label_i = label.item()
                pred_i = pred.item()
                total_per_class[label_i] += 1
                if label_i == pred_i:
                    correct_per_class[label_i] += 1

    per_class_acc = {}
    for i in range(num_classes):
        cls_name = idx_to_class[i]
        total_i = total_per_class[i]
        acc_i = correct_per_class[i] / total_i if total_i > 0 else 0.0
        per_class_acc[cls_name] = {
            "correct": correct_per_class[i],
            "total": total_i,
            "accuracy": acc_i,
        }

    return per_class_acc


# TRAINING LOOP

best_val_acc = 0.0

run = wandb.init(
    project="costume-recognition",
    config=config,
    name="0421_model_v2",
)
run.watch(model)
wandb.config.update(dataset_info)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

    scheduler.step(val_acc)

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

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "use_pretrained": USE_PRETRAINED,
                "image_size": IMAGE_SIZE,
            },
            SAVE_PATH,
        )

checkpoint = torch.load(SAVE_PATH, map_location=device)
if checkpoint.get("use_pretrained", False):
    best_model = resnet18(weights="DEFAULT")
else:
    best_model = resnet18(weights=None)

best_model.fc = nn.Linear(best_model.fc.in_features, checkpoint["num_classes"])
best_model.load_state_dict(checkpoint["model_state_dict"])
best_model = best_model.to(device)

test_loss, test_acc = run_epoch(best_model, test_loader, criterion, optimizer=None)

wandb.log(
    {
        "test/loss": test_loss,
        "test/accuracy": test_acc,
    }
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

wandb.save(SAVE_PATH)
wandb.finish()

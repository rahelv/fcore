import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import wandb
import os
from pathlib import Path

RUN = "0511_model_v04_more_epochs"

DATA_DIR = Path("/home/ubuntu/data/robust_dataset_split")
# DATA_DIR = Path("/Users/rahel/code/FS26/playground/data/robust_dataset_split")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMAGE_SIZE = 224  # TODO: explain / text why this image size
BATCH_SIZE = 64  # TODO:
EPOCHS = 50  # TODO:
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = min(4, os.cpu_count() or 1)

SAVE_PATH = "/home/ubuntu/data/models/0428_v04.pt"
# SAVE_PATH = "/Users/rahel/code/FS26/playground/data/models/0421_model.pt"
LABELS_PATH = DATA_DIR / "labels.json"

SEED = 42  # for reproducibility

config = {
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "classification_model": "resnet18",
}

# TRANSFORMS

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),  # resize and crop
        # transforms.RandomHorizontalFlip(),  # spiegeln
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # TODO: https://stackoverflow.com/questions/65467621/what-are-the-numbers-in-torch-transforms-normalize-and-how-to-select-them
    ]
)

eval_transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),  # no randomness for test / val
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # TODO: keep same as in training set
    ]
)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DATASETS

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

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)  # update the weights after backpropagation

# TODO: maybe try other schedulers
# Sets the learning rate to the initial LR decayed by 0.5 every 10 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

dataset_info = {
    "num_classes": num_classes,
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "test_size": len(test_dataset),
    "transforms": str(train_transforms),
    "scheduler": str(scheduler),
}

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
        for images, labels in loader:  # TODO: shape
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)  # forward pass  TODO: shape
            loss = criterion(outputs, labels)  # compute the loss

            if is_train:  # backpropagation and optimizer step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(
                dim=1
            )  # converts classification_model scores to predicted class IDs

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

wandb_config = {
    **config,
    **dataset_info,
}

run = wandb.init(
    project="costume_recognition_model",
    config=wandb_config,
    name=RUN,
)

run.watch(model)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

    scheduler.step()

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
                "image_size": IMAGE_SIZE,
            },
            SAVE_PATH,
        )

checkpoint = torch.load(SAVE_PATH, map_location=device)

best_model = resnet18(weights=None)

best_model.fc = nn.Linear(best_model.fc.in_features, checkpoint["num_classes"])
best_model.load_state_dict(checkpoint["model_state_dict"])
best_model = best_model.to(device)

test_loss, test_acc = run_epoch(best_model, test_loader, criterion, optimizer=None)
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
        f"{cls_name}: "
        f"{stats['correct']}/{stats['total']} "
        f"accuracy={stats['accuracy']:.4f}"
    )

wandb.save(SAVE_PATH)
wandb.finish()

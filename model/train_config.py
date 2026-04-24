import os
from pathlib import Path

from torchvision import transforms

DATA_DIR = Path("/home/ubuntu/data/robust_dataset_split")
# DATA_DIR = Path("/Users/rahel/code/FS26/playground/data/robust_dataset_split")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMAGE_SIZE = 224  # TODO: explain / text why this image size
BATCH_SIZE = 32  # TODO:
EPOCHS = 30  # TODO:
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = min(4, os.cpu_count() or 1)

SAVE_PATH = "/home/ubuntu/data/models/0421_v1.pt"
# SAVE_PATH = "/Users/rahel/code/FS26/playground/data/models/0421_model.pt"
LABELS_PATH = DATA_DIR / "labels.json"

SEED = 42  # for reproducibility

config = {
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "model": "resnet18",
}

# TRANSFORMS

train_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.RandomResizedCrop(IMAGE_SIZE),  # resize and crop
        # transforms.RandomHorizontalFlip(),  # spiegeln
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
            mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
        ),  # TODO: keep same as in training set
    ]
)

# MODEL

# LOSS / OPTIMIZER

# SCHEDULER
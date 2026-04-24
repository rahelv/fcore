# TODO: instead of split, keep organization in json file maybe ?

# splits folder structure into test/train/val
# TODO: ignore folder "originals"

from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("/home/ubuntu/data/robust_dataset")  # current folder
TARGET_DIR = Path("/home/ubuntu/data/robust_dataset_split")  # output folder

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-8

for class_dir in SOURCE_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        continue

    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split_name, split_files in splits.items():
        out_class_dir = TARGET_DIR / split_name / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for src in split_files:
            shutil.copy2(src, out_class_dir / src.name)

print("Done.")

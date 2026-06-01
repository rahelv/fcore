from PIL import Image
from pathlib import Path

data_dir = Path("/home/ubuntu/data/robust_dataset_split_2")

converted = 0
for img_path in data_dir.rglob("*.png"):
    img = Image.open(img_path)
    if img.mode != "RGB":
        print(f"Converting {img_path} — mode was {img.mode}")
        img.convert("RGB").save(img_path)
        converted += 1

print(f"\nDone — converted {converted} images")

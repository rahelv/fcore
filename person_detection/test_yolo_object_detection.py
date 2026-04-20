from pathlib import Path

import torch
from ultralytics import YOLO


# ---------------------------
# Config
# ---------------------------
INPUT_DIR = Path("../data/ollama_test_images")
OUTPUT_DIR = Path("/home/ubuntu/data/test_yolo_detect")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_NAME = "yolo26s.pt"  # or "yolo26n.pt" for faster inference
GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Basic checks
# ---------------------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Using device:", DEVICE)

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

image_paths = sorted(
    [
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)

if not image_paths:
    raise RuntimeError(f"No image files found in: {INPUT_DIR}")

print(f"Found {len(image_paths)} image(s).")


# ---------------------------
# Load model
# ---------------------------
model = YOLO(MODEL_NAME)


# ---------------------------
# Process all images
# ---------------------------
for image_path in image_paths:
    print(f"\nProcessing: {image_path}")

    try:
        results = model.predict(
            source=str(image_path),
            device=DEVICE,
            conf=0.25,
            save=True,
            project=str(OUTPUT_DIR),
            name="predictions",
            exist_ok=True,
            verbose=False,
        )

        if not results:
            print(f"No results returned for: {image_path.name}")
            continue

        print(f"Saved annotated detection image for: {image_path.name}")

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

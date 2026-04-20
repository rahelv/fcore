from pathlib import Path

import cv2
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# ---------------------------
# Config
# ---------------------------
REPO_ID = "RyanJames/yolo12l-person-seg"
FILENAME = "yolo12l-person-seg-extended.pt"  # recommended by model card

INPUT_DIR = Path("../data/ollama_test_images")
OUTPUT_DIR = Path("/home/ubuntu/data/test_segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 0  # GPU 0


# ---------------------------
# Basic checks
# ---------------------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Check your PyTorch installation.")

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")


# ---------------------------
# Download model from Hugging Face
# ---------------------------
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
)

print("Model downloaded to:", model_path)


# ---------------------------
# Load model
# ---------------------------
model = YOLO(model_path)


# ---------------------------
# Collect images
# ---------------------------
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = sorted(
    [
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
)

if not image_paths:
    raise RuntimeError(f"No images found in: {INPUT_DIR}")

print(f"Found {len(image_paths)} image(s).")


# ---------------------------
# Run segmentation for each image
# ---------------------------
for image_path in image_paths:
    print(f"\nProcessing: {image_path}")

    try:
        results = model.predict(
            source=str(image_path),
            device=DEVICE,
            conf=0.25,
            save=False,
        )

        if not results:
            print(f"No results returned for: {image_path.name}")
            continue

        result = results[0]

        # Save only annotated visualization
        annotated = result.plot()
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        annotated_path = OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"
        success = cv2.imwrite(str(annotated_path), annotated_bgr)

        if not success:
            print(f"Failed to save annotated image: {annotated_path}")
            continue

        print(f"Saved annotated image to: {annotated_path}")

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

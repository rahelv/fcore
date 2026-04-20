from pathlib import Path

import cv2
import numpy as np
import torch
from fashn_human_parser import FashnHumanParser


# ---------------------------
# Config
# ---------------------------
INPUT_DIR = Path("../data/ollama_test_images")
OUTPUT_DIR = Path("/home/ubuntu/data/test_fashn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Welche Labels sollen sichtbar markiert werden?
# IDs laut Model Card:
# 0 background
# 1 face
# 2 hair
# 3 top
# 4 dress
# 5 skirt
# 6 pants
# 7 belt
# 8 bag
# 9 hat
# 10 scarf
# 11 glasses
# 12 arms
# 13 hands
# 14 legs
# 15 feet
# 16 torso
# 17 jewelry
LABEL_NAMES = {
    0: "background",
    1: "face",
    2: "hair",
    3: "top",
    4: "dress",
    5: "skirt",
    6: "pants",
    7: "belt",
    8: "bag",
    9: "hat",
    10: "scarf",
    11: "glasses",
    12: "arms",
    13: "hands",
    14: "legs",
    15: "feet",
    16: "torso",
    17: "jewelry",
}

# Farben in BGR für OpenCV
LABEL_COLORS = {
    1: (80, 180, 255),  # face
    2: (60, 60, 180),  # hair
    3: (0, 255, 0),  # top
    4: (255, 0, 255),  # dress
    5: (255, 100, 255),  # skirt
    6: (255, 0, 0),  # pants
    7: (0, 255, 255),  # belt
    8: (0, 165, 255),  # bag
    9: (128, 0, 255),  # hat
    10: (255, 255, 0),  # scarf
    11: (100, 255, 100),  # glasses
    12: (180, 180, 0),  # arms
    13: (180, 220, 0),  # hands
    14: (0, 128, 255),  # legs
    15: (0, 80, 200),  # feet
    16: (80, 255, 80),  # torso
    17: (200, 200, 255),  # jewelry
}


def build_overlay(
    segmentation: np.ndarray, image_shape: tuple[int, int, int]
) -> np.ndarray:
    """
    Create a color overlay image from the segmentation mask.
    segmentation: HxW array of class IDs
    image_shape: shape of original BGR image
    """
    h, w = segmentation.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in LABEL_COLORS.items():
        overlay[segmentation == class_id] = color

    return overlay


def draw_legend(image: np.ndarray, present_labels: list[int]) -> np.ndarray:
    """
    Draw a simple legend in the top-left corner.
    """
    output = image.copy()
    x = 15
    y = 20
    box_size = 16
    line_height = 24

    for class_id in present_labels:
        if class_id == 0:
            continue
        color = LABEL_COLORS.get(class_id, (255, 255, 255))
        label = LABEL_NAMES.get(class_id, str(class_id))

        cv2.rectangle(output, (x, y - 12), (x + box_size, y + 4), color, -1)
        cv2.putText(
            output,
            label,
            (x + box_size + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_height

    return output


# ---------------------------
# Basic checks
# ---------------------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

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
# Load parser
# ---------------------------
# Laut Model Card auto-detectet das Paket GPU.
parser = FashnHumanParser()

# ---------------------------
# Process all images
# ---------------------------
for image_path in image_paths:
    print(f"\nProcessing: {image_path}")

    try:
        original_bgr = cv2.imread(str(image_path))
        if original_bgr is None:
            print(f"Could not read image: {image_path}")
            continue

        # segmentation: HxW numpy array mit Klassen 0-17
        segmentation = parser.predict(str(image_path))

        if segmentation is None:
            print(f"No segmentation returned for: {image_path.name}")
            continue

        if segmentation.shape[:2] != original_bgr.shape[:2]:
            segmentation = cv2.resize(
                segmentation.astype(np.uint8),
                (original_bgr.shape[1], original_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        overlay = build_overlay(segmentation, original_bgr.shape)

        # Original + Overlay mischen
        alpha = 0.45
        annotated = cv2.addWeighted(original_bgr, 1.0 - alpha, overlay, alpha, 0)

        present_labels = sorted(int(x) for x in np.unique(segmentation) if int(x) != 0)
        annotated = draw_legend(annotated, present_labels)

        out_path = OUTPUT_DIR / f"{image_path.stem}_annotated.png"
        ok = cv2.imwrite(str(out_path), annotated)

        if ok:
            print(f"Saved annotated image to: {out_path}")
            print("Present labels:", [LABEL_NAMES[i] for i in present_labels])
        else:
            print(f"Failed to save: {out_path}")

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = Path("classes_originals")
OUTPUT_FILE = "classes_originals_grid.png"

COLS = 5
IMAGE_SIZE = 224        # Size of each image
LABEL_HEIGHT = 32
PADDING = 5
BACKGROUND = "white"

# -----------------------------
# Find images
# -----------------------------
extensions = ("*.webp", "*.png", "*.jpg", "*.jpeg")
image_files = []

for ext in extensions:
    image_files.extend(INPUT_DIR.glob(ext))

image_files = sorted(image_files)

if len(image_files) == 0:
    raise RuntimeError("No images found!")

ROWS = (len(image_files) + COLS - 1) // COLS

canvas_width = COLS * IMAGE_SIZE + (COLS + 1) * PADDING
canvas_height = ROWS * (IMAGE_SIZE + LABEL_HEIGHT) + (ROWS + 1) * PADDING

canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
draw = ImageDraw.Draw(canvas)

# Try to use a nicer font
try:
    font = ImageFont.truetype("Arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()

# -----------------------------
# Draw images
# -----------------------------
for idx, img_path in enumerate(image_files):

    row = idx // COLS
    col = idx % COLS

    x = PADDING + col * IMAGE_SIZE
    y = PADDING + row * (IMAGE_SIZE + LABEL_HEIGHT)

    img = Image.open(img_path).convert("RGBA")
    img.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    # Create white background
    cell = Image.new(
        "RGBA",
        (IMAGE_SIZE, IMAGE_SIZE),
        (255, 255, 255, 255)
    )

    # Center image
    offset = (
        (IMAGE_SIZE - img.width) // 2,
        (IMAGE_SIZE - img.height) // 2
    )

    # Paste respecting transparency
    cell.alpha_composite(img, offset)

    # Convert back to RGB before saving
    cell = cell.convert("RGB")

    # Put into grid
    canvas.paste(cell, (x, y))

    label = img_path.stem.replace("_", " ")

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]

    draw.text(
        (x + (IMAGE_SIZE - text_width) / 2, y + IMAGE_SIZE + 6),
        label,
        fill="black",
        font=font,
    )

canvas.save(OUTPUT_FILE)

print(f"Saved {OUTPUT_FILE}")
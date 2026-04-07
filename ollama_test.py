import base64
import json
from pathlib import Path

import requests

IMAGE_DIR = Path("~/code_playground/ollama_test_images").expanduser()
OUTPUT_FILE = IMAGE_DIR / "results.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:7b"

PROMPT = (
    "Analyze this cosplay image and return JSON with keys: "
    "costume_label, character_name, franchise, description, "
    "garments, accessories, confidence, uncertain. "
    "If identity is unclear, use unknown. "
    "Describe only visible details."
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def run_image(image_path: Path) -> dict:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [encode_image(image_path)],
        "format": "json",
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    raw = response.json()["response"]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "costume_label": "",
            "character_name": "unknown",
            "franchise": "unknown",
            "description": raw,
            "garments": [],
            "accessories": [],
            "confidence": 0.0,
            "uncertain": True,
        }

    parsed["image"] = image_path.name
    return parsed


with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for image_path in sorted(IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS or not image_path.is_file():
            continue

        print(f"Processing {image_path.name}...")
        try:
            result = run_image(image_path)
        except Exception as e:
            result = {
                "image": image_path.name,
                "costume_label": "",
                "character_name": "unknown",
                "franchise": "unknown",
                "description": f"ERROR: {e}",
                "garments": [],
                "accessories": [],
                "confidence": 0.0,
                "uncertain": True,
            }

        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Saved results to {OUTPUT_FILE}")


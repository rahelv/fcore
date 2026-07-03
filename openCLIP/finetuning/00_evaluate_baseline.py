#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_metadata(csv_path):
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "filepath" not in reader.fieldnames or "caption" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: filepath, caption")

        for row in reader:
            rows.append(
                {
                    "filepath": row["filepath"],
                    "caption": row["caption"],
                }
            )

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    return rows


@torch.no_grad()
def encode_texts(model, tokenizer, captions, device):
    tokens = tokenizer(captions).to(device)

    with torch.amp.autocast(
        device_type="cuda",
        enabled=device.type == "cuda"
    ):
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device):
    images = []
    valid_paths = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(preprocess(img))
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not images:
        return None, []

    batch = torch.stack(images).to(device)

    with torch.amp.autocast(
        device_type="cuda",
        enabled=device.type == "cuda"
    ):
        image_features = model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy(), valid_paths


def load_model(model_id, checkpoint, device):
    model, preprocess = create_model_from_pretrained(model_id)
    tokenizer = get_tokenizer(model_id)

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    model = model.to(device).eval()
    return model, preprocess, tokenizer


def evaluate(args):
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    rows = read_metadata(metadata_path)

    captions = sorted(set(row["caption"] for row in rows))
    caption_to_idx = {caption: i for i, caption in enumerate(captions)}

    print(f"Metadata rows: {len(rows)}")
    print(f"Unique captions/classes: {len(captions)}")

    model, preprocess, tokenizer = load_model(
        args.model,
        args.checkpoint,
        device
    )

    text_matrix = encode_texts(model, tokenizer, captions, device)

    top1_correct = 0
    top5_correct = 0
    total = 0

    per_caption = defaultdict(lambda: {
        "total": 0,
        "top1_correct": 0,
        "top5_correct": 0,
        "confused_as": defaultdict(int),
    })

    batch_paths = []
    batch_captions = []

    def process_batch(paths, true_captions):
        nonlocal top1_correct, top5_correct, total

        img_matrix, valid_paths = encode_images(model, preprocess, paths, device)

        if img_matrix is None:
            return

        valid_true_captions = true_captions[:len(valid_paths)]
        sims = img_matrix @ text_matrix.T

        for i, true_caption in enumerate(valid_true_captions):
            top5_idx = np.argsort(sims[i])[::-1][:5]
            top5_preds = [captions[j] for j in top5_idx]
            top1_pred = top5_preds[0]

            hit1 = top1_pred == true_caption
            hit5 = true_caption in top5_preds

            top1_correct += int(hit1)
            top5_correct += int(hit5)
            total += 1

            per_caption[true_caption]["total"] += 1
            per_caption[true_caption]["top1_correct"] += int(hit1)
            per_caption[true_caption]["top5_correct"] += int(hit5)

            if not hit1:
                per_caption[true_caption]["confused_as"][top1_pred] += 1

    for row in tqdm(rows, desc="Evaluating", unit="img"):
        batch_paths.append(row["filepath"])
        batch_captions.append(row["caption"])

        if len(batch_paths) >= args.batch_size:
            process_batch(batch_paths, batch_captions)
            batch_paths.clear()
            batch_captions.clear()

    if batch_paths:
        process_batch(batch_paths, batch_captions)

    sorted_captions = sorted(
        per_caption.keys(),
        key=lambda c: per_caption[c]["top1_correct"] / max(per_caption[c]["total"], 1),
        reverse=True,
    )

    output = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "metadata": str(metadata_path),
        "total_images": total,
        "total_captions": len(captions),
        "overall": {
            "top1_accuracy": round(top1_correct / total * 100, 2),
            "top1_correct": top1_correct,
            "top5_accuracy": round(top5_correct / total * 100, 2),
            "top5_correct": top5_correct,
            "total": total,
        },
        "per_caption": {
            caption: {
                "total": per_caption[caption]["total"],
                "top1_correct": per_caption[caption]["top1_correct"],
                "top1_accuracy": round(
                    per_caption[caption]["top1_correct"]
                    / per_caption[caption]["total"]
                    * 100,
                    2,
                ),
                "top5_correct": per_caption[caption]["top5_correct"],
                "top5_accuracy": round(
                    per_caption[caption]["top5_correct"]
                    / per_caption[caption]["total"]
                    * 100,
                    2,
                ),
                "confused_as": dict(
                    sorted(
                        per_caption[caption]["confused_as"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ),
            }
            for caption in sorted_captions
        },
    }

    print("\n" + "=" * 60)
    print("CSV CAPTION VAL EVALUATION")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Checkpoint: {args.checkpoint or 'pretrained baseline'}")
    print(f"Metadata:   {metadata_path}")
    print(f"Top-1:      {output['overall']['top1_accuracy']}% ({top1_correct}/{total})")
    print(f"Top-5:      {output['overall']['top5_accuracy']}% ({top5_correct}/{total})")
    print("=" * 60)

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved results to: {output_path}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--metadata",
        default="/home/ubuntu/data/robust_dataset_split_30c_v2/val_metadata.csv",
    )
    p.add_argument(
        "--model",
        default="hf-hub:timm/ViT-B-16-SigLIP2-256",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
    )
    p.add_argument(
        "--output",
        default="results_val_pretrained_siglip2.json",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--cpu",
        action="store_true",
    )

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
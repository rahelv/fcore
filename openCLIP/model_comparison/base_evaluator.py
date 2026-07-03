"""
base_evaluator.py — shared val-split accuracy evaluation, no database
========================================================================
For every image in VAL_DIR (ImageFolder layout, folder name = label):
  1. Image search  : embed the image, compare to all label text-embeddings,
                      record whether the true label is top-1 / top-5.
  2. Prompt search  : embed each label as text, rank all val images by
                      similarity, compute P@K / R@K / AP / mAP.

Both directions reuse the same image + label embeddings (one pass over the
val set), computed entirely in memory — no Postgres/pgvector involved.

Subclasses only need to implement `load_model()`, `encode_images_batch()`,
and `encode_text_batch()`; everything else (data loading, metrics, printing,
saving) lives here.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from torchvision import datasets
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def average_precision(relevant: np.ndarray) -> float:
    """relevant[i] = True if the i-th ranked result is the correct class."""
    hits, ap = 0, 0.0
    n_relevant = relevant.sum()
    if n_relevant == 0:
        return 0.0
    for i, hit in enumerate(relevant):
        if hit:
            hits += 1
            ap += hits / (i + 1)
    return ap / n_relevant


class BaseEvaluator:
    """Override load_model / encode_images_batch / encode_text_batch per model."""

    def __init__(self, val_dir, device, batch_size=64, top_k=(5, 10, 20)):
        self.val_dir = Path(val_dir)
        self.device = device
        self.batch_size = batch_size
        self.top_k = list(top_k)
        self.model, self.preprocess, self.tokenizer = self.load_model()

    # -- to be implemented by subclasses --------------------------------
    def load_model(self):
        """Return (model, preprocess, tokenizer), model already on self.device and .eval()."""
        raise NotImplementedError

    def encode_images_batch(self, pil_images: list) -> np.ndarray:
        """Return l2-normalised [N, D] embeddings for a batch of PIL images."""
        raise NotImplementedError

    def encode_text_batch(self, texts: list) -> np.ndarray:
        """Return l2-normalised [N, D] embeddings for a batch of strings."""
        raise NotImplementedError

    # -- shared evaluation logic -----------------------------------------
    def _embed_val_set(self):
        val_ds = datasets.ImageFolder(self.val_dir)
        idx_to_class = {v: k for k, v in val_ds.class_to_idx.items()}
        labels = val_ds.classes
        print(f"Val set: {len(val_ds)} images across {len(labels)} classes")

        img_vecs, img_labels = [], []
        batch_paths, batch_labels = [], []
        errors = []

        def flush():
            if not batch_paths:
                return
            pil_images, valid_labels = [], []
            for path, lbl in zip(batch_paths, batch_labels):
                try:
                    pil_images.append(Image.open(path).convert("RGB"))
                    valid_labels.append(lbl)
                except Exception as e:
                    errors.append((str(path), str(e)))
            if not pil_images:
                return
            img_vecs.append(self.encode_images_batch(pil_images))
            img_labels.extend(valid_labels)

        for img_path, label_idx in tqdm(val_ds.imgs, desc="Embedding val images", unit="img"):
            batch_paths.append(img_path)
            batch_labels.append(idx_to_class[label_idx])
            if len(batch_paths) == self.batch_size:
                flush()
                batch_paths.clear()
                batch_labels.clear()
        flush()

        img_matrix = np.concatenate(img_vecs, axis=0)
        img_labels_arr = np.array(img_labels)
        label_matrix = self.encode_text_batch(labels)

        if errors:
            print(f"Skipped {len(errors)} images due to load errors.")

        return labels, img_matrix, img_labels_arr, label_matrix, errors

    def _image_search_metrics(self, labels, img_matrix, img_labels_arr, label_matrix):
        sims = img_matrix @ label_matrix.T  # [N, C]
        top1_correct = top5_correct = 0
        per_class = defaultdict(lambda: {
            "top1_correct": 0, "top5_correct": 0, "total": 0,
            "confused_as": defaultdict(int),
        })

        for i, true_label in enumerate(img_labels_arr):
            order = np.argsort(sims[i])[::-1]
            top5_preds = [labels[j] for j in order[:5]]
            top1_pred = top5_preds[0]
            hit1 = top1_pred == true_label
            hit5 = true_label in top5_preds

            top1_correct += int(hit1)
            top5_correct += int(hit5)
            d = per_class[true_label]
            d["total"] += 1
            d["top1_correct"] += int(hit1)
            d["top5_correct"] += int(hit5)
            if not hit1:
                d["confused_as"][top1_pred] += 1

        total = len(img_labels_arr)
        return {
            "total_images": total,
            "top1_accuracy": round(top1_correct / total * 100, 2),
            "top1_correct": top1_correct,
            "top5_accuracy": round(top5_correct / total * 100, 2),
            "top5_correct": top5_correct,
            "per_class": {
                cls: {
                    "total": d["total"],
                    "top1_correct": d["top1_correct"],
                    "top1_accuracy": round(d["top1_correct"] / d["total"] * 100, 2),
                    "top5_correct": d["top5_correct"],
                    "top5_accuracy": round(d["top5_correct"] / d["total"] * 100, 2),
                    "confused_as": dict(sorted(d["confused_as"].items(), key=lambda x: x[1], reverse=True)),
                }
                for cls, d in per_class.items()
            },
        }

    def _prompt_search_metrics(self, labels, img_matrix, img_labels_arr, label_matrix):
        ks = self.top_k
        sims = label_matrix @ img_matrix.T  # [C, N]
        per_class = {}
        ap_scores = []

        for c, label in enumerate(labels):
            ranked_idx = np.argsort(sims[c])[::-1]
            ranked_labels = img_labels_arr[ranked_idx]
            relevant = ranked_labels == label

            ap = average_precision(relevant)
            ap_scores.append(ap)
            n_class = int((img_labels_arr == label).sum())

            max_k = max(ks)
            top_k_labels = ranked_labels[:max_k]
            precision_at_k, recall_at_k = {}, {}
            for k in ks:
                n_correct = int((ranked_labels[:k] == label).sum())
                precision_at_k[k] = {"correct": n_correct, "k": k, "fraction": f"{n_correct}/{k}", "percent": round(n_correct / k * 100, 1)}
                recall_at_k[k] = {"correct": n_correct, "total_in_val": n_class, "fraction": f"{n_correct}/{n_class}", "percent": round(n_correct / max(n_class, 1) * 100, 1)}

            fp_counts = defaultdict(int)
            for lbl in top_k_labels:
                if lbl != label:
                    fp_counts[lbl] += 1
            top_fp = sorted(fp_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            per_class[label] = {
                "total_in_val": n_class,
                "ap": round(ap, 4),
                "precision_at_k": {str(k): v for k, v in precision_at_k.items()},
                "recall_at_k": {str(k): v for k, v in recall_at_k.items()},
                "top_false_positives": [{"label": l, "count": cnt} for l, cnt in top_fp],
            }

        mAP = float(np.mean(ap_scores))
        return {
            "top_k_values": ks,
            "mAP": round(mAP * 100, 2),
            "per_class": per_class,
        }

    def run(self) -> dict:
        labels, img_matrix, img_labels_arr, label_matrix, errors = self._embed_val_set()
        image_search = self._image_search_metrics(labels, img_matrix, img_labels_arr, label_matrix)
        prompt_search = self._prompt_search_metrics(labels, img_matrix, img_labels_arr, label_matrix)

        print("\n" + "=" * 60)
        print(f"  VAL EVALUATION — {self.describe()}")
        print("=" * 60)
        print(f"  Val dir    : {self.val_dir}")
        print(f"  Top-1 acc  : {image_search['top1_accuracy']:.2f}%  ({image_search['top1_correct']}/{image_search['total_images']})")
        print(f"  Top-5 acc  : {image_search['top5_accuracy']:.2f}%  ({image_search['top5_correct']}/{image_search['total_images']})")
        print(f"  mAP        : {prompt_search['mAP']:.2f}%")
        print("=" * 60)

        return {
            "model": self.describe(),
            "val_dir": str(self.val_dir),
            "total_classes": len(labels),
            "image_search": image_search,
            "prompt_search": prompt_search,
            "errors": errors,
        }

    def describe(self) -> str:
        return self.__class__.__name__

    def save(self, output: dict, path: str):
        Path(path).write_text(json.dumps(output, indent=2))
        print(f"\nSaved results to {path}")

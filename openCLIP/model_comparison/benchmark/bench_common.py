"""
bench_common.py — shared logic for the per-model Jetson benchmark scripts
=============================================================================
Not run directly. Imported by bench_xlm_roberta.py and bench_siglip2.py.

Matches the real deployment use case exactly:
  1. Embed the 30 class labels as text ONCE, upfront (deployment precomputes this offline).
  2. For every val image, one at a time: embed the image (batch size 1), compare
     against the 30 fixed text embeddings, get the prediction.
  3. Record per-image latency for all 779 images, then report mean/median/std/p95.

Accuracy (Top-1 / Top-5) falls out of the same loop for free, and matches
base_evaluator.py's image_search logic (raw folder names as labels, no prompt
templates), so it's directly comparable to the thesis table.

Only needs VAL_DIR (ImageFolder layout) — the train split is never read.

Reproducibility: seeds set for torch/numpy/random, cuDNN in deterministic mode.

Dependency: scikit-learn (pip install scikit-learn) if not already present.
"""

import random
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import top_k_accuracy_score
from torchvision import datasets
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

VAL_DIR = "/home/ubuntu/data/robust_dataset_split_30c_v2/val"
WARMUP_ITERS = 20  # excluded from timing stats, not from accuracy (accuracy uses every image regardless)


def load_model(cfg, device):
    import open_clip

    if cfg["loader"] == "openclip_pretrained":
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=cfg["model_name"], pretrained=cfg["pretrained"],
        )
        tokenizer = open_clip.get_tokenizer(cfg["model_name"])
    else:
        from open_clip import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained(cfg["model_id"])
        tokenizer = get_tokenizer(cfg["model_id"])

    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_one_image(model, preprocess, pil_image, device):
    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    feats = model.encode_image(tensor)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def encode_text(model, tokenizer, texts, device):
    tokens = tokenizer(texts).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def run_image_to_label(model, preprocess, tokenizer, device):
    val_ds = datasets.ImageFolder(VAL_DIR)
    idx_to_class = {v: k for k, v in val_ds.class_to_idx.items()}
    labels = val_ds.classes
    n_images = len(val_ds)
    print(f"Val set: {n_images} images across {len(labels)} classes")

    is_cuda = device.type == "cuda"

    # 1. Text embeddings computed ONCE, upfront — matches deployment (offline text tower)
    label_matrix = encode_text(model, tokenizer, labels, device)
    if is_cuda:
        torch.cuda.synchronize()

    # 2. Warmup — not counted in timing stats. Images loaded and discarded one at a
    # time (not preloaded — preloading all 779 into RAM at once can OOM the Jetson).
    warmup_paths = [p for p, _ in val_ds.imgs[:WARMUP_ITERS]]
    for p in warmup_paths:
        img = Image.open(p).convert("RGB")
        encode_one_image(model, preprocess, img, device)
        del img
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 3. Real per-image loop over the FULL val set — this is the actual benchmark.
    # Only the encode_one_image() call is timed, so disk I/O doesn't pollute the
    # latency measurement, without needing to hold everything in memory at once.
    latencies_ms = []
    sims_rows = []
    y_true = []
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    for img_path, label_idx in tqdm(val_ds.imgs, desc="Per-image inference", unit="img"):
        true_label = idx_to_class[label_idx]
        pil_img = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        feat = encode_one_image(model, preprocess, pil_img, device)
        if is_cuda:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1e3)

        sim = (feat @ label_matrix.T).squeeze(0).cpu().numpy()  # [30]
        sims_rows.append(sim)
        y_true.append(label_to_idx[true_label])
        del pil_img

    sims = np.stack(sims_rows)  # [N, 30]
    y_true = np.array(y_true)
    top1_acc = top_k_accuracy_score(y_true, sims, k=1, labels=np.arange(len(labels))) * 100
    top5_acc = top_k_accuracy_score(y_true, sims, k=5, labels=np.arange(len(labels))) * 100

    lat = np.array(latencies_ms)
    print(f"\nTop-1: {top1_acc:.2f}%   Top-5: {top5_acc:.2f}%   (n={n_images})")
    print(f"Per-image latency (bs=1, n={len(lat)}, {WARMUP_ITERS} warmup iters excluded):")
    print(f"  mean   = {lat.mean():7.2f} ms")
    print(f"  median = {np.median(lat):7.2f} ms")
    print(f"  std    = {lat.std():7.2f} ms")
    print(f"  p95    = {np.percentile(lat, 95):7.2f} ms")
    print(f"  min    = {lat.min():7.2f} ms")
    print(f"  max    = {lat.max():7.2f} ms")
    print(f"  throughput (1 / mean) = {1000 / lat.mean():.2f} img/s")
    if is_cuda:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  peak GPU memory       = {peak_mem:.2f} GB")


def run_one_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("\n" + "=" * 70)
    print(f"MODEL: {cfg['key']}")
    print("=" * 70)

    model, preprocess, tokenizer = load_model(cfg, device)
    run_image_to_label(model, preprocess, tokenizer, device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

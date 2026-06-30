"""
download_test_images.py
================================
Downloads 250 images per character label using multiple DuckDuckGo queries.
No API key, no account, no cost.

Install:
    pip install ddgs requests Pillow tqdm

Usage:
    python download_test_images.py
    python download_test_images.py --n 250 --workers 8
    python download_test_images.py --labels "Raiden Shogun from Genshin Impact"
"""

import argparse
import io
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

try:
    from ddgs import DDGS
except ImportError:
    raise ImportError("Run: pip install ddgs")


# ── config ────────────────────────────────────────────────────────────────────

DEFAULT_DATASET_ROOT = Path("/home/ubuntu/data/robust_dataset")
DEFAULT_OUTPUT_ROOT  = Path("/home/ubuntu/data/robust_dataset_domain_transfer")
IMAGES_PER_LABEL     = 250
IGNORED_FOLDERS      = {"originals"}
MIN_SIZE             = (64, 64)
REQUEST_TIMEOUT      = 8
DEFAULT_WORKERS      = 8
PAUSE_BETWEEN_QUERIES = (2.0, 4.0)   # pause between DDG queries (avoid rate limit)
PAUSE_BETWEEN_LABELS  = (4.0, 8.0)   # pause between labels
DDG_RETRY_DELAYS      = [15, 30, 60] # backoff on rate limit


# ── query templates ───────────────────────────────────────────────────────────

def build_queries(label: str) -> list[str]:
    """
    'Raiden Shogun from Genshin Impact' →
      1. 'Raiden Shogun'                        (character name only)
      2. 'Raiden Shogun Genshin Impact'          (character name + source)
      3. 'Raiden Shogun image'                   (character name + image)
      4. 'Raiden Shogun fanart'
      5. 'Raiden Shogun official art'
      6. 'Raiden Shogun wallpaper'
      7. 'Raiden Shogun screenshot'
      8. 'Raiden Shogun render'
    """
    if " from " in label:
        name, source = label.split(" from ", 1)
        name, source = name.strip(), source.strip()
    else:
        name, source = label.strip(), ""

    # your 3 required queries first
    queries = [
        name,
        f"{name} {source}".strip(),
        f"{name} image",
    ]

    # extra queries to reach 250
    extras = ["fanart", "official art", "wallpaper", "screenshot", "render"]
    for extra in extras:
        queries.append(f"{name} {extra}")

    # deduplicate while preserving order
    seen = set()
    result = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            result.append(q)
    return result


# ── DDG fetching ──────────────────────────────────────────────────────────────

def fetch_urls_for_query(query: str) -> list[str]:
    """Fetch image URLs for one query with retry on rate limit."""
    for attempt, delay in enumerate([0] + DDG_RETRY_DELAYS):
        if delay:
            print(f"    ⏳ Rate limited — waiting {delay}s (retry {attempt})...")
            time.sleep(delay)
        try:
            results = DDGS().images(query, max_results=100, safesearch="off")
            return [r["image"] for r in results if r.get("image")]
        except Exception as e:
            err = str(e)
            if "Ratelimit" in err or "429" in err or "403" in err:
                if attempt == len(DDG_RETRY_DELAYS):
                    print(f"    ✗ Gave up after retries.")
                    return []
                continue
            else:
                print(f"    ✗ Error: {e}")
                return []
    return []


def collect_urls(label: str, needed: int) -> list[str]:
    """Run queries until we have enough unique URLs."""
    queries = build_queries(label)
    all_urls = []
    seen = set()

    for i, query in enumerate(queries):
        if len(all_urls) >= needed * 1.5:
            break  # have plenty, stop early

        urls = fetch_urls_for_query(query)
        new = [u for u in urls if u not in seen]
        seen.update(new)
        all_urls.extend(new)
        print(f"    [{i+1}/{len(queries)}] '{query}' → {len(urls)} urls ({len(new)} new, {len(all_urls)} total)")

        if i < len(queries) - 1 and len(all_urls) < needed * 1.5:
            time.sleep(random.uniform(*PAUSE_BETWEEN_QUERIES))

    return all_urls


# ── image downloading ─────────────────────────────────────────────────────────

def download_image(url: str, save_path: Path) -> bool:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)
        r.raise_for_status()
        data = b"".join(r.iter_content(chunk_size=8192))
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        if img.size[0] < MIN_SIZE[0] or img.size[1] < MIN_SIZE[1]:
            return False
        img.convert("RGB").save(save_path, "JPEG", quality=90)
        return True
    except Exception:
        if save_path.exists():
            save_path.unlink()
        return False


def download_all(urls: list[str], save_dir: Path, n: int, workers: int) -> tuple[int, int]:
    save_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counter = {"saved": 0, "failed": 0}

    def _download_one(url: str) -> bool:
        with lock:
            if counter["saved"] >= n:
                return False
            idx = counter["saved"] + 1
            save_path = save_dir / f"{idx:04d}.jpg"
            counter["saved"] += 1

        ok = download_image(url, save_path)
        if not ok:
            with lock:
                counter["saved"] -= 1
                counter["failed"] += 1
        return ok

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, url): url for url in urls}
        with tqdm(total=min(n, len(urls)), desc="  Downloading", leave=False, unit="img") as pbar:
            for future in as_completed(futures):
                if future.result():
                    pbar.update(1)
                if counter["saved"] >= n:
                    for f in futures:
                        f.cancel()
                    break

    # rename sequentially to fill any gaps
    files = sorted(save_dir.glob("*.jpg"))
    for i, f in enumerate(files, start=1):
        target = save_dir / f"{i:04d}.jpg"
        if f != target:
            f.rename(target)

    return counter["saved"], counter["failed"]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root",  type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n",            type=int,  default=IMAGES_PER_LABEL,
                        help="Target images per label (default: 250)")
    parser.add_argument("--workers",      type=int,  default=DEFAULT_WORKERS,
                        help="Parallel download threads (default: 8)")
    parser.add_argument("--labels",       nargs="+", default=None)
    args = parser.parse_args()

    labels = args.labels or [
        d.name for d in sorted(args.dataset_root.iterdir())
        if d.is_dir() and d.name not in IGNORED_FOLDERS
    ]
    if not labels:
        print(f"No labels found in {args.dataset_root}")
        return

    test_root = args.output_root / "test"
    test_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root  : {args.dataset_root}")
    print(f"Output (test) : {test_root}")
    print(f"Labels        : {len(labels)}")
    print(f"Images/label  : {args.n}")
    print(f"Workers       : {args.workers}")
    print(f"Cost          : free")
    print(f"Time est.     : ~45-60 min for all labels\n")

    summary = []

    for i, label in enumerate(labels):
        print(f"\n[{i+1}/{len(labels)}] {label}")
        save_dir = test_root / label

        # resume: skip if already complete
        existing = list(save_dir.glob("*.jpg")) if save_dir.exists() else []
        if len(existing) >= args.n:
            print(f"  ✓ Already have {len(existing)} images, skipping.")
            summary.append((label, len(existing), 0))
            continue

        urls = collect_urls(label, args.n)
        print(f"  Total unique URLs : {len(urls)}")

        if not urls:
            print(f"  ✗ No URLs found.")
            summary.append((label, 0, 0))
            continue

        saved, failed = download_all(urls, save_dir, args.n, args.workers)
        summary.append((label, saved, failed))
        print(f"  ✓ {saved}/{args.n} saved  ({failed} failed/skipped)")

        if i < len(labels) - 1:
            pause = random.uniform(*PAUSE_BETWEEN_LABELS)
            print(f"  Pausing {pause:.1f}s...")
            time.sleep(pause)

    print("\n" + "═" * 62)
    print("SUMMARY")
    print("═" * 62)
    total = 0
    for label, saved, failed in summary:
        flag = "⚠" if saved < args.n * 0.8 else "✓"
        print(f"  {flag} {label:<45} {saved:>3}/{args.n}")
        total += saved
    print(f"\n  Total images : {total}")
    print(f"  Location     : {test_root}")


if __name__ == "__main__":
    main()
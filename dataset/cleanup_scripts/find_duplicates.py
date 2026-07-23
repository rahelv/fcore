#!/usr/bin/env python3

from pathlib import Path
from PIL import Image
import imagehash
import hashlib
import argparse
import csv
import shutil
import sys

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(folder: Path):
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and "_duplicates_found" not in p.parts
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_image_info(path: Path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size

            return {
                "path": path,
                "sha256": sha256(path),
                "phash": imagehash.phash(img),
                "dhash": imagehash.dhash(img),
                "whash": imagehash.whash(img),
                "width": width,
                "height": height,
                "pixels": width * height,
                "file_size": path.stat().st_size,
            }

    except Exception as e:
        print(f"Skipping unreadable file: {path}")
        print(f"Reason: {e}")
        return None


def is_duplicate(a, b, mode: str):
    if a["sha256"] == b["sha256"]:
        return True, "exact", 0, 0, 0

    phash_diff = a["phash"] - b["phash"]
    dhash_diff = a["dhash"] - b["dhash"]
    whash_diff = a["whash"] - b["whash"]

    if mode == "strict":
        duplicate = phash_diff <= 4 and dhash_diff <= 6 and whash_diff <= 8

    elif mode == "aggressive":
        duplicate = (
            sum(
                [
                    phash_diff <= 12,
                    dhash_diff <= 14,
                    whash_diff <= 16,
                ]
            )
            >= 2
        )

    else:
        # balanced
        duplicate = (
            sum(
                [
                    phash_diff <= 8,
                    dhash_diff <= 10,
                    whash_diff <= 12,
                ]
            )
            >= 2
        )

    return duplicate, "near-duplicate", phash_diff, dhash_diff, whash_diff


def choose_keeper(a, b):
    # Keep higher-resolution image
    if a["pixels"] > b["pixels"]:
        return a, b
    if b["pixels"] > a["pixels"]:
        return b, a

    # If same resolution, keep larger file
    if a["file_size"] >= b["file_size"]:
        return a, b
    return b, a


def unique_target_path(path: Path):
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    counter = 1

    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    parser = argparse.ArgumentParser(
        description="Find duplicates and near-duplicates inside one image folder."
    )

    parser.add_argument("folder", help="Folder to deduplicate")

    parser.add_argument(
        "--mode",
        choices=["strict", "balanced", "aggressive"],
        default="balanced",
        help="Detection strength. Default: balanced",
    )

    parser.add_argument(
        "--action",
        choices=["report", "move", "delete"],
        default="report",
        help="Default: report only. Use move before delete.",
    )

    parser.add_argument(
        "--report",
        default="duplicate_report.csv",
        help="CSV report path. Default: duplicate_report.csv",
    )

    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        print(f"Error: not a folder: {folder}")
        sys.exit(1)

    print(f"Folder: {folder}")
    print(f"Mode: {args.mode}")
    print(f"Action: {args.action}")
    print()

    image_paths = find_images(folder)
    print(f"Found {len(image_paths)} images.")

    infos = []
    for path in image_paths:
        info = get_image_info(path)
        if info is not None:
            infos.append(info)

    print(f"Usable images: {len(infos)}")
    print("Comparing all image pairs...")
    print()

    duplicate_pairs = []
    duplicate_paths = set()

    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            a = infos[i]
            b = infos[j]

            duplicate, reason, phash_diff, dhash_diff, whash_diff = is_duplicate(
                a, b, args.mode
            )

            if not duplicate:
                continue

            keeper, duplicate_img = choose_keeper(a, b)

            if duplicate_img["path"] in duplicate_paths:
                continue

            duplicate_paths.add(duplicate_img["path"])

            duplicate_pairs.append(
                {
                    "reason": reason,
                    "phash_diff": phash_diff,
                    "dhash_diff": dhash_diff,
                    "whash_diff": whash_diff,
                    "keep": str(keeper["path"]),
                    "duplicate": str(duplicate_img["path"]),
                }
            )

    report_path = Path(args.report).expanduser().resolve()

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reason",
                "phash_diff",
                "dhash_diff",
                "whash_diff",
                "keep",
                "duplicate",
            ],
        )
        writer.writeheader()
        writer.writerows(duplicate_pairs)

    print(f"Duplicate candidates found: {len(duplicate_pairs)}")
    print(f"Unique files marked as duplicates: {len(duplicate_paths)}")
    print(f"Report written to: {report_path}")
    print()

    for pair in duplicate_pairs[:30]:
        print(f"[{pair['reason']}]")
        print(f"KEEP:      {pair['keep']}")
        print(f"DUPLICATE: {pair['duplicate']}")
        print(
            f"pHash={pair['phash_diff']} "
            f"dHash={pair['dhash_diff']} "
            f"wHash={pair['whash_diff']}"
        )
        print()

    if len(duplicate_pairs) > 30:
        print(f"... showing first 30 of {len(duplicate_pairs)}")
        print()

    if args.action == "report":
        print("No files were changed.")
        print("Check duplicate_report.csv first, then run with --action move.")
        return

    if args.action == "move":
        duplicates_folder = folder / "_duplicates_found"
        duplicates_folder.mkdir(exist_ok=True)

        moved = 0

        for duplicate_path in sorted(duplicate_paths):
            duplicate_path = Path(duplicate_path)

            relative = duplicate_path.relative_to(folder)
            target = unique_target_path(duplicates_folder / relative)

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(duplicate_path), str(target))
            moved += 1

        print(f"Moved {moved} duplicates to:")
        print(duplicates_folder)

    elif args.action == "delete":
        deleted = 0

        for duplicate_path in sorted(duplicate_paths):
            Path(duplicate_path).unlink()
            deleted += 1

        print(f"Deleted {deleted} duplicate files.")

if __name__ == "__main__":
    main()


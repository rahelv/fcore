#!/usr/bin/env python3
"""
Build the 6 balanced subsets for the mixed domain-transfer test.

Design: 3 complementary pairs of 5-class subsets drawn from the 10 classes.
For each pair (S, S_complement) we run two finetunings, so across the 6 runs
every class is finetuned in EXACTLY 3 runs and held-out (character-only) in
EXACTLY 3 runs. This balance is guaranteed by the complementary-pair
construction, independent of which split is chosen.

For each run this script filters the full cosplay training CSV down to the
rows whose caption is one of that run's 5 finetuned classes, and writes:
  - cosplay_subset_run{k}.csv   (training data for run k)
  - subsets_manifest.json / .csv (which classes are finetuned / held-out)
"""
import argparse
import csv
import json
from pathlib import Path

# The 10 classes (must match the captions used in val/test metadata exactly).
CLASSES = [
    "Ai Hoshino from Oshi no Ko",                                     # 0
    "Asuka Langley Soryu in Red Suit from Neon Genesis Evangelion",   # 1
    "Asuka Langley Soryu in School Uniform from Neon Genesis Evangelion",  # 2
    "Deadpool from Marvel",                                           # 3
    "Frieren from Frieren",                                           # 4
    "Furina from Genshin Impact",                                     # 5
    "Kafka from Honkai Star Rail",                                    # 6
    "Sailor Moon from Sailor Moon",                                   # 7
    "Spider Man from Marvel",                                         # 8
    "Yor Forger from Spy x Family",                                   # 9
]

# 3 complementary pairs -> 6 runs. Each entry lists the FINETUNED class indices.
# run1 = S1, run2 = complement(S1), run3 = S2, run4 = complement(S2), ...
FINETUNED_IDX = [
    [0, 1, 2, 3, 4],   # run 1  (S1)
    [5, 6, 7, 8, 9],   # run 2  (S1 complement)
    [0, 2, 5, 6, 8],   # run 3  (S2)
    [1, 3, 4, 7, 9],   # run 4  (S2 complement)
    [0, 3, 5, 7, 9],   # run 5  (S3)
    [1, 2, 4, 6, 8],   # run 6  (S3 complement)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cosplay-csv",
        default="/home/ubuntu/data/domain_transfer_10c_split/cosplay_metadata.csv",
        help="Full cosplay training metadata (all 10 classes).",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/ubuntu/data/domain_transfer_10c_split/mixed_test_subsets",
        help="Where to write the filtered per-run training CSVs + manifest.",
    )
    ap.add_argument("--caption-key", default="caption")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full cosplay training rows.
    with open(args.cosplay_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if args.caption_key not in fieldnames:
        raise SystemExit(f"'{args.caption_key}' not in columns {fieldnames}")

    found_caps = sorted(set(r[args.caption_key] for r in rows))
    print(f"Loaded {len(rows)} rows, {len(found_caps)} unique captions from {args.cosplay_csv}")

    # Sanity: every class name should exist in the training captions.
    missing = [c for c in CLASSES if c not in found_caps]
    if missing:
        print("\n!!! WARNING: these class names were NOT found in the training CSV captions:")
        for m in missing:
            print("   ", repr(m))
        print("Training captions present:")
        for c in found_caps:
            print("   ", repr(c))
        print("Fix CLASSES to match the training captions before running, else "
              "subsets will be empty.\n")

    # Balance check.
    counts = {c: 0 for c in CLASSES}
    for idxs in FINETUNED_IDX:
        for i in idxs:
            counts[CLASSES[i]] += 1
    assert all(v == 3 for v in counts.values()), f"Unbalanced design: {counts}"

    manifest = {"classes": CLASSES, "runs": []}
    for k, idxs in enumerate(FINETUNED_IDX, start=1):
        finetuned = [CLASSES[i] for i in idxs]
        heldout = [c for c in CLASSES if c not in finetuned]
        sub_rows = [r for r in rows if r[args.caption_key] in set(finetuned)]

        out_csv = out_dir / f"cosplay_subset_run{k}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(sub_rows)

        per_class_counts = {c: sum(1 for r in sub_rows if r[args.caption_key] == c)
                            for c in finetuned}
        print(f"run {k}: {len(sub_rows):5d} train imgs  finetuned={finetuned}")
        for c, n in per_class_counts.items():
            if n == 0:
                print(f"    !! 0 images for finetuned class {c!r}")

        manifest["runs"].append({
            "run": k,
            "train_csv": str(out_csv),
            "finetuned": finetuned,
            "heldout": heldout,
            "n_train_images": len(sub_rows),
            "per_class_train_counts": per_class_counts,
        })

    (out_dir / "subsets_manifest.json").write_text(json.dumps(manifest, indent=2))

    with open(out_dir / "subsets_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "class", "role"])
        for r in manifest["runs"]:
            for c in r["finetuned"]:
                w.writerow([r["run"], c, "finetuned"])
            for c in r["heldout"]:
                w.writerow([r["run"], c, "heldout"])

    print(f"\nWrote manifest + 6 subset CSVs to {out_dir}")
    print("Balance (times each class is finetuned across the 6 runs):")
    for c, v in counts.items():
        print(f"   {v}x  {c}")


if __name__ == "__main__":
    main()

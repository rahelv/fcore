#!/usr/bin/env python3
"""
Aggregate the 6 mixed-test TEST results.

Per run, split the 10 per-class accuracies into the 5 FINETUNED-on-cosplay
classes vs the 5 HELD-OUT (character-only) classes, using the balanced design.
Reports:
  - per-run: finetuned-group vs held-out-group top-1 (macro = mean of per-class,
    and micro = pooled over images)
  - across the 6 runs: mean +/- std for each group
  - per-class: mean top-1 over the 3 runs where the class was finetuned vs the
    3 runs where it was held-out, next to the two reference endpoints
    (character-only and full character->cosplay).
Writes mixed_test_summary.json and mixed_test_summary.csv.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

CLASSES = [
    "Ai Hoshino from Oshi no Ko",
    "Asuka Langley Soryu in Red Suit from Neon Genesis Evangelion",
    "Asuka Langley Soryu in School Uniform from Neon Genesis Evangelion",
    "Deadpool from Marvel",
    "Frieren from Frieren",
    "Furina from Genshin Impact",
    "Kafka from Honkai Star Rail",
    "Sailor Moon from Sailor Moon",
    "Spider Man from Marvel",
    "Yor Forger from Spy x Family",
]
FINETUNED_IDX = [
    [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 2, 5, 6, 8],
    [1, 3, 4, 7, 9], [0, 3, 5, 7, 9], [1, 2, 4, 6, 8],
]


def macro_micro(pc, classes):
    accs = [pc[c]["top1_accuracy"] for c in classes]
    corr = sum(pc[c]["top1_correct"] for c in classes)
    tot = sum(pc[c]["total"] for c in classes)
    return round(sum(accs) / len(accs), 2), round(corr / tot * 100, 2)


def load_pc(path):
    return json.load(open(path))["per_caption"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/mixed_test")
    ap.add_argument("--char-only-test",
                    default="results/10c_character/10c_character_epoch_8_test.json")
    ap.add_argument("--char-then-cosplay-test",
                    default="results/10c_character_then_cosplay/10c_char_then_cosplay_epoch_9_test.json")
    args = ap.parse_args()

    base = Path(args.results_dir)

    runs = []
    for k, idxs in enumerate(FINETUNED_IDX, start=1):
        p = base / f"run{k}" / f"run{k}_best_test.json"
        if not p.exists():
            print(f"missing {p}, skipping run {k}")
            continue
        pc = load_pc(p)
        ft = [CLASSES[i] for i in idxs]
        ho = [c for c in CLASSES if c not in ft]
        ft_macro, ft_micro = macro_micro(pc, ft)
        ho_macro, ho_micro = macro_micro(pc, ho)
        ov_macro, ov_micro = macro_micro(pc, CLASSES)
        runs.append({"run": k, "finetuned": ft, "heldout": ho, "pc": pc,
                     "ft_macro": ft_macro, "ft_micro": ft_micro,
                     "ho_macro": ho_macro, "ho_micro": ho_micro,
                     "overall_macro": ov_macro, "overall_micro": ov_micro})

    print("\n=== Per run (test top-1) ===")
    print(f"{'run':>3}  {'finetuned-5 (macro/micro)':>26}  {'heldout-5 (macro/micro)':>24}  {'overall':>7}")
    for r in runs:
        print(f"{r['run']:>3}  {r['ft_macro']:>11.2f} /{r['ft_micro']:>7.2f}     "
              f"{r['ho_macro']:>11.2f} /{r['ho_micro']:>7.2f}     {r['overall_macro']:>6.2f}")

    def ms(key):
        vals = [r[key] for r in runs]
        return (round(st.mean(vals), 2), round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0)

    agg = {k: ms(k) for k in ["ft_macro", "ft_micro", "ho_macro", "ho_micro",
                              "overall_macro", "overall_micro"]}
    print("\n=== Across 6 runs (mean +/- std) ===")
    print(f"finetuned-5 macro : {agg['ft_macro'][0]:.2f} +/- {agg['ft_macro'][1]:.2f}")
    print(f"held-out-5  macro : {agg['ho_macro'][0]:.2f} +/- {agg['ho_macro'][1]:.2f}")
    print(f"overall     macro : {agg['overall_macro'][0]:.2f} +/- {agg['overall_macro'][1]:.2f}")

    # Per-class: finetuned vs held-out averages + reference endpoints.
    char_pc = load_pc(args.char_only_test) if Path(args.char_only_test).exists() else {}
    ctc_pc = load_pc(args.char_then_cosplay_test) if Path(args.char_then_cosplay_test).exists() else {}

    per_class = []
    for c in CLASSES:
        ft_vals = [r["pc"][c]["top1_accuracy"] for r in runs if c in r["finetuned"]]
        ho_vals = [r["pc"][c]["top1_accuracy"] for r in runs if c in r["heldout"]]
        row = {
            "class": c,
            "n_finetuned_runs": len(ft_vals),
            "n_heldout_runs": len(ho_vals),
            "finetuned_mean": round(st.mean(ft_vals), 2) if ft_vals else None,
            "heldout_mean": round(st.mean(ho_vals), 2) if ho_vals else None,
            "ref_char_only": char_pc.get(c, {}).get("top1_accuracy"),
            "ref_char_then_cosplay": ctc_pc.get(c, {}).get("top1_accuracy"),
        }
        if row["finetuned_mean"] is not None and row["heldout_mean"] is not None:
            row["delta_ft_minus_ho"] = round(row["finetuned_mean"] - row["heldout_mean"], 2)
        per_class.append(row)

    print("\n=== Per class: finetuned vs held-out (mean test top-1 over 3 runs each) ===")
    print(f"{'class':52} {'FT':>6} {'HO':>6} {'d':>6}  {'charOnly':>8} {'char>cos':>8}")
    for r in per_class:
        d = r.get("delta_ft_minus_ho")
        print(f"{r['class']:52} {r['finetuned_mean']:>6} {r['heldout_mean']:>6} "
              f"{(d if d is not None else ''):>6}  "
              f"{str(r['ref_char_only']):>8} {str(r['ref_char_then_cosplay']):>8}")

    summary = {"per_run": [{k: v for k, v in r.items() if k != "pc"} for r in runs],
               "aggregate": agg, "per_class": per_class}
    (base / "mixed_test_summary.json").write_text(json.dumps(summary, indent=2))

    with open(base / "mixed_test_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "finetuned_mean", "heldout_mean", "delta_ft_minus_ho",
                    "ref_char_only", "ref_char_then_cosplay"])
        for r in per_class:
            w.writerow([r["class"], r["finetuned_mean"], r["heldout_mean"],
                        r.get("delta_ft_minus_ho"), r["ref_char_only"],
                        r["ref_char_then_cosplay"]])

    print(f"\nWrote {base/'mixed_test_summary.json'} and .csv")


if __name__ == "__main__":
    main()

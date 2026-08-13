#!/usr/bin/env python3
"""
For each mixed-test run: pick the best epoch by VAL top-1 (earliest epoch on
ties), then evaluate that checkpoint on the TEST set (all 10 classes) by
calling 00_evaluate.py. Writes results/mixed_test/run{k}/run{k}_best_test.json.
"""
import argparse
import glob
import json
import re
import subprocess
from pathlib import Path

TEST_METADATA = "/home/ubuntu/data/domain_transfer_10c_split/test_metadata.csv"
CKPT_TMPL = "/home/ubuntu/open_clip/logs/mixed_test_run{k}_finetuning/mixed_test_run{k}_train/checkpoints/epoch_{e}.pt"


def best_epoch(run_dir):
    best_e, best_acc = None, -1.0
    for f in glob.glob(str(run_dir / "run*_epoch_*_val.json")):
        e = int(re.search(r"epoch_(\d+)_val", f).group(1))
        acc = json.load(open(f))["overall"]["top1_accuracy"]
        if acc > best_acc or (acc == best_acc and (best_e is None or e < best_e)):
            best_acc, best_e = acc, e
    return best_e, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/mixed_test")
    ap.add_argument("--eval-script", default="00_evaluate.py")
    args = ap.parse_args()

    base = Path(args.results_dir)
    summary = []
    for k in range(1, 7):
        run_dir = base / f"run{k}"
        e, acc = best_epoch(run_dir)
        if e is None:
            print(f"run {k}: no val json found, skipping")
            continue
        ckpt = CKPT_TMPL.format(k=k, e=e)
        out = run_dir / f"run{k}_best_test.json"
        print(f"run {k}: best epoch {e} (val {acc:.2f}) -> test eval")
        subprocess.run([
            "python3", args.eval_script,
            "--metadata", TEST_METADATA,
            "--checkpoint", ckpt,
            "--output", str(out),
        ], check=True)
        test_acc = json.load(open(out))["overall"]["top1_accuracy"]
        summary.append({"run": k, "best_epoch": e, "val_top1": acc,
                        "test_top1": test_acc, "test_json": str(out)})

    (base / "best_epochs_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nrun  best_epoch  val_top1  test_top1")
    for s in summary:
        print(f"{s['run']:>3}  {s['best_epoch']:>10}  {s['val_top1']:>8.2f}  {s['test_top1']:>9.2f}")


if __name__ == "__main__":
    main()

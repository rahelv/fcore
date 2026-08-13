#!/bin/bash
# Evaluate every epoch of each mixed-test run on the FULL 10-class VAL set.
# (All 10 captions are in the CSV, so predictions are 10-way -- finetuned-5 and
#  held-out-5 are scored in the same decision space.)
# After this, run select_and_test.py to pick the best epoch per run (by val
# top-1) and evaluate it on the test set.
set -e

cd /home/ubuntu/fcore/openCLIP/domain_transfer   # location of 00_evaluate.py

VAL_METADATA=/home/ubuntu/data/domain_transfer_10c_split/val_metadata.csv
EPOCHS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

for k in 1 2 3 4 5 6; do
  CKPT_DIR=/home/ubuntu/open_clip/logs/mixed_test_run${k}_finetuning/mixed_test_run${k}_train/checkpoints
  OUT_DIR=results/mixed_test/run${k}
  mkdir -p "$OUT_DIR"
  for e in $EPOCHS; do
    [ -f "${CKPT_DIR}/epoch_${e}.pt" ] || continue
    python3 00_evaluate.py \
      --metadata "$VAL_METADATA" \
      --checkpoint "${CKPT_DIR}/epoch_${e}.pt" \
      --output "${OUT_DIR}/run${k}_epoch_${e}_val.json"
  done
done

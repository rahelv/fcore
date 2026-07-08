#!/bin/bash
set -e

CKPT_DIR=/home/ubuntu/open_clip/logs/10c_character_finetuning/10c_character_train/checkpoints
METADATA=/home/ubuntu/data/domain_transfer_10c_split/val_metadata.csv

for e in 1 2 3 4 5  6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  python3 00_evaluate.py \
    --metadata "$METADATA" \
    --checkpoint "$CKPT_DIR/epoch_${e}.pt" \
    --output "results/10c_character/10c_character_epoch_${e}_val.json"
done
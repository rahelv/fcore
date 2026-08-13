#!/bin/bash
set -e

CKPT_DIR=/home/ubuntu/open_clip/logs/40c_extended_finetuning/40c_cosplay_and_character_train/checkpoints/
METADATA=/home/ubuntu/data/40c_extended_w_char_dataset/val_metadata_plain.csv

for e in 1 2 3 4 5  6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  python3 00_evaluate.py \
    --metadata "$METADATA" \
    --checkpoint "$CKPT_DIR/epoch_${e}.pt" \
    --output "results/40c_extended/40c_extended_epoch_${e}_val.json"
done
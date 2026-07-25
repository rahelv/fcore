#!/bin/bash
set -e

#CKPT_DIR=/home/ubuntu/open_clip/logs/V3_plain_siglip2_finetuning_seed42/V3_plain_siglip2_train_seed42/checkpoints
CKPT_DIR=/home/ubuntu/open_clip/logs/V4_plain_siglip2_finetuning_seed42/V4_plain_siglip2_train_seed42/checkpoints
METADATA=/home/ubuntu/data/robust_dataset_split_30c_v2/val_metadata_plain.csv

for e in 1 2 3 4 5; do
  python3 00_evaluate_baseline.py \
    --metadata "$METADATA" \
    --checkpoint "$CKPT_DIR/epoch_${e}.pt" \
    --output "V4_plain_epoch_${e}_val.json"
done
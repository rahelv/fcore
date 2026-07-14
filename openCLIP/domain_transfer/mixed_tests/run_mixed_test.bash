#!/bin/bash
# Mixed domain-transfer test: 6 balanced runs.
# Each run starts from the 10-class CHARACTER-pretrained checkpoint (epoch_8)
# and finetunes on COSPLAY images for that run's 5 finetuned classes only.
# Hyperparameters are identical to 10c_character_then_cosplay (default InfoNCE
# loss -- NO --siglip flag).
set -e

source /home/ubuntu/fcore/.venv/bin/activate
cd /home/ubuntu/open_clip/src

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=fcore_openclip_finetuning

# Character-pretrained base checkpoint (all 10 classes).
PRETRAINED=/home/ubuntu/open_clip/logs/10c_character_finetuning/10c_character_train/checkpoints/epoch_8.pt

# Directory holding the per-run filtered training CSVs (from make_subsets.py).
SUBSET_DIR=/home/ubuntu/data/domain_transfer_10c_split/mixed_test_subsets

for k in 1 2 3 4 5 6; do
  echo "================ MIXED TEST RUN ${k} ================"
  torchrun --nproc_per_node 2 -m open_clip_train.main -- \
    --name mixed_test_run${k}_train \
    --seed 42 \
    --dataset-type csv \
    --train-data ${SUBSET_DIR}/cosplay_subset_run${k}.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator "," \
    --model hf-hub:timm/ViT-B-16-SigLIP2-256 \
    --pretrained ${PRETRAINED} \
    --batch-size 32 \
    --precision amp \
    --workers 4 \
    --warmup 50 \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 10 \
    --save-frequency 1 \
    --report-to wandb \
    --logs /home/ubuntu/open_clip/logs/mixed_test_run${k}_finetuning
done

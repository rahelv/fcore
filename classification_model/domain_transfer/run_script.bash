#!/usr/bin/env bash

# Example script for running the training. Here:
# C10-Character (scratch), then (on success) C10-Character→Cosplay.


set -euo pipefail

source /home/ubuntu/fcore/.venv/bin/activate
cd /home/ubuntu/fcore/classification_model/domain_transfer

export CUDA_VISIBLE_DEVICES=0

python3 train_domain_transfer.py --config configs/c10_character.yaml
python3 train_domain_transfer.py --config configs/c10_character_to_cosplay.yaml

"""Launch a CutMix sweep agent on this GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 run_agent_3b_cutmix.py <sweep_id> <count>
    CUDA_VISIBLE_DEVICES=1 python3 run_agent_3b_cutmix.py <sweep_id> <count>
"""
import sys
import wandb
from train_sweep_3b_cutmix import train, PROJECT, ENTITY

if len(sys.argv) < 3:
    print("Usage: python3 run_agent_3b_cutmix.py <sweep_id> <count>")
    sys.exit(1)

sweep_id = sys.argv[1]
count    = int(sys.argv[2])

wandb.agent(f"{ENTITY}/{PROJECT}/{sweep_id}", function=train, count=count)

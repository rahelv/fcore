"""
Run this ONCE to create the sweep and get the sweep ID.
Then pass that ID to run_agent.py.

Usage:
    python3 create_sweep.py
"""
import wandb
from train_sweep import sweep_config, PROJECT, ENTITY

sweep_id = wandb.sweep(
    sweep=sweep_config,
    project=PROJECT,
    entity=ENTITY,
)

print(f"\n✅ Sweep created!")
print(f"   Sweep ID : {sweep_id}")
print(f"   Dashboard: https://wandb.ai/{ENTITY}/{PROJECT}/sweeps/{sweep_id}")
print(f"\nNow launch agents with:")
print(f"   CUDA_VISIBLE_DEVICES=0 python3 run_agent.py {sweep_id}")
print(f"   CUDA_VISIBLE_DEVICES=1 python3 run_agent.py {sweep_id}")

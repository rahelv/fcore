"""
Run this once per GPU to launch an agent on an existing sweep.
Each agent picks up different runs from the same sweep queue.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 run_agent.py <sweep_id>
    CUDA_VISIBLE_DEVICES=1 python3 run_agent.py <sweep_id>
"""

import sys
import wandb
from train_sweep_v2 import train, PROJECT, ENTITY

if len(sys.argv) != 2:
    print("Usage: python3 run_agent.py <sweep_id>")
    sys.exit(1)

sweep_id = sys.argv[1]

print(f"Starting agent for sweep: {sweep_id}")

wandb.agent(
    sweep_id,
    function=train,
    entity=ENTITY,
    project=PROJECT,
    count=40,  # this agent will run 20 experiments
    # two agents × 20 = 40 total runs
)

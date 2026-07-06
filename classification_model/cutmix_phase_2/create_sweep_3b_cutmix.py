"""Run once to register the CutMix sweep with W&B."""
import wandb
from train_sweep_3b_cutmix import sweep_config, PROJECT, ENTITY

sweep_id = wandb.sweep(sweep_config, project=PROJECT, entity=ENTITY)
print(f"\nSweep created: {sweep_id}")
print(f"\nRun on each GPU:")
print(f"  CUDA_VISIBLE_DEVICES=0 python3 run_agent_3b_cutmix.py {sweep_id} 25")
print(f"  CUDA_VISIBLE_DEVICES=1 python3 run_agent_3b_cutmix.py {sweep_id} 25")

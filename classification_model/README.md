Goal: 
- Hyperparameter Sweep 
- Data Augmentations and other Regularization Techniques 
- Final Models (Pretrained + Not Pretrained)

wandb sweep configuration is stored in yaml files. 

The wandb CLI can be used to initialize a sweep: 

wandb sweep --project sweeps_demo config.yaml 

this returns sweep ID which can be used to run agents. 

https://docs.wandb.ai/models/sweeps/initialize-sweeps#cli

Run Sweep: 

wandb agent sweep_id

example for cutmix_phase_2 :

wandb sweep sweep_configurations/30c_v2_cutmix_sweep_phase2.yaml

One agent per GPU: CUDA_VISIBLE_DEVICES=0 wandb agent <entity>/<project>/<sweep_id>
start the agent from cutmix_phase_2 

run_cap stops the whole sweep at 50. 


20c and 30c not runnable bc historical (different configuration, can be found in repo history)
source /home/ubuntu/fcore/.venv/bin/activate
# enter the src folder of the open_clip repository
cd /home/ubuntu/open_clip/src

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=fcore_openclip_finetuning

torchrun --nproc_per_node 2 -m open_clip_train.main -- \
  --name V4_plain_siglip2_train \
  --dataset-type csv \
  --train-data /home/ubuntu/data/robust_dataset_split_30c_v2/train_metadata_plain.csv \
  --val-data /home/ubuntu/data/robust_dataset_split_30c_v2/val_metadata_plain.csv \
  --val-frequency 1 \
  --csv-img-key filepath \
  --csv-caption-key caption \
  --csv-separator "," \
  --model hf-hub:timm/ViT-B-16-SigLIP2-256 \
  --siglip \
  --batch-size 32 \
  --precision amp \
  --workers 4 \
  --warmup 50 \
  --lr 1e-6 \
  --wd 0.1 \
  --epochs 5 \
  --save-frequency 1 \
  --report-to wandb \
  --logs /home/ubuntu/open_clip/logs/V4_plain_siglip2_finetuning
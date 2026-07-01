# enter the src folder of the open_clip repository
cd
open_clip / src

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=openclip-costume-recognition

torchrun --nproc_per_node 2 -m open_clip_train.main -- \
    --name 02_siglip_train_only \
    --dataset-type csv \
    --train-data /home/ubuntu/data/robust_dataset_split_2/train_metadata.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator "," \
    --model hf-hub:timm/ViT-B-16-SigLIP2-256 \
    --batch-size 32 \
    --precision amp \
    --workers 4 \
    --warmup 50 \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 5 \
    --save-frequency 1 \
     --report-to wandb \
    --logs /home/ubuntu/open_clip/logs/costume_siglip2
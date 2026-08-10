# Domain Transfer (10-class subset, ResNet18 from scratch)

Four experiments comparing training on curated cosplay photographs vs.
mostly-unsorted character artwork. All runs use the best from-scratch
hyperparameters (s8i1hx30, val 0.819), `pretrained: false`, and are
evaluated on the shared 10-class val/test split.

| Experiment | Config | Training data |
|---|---|---|
| C10-Cosplay | `configs/c10_cosplay.yaml` | cosplay only (1,351) |
| C10-Character | `configs/c10_character.yaml` | character only (1,626) |
| C10-Character→Cosplay | `configs/c10_character_to_cosplay.yaml` | init from C10-Character, fine-tune on cosplay (lr/10) |
| C10-Cosplay→Character | `configs/c10_cosplay_to_character.yaml` | init from C10-Cosplay, fine-tune on character (lr/10) |

## Pretrained variant

The same four experiments with ImageNet initialisation
(`configs/*_pretrained.yaml`, run names `*-Pretrained`). Hyperparameters
are identical (s8i1hx30) so the only difference to the from-scratch set
is the initialisation — scratch vs. pretrained is directly comparable
per experiment. Stage-2 pretrained runs load the *pretrained* stage-1
checkpoints (`C10-Cosplay-Pretrained_best.pt` etc.); `pretrained: false`
there because weights come from `init_checkpoint`.

## Run order

Stage 1 (independent, can run in parallel on separate GPUs):

    CUDA_VISIBLE_DEVICES=0 python3 train_domain_transfer.py --config configs/c10_cosplay.yaml
    CUDA_VISIBLE_DEVICES=1 python3 train_domain_transfer.py --config configs/c10_character.yaml

Stage 2 (after stage 1 — they load the `*_best.pt` checkpoints via `init_checkpoint`):

    CUDA_VISIBLE_DEVICES=0 python3 train_domain_transfer.py --config configs/c10_character_to_cosplay.yaml
    CUDA_VISIBLE_DEVICES=1 python3 train_domain_transfer.py --config configs/c10_cosplay_to_character.yaml

Same two-stage order applies to the `*_pretrained.yaml` configs.

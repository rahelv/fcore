# Mixed domain-transfer test

**Question.** Start from the 10-class model pretrained on *character* images.
Finetune only a 5-class subset on *cosplay*, then evaluate all 10 classes on the
cosplay test set. Do the finetuned-5 climb toward cosplay accuracy, and do the
held-out-5 (character-only) degrade from interference?

**Design.** 6 runs = 3 complementary pairs of 5-class subsets. Because each pair
is a subset and its complement, every class is finetuned in exactly 3 runs and
held-out in exactly 3 runs (perfectly balanced). Everything else is identical to
`10c_character_then_cosplay` (same base checkpoint `epoch_8`, lr 1e-6, wd 0.1,
warmup 50, batch 32, seed 42, default InfoNCE loss — no `--siglip`).

## Run order

1. **Build the subsets + filtered training CSVs** (edit paths if needed):
   ```bash
   python3 make_subsets.py \
     --cosplay-csv /home/ubuntu/data/domain_transfer_10c_split/cosplay_metadata.csv \
     --out-dir     /home/ubuntu/data/domain_transfer_10c_split/mixed_test_subsets
   ```
   Check the printout: each class must be finetuned 3× and no subset should have
   0 images. If class names don't match the training captions, fix `CLASSES` at
   the top of `make_subsets.py`.

2. **Train the 6 runs:**
   ```bash
   bash run_mixed_test.bash
   ```

3. **Evaluate every epoch on the 10-class val set:**
   ```bash
   bash eval_mixed_test.bash
   ```

4. **Pick best epoch per run (by val) and evaluate it on test:**
   ```bash
   cd /home/ubuntu/fcore/openCLIP/domain_transfer
   python3 mixed_test/select_and_test.py
   ```

5. **Aggregate finetuned-5 vs held-out-5:**
   ```bash
   python3 mixed_test/aggregate_mixed_test.py
   ```
   Produces `results/mixed_test/mixed_test_summary.{json,csv}` and prints
   per-run, across-run (mean±std), and per-class tables. Per-class results sit
   next to the two reference endpoints: character-only (0% adapted) and full
   character→cosplay (100% adapted).

## Files
- `make_subsets.py` — balanced subset generator + CSV filter + manifest
- `run_mixed_test.bash` — 6 finetuning runs
- `eval_mixed_test.bash` — per-epoch val evaluation (all 10 classes)
- `select_and_test.py` — best-epoch selection + test evaluation
- `aggregate_mixed_test.py` — subgroup + per-class aggregation

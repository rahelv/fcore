"""
Create the combined, seeded Phase-2 sweep (fcore_hyperparam_sweep_3c) and
warm-start its Bayesian optimiser with the existing (unseeded) Phase-2 runs.

W&B can seed a new sweep with existing runs via `wandb.sweep(prior_runs=...)`,
but prior runs must live in the sweep's own project. The old Phase-2 runs are
split across two projects (3b and 3b_cutmix) and the 3b runs never logged a
`cutmix_alpha` key, so this script:

  1. fetches all sweep runs from fcore_hyperparam_sweep_3b and
     fcore_hyperparam_sweep_3b_cutmix,
  2. copies each run's swept hyperparameters + best val/accuracy into
     lightweight "prior" runs in the new project (cutmix_alpha=0.0 is filled
     in for the non-CutMix sweep, tagged `prior` + `unseeded`),
  3. creates the new sweep from the YAML with those runs as priors.

The Gaussian Process starts informed and will not re-explore regions the old
sweeps already covered. Note the priors are unseeded and therefore noisy
observations; final rankings should be based on the new seeded runs only.

Usage (from classification_model/):
    python seed_sweep_from_priors.py --dry-run   # preview what gets copied
    python seed_sweep_from_priors.py             # copy priors + create sweep
    CUDA_VISIBLE_DEVICES=0 wandb agent <printed sweep path>
"""

import argparse
from pathlib import Path

import wandb
import yaml

ENTITY = "rahelvalerie-universit-t-basel"
NEW_PROJECT = "fcore_phase2"
SWEEP_YAML = Path(__file__).parent / "sweep_configurations" / "04_30c_v2_phase2_rerun.yaml"

# source project -> forced cutmix_alpha (None = take from run config)
SOURCE_PROJECTS = {
    "fcore_hyperparam_sweep_3b": 0.0,          # unseeded, no cutmix_alpha key
    "fcore_hyperparam_sweep_3b_cutmix": None,  # unseeded
}

# skip runs that crashed before reaching this epoch (too little signal for the GP)
MIN_EPOCHS = 40

# the parameters the Bayesian optimiser models — must match the YAML
SWEPT_KEYS = [
    "lr", "weight_decay", "dropout_p", "label_smoothing",
    "jitter_strength", "rotation_degrees", "aug_erasing", "cutmix_alpha",
]


def best_val(run):
    """Best validation accuracy of a run (handles both summary layouts)."""
    val = run.summary.get("best_val_acc")
    if val is None:
        acc = run.summary.get("val/accuracy")
        val = acc.get("max") if isinstance(acc, dict) else acc
    return val


def collect_priors():
    api = wandb.Api()
    priors = []
    for project, forced_alpha in SOURCE_PROJECTS.items():
        for run in api.runs(f"{ENTITY}/{project}"):
            if run.sweep is None:
                continue  # skip manual reruns stored in the same project
            val = best_val(run)
            if val is None:
                continue  # crashed before any validation epoch
            if run.summary.get("epoch", 0) < MIN_EPOCHS:
                continue  # crashed too early to be a useful observation
            config = {k: run.config[k] for k in SWEPT_KEYS if k in run.config}
            if forced_alpha is not None:
                config["cutmix_alpha"] = forced_alpha
            config["prior_source"] = f"{project}/{run.id}"
            priors.append((run.name, config, val))
    return priors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be copied, create nothing")
    args = parser.parse_args()

    priors = collect_priors()
    print(f"collected {len(priors)} prior runs")

    if args.dry_run:
        for name, config, val in priors:
            print(f"  {name:<24} val={val:.4f}  {config['prior_source']}")
        return

    prior_ids = []
    for name, config, val in priors:
        run = wandb.init(
            entity=ENTITY, project=NEW_PROJECT, name=f"prior_{name}",
            config=config, tags=["prior", "unseeded"],
            settings=wandb.Settings(silent=True, console="off"),
        )
        run.log({"val/accuracy": val})
        run.finish()
        prior_ids.append(run.id)
        print(f"  copied {name} ({val:.4f}) -> {run.id}")

    sweep_config = yaml.safe_load(SWEEP_YAML.read_text())
    sweep_id = wandb.sweep(
        sweep_config, entity=ENTITY, project=NEW_PROJECT, prior_runs=prior_ids,
    )
    print(f"\nsweep created with {len(prior_ids)} priors:")
    print(f"  CUDA_VISIBLE_DEVICES=0 wandb agent {ENTITY}/{NEW_PROJECT}/{sweep_id}")
    print(f"  CUDA_VISIBLE_DEVICES=1 wandb agent {ENTITY}/{NEW_PROJECT}/{sweep_id}")


if __name__ == "__main__":
    main()

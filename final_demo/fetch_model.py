"""
fetch_model.py
==============

Run this ONCE on a machine with internet. It pulls everything
clip_classifier.py needs from the Hugging Face hub into ./hf_cache, next to
your code, so the demo starts with no network at all.

    python fetch_model.py

Then pull the plug and check it for real:

    python fetch_model.py --verify-offline

That second command is the one that matters. It re-runs the app's exact load
path in a FRESH process with HF_HUB_OFFLINE=1 set before anything is imported
-- which is the only way to test it honestly, because huggingface_hub reads
that variable once at import time and ignores it if you set it later.

Copy the whole final_demo/ folder (including hf_cache/) to the Jetson and it
works offline. If the Jetson has internet right now, just run this there.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# keep this in sync with MODEL_ID in clip_classifier.py (minus the hf-hub: prefix)
REPO_ID = "timm/ViT-B-16-SigLIP2-256"

VERIFY_FLAG = "--verify-offline"
_VERIFYING = VERIFY_FLAG in sys.argv

# ─── environment, set BEFORE any HF import ───────────────────────────────────
# Same variables clip_classifier.py uses. Downloading needs them off; verifying
# needs them on. Either way they have to be set up here, above the imports.
CACHE = Path(__file__).with_name("hf_cache")
CACHE.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_OFFLINE"] = "1" if _VERIFYING else "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1" if _VERIFYING else "0"
# ─────────────────────────────────────────────────────────────────────────────


def load_like_the_app():
    """Exactly what ClipCostumeClassifier.__init__ does, nothing more."""
    from open_clip import create_model_from_pretrained, get_tokenizer

    model_id = f"hf-hub:{REPO_ID}"
    create_model_from_pretrained(model_id)
    get_tokenizer(model_id)          # the call that fails offline


def verify_offline() -> int:
    """Child process: the env block above already ran with OFFLINE=1."""
    try:
        load_like_the_app()
    except Exception as e:
        print("OFFLINE LOAD FAILED\n")
        print(f"{type(e).__name__}: {e}")
        return 1
    print("OFFLINE LOAD OK -- model and tokenizer both came from the cache.")
    return 0


def report_cache():
    files = [f for f in CACHE.rglob("*") if f.is_file()]
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"  {len(files)} files, {size_mb:.0f} MB in {CACHE}")

    # .no_exist markers are how the hub remembers "this file isn't in the repo".
    # They're normal, but they turn into confusing LocalEntryNotFoundError
    # messages offline, so it helps to know they're there.
    absent = sorted({f.name for f in CACHE.rglob("*") if ".no_exist" in f.parts})
    if absent:
        print(f"  cached as absent from the repo: {', '.join(absent)}")


def download() -> int:
    from huggingface_hub import snapshot_download

    print(f"Cache directory : {CACHE}")
    print(f"Downloading     : {REPO_ID}")
    try:
        path = snapshot_download(repo_id=REPO_ID)
    except Exception as e:
        print(f"\nDownload failed: {type(e).__name__}: {e}")
        print("This step needs internet -- that's why it's a separate script.")
        return 1
    print(f"Snapshot at     : {path}")

    # snapshot_download grabs the repo, but AutoTokenizer also looks up sibling
    # files (config.json, tokenizer_config.json) through its own code path.
    # Running the real load once, online, caches whatever that touches.
    print("\nWarming the cache through the app's own load path…")
    try:
        load_like_the_app()
    except Exception as e:
        print(f"Load failed even WITH internet: {type(e).__name__}: {e}")
        return 1
    print("Done.")
    report_cache()

    # The honest test: a fresh process with offline set from the start.
    print("\nVerifying offline in a separate process…")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), VERIFY_FLAG],
                        env=env).returncode
    if rc != 0:
        print("\nThe cache is still incomplete. The error above names the file "
              "that's missing.")
        return rc

    ckpt = Path(__file__).with_name("40c_epoch8.pt")
    if not ckpt.exists():
        print(f"\nNote: {ckpt.name} isn't next to this script. That's your own "
              "finetuned checkpoint, not a hub download -- copy it in yourself.")
    print("\nReady. You can disconnect the network now.")
    return 0


if __name__ == "__main__":
    sys.exit(verify_offline() if _VERIFYING else download())

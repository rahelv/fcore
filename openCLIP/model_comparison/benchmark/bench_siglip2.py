"""
bench_siglip2.py — Image -> Label benchmark for ViT-B-16-SigLIP2-256 only

Usage
-----
    python -u bench_siglip2.py | tee jetson_siglip2_results.txt
"""

from bench_common import run_one_model

CFG = dict(
    key="ViT-B-16-SigLIP2-256",
    loader="openclip_hf",
    model_id="hf-hub:timm/ViT-B-16-SigLIP2-256",
)

if __name__ == "__main__":
    run_one_model(CFG)

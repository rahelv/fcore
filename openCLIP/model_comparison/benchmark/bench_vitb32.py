"""
bench_vitb32.py — Image -> Label benchmark for xlm-roberta-base-ViT-B-32 only

Usage
-----
    python -u bench_vitb32.py | tee jetson_vitb32_results.txt
"""

from bench_common import run_one_model

CFG = dict(
    key="xlm-roberta-base-ViT-B-32",
    loader="openclip_pretrained",
    model_name="xlm-roberta-base-ViT-B-32",
    pretrained="laion5b_s13b_b90k",
)

if __name__ == "__main__":
    run_one_model(CFG)

"""
evaluate_768.py — val-split accuracy evaluation (768-dim, SigLIP2)
=====================================================================
See base_evaluator.py for the shared image-search / prompt-search logic.

Baseline (pretrained) evaluation only.

Usage
-----
    python evaluate_768.py
    python evaluate_768.py --cpu --val-dir /path/to/val
"""

import argparse

import torch
from open_clip import create_model_from_pretrained, get_tokenizer

from base_evaluator import BaseEvaluator

DEFAULT_VAL_DIR = "/home/ubuntu/data/robust_dataset_split_30c_v2/val"
DEFAULT_MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP2-256"


class Evaluator768(BaseEvaluator):
    def __init__(self, val_dir, device, batch_size, top_k, model_id):
        self.model_id = model_id
        super().__init__(val_dir, device, batch_size, top_k)

    def load_model(self):
        model, preprocess = create_model_from_pretrained(self.model_id)
        tokenizer = get_tokenizer(self.model_id)
        model = model.to(self.device).eval()
        return model, preprocess, tokenizer

    @torch.no_grad()
    def encode_images_batch(self, pil_images):
        tensors = [self.preprocess(img) for img in pil_images]
        batch = torch.stack(tensors).to(self.device)
        with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            features = self.model.encode_image(batch)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def encode_text_batch(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            features = self.model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def describe(self):
        return f"{self.model_id} (pretrained)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-dir", default=DEFAULT_VAL_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--output", default="results_768_val.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    evaluator = Evaluator768(args.val_dir, device, args.batch_size, args.top_k, args.model)
    output = evaluator.run()
    evaluator.save(output, args.output)
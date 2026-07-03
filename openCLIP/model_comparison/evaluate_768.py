"""
evaluate_768.py — val-split accuracy evaluation (768-dim, SigLIP2)
=====================================================================
See base_evaluator.py for the shared image-search / prompt-search logic.

Supports --checkpoint so the same script works for a finetuned SigLIP2
checkpoint later, without touching the database or the base class.

Usage
-----
    python evaluate_768.py
    python evaluate_768.py --checkpoint /path/to/epoch.pt
    python evaluate_768.py --cpu --val-dir /path/to/val
"""

import argparse

import torch
from open_clip import create_model_from_pretrained, get_tokenizer

from base_evaluator import BaseEvaluator

DEFAULT_VAL_DIR = "/home/ubuntu/data/robust_dataset_split_30c_v2/val"
DEFAULT_MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP2-256"


class Evaluator768(BaseEvaluator):
    def __init__(self, val_dir, device, batch_size, top_k, model_id, checkpoint):
        self.model_id = model_id
        self.checkpoint = checkpoint
        super().__init__(val_dir, device, batch_size, top_k)

    def load_model(self):
        model, preprocess = create_model_from_pretrained(self.model_id)
        tokenizer = get_tokenizer(self.model_id)

        if self.checkpoint:
            print(f"Loading checkpoint: {self.checkpoint}")
            ckpt = torch.load(self.checkpoint, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            cleaned = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

        model = model.to(self.device).eval()
        return model, preprocess, tokenizer

    @torch.no_grad()
    def encode_images_batch(self, pil_images):
        tensors = [self.preprocess(img) for img in pil_images]
        batch = torch.stack(tensors).to(self.device)
        features = self.model.encode_image(batch)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def encode_text_batch(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def describe(self):
        return f"{self.model_id}" + (f" (checkpoint: {self.checkpoint})" if self.checkpoint else " (pretrained)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-dir", default=DEFAULT_VAL_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default="results_768_val.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    evaluator = Evaluator768(args.val_dir, device, args.batch_size, args.top_k, args.model, args.checkpoint)
    output = evaluator.run()
    evaluator.save(output, args.output)

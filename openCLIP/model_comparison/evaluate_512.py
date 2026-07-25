"""
evaluate_512.py — val-split accuracy evaluation (512-dim, xlm-roberta-base-ViT-B-32)
=======================================================================================
See base_evaluator.py for the shared image-search / prompt-search logic.

Usage
-----
    python evaluate_512.py
    python evaluate_512.py --cpu
    python evaluate_512.py --val-dir /path/to/val --batch-size 64
"""

import argparse

import open_clip
import torch

from base_evaluator import BaseEvaluator

DEFAULT_VAL_DIR = "/home/ubuntu/data/robust_dataset_split_30c_v2/val"
MODEL_NAME = "xlm-roberta-base-ViT-B-32"
PRETRAINED = "laion5b_s13b_b90k"


class Evaluator512(BaseEvaluator):
    def load_model(self):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=MODEL_NAME, pretrained=PRETRAINED,
        )
        model = model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
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
        return f"{MODEL_NAME} / {PRETRAINED}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-dir", default=DEFAULT_VAL_DIR)
    p.add_argument("--output", default="results_512_val.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    evaluator = Evaluator512(args.val_dir, device, args.batch_size, args.top_k)
    output = evaluator.run()
    evaluator.save(output, args.output)
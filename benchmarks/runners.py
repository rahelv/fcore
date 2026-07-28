"""
runners.py — model loading + forward pass for the ZED Box benchmarks

three models to benchmark:
  vitb32   xlm-roberta-base-ViT-B-32   open_clip, 224px    -> encode_image
  siglip2  ViT-B-16-SigLIP2-256        open_clip, 256px    -> encode_image
  resnet   resnet18 classifier         .pt checkpoint      -> model(x)

Every runner exposes the same interface:
    .name                    label for printing
    .input_size              H = W the model runs at
    .make_input(bs, device)  a random [bs, 3, H, W] tensor (only the shape
                             matters for timing; the pixel content does not)
    .forward(x)              the timed forward pass (no_grad, eval mode)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ResNet checkpoint. TODO: hardcoded, adjust path before running.
RESNET_CKPT = "/home/ubuntu/data/models/resnet_pretrained_best_9jglds8d.pt"

CLIP_MODELS = {
    "vitb32": dict(
        name="xlm-roberta-base-ViT-B-32",
        loader="pretrained",
        model_name="xlm-roberta-base-ViT-B-32",
        pretrained="laion5b_s13b_b90k",
    ),
    "siglip2": dict(
        name="ViT-B-16-SigLIP2-256",
        loader="hf",
        model_id="hf-hub:timm/ViT-B-16-SigLIP2-256",
    ),
}

MODEL_KEYS = list(CLIP_MODELS) + ["resnet"]

class ClipRunner:
    """CLIP image tower (vitb32 / siglip2). Timed call is encode_image()."""

    def __init__(self, cfg, device):
        import open_clip
        self.name = cfg["name"]

        if cfg["loader"] == "pretrained":
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=cfg["model_name"], pretrained=cfg["pretrained"])
        else:
            from open_clip import create_model_from_pretrained
            model, preprocess = create_model_from_pretrained(cfg["model_id"])

        self.model = model.to(device).eval()
        # Read the model's real input size from its own preprocess transform,
        self.input_size = self._infer_input_size(preprocess)

    @staticmethod
    def _infer_input_size(preprocess):
        import numpy as np
        from PIL import Image
        probe = Image.fromarray((np.random.rand(320, 320, 3) * 255).astype("uint8"))
        return int(preprocess(probe).shape[-1])

    def make_input(self, bs, device):
        return torch.randn(bs, 3, self.input_size, self.input_size, device=device)

    @torch.no_grad()
    def forward(self, x):
        return self.model.encode_image(x)


class ResnetRunner:
    """torchvision resnet18 classifier from a checkpoint. Timed call is model(x)."""

    def __init__(self, ckpt, device):
        from torchvision.models import resnet18
        self.name = "ResNet-18 (classifier)"

        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = raw.get("model_state_dict", raw.get("state_dict", raw))
        state = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state.items()}
        self.input_size = int(raw.get("image_size", 224))

        # Rebuild the head that was trained, reading its shape from the checkpoint:
        # plain Linear (fc.weight) or Dropout+Linear (fc.1.weight), as saved by
        # classification_model/train.py.
        model = resnet18(weights=None)
        in_f = model.fc.in_features
        if "fc.weight" in state:
            model.fc = nn.Linear(in_f, state["fc.weight"].shape[0])
        elif "fc.1.weight" in state:
            model.fc = nn.Sequential(nn.Dropout(0.0),
                                     nn.Linear(in_f, state["fc.1.weight"].shape[0]))

        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()

    def make_input(self, bs, device):
        return torch.randn(bs, 3, self.input_size, self.input_size, device=device)

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)

def build_runner(model, device, ckpt=None):
    if model in CLIP_MODELS:
        return ClipRunner(CLIP_MODELS[model], device)
    if model == "resnet":
        return ResnetRunner(ckpt or RESNET_CKPT, device)
    raise SystemExit(f"unknown model '{model}'. choices: {MODEL_KEYS}")

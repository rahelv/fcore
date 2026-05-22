import io
from typing import List
import pathlib
import numpy as np
import open_clip
import torch
from PIL import Image

model_name = "xlm-roberta-base-ViT-B-32"
pretrained = "laion5b_s13b_b90k"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name = model_name,
    pretrained= pretrained,
)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer(model_name)

print(f"Device is {device}")

def _clip_text(text: str) -> np.ndarray:

    text = tokenizer(text).to(device)


    with torch.no_grad():
        features = model.encode_text(text)
        features /= features.norm(dim=-1, keepdim=True)
        vec =features.cpu().numpy().flatten()
        # print(vec)
        return vec

    vec = feats[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
    return vec

def _clip_image(image):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

l1_text = "Aayla Secura from Star Wars"
l2_text = "Ariel from The Little Mermaid"
l3_text = "Frieren from Frieren: Beyond Journeys End"

l1_vec = _clip_text(l1_text)
l2_vec = _clip_text(l2_text)
l3_vec = _clip_text(l3_text)

cos_l1 = cosine_sim(l1_vec, l1_vec)
print(f"Cosine similarity between '{l1_text}' and itself: {cos_l1}")

import itertools
mypath = pathlib.Path("E:/ImagesRahel")
matched = list(
    itertools.chain.from_iterable(
        mypath.glob(pattern) for pattern in ["**/*.jpg", "**/*.png"]
    )
)

for path in matched:
    #Load image with pil
    img = Image.open(path)
    vec = _clip_image(img)
    cos_l1 = cosine_sim(l1_vec, vec)
    cos_l2 = cosine_sim(l2_vec, vec)
    cos_l3 = cosine_sim(l3_vec, vec)
    print(f"Cosine similarity between '{l1_text}' and '{path}': {cos_l1}")
    print(f"Cosine similarity between '{l2_text}' and '{path}': {cos_l2}")
    print(f"Cosine similarity between '{l3_text}' and '{path}': {cos_l3}")


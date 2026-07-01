import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = create_model_from_pretrained(
    "hf-hub:timm/ViT-B-16-SigLIP2-256"
)
tokenizer = get_tokenizer("hf-hub:timm/ViT-B-16-SigLIP2-256")

model = model.to(device).eval()

img_path = "/home/ubuntu/data/robust_dataset/Frieren from Frieren/00075_1qqj0h0_1_gallery_1.jpg"

labels = [
    "a cosplay costume of Frieren from Frieren",
    "a cosplay costume of Batman from DC Comics",
    "a cosplay costume of Yor Forger from Spy x Family",
]

image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
text = tokenizer(labels).to(device)

with torch.no_grad(), torch.amp.autocast(
    device_type="cuda",
    enabled=device == "cuda" # enabled only true if the selected device is cuda
):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(list(zip(labels, probs[0].cpu().tolist())))
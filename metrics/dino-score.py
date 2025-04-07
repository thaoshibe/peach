import requests
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from transformers import ViTModel

# DINO Transforms
T = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

images = [
    T(Image.open(image_path).convert('RGB'))
    for image_path in ['./pets/rose-3.jpg', './pets/max-3.jpg']
]
inputs = torch.stack(images) # (2, 3, 224, 224). Batchsize = 2

# Load DINO ViT-S/16< Madison23*
model = ViTModel.from_pretrained('facebook/dino-vits16')

# Get DINO features
with torch.no_grad():
    outputs = model(inputs)

last_hidden_states = outputs.last_hidden_state # ViT backbone features
emb_img1, emb_img2 = last_hidden_states[0, 0], last_hidden_states[1, 0] # Get cls token (0-th token) for each img
metric = F.cosine_similarity(emb_img1, emb_img2, dim=0)
print(f'''
Calculated: {metric.item():.3f}''')


filename = "server.py"
content = """
import io
import base64
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from safetensors.torch import load_file
from safetensors import safe_open
import numpy as np
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Generator(nn.Module):
    def __init__(self, codings_size, image_size, image_channels):
        super(Generator, self).__init__()
        self.fc = nn.Linear(codings_size, 6 * 6 * 256, bias=False)
        self.bn1 = nn.BatchNorm1d(6 * 6 * 256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv_transpose1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv_transpose2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv_transpose3 = nn.ConvTranspose2d(64, image_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.fc(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = x.view(-1, 256, 6, 6)
        x = self.conv_transpose1(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv_transpose2(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv_transpose3(x)
        x = self.tanh(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
metadata = None
def load_generator(model_path='model.safetensors'):
    global model, metadata
    state_dict = load_file(model_path)
    with safe_open(model_path, framework="pt", device=str(device)) as f:
        metadata = f.metadata()
    codings_size = int(metadata['codings_size'])
    image_size = int(metadata['image_size'])
    image_channels = int(metadata['image_channels'])
    model = Generator(codings_size, image_size, image_channels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Generator loaded: codings_size={codings_size}, image_size={image_size}, channels={image_channels}")
load_generator()
class GenerateRequest(BaseModel):
    seed: int = None
    num_images: int = 1
@app.get("/")
def root():
    return {"message": "Pixel API", "status": "running"}
@app.get("/info")
def get_info():
    return {
        "codings_size": metadata['codings_size'],
        "image_size": metadata['image_size'],
        "image_channels": metadata['image_channels'],
        "device": str(device)
    }
@app.post("/generate")
def generate_images(request: GenerateRequest):
    try:
        if request.seed is not None:
            torch.manual_seed(request.seed)
        num_images = min(max(1, request.num_images), 16)
        codings_size = int(metadata['codings_size'])
        with torch.no_grad():
            noise = torch.randn(num_images, codings_size, device=device)
            generated = model(noise)
        generated = generated.cpu()
        generated = (generated + 1) / 2.0
        generated = torch.clamp(generated, 0, 1)
        images_base64 = []
        for i in range(num_images):
            img_tensor = generated[i]
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='RGBA' if int(metadata['image_channels']) == 4 else 'RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            images_base64.append(img_base64)
        return {
            "images": images_base64,
            "count": num_images,
            "seed": request.seed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

with open(filename, "w") as f:
    f.write(content)

import os
from .quant3 import convert_gguf_to_safetensors
from .quant4 import add_metadata_to_safetensors

gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("GGUF file(s) available. Select which one to use:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice2 = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice2)-1
        selected_model_file=gguf_files[choice_index]
        print(f"Model file: {selected_model_file} is selected!")
        selected_file_path=selected_model_file
        print(f"Prepare to dequantize: {selected_file_path}")
        convert_gguf_to_safetensors(selected_file_path, "model.safetensors", use_bf16=False)
        add_metadata_to_safetensors("model.safetensors", {'image_size': '24', 'codings_size': '100', 'image_channels': '4'})
        os.system("uvicorn server:app --reload --host 0.0.0.0 --port 8000")
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

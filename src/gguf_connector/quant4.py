
import torch # need torch to work; pip install torch
from safetensors.torch import load_file
import os
def convert_safetensors_to_pth(safetensors_path):
    print(f"Loading .safetensors file: {safetensors_path}")
    state_dict = load_file(safetensors_path)
    output_path = os.path.splitext(safetensors_path)[0] + ".pth"
    print(f"Packing to .pth file: {output_path}")
    torch.save(state_dict, output_path

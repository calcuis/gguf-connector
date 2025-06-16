
import torch # need torch to work; pip install torch
from safetensors.torch import load_file

def convert_safetensors_to_pth(safetensors_path, output_path):
    print(f"Loading .safetensors file: {safetensors_path}")
    state_dict = load_file(safetensors_path)
    print(f"Packing to .pth file: {output_path}")
    torch.save(state_dict, output_path)

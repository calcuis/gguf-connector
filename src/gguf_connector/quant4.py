
import torch, os # need torch to work; pip install torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

def convert_safetensors_to_pth(safetensors_path, output_path):
    print(f"Loading .safetensors file: {safetensors_path}")
    state_dict = load_file(safetensors_path)
    print(f"Packing to .pth file: {output_path}")
    torch.save(state_dict, output_path)

def add_metadata_to_safetensors(file_path, new_metadata):
    with safe_open(file_path, framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
        metadata = f.metadata() or {}
    metadata.update(new_metadata)
    temp_path = file_path + ".tmp"
    save_file(tensors, temp_path, metadata)
    os.replace(temp_path, file_path)

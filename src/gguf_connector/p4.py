
import torch # need torch to work; pip install torch
from safetensors.torch import load_file
import os, glob

def convert_safetensors_to_pth(safetensors_path: str, output_path: str = None):
    if not safetensors_path.endswith(".safetensors"):
        raise ValueError("Input file must be a .safetensors file.")
    # Load tensors
    print(f"Loading .safetensors file: {safetensors_path}")
    state_dict = load_file(safetensors_path)
    # Set output filename
    if output_path is None:
        output_path = os.path.splitext(safetensors_path)[0] + ".pth"
    # Save as .pth
    print(f"Saving to .pth file: {output_path}")
    torch.save(state_dict, output_path)
    print("Conversion complete.")

st_files = glob.glob("*.safetensors")
if not st_files:
    print("No .safetensors files found in the current directory.")
else:
    for idx, file in enumerate(st_files):
        print(f"[{idx}] {file}")
    selected = int(input("Select a .safetensors file to convert (by index): "))
    convert_safetensors_to_pth(st_files[selected])

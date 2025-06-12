
import torch # need torch to work; pip install torch
from safetensors.torch import save_file
import os, glob

def convert_pth_to_safetensors(pth_path: str, output_path: str = None):
    if not pth_path.endswith(".pth"):
        raise ValueError("Input file must be a .pth file.")
    # Load the state dict
    print(f"Loading .pth file: {pth_path}")
    state_dict = torch.load(pth_path, map_location="cpu")
    # Some .pth files may contain more than just state_dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    tensor_dict = {
        k: v.cpu() if isinstance(v, torch.Tensor) else torch.tensor(v)
        for k, v in state_dict.items()
        if isinstance(v, (torch.Tensor, float, int))
    }
    # Output path
    if output_path is None:
        output_path = os.path.splitext(pth_path)[0] + ".safetensors"
    # Save as .safetensors
    print(f"Saving to .safetensors file: {output_path}")
    save_file(tensor_dict, output_path)
    print("Conversion complete.")

pth_files = glob.glob("*.pth")
if not pth_files:
    print("No .pth files found in the current directory.")
else:
    for idx, file in enumerate(pth_files):
        print(f"[{idx}] {file}")
    selected = int(input("Select a .pth file to convert (by index): "))
    convert_pth_to_safetensors(pth_files[selected])

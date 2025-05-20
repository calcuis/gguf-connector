
from safetensors.torch import load_file, save_file # optional (need torch to work; pip install torch)
from tqdm import tqdm
import os

def list_safetensors_files():
    files = [f for f in os.listdir() if f.endswith(".safetensors")]
    if not files:
        print("No .safetensors files found in the current directory.")
        return None
    print("Select a .safetensors file to split:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}: {file}")
    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.")

def split_safetensors(file_path):
    tensors = load_file(file_path)
    unidim = {}
    multidim = {}

    for name in tqdm(tensors, desc="Splitting tensors"):
        tensor = tensors[name]
        if tensor.dim() <= 4:
            unidim[name] = tensor
        else:
            multidim[name] = tensor

    save_file(unidim, "4d.safetensors")
    save_file(multidim, "5d.safetensors")
    print("Saved: 4d.safetensors and 5d.safetensors")

selected_file = list_safetensors_files()
if selected_file:
    split_safetensors(selected_file)

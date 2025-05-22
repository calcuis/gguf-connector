
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
    dim2 = {}
    dimy = {}

    for name in tqdm(tensors, desc="Splitting tensors"):
        tensor = tensors[name]
        if tensor.dim() == 2:
            dim2[name] = tensor
        else:
            dimy[name] = tensor

    save_file(dim2, "tensor2d.safetensors")
    save_file(dimy, "except2d.safetensors")
    print("Saved: except2d.safetensors and tensor2d.safetensors")

selected_file = list_safetensors_files()
if selected_file:
    split_safetensors(selected_file)

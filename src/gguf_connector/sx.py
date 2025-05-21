
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
    dim1 = {}
    dim2 = {}
    dim3 = {}
    dim4 = {}
    dim5 = {}
    dimx = {}

    for name in tqdm(tensors, desc="Splitting tensors"):
        tensor = tensors[name]
        if tensor.dim() == 1:
            dim1[name] = tensor
        elif tensor.dim() == 2:
            dim2[name] = tensor
        elif tensor.dim() == 3:
            dim3[name] = tensor
        elif tensor.dim() == 4:
            dim4[name] = tensor
        elif tensor.dim() == 5:
            dim5[name] = tensor
        else:
            dimx[name] = tensor

    save_file(dim1, "1d.safetensors")
    save_file(dim2, "2d.safetensors")
    save_file(dim3, "3d.safetensors")
    save_file(dim4, "4d.safetensors")
    save_file(dim5, "5d.safetensors")
    save_file(dimx, "xd.safetensors")
    print("Saved: 1d, 2d, 3d, 4d, 5d and xd.safetensors")

selected_file = list_safetensors_files()
if selected_file:
    split_safetensors(selected_file)

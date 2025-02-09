
from safetensors.torch import load_file # optional (need torch to work; pip install torch)

def tensor_reader(file_path):
    try:
        tensors = load_file(file_path)
        tensor_info = {name: tensor.shape for name, tensor in tensors.items()}
        return tensor_info
    except Exception as e:
        return f"Error reading tensor: {e}", {}

import os
files = [file for file in os.listdir() if file.endswith('.safetensors')]

if files:
    print("Available .safetensors files:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")
    choice = input("Enter the number of the file to read tensor info from: ")
    try:
        choice = int(choice)
        if 1 <= choice <= len(files):
            file_path = files[choice - 1]
            tensor_info = tensor_reader(file_path)
            print("Tensors:")
            for name, shape in tensor_info.items():
                print(f"  {name}: {shape}")
        else:
            print("Invalid selection.")
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No safetensors files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

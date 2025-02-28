
import torch # optional (if you want this conversion tool; pip install torch)
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def convert_safetensors_to_fp32(file_path):
    print(f"Loading: {file_path}")
    tensors = load_file(file_path)
    converted_tensors = {}
    for key, tensor in tqdm(tensors.items(), desc="Converting tensors", unit="tensor"):
        converted_tensors[key] = tensor.to(torch.float32)
    output_file = file_path.replace(".safetensors", "_fp32.safetensors")
    save_file(converted_tensors, output_file)
    print(f"Converted and saved to: {output_file}")

def upscale_to_fp32():
    import os
    safetensors_files = [f for f in os.listdir('.') if f.endswith('.safetensors')]
    if not safetensors_files:
        print("No .safetensors files found in the current directory.")
        return
    print("Select a file to convert:")
    for idx, file in enumerate(safetensors_files, 1):
        print(f"{idx}. {file}")
    try:
        choice = int(input("Enter the number of the file: ")) - 1
        if choice < 0 or choice >= len(safetensors_files):
            print("Invalid selection.")
            return
        convert_safetensors_to_fp32(safetensors_files[choice])
    except ValueError:
        print("Please enter a valid number.")

upscale_to_fp32()

from safetensors import safe_open
from safetensors.torch import save_file
import os

def list_safetensors_files():
    return [f for f in os.listdir('.') if f.endswith('.safetensors')]

def add_metadata_to_safetensors(file_path, new_metadata):
    with safe_open(file_path, framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
        metadata = f.metadata() or {}
    # Update metadata
    metadata.update(new_metadata)
    # Save back to the same file (or change to a new one to avoid overwrite)
    temp_path = file_path + ".tmp"
    save_file(tensors, temp_path, metadata)
    # Replace original file
    os.replace(temp_path, file_path)
    print(f"Metadata added to {file_path}: {new_metadata}")

files = list_safetensors_files()
if not files:
    print("No .safetensors files found in the current directory.")
else:
    print("Select a .safetensors file:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")
    try:
        choice = int(input("Enter number: ")) - 1
        selected_file = files[choice]
        add_metadata_to_safetensors(selected_file, {'format': 'pt'})
    except (ValueError, IndexError):
        print("Invalid selection.")

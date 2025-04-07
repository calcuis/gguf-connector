
import os, json
from safetensors import safe_open
from safetensors.torch import save_file

def list_safetensors():
    files = [f for f in os.listdir() if f.endswith(".safetensors")]
    if not files:
        print("No .safetensors files found in the current directory.")
        return None
    print("Available .safetensors files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    while True:
        try:
            choice = int(input("Select a file by number: "))
            return files[choice - 1]
        except (ValueError, IndexError):
            print("Invalid selection. Try again.")

def get_cutoff_size():
    while True:
        try:
            mb = float(input("Enter cutoff size for each part (in MB): "))
            return int(mb * 1024 * 1024)
        except ValueError:
            print("Please enter a valid number.")

def split_safetensors(file_name, cutoff_size):
    with safe_open(file_name, framework="pt") as f:
        metadata = f.metadata()
        keys = f.keys()
        tensors = {key: f.get_tensor(key) for key in keys}

    total_size = 0
    tensor_sizes = {}
    for k, v in tensors.items():
        size = v.element_size() * v.nelement()
        tensor_sizes[k] = size
        total_size += size

    part_id = 1
    current_size = 0
    current_tensors = {}
    weight_map = {}
    num_parts = 0
    all_parts = []

    for key, tensor in tensors.items():
        tensor_size = tensor_sizes[key]

        if current_size + tensor_size > cutoff_size and current_tensors:
            part_name = f"model-{part_id:05d}-of-XXXX.safetensors"
            save_file(current_tensors, part_name, metadata={})
            all_parts.append(part_name)
            for k in current_tensors:
                weight_map[k] = part_name
            part_id += 1
            current_tensors = {}
            current_size = 0

        current_tensors[key] = tensor
        current_size += tensor_size

    if current_tensors:
        part_name = f"model-{part_id:05d}-of-XXXX.safetensors"
        save_file(current_tensors, part_name, metadata={})
        all_parts.append(part_name)
        for k in current_tensors:
            weight_map[k] = part_name

    num_parts = len(all_parts)
    final_files = []

    for i, old_name in enumerate(all_parts):
        new_name = f"model-{i+1:05d}-of-{num_parts:05d}.safetensors"
        os.rename(old_name, new_name)
        for k in weight_map:
            if weight_map[k] == old_name:
                weight_map[k] = new_name
        final_files.append(new_name)

    index_json = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }

    with open("model.safetensors.index.json", "w") as f:
        json.dump(index_json, f, indent=2)

    print(f"\nSplit complete! {num_parts} parts created.")
    print("Index file: model.safetensors.index.json")

file = list_safetensors()
if file:
    cutoff = get_cutoff_size()
    split_safetensors(file, cutoff)

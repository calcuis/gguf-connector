
import os, json
from tqdm import tqdm

def list_gguf_files():
    files = [f for f in os.listdir() if f.endswith(".gguf")]
    if not files:
        print("No .gguf files found in the current directory.")
        return None
    print("Available .gguf files:")
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

def split_gguf_file(filename, cutoff_size):
    with open(filename, "rb") as f:
        data = f.read()

    total_size = len(data)
    parts = []
    offset = 0
    part_id = 1

    print("\nSplitting file into parts...")
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Splitting") as pbar:
        while offset < total_size:
            end = min(offset + cutoff_size, total_size)
            part_data = data[offset:end]
            part_name = f"model-{part_id:05d}-of-XXXX.gguf"
            with open(part_name, "wb") as pf:
                pf.write(part_data)
            parts.append(part_name)
            offset = end
            part_id += 1
            pbar.update(len(part_data))

    num_parts = len(parts)
    final_files = []

    print("\nRenaming parts with total count info...")
    for i, old_name in tqdm(list(enumerate(parts)), desc="Renaming", unit="file"):
        new_name = f"model-{i+1:05d}-of-{num_parts:05d}.gguf"
        os.rename(old_name, new_name)
        final_files.append(new_name)

    index = {
        "metadata": {
            "total_size": total_size
        },
        "file_map": {
            f"part_{i+1}": name for i, name in enumerate(final_files)
        }
    }

    with open("model.gguf.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nSplit complete! {num_parts} parts created.")
    print("Index file: model.gguf.index.json")

file = list_gguf_files()
if file:
    cutoff = get_cutoff_size()
    split_gguf_file(file, cutoff)

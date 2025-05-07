
from safetensors.torch import load_file, save_file
import os

def split_safetensors_file(input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tensors = load_file(input_path)
    components = {
        "model": {},
        "encoder": {},
        "vae": {}
    }
    print(f"Splitting {input_path} into components...")
    for key, tensor in tensors.items():
        for comp in components:
            if key.startswith(comp):
                components[comp][key] = tensors[key]
                break  # Assign only to the first matching component

    for comp_name, comp_tensors in components.items():
        if comp_tensors:
            output_path = os.path.join(output_dir, f"{comp_name}.safetensors")
            save_file(comp_tensors, output_path)
            print(f"✅ Saved {comp_name} to {output_path} ({len(comp_tensors)} tensors)")
        else:
            print(f"⚠️ No tensors found for component: {comp_name}")

safetensors_files = [file for file in os.listdir() if file.endswith('.safetensors')]

if safetensors_files:
    print("Safetensors file(s) available. Select which one to split:")
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(safetensors_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=safetensors_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        split_safetensors_file(selected_file, "split_output")
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No safetensors files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

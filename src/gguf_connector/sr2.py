
from .rky import unet_to_base_map, base_to_unet_map, rename_key

import os
safetensors_files = [file for file in os.listdir() if file.endswith('.safetensors')]

if safetensors_files:
    print("Safetensors file(s) available. Select which one to convert:")
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(safetensors_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=safetensors_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        input_path=selected_file
        from safetensors.torch import safe_open
        with safe_open(selected_file, framework="pt") as f:
            tensor_names = f.keys()
            first_tensor_name = next(iter(tensor_names), None)
            if first_tensor_name.startswith("lora"):
                print("unet detected! swapping unet to base...")
                rename_rules = unet_to_base_map
                output_path = f"{os.path.splitext(input_path)[0]}_base_converted.safetensors"
            elif first_tensor_name.startswith("base"):
                print("base detected! swapping base to unet...")
                rename_rules = base_to_unet_map
                output_path = f"{os.path.splitext(input_path)[0]}_unet_converted.safetensors"
            else:
                print("nothing detected! quit without doing anything...")
                quit()
        from safetensors.torch import load_file, save_file
        tensors = load_file(input_path)
        new_tensors = {rename_key(k, rename_rules): v for k, v in tensors.items()}
        save_file(new_tensors, output_path)
        print(f"Saved modified tensors with renamed keys to: {output_path}")
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No safetensors files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

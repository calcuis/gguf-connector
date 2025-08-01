
from .rky import rename_key

import os
safetensors_files = [file for file in os.listdir() if file.endswith('.safetensors')]

if safetensors_files:
    print("Safetensors file(s) available. Select which one to perform batch rename (search & replace):")
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(safetensors_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=safetensors_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        search_term = input("Enter a search term: ")
        replace_term = input("Enter a replace term: ")
        rename_rules = [(search_term, replace_term)]
        input_path=selected_file
        output_path = f"{os.path.splitext(input_path)[0]}-renamed.safetensors"
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

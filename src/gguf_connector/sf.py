
import os
safetensors_files = [file for file in os.listdir() if file.endswith('.safetensors')]

if safetensors_files:
    print("Safetensors file(s) available. Select which one to add suffix:")
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(safetensors_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=safetensors_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        suffix = input("Enter a suffix: ")
        input_path=selected_file
        output_path = f"{os.path.splitext(input_path)[0]}-suffix-added.safetensors"

        from safetensors.torch import load_file, save_file
        tensors = load_file(input_path)
        new_tensors = {f"{k}{suffix}": v for k, v in tensors.items()}
        save_file(new_tensors, output_path)

        print(f"Saved modified tensors with suffix '{suffix}' to: {output_path}")
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No safetensors files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

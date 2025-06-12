
from .quant3 import convert_gguf_to_safetensors

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files:
    print("GGUF file(s) available. Select which one to convert:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=gguf_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        input_path=selected_file
        use_bf16 = False
        out_path = f"{os.path.splitext(input_path)[0]}-f32.safetensors"
        convert_gguf_to_safetensors(input_path, out_path, use_bf16)
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

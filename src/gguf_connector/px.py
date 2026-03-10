
import os
from .quant3 import convert_gguf_to_safetensors
from .quant4 import add_metadata_to_safetensors

gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("GGUF file(s) available. Select which one to use:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice2 = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice2)-1
        selected_model_file=gguf_files[choice_index]
        print(f"Model file: {selected_model_file} is selected!")
        selected_file_path=selected_model_file
        print(f"Prepare to dequantize: {selected_file_path}")
        convert_gguf_to_safetensors(selected_file_path, "model.safetensors", use_bf16=False)
        add_metadata_to_safetensors("model.safetensors", {'image_size': '24', 'codings_size': '100', 'image_channels': '4'})
        num_avatar = input(f"How many (Enter a number): ")
        os.system(f'pixel --num_images {num_avatar}')
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

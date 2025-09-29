
import torch # need torch to work; pip install torch
from .q4 import launch_image_edit_lite_app

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print('GGUF file(s) available. Select which one to use:')
    for index, file_name in enumerate(gguf_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(gguf_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = gguf_files[choice_index]
        print(f'Model file: {selected_file} is selected!')
        input_path = selected_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        if device == "cuda":
            print(f"running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        launch_image_edit_lite_app(input_path,dtype)
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

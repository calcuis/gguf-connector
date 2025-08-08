
import torch # need torch to work; pip install torch
from .x1 import launch_ltxv_app, launch_ltxv_13b_app

# connector selection (2b by default)
ask=input("Use 13b instead of 2b (Y/n)? ")
if ask.lower() == 'y':
    selected = "13b"
    print(f'connector choice: {selected}\n')
else:
    selected = "2b"
    print(f'connector choice: {selected}\n')

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
        print(f'Model file: {selected_file} is selected as ltxv {selected} video generator!')
        input_path = selected_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        if selected == '13b':
            launch_ltxv_13b_app(input_path, dtype)
        else:
            launch_ltxv_app(input_path, dtype)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

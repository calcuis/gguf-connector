
import torch # need torch to work; pip install torch
from .q4 import launch_qi_app, launch_qi_distill_app

# connector selection (2b by default)
ask=input("Use distilled model instead (Y/n)? ")
if ask.lower() == 'y':
    selected = "distill"
    print(f'connector choice: {selected}\n')
else:
    selected = "original"
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
        print(f'Model file: {selected_file} is selected as qwen image generator {selected}!')
        input_path = selected_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        if selected == 'distill':
            launch_qi_distill_app(input_path, dtype)
        else:
            launch_qi_app(input_path, dtype)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

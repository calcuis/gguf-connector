
import torch # need torch to work; pip install torch
from .s4 import launch_sd3_app, launch_sd3m_app

# connector selection (2b/medium by default)
ask=input("Use large (8b) model instead (Y/n)? ")
if ask.lower() == 'y':
    selected = "large"
    print(f'connector choice: {selected}\n')
else:
    selected = "medium"
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
        print(f'Model file: {selected_file} is selected as s3 image generator - {selected}!')
        input_path = selected_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        if device == "cuda":
            print(f"running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        if selected == 'large':
            launch_sd3_app(input_path, dtype)
        else:
            launch_sd3m_app(input_path, dtype)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

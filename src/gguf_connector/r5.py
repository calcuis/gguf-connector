
import torch # optional (need torch to work; pip install torch)
import os
files = [file for file in os.listdir() if file.endswith('.pt')]

if files:
    print("Available .pt files:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")
    choice = input("Enter the number of the file to read tensor info from: ")
    try:
        choice = int(choice)
        if 1 <= choice <= len(files):
            file_path = files[choice - 1]
            loaded_data = torch.load(file_path, weights_only=False)
            print(f"{loaded_data}")
        else:
            print("Invalid selection.")
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No .pt files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

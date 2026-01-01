
from safetensors import safe_open
import os

def list_safetensors_files():
    files = [f for f in os.listdir() if f.endswith(".safetensors")]
    if not files:
        print("No .safetensors files found in the current directory.")
        return None
    print("Select a .safetensors file to read metadata:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}: {file}")
    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.")

selected_file = list_safetensors_files()
if selected_file:
    with safe_open(selected_file, framework="pt") as f:
        print(f.metadata())

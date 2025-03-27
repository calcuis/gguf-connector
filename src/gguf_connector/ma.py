
import os, safetensors.torch # need torch to work; pip install torch

def merge_safetensors(output):
    files = [f for f in os.listdir() if f.endswith('.safetensors') and f != output]
    if not files:
        print("No safetensors files found.")
        return

    merged_data = {}

    for file in files:
        print(f"Loading {file}...")
        data = safetensors.torch.load_file(file)
        for key, value in data.items():
            if key in merged_data:
                print(f"Warning: Duplicate key '{key}' in {file}. Skipping.")
            else:
                merged_data[key] = value

    print(f"Saving merged data to {output}...")
    safetensors.torch.save_file(merged_data, output)
    print("Merge completed successfully.")

ask=input("Assign a name other than model.safetensors (Y/n)? ")
if ask.lower() == 'y':
    given = input("Enter a file name: ")
else:
    given = 'model.safetensors'
merge_safetensors(given)

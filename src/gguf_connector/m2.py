
import os
from tqdm import tqdm

def merge_gguf_files():
    gguf_files = [f for f in os.listdir('.') if f.endswith('.gguf')]

    if not gguf_files:
        print("No .gguf files found in the current directory.")
        return

    filename = input("Enter the output file name (without .gguf): ").strip()
    if not filename:
        filename = "model"
    output_file = f"{filename}.gguf"

    with open(output_file, 'wb') as outfile:
        for fname in tqdm(gguf_files, desc="Merging files", unit="file"):
            with open(fname, 'rb') as infile:
                outfile.write(infile.read())

    print(f"\nAll files merged into: {output_file}")

merge_gguf_files()

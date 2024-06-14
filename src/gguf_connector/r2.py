
def read_gguf_file(gguf_file_path):
    from gguf_connector.reader2 import GGUFReader, ReaderError
    reader = GGUFReader(gguf_file_path)
    try:
        reader.read()
    except ReaderError as e:
        print(f"Error: {e}")
    else:
        reader.print()

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("GGUF file(s) available. Select which one to read:")   
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=gguf_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        ModelPath=selected_file
        read_gguf_file(ModelPath)

    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

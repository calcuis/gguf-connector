
from .const import GGML_QUANT_VERSION, LlamaFileType
from .reader import GGUFReader
from .writer import GGUFWriter

def get_arch_str(reader):
    field = reader.get_field("general.architecture")
    return str(field.parts[field.data[-1]], encoding="utf-8")

def get_file_type(reader):
    field = reader.get_field("general.file_type")
    ft = int(field.parts[field.data[-1]])
    return LlamaFileType(ft)

def extract_by_component(tensor_head, input_path, output_path):
    with open(input_path, "rb") as f:
        reader = GGUFReader(f)
        arch = get_arch_str(reader)
        file_type = get_file_type(reader)
        print(f"Detected arch: '{arch}' (ftype: {str(file_type)})")
        writer = GGUFWriter(path=None, arch=arch)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_type)
        count = 0
        for tensor in reader.tensors:
            if tensor.name.startswith(tensor_head):
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
                count += 1
        if count > 0:
            with open(output_path, "wb"):
                writer.write_header_to_file(path=output_path)
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file(progress=True)
                writer.close()
            print(f"Extracted {count} tensor(s) to: {output_path}")
        else:
            print(f"No tensors found for component '{tensor_head}'.")

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files:
    print("Available GGUF files:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    try:
        choice = int(input(f"Select a file to extract from (1-{len(gguf_files)}): "))
        input_path = gguf_files[choice - 1]
        print(f"Selected: {input_path}")
        num_components = int(input("How many components to extract? "))
        for i in range(num_components):
            keyword = input(f"Enter keyword for component #{i+1}: ").strip()
            output_path = f"{keyword}-extracted.gguf"
            extract_by_component(keyword, input_path, output_path)
    except (ValueError, IndexError):
        print("Invalid input. Please enter valid numbers.")
else:
    print("No GGUF files found in the current directory.")
input("--- Press ENTER to exit ---")

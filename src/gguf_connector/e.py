
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

def extract_weight_tensors(input_path, output_path):
    with open(input_path, "rb") as f1:
        reader1 = GGUFReader(f1)
        arch = get_arch_str(reader1)
        file_type = get_file_type(reader1)
        print(f"Detected arch: '{arch}' (ftype: {str(file_type)})")
        writer = GGUFWriter(path=None, arch=arch)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_type)
        for tensor in reader1.tensors:
            if tensor.name.endswith(".weight"):
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        with open(output_path, "wb"):
            writer.write_header_to_file(path=output_path)
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
            writer.close()

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("GGUF file(s) available. Select which one to extract:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=gguf_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        input_path=selected_file
        output_path = f"{os.path.splitext(input_path)[0]}-weight-extracted.gguf"
        extract_weight_tensors(input_path, output_path)
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

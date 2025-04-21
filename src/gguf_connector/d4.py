
import torch # need torch to work
from .const import GGML_QUANT_VERSION, LlamaFileType
from .reader import GGUFReader, GGMLQuantizationType
from .writer import GGUFWriter
from .quant import quantize
from tqdm import tqdm

def get_arch_str(reader):
    field = reader.get_field("general.architecture")
    return str(field.parts[field.data[-1]], encoding="utf-8")

def get_file_type(reader):
    field = reader.get_field("general.file_type")
    ft = int(field.parts[field.data[-1]])
    return LlamaFileType(ft)

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("GGUF file(s) available. Select which one to fix:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=gguf_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        input_path=selected_file
        output_path = f"fixed-{os.path.splitext(input_path)[0]}.gguf"
        reader = GGUFReader(input_path)
        arch = get_arch_str(reader)
        file_type = get_file_type(reader)
        print(f"Detected arch: '{arch}' (ftype: {str(file_type)})")
        sd5d = torch.load(f"fix_5d_tensors_{arch}.pt", weights_only=False)
        print("5D:", sd5d.keys())
        writer = GGUFWriter(path=None, arch=arch)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_type)
        added = []
        def add_extra_key(writer, key, data):
            # old_dtype = data.dtype
            data_qtype = GGMLQuantizationType.F32
            # n_dims = len(data.shape)
            data_shape = data.shape
            data = quantize(data, data_qtype)
            tqdm.write(f"Adding key {key} ({data_shape})")
            writer.add_tensor(key, data, raw_dtype=data_qtype)
            global added
            added.append(key)
        for tensor in tqdm(reader.tensors):
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            key5d = tensor.name.replace(".bias", ".weight")
            if key5d in sd5d.keys():
                add_extra_key(writer, key5d, sd5d[key5d])
        for key, data in sd5d.items():
            if key not in added:
                add_extra_key(writer, key, data)
        writer.write_header_to_file(path=output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

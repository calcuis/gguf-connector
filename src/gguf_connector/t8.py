
import torch # optional (if you want this conversion tool; pip install torch)
from gguf_connector.writer import GGUFWriter, GGMLQuantizationType
from gguf_connector.const import GGML_QUANT_VERSION, LlamaFileType
from safetensors.torch import load_file
from tqdm import tqdm
import numpy as np

MAX_TENSOR_NAME_LENGTH = 127  # Max allowed length for tensor names

def load_state_dict(path):
    state_dict = load_file(path)
    return {k: v for k, v in state_dict.items()}

def load_model(path, model_arch):
    state_dict = load_state_dict(path)
    writer = GGUFWriter(path=None, arch=model_arch)
    return writer, state_dict, model_arch

def is_tensor_valid(data, key):
    if data is None:
        print(f"[WARNING] Skipping tensor '{key}': Empty or NULL tensor.")
        return False
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"[WARNING] Skipping tensor '{key}': Contains NaN or Inf values.")
        return False
    return True

def handle_tensors(writer, state_dict):
    name_lengths = [(key, len(key)) for key in state_dict.keys()]
    if not name_lengths:
        return
    max_name_len = max(name_lengths, key=lambda x: x[1])[1]

    for key, data in tqdm(state_dict.items(), desc="Processing Tensors"):
        old_dtype = data.dtype
        print(f"[INFO] Processing: {key} | Original dtype: {old_dtype} | Shape: {data.shape}")

        data = data.to(torch.float32).numpy()

        if not is_tensor_valid(data, key):
            continue  # Skip if tensor is invalid

        data_qtype = GGMLQuantizationType.F32  # Force F32 for all tensors
        shape_str = f"{{{', '.join(map(str, reversed(data.shape)))}}}"
        print(f"[INFO] Writing: {key.ljust(max_name_len)} | {old_dtype} -> {data_qtype.name} | Shape: {shape_str}")
        writer.add_tensor(key, data, raw_dtype=data_qtype)

import os
safetensors_files = [file for file in os.listdir() if file.endswith('.safetensors')]

if safetensors_files:
    print("Safetensors file(s) available. Select which one to convert:")
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(safetensors_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=safetensors_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        path=selected_file
        ask=input("Assign a name for the model (Y/n)? ")
        if ask.lower() == 'y':
            given = input("Enter a model name: ")
        else:
            given = None
        writer, state_dict, _ = load_model(path,given)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        out_path = f"{os.path.splitext(path)[0]}-f32.gguf"
        writer.add_file_type(LlamaFileType.MOSTLY_F16)
        if os.path.isfile(out_path):
            input("Output file exists. Press Enter to overwrite or Ctrl+C to abort.")
        handle_tensors(writer, state_dict)
        writer.write_header_to_file(path=out_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        print(f"Conversion completed: {out_path}")
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No safetensors files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

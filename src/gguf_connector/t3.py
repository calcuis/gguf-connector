
import torch # optional (if you want this tool; pip install torch)
import numpy as np
from safetensors.torch import save_file
from typing import Dict, Tuple
from .quant import dequantize
from .reader import GGUFReader
from .const import Keys

def load_gguf_and_extract_metadata(gguf_path: str) -> Tuple[GGUFReader, list]:
    reader = GGUFReader(gguf_path)
    tensors_metadata = []
    for tensor in reader.tensors:
        tensor_metadata = {
            'name': tensor.name,
            'shape': tuple(tensor.shape.tolist()),
            'n_elements': tensor.n_elements,
            'n_bytes': tensor.n_bytes,
            'data_offset': tensor.data_offset,
            'type': tensor.tensor_type,
        }
        tensors_metadata.append(tensor_metadata)
    return reader, tensors_metadata

def convert_gguf_to_safetensors(gguf_path: str, output_path: str, use_bf16: bool) -> None:
    reader, tensors_metadata = load_gguf_and_extract_metadata(gguf_path)
    print(f"Extracted {len(tensors_metadata)} tensors from GGUF file")
    tensors_dict: dict[str, torch.Tensor] = {}
    for i, tensor_info in enumerate(tensors_metadata):
        tensor_name = tensor_info['name']
        tensor_data = reader.get_tensor(i)
        weights = dequantize(tensor_data.data, tensor_data.tensor_type).copy()
        try:
            if use_bf16:
                print(f"Attempting BF16 conversion")
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.float32)
                weights_tensor = weights_tensor.to(torch.bfloat16)
            else:
                print("Using FP16 conversion.")
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.float16)
            weights_hf = weights_tensor
        except Exception as e:
            print(f"Error during BF16 conversion for tensor '{tensor_name}': {e}")
            weights_tensor = torch.from_numpy(weights.astype(np.float32)).to(torch.float16)
            weights_hf = weights_tensor
        print(f"dequantize tensor: {tensor_name} | Shape: {weights_hf.shape} | Type: {weights_tensor.dtype}")
        del weights_tensor
        del weights
        tensors_dict[tensor_name] = weights_hf
        del weights_hf
    metadata = {"modelspec.architecture": f"{reader.get_field(Keys.General.FILE_TYPE)}", "description": "Model converted from gguf."}
    save_file(tensors_dict, output_path, metadata=metadata)
    print("Conversion complete!")

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("GGUF file(s) available. Select which one to convert:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=gguf_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        input_path=selected_file
        ask=input("Convert tensors to BF16 format instead of FP16 (Y/n)? ")
        if ask.lower() == 'y':
            use_bf16 = True
            out_path = f"{os.path.splitext(input_path)[0]}-bf16.safetensors"
        else:
            use_bf16 = False
            out_path = f"{os.path.splitext(input_path)[0]}-fp16.safetensors"
        convert_gguf_to_safetensors(input_path, out_path, use_bf16)
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No safetensors files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

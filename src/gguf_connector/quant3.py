
import torch # optional (need torch to work; pip install torch)
import numpy as np
from tqdm import tqdm
from typing import Tuple
from safetensors.torch import save_file
from .reader import GGUFReader
from .quant import dequantize

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
    for i, tensor_info in enumerate(tqdm(tensors_metadata, desc="Dequantizing tensors", unit="tensor")):
        tensor_name = tensor_info['name']
        tensor_data = reader.get_tensor(i)
        weights = dequantize(tensor_data.data, tensor_data.tensor_type).copy()
        try:
            if use_bf16:
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.float32)
                weights_tensor = weights_tensor.to(torch.bfloat16)
            else:
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.float32)
            weights_hf = weights_tensor
        except Exception as e:
            print(f"Error during dequantization for tensor '{tensor_name}': {e}")
            weights_tensor = torch.from_numpy(weights.astype(np.float32)).to(torch.float16)
            weights_hf = weights_tensor
        tensors_dict[tensor_name] = weights_hf
    metadata = {key: str(reader.get_field(key)) for key in reader.fields}
    save_file(tensors_dict, output_path, metadata=metadata)

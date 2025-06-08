
import torch # optional (need torch to work; pip install torch)
from .writer import GGUFWriter, GGMLQuantizationType
from .quant import quantize, QuantError
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

def handle_tensors(writer, state_dict, fp32):
    name_lengths = [(key, len(key)) for key in state_dict.keys()]
    if not name_lengths:
        return
    max_name_len = max(name_lengths, key=lambda x: x[1])[1]
    for key, data in tqdm(state_dict.items(), desc="Processing Tensors"):
        old_dtype = data.dtype
        if fp32:
            print(f"[INFO] Processing: {key} | Original dtype: {old_dtype} | Shape: {data.shape}")
            data = data.to(torch.float32).numpy()
            if not is_tensor_valid(data, key):
                continue  # Skip if tensor is invalid
            data_qtype = GGMLQuantizationType.F32  # Force F32 for all tensors
        else:
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float32).numpy()
            elif data.dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
                data = data.to(torch.float16).numpy()
            else:
                data = data.numpy()
            n_dims = len(data.shape)
            data_shape = data.shape
            data_qtype = getattr(
                GGMLQuantizationType,
                "BF16" if old_dtype == torch.bfloat16 else "F16"
            )
            n_params = 1
            for dim_size in data_shape:
                n_params *= dim_size
            blacklist = {
                "time_embedding.",
                "add_embedding.",
                "time_in.",
                "txt_in.",
                "vector_in.",
                "img_in.",
                "guidance_in.",
                "final_layer.",
            }
            if old_dtype in (torch.float32, torch.bfloat16):
                if n_dims == 1:
                    data_qtype = GGMLQuantizationType.F32
                elif ".weight" in key and any(x in key for x in blacklist):
                    data_qtype = GGMLQuantizationType.F32
            try:
                data = quantize(data, data_qtype)
            except (AttributeError, QuantError) as e:
                tqdm.write(f"falling back to F16: {e}")
                data_qtype = GGMLQuantizationType.F16
                data = quantize(data, data_qtype)
        shape_str = f"{{{', '.join(map(str, reversed(data.shape)))}}}"
        print(f"[INFO] Writing: {key.ljust(max_name_len)} | {old_dtype} -> {data_qtype.name} | Shape: {shape_str}")
        writer.add_tensor(key, data, raw_dtype=data_qtype)

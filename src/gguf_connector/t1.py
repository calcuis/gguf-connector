
import torch # optional (if you want this conversion tool; pip install torch)
from .writer import GGUFWriter, GGMLQuantizationType
from .quant import quantize, QuantError
from .const import GGML_QUANT_VERSION, LlamaFileType
from safetensors.torch import load_file # optional as well; pip install safetensors
from tqdm import tqdm

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127

def load_state_dict(path):
    state_dict = load_file(path)
    prefix = None
    for pfx in ["model.diffusion_model.", "model."]:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break
    sd = {}
    for k, v in state_dict.items():
        if prefix and prefix not in k:
            continue
        if prefix:
            k = k.replace(prefix, "")
        sd[k] = v
    return sd

def load_model(path,model_arch):
    state_dict = load_state_dict(path)
    writer = GGUFWriter(path=None, arch=model_arch)
    return (writer, state_dict, model_arch)

def handle_tensors(args, writer, state_dict, model_arch):
    name_lengths = tuple(sorted(
        ((key, len(key)) for key in state_dict.keys()),
        key=lambda item: item[1],
        reverse=True,
    ))
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]
    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        bad_list = ", ".join(f"{key!r} ({namelen})" for key, namelen in name_lengths if namelen > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad_list}")
    for key, data in tqdm(state_dict.items()):
        old_dtype = data.dtype
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

            elif n_params <= QUANTIZATION_THRESHOLD:
                data_qtype = GGMLQuantizationType.F32

            elif ".weight" in key and any(x in key for x in blacklist):
                data_qtype = GGMLQuantizationType.F32

        try:
            data = quantize(data, data_qtype)
        except (AttributeError, QuantError) as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = GGMLQuantizationType.F16
            data = quantize(data, data_qtype)

        # ## t1 = t2 plus #####################################################################
        orig_shape = data.shape
        # data = data.reshape(n_params // 256, 256) # skip this as n_params might be empty
        writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))
        # ##### ###############################################################################

        new_name = key
        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len + 4}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(new_name, data, raw_dtype=data_qtype)

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
        writer, state_dict, model_arch = load_model(path,given)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        if next(iter(state_dict.values())).dtype == torch.bfloat16:
            out_path = f"{os.path.splitext(path)[0]}-bf16.gguf"
            writer.add_file_type(LlamaFileType.MOSTLY_BF16)
        else:
            out_path = f"{os.path.splitext(path)[0]}-f16.gguf"
            writer.add_file_type(LlamaFileType.MOSTLY_F16)
        if os.path.isfile(out_path):
            input("Output exists enter to continue or ctrl+c to abort!")
        handle_tensors(path, writer, state_dict, model_arch)
        writer.write_header_to_file(path=out_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No safetensors files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

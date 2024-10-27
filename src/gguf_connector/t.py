
import torch
from .writer import GGUFWriter, GGMLQuantizationType
from .quant import quantize, QuantError
from .const import GGML_QUANT_VERSION, LlamaFileType
from safetensors.torch import load_file
from tqdm import tqdm

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127

class ModelTemplate:
    arch = "invalid"
    shape_fix = False
    keys_detect = []
    keys_banned = []

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]

class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ), # Non-diffusers
        ("label_emb.0.0.weight",),
    ]

class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = False
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ), # Non-diffusers
    ]

# The architectures are checked in order and the first successful match terminates the search.
arch_list = [ModelFlux, ModelSD3, ModelSDXL, ModelSD1]

def is_model_arch(model, state_dict):
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, "Model architecture not allowed for conversion! (i.e. reference VS diffusers format)"
    return matched

def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch
            break
    assert model_arch is not None, "Unknown model architecture!"
    return model_arch

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = state_dict.get("model", state_dict)
    else:
        state_dict = load_file(path)
    sd = {}
    has_prefix = any(["model.diffusion_model." in x for x in state_dict.keys()])
    for k, v in state_dict.items():
        if has_prefix and "model.diffusion_model." not in k:
            continue
        if has_prefix:
            k = k.replace("model.diffusion_model.", "")
        sd[k] = v
    return sd

def load_model(path):
    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    print(f"* Architecture detected from input: {model_arch.arch}")
    writer = GGUFWriter(path=None, arch=model_arch.arch)
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
            # gguf.GGMLQuantizationType,
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
                # one-dimensional tensors should be kept in f32
                # data_qtype = gguf.GGMLQuantizationType.F32
                data_qtype = GGMLQuantizationType.F32

            elif n_params <= QUANTIZATION_THRESHOLD:
                # data_qtype = gguf.GGMLQuantizationType.F32
                data_qtype = GGMLQuantizationType.F32

            elif ".weight" in key and any(x in key for x in blacklist):
                # data_qtype = gguf.GGMLQuantizationType.F32
                data_qtype = GGMLQuantizationType.F32

        if (model_arch.shape_fix
            and n_dims > 1
            and n_params >= REARRANGE_THRESHOLD
            and (n_params / 256).is_integer()
            and not (data.shape[-1] / 256).is_integer()
        ):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))

        try:
            # data = gguf.quants.quantize(data, data_qtype)
            data = quantize(data, data_qtype)
        # except (AttributeError, gguf.QuantError) as e:
        except (AttributeError, QuantError) as e:
            tqdm.write(f"falling back to F16: {e}")
            # data_qtype = gguf.GGMLQuantizationType.F16
            # data = gguf.quants.quantize(data, data_qtype)
            data_qtype = GGMLQuantizationType.F16
            data = quantize(data, data_qtype)

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

        writer, state_dict, model_arch = load_model(path)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        if next(iter(state_dict.values())).dtype == torch.bfloat16:
            out_path = f"{os.path.splitext(path)[0]}-bf16.gguf"
            writer.add_file_type(LlamaFileType.MOSTLY_BF16)
        else:
            out_path = f"{os.path.splitext(path)[0]}-f16.gguf"
            writer.add_file_type(LlamaFileType.MOSTLY_F16)
            # out_path = args.dst or out_path
            if os.path.isfile(out_path):
                input("Output exists enter to continue or ctrl+c to abort!")
            # handle_tensors(path, writer, state_dict, model_arch)
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

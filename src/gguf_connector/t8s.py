
import torch # optional (if you want this conversion tool; pip install torch)
from .writer import GGUFWriter, GGMLQuantizationType
from .const import GGML_QUANT_VERSION, LlamaFileType
from safetensors.torch import load_file
from tqdm import tqdm

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch = "invalid"
    shape_fix = False
    keys_detect = []  # list of lists to match in state dict
    keys_banned = []  # list of keys that should mark model as invalid for conversion
    keys_hiprec = []  # list of keys that need to be kept in fp32 for some reason
    def handle_nd_tensor(self, key, data):
        raise NotImplementedError(f"Tensor detected that exceeds dims supported by C++ code! ({key} @ {data.shape})")

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

class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]

class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]

    def handle_nd_tensor(self, key, data):
        path = f"./fix_5d_tensors_{self.arch}.safetensors"
        if os.path.isfile(path):
            raise RuntimeError(f"5D tensor fix file already exists! {path}")
        fsd = {key: torch.from_numpy(data)}
        tqdm.write(f"5D key found in state dict! Manual fix required! - {key} {data.shape}")
        torch.save(fsd, path)

class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = [
        (
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        )
    ]

class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = [
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        )
    ]
    keys_hiprec = [
        "scale_shift_table" # nn.parameter, can't load from BF16 base quant
    ]

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
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ), # Non-diffusers
    ]

arch_list = [ModelFlux, ModelSD3, ModelAura, ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1]

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
            model_arch = arch()
            break
    assert model_arch is not None, "Unknown model architecture!"
    return model_arch

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ["model", "module"]:
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20:
            raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
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
        data = data.to(torch.float32).numpy()
        n_dims = len(data.shape)
        data_shape = data.shape
        data_qtype = getattr(
            GGMLQuantizationType,
            "BF16" if old_dtype == torch.bfloat16 else "F16"
        )
        if len(data.shape) > MAX_TENSOR_DIMS:
            model_arch.handle_nd_tensor(key, data)
            continue # needs to be added back later
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size
        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1:
                data_qtype = GGMLQuantizationType.F32
            elif n_params <= QUANTIZATION_THRESHOLD:
                data_qtype = GGMLQuantizationType.F32
            elif any(x in key for x in model_arch.keys_hiprec):
                data_qtype = GGMLQuantizationType.F32
        new_name = key
        data_qtype = GGMLQuantizationType.F32  # Force F32 for all tensors
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
            out_path = f"{os.path.splitext(path)[0]}-f32.gguf"
            writer.add_file_type(LlamaFileType.MOSTLY_BF16)
        else:
            out_path = f"{os.path.splitext(path)[0]}-f32.gguf"
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

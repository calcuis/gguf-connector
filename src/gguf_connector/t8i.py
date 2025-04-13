
import torch # need torch to work
import os, argparse
from tqdm import tqdm
from gguf_connector.reader import GGUFReader, GGMLQuantizationType
from gguf_connector.writer import GGUFWriter
from gguf_connector.quant import quantize
from gguf_connector.const import GGML_QUANT_VERSION, LlamaFileType

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--fix", required=False, help="Defaults to ./fix_5d_tensors_[arch].safetensors")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not os.path.isfile(args.src):
        parser.error(f"Invalid source file '{args.src}'")
    if not args.overwrite and os.path.exists(args.dst):
        parser.error(f"Output exists, use '--overwrite' ({args.dst})")
    return args

def get_arch_str(reader):
    field = reader.get_field("general.architecture")
    return str(field.parts[field.data[-1]], encoding="utf-8")

def get_file_type(reader):
    field = reader.get_field("general.file_type")
    ft = int(field.parts[field.data[-1]])
    return LlamaFileType(ft)

if __name__ == "__main__":
    args = get_args()
    reader = GGUFReader(args.src)
    arch = get_arch_str(reader)
    file_type = get_file_type(reader)
    print(f"Detected arch: '{arch}' (ftype: {str(file_type)})")
    sd5d = torch.load(f"fix_5d_tensors_{arch}.safetensors", weights_only=False)
    sd5d = {k:v.numpy() for k,v in sd5d.items()}
    print("5D tensors:", sd5d.keys())
    writer = GGUFWriter(path=None, arch=arch)
    writer.add_quantization_version(GGML_QUANT_VERSION)
    writer.add_file_type(file_type)
    added = []
    def add_extra_key(writer, key, data):
        global added
        data_qtype = GGMLQuantizationType.F32
        data = quantize(data, data_qtype)
        tqdm.write(f"Adding key {key} ({data.shape})")
        writer.add_tensor(key, data, raw_dtype=data_qtype)
        added.append(key)
    for tensor in tqdm(reader.tensors):
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        key5d = tensor.name.replace(".bias", ".weight")
        if key5d in sd5d.keys():
            add_extra_key(writer, key5d, sd5d[key5d])
    for key, data in sd5d.items():
        if key not in added:
            add_extra_key(writer, key, data)
    writer.write_header_to_file(path=args.dst)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

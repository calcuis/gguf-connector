
from .const import GGML_QUANT_VERSION, LlamaFileType
from .reader import GGUFReader
from .writer import GGUFWriter
from glob import glob

def get_arch_str(reader):
    field = reader.get_field("general.architecture")
    return str(field.parts[field.data[-1]], encoding="utf-8")

def get_file_type(reader):
    field = reader.get_field("general.file_type")
    ft = int(field.parts[field.data[-1]])
    return LlamaFileType(ft)

def merge_gguf_files(master_file, output_file):
    with open(master_file, "rb") as f_master:
        reader_master = GGUFReader(f_master)
        arch = get_arch_str(reader_master)
        file_type = get_file_type(reader_master)
        print(f"Using master file: {master_file}")
        print(f"Detected arch: '{arch}' (ftype: {str(file_type)})")
        writer = GGUFWriter(path=None, arch=arch)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_type)
        for tensor in reader_master.tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    for file in glob("*.gguf"):
        if file == master_file:
            continue
        print(f"Merging from: {file}")
        with open(file, "rb") as f:
            reader = GGUFReader(f)
            for tensor in reader.tensors:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    with open(output_file, "wb"):
        writer.write_header_to_file(path=output_file)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
    print(f"Saved merged file as: {output_file}")

all_gguf_files = sorted(glob("*.gguf"))
if not all_gguf_files:
    print("No .gguf files found in the current directory.")
else:
    print("Available .gguf files:")
    for idx, fname in enumerate(all_gguf_files):
        print(f"{idx + 1}: {fname}")
    selected = int(input("Select the master file by number: ")) - 1
    master_file = all_gguf_files[selected]
    ask=input("Assign a name other than output.gguf (Y/n)? ")
    if ask.lower() == 'y':
        given = input("Enter a file name: ")
    else:
        given = 'output.gguf'
    merge_gguf_files(master_file, given)

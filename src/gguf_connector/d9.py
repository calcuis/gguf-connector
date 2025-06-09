
import torch # optional (need torch to work; pip install torch)
from .const import GGML_QUANT_VERSION, LlamaFileType
from .quant1 import load_model, handle_tensors
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
        ask2=input("Convert to f/bf16 instead of 32 (Y/n)? ")
        if ask2.lower() == 'y':
            fp32 = False
        else:
            fp32 = True
        writer, state_dict, _ = load_model(path,given)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        if fp32:
            out_path = f"{os.path.splitext(path)[0]}-f32.gguf"
            writer.add_file_type(LlamaFileType.MOSTLY_F16)
        else:
            if next(iter(state_dict.values())).dtype == torch.bfloat16:
                out_path = f"{os.path.splitext(path)[0]}-bf16.gguf"
                writer.add_file_type(LlamaFileType.MOSTLY_BF16)
            else:
                out_path = f"{os.path.splitext(path)[0]}-f16.gguf"
                writer.add_file_type(LlamaFileType.MOSTLY_F16)
        if os.path.isfile(out_path):
            input("Output exists enter to continue or ctrl+c to abort!")
        handle_tensors(writer, state_dict, fp32)
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

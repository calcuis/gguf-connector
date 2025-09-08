
import torch, os # need torch to work; pip install torch
from .f4 import launch_fastvlm9_app, get_hf_cache_hub_path

from .quant3 import convert_gguf_to_safetensors
from .quant4 import add_metadata_to_safetensors
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files:
    print("\nGGUF file(s) available. Select which one to use:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice2 = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice2)-1
        selected_model_file=gguf_files[choice_index]
        print(f"Model file: {selected_model_file} is selected!")
        selected_file_path=selected_model_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        selected = "0.5b"
        model_path = get_hf_cache_hub_path(selected)
        if device == "cuda":
            print(f"Running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            use_bf16 = True
        else:
            use_bf16 = False
        print(f"Prepare to dequantize: {selected_file_path}")
        convert_gguf_to_safetensors(selected_file_path, model_path, use_bf16)
        add_metadata_to_safetensors(model_path, {'format': 'pt'})
        launch_fastvlm9_app()
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

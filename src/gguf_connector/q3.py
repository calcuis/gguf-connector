
import torch # optional (if you want this conversion tool; pip install torch)
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def quantize_to_fp8_e5m2(tensor):
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Input tensor must be in BF16 format.")
    tensor = tensor.to(torch.float16)
    # FP8 E5M2 has larger dynamic range but lower precision than E4M3FN
    fp8_max = 57344.0  # Approximate max for FP8 E5M2 (2^15 * 1.75)
    fp8_min = -fp8_max
    clamped_tensor = tensor.clamp(min=fp8_min, max=fp8_max)
    # Scale to fit within representable range (adaptive scaling)
    scale = fp8_max / torch.max(torch.abs(clamped_tensor))
    quantized_tensor = torch.round(clamped_tensor * scale) / scale
    return quantized_tensor.to(torch.float8_e5m2)

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
        input_file = selected_file
        output_file = f"{os.path.splitext(input_file)[0]}_fp8_e5m2.safetensors"
        data = load_file(input_file)
        quantized_data = {}
        print("Starting quantization process...")
        for key, tensor in tqdm(data.items(), desc="Quantizing tensors", unit="tensor"):
            # Move tensor to GPU
            tensor = tensor.to(dtype=torch.bfloat16, device="cuda")
            quantized_tensor = quantize_to_fp8_e5m2(tensor)
            quantized_data[key] = quantized_tensor.cpu()
        save_file(quantized_data, output_file)
        print(f"Quantized safetensors saved to {output_file}.")
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No safetensors files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

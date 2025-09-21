
import torch # need torch to work; pip install torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import gradio as gr

def launch_docling_app():
    # Load processor and model
    MODEL_ID = "callgg/docling-bf16"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to(DEVICE)
    def convert_image_to_docling(image,num_tokens):
        """Convert an uploaded image into Docling format."""
        if image is None:
            return "Please upload an image first."
        # Create message with instruction
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."}
                ]
            },
        ]
        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)
        # Generate outputs
        generated_ids = model.generate(**inputs, max_new_tokens=num_tokens)
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]
        doctags = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=False,
        )[0].lstrip()
        return doctags
    # Gradio UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 📝 Docling Converter")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload an image")
                convert_btn = gr.Button("Convert")
                num_tokens = gr.Slider(minimum=64, maximum=2048, value=128, step=1, label="Output Token")
            with gr.Column():
                output_box = gr.Textbox(label="Docling Output", lines=15)
        convert_btn.click(
            fn=convert_image_to_docling,
            inputs=[image_input,num_tokens],
            outputs=output_box
        )
    block.launch()

from pathlib import Path
def get_hf_cache_hub_path():
    home_dir = Path.home()
    hf_cache_path = home_dir / ".cache" / "huggingface" / "hub" / "models--callgg--docling-bf16" / "blobs" / "1cdad234deb1cde18ee6a586f849057f19851daf1fedce2e40aff791dbe46f61"
    return str(hf_cache_path)

import os
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

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        print(f"Device detected: {DEVICE}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")

        model_path = get_hf_cache_hub_path()
        if DEVICE == "cuda":
            print(f"Running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            use_bf16 = True
        else:
            use_bf16 = False
        print(f"Prepare to dequantize: {selected_file_path}")
        convert_gguf_to_safetensors(selected_file_path, model_path, use_bf16)
        add_metadata_to_safetensors(model_path, {'format': 'pt'})
        launch_docling_app()
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

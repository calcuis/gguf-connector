
import torch # need torch to work; pip install torch
from vtoo import process_vision_info # need vtoo; pip install vtoo
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gradio as gr

def launch_holo_app():
    MODEL_ID = "callgg/holo-bf16"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def describe_image(image, prompt, num_tokens):
        if image is None:
            return "Please upload an image first."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        generated_ids = model.generate(**inputs, max_new_tokens=num_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else "No description generated."
    sample_prompts = ['Describe this image.',
                    'Describe what you see in one sentence.']
    sample_prompts = [[x] for x in sample_prompts]
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🖼️ Image Descriptor (h3)")
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="filepath", label="Input Image")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                btn = gr.Button("Submit", variant="primary")
                num_tokens = gr.Slider(minimum=64, maximum=1024, value=128, step=1, label="Output Token")
            with gr.Column():
                output = gr.Textbox(label="Description", lines=5)
        btn.click(describe_image, inputs=[img_input,prompt,num_tokens], outputs=output)
    block.launch()

from pathlib import Path
def get_hf_cache_hub_path():
    home_dir = Path.home()
    hf_cache_path = home_dir / ".cache" / "huggingface" / "hub" / "models--callgg--holo-bf16" / "blobs" / "89deede1bb73ad52f74e45cae86d6252750b97c9b5d90e567dfc18df252fb417"
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
        launch_holo_app()
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

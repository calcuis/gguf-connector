
import torch # need torch to work; pip install torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import gradio as gr

def launch_fastvlm_app():
    MODEL_ID = "callgg/fastvlm-0.5b-bf16"
    IMAGE_TOKEN_INDEX = -200
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    def describe_image(img: Image.Image) -> str:
        if img is None:
            return "Please upload an image."
        messages = [{"role": "user", "content": "<image>\nDescribe this image in detail."}]
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)
        pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
        px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
        px = px.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            out = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=256
            )
        return tok.decode(out[0], skip_special_tokens=True)
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🖼️ Image Descriptor (f6)")
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Input Image")
                btn = gr.Button("Describe Image", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Description", lines=5)
        btn.click(fn=describe_image, inputs=img_input, outputs=output)
    block.launch()

from pathlib import Path
def get_hf_cache_hub_path():
    home_dir = Path.home()
    hf_cache_path = home_dir / ".cache" / "huggingface" / "hub" / "models--callgg--fastvlm-0.5b-bf16" / "blobs" / "4c70bf1ae23f362b9a623706bffbd0830999ea55c821c0b3545c49bd988095a7"
    return str(hf_cache_path)

import os
from .quant3 import convert_gguf_to_safetensors
from .quant4 import add_metadata_to_safetensors
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

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
        model_path = get_hf_cache_hub_path()
        if DEVICE == "cuda":
            use_bf16 = True
        else:
            use_bf16 = False
        print(f"Prepare to dequantize: {selected_file_path}")
        convert_gguf_to_safetensors(selected_file_path, model_path, use_bf16)
        add_metadata_to_safetensors(model_path, {'format': 'pt'})
        launch_fastvlm_app()
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

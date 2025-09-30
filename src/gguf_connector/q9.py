
import torch # need torch, transformers and dequantor to work
import gradio as gr
from dequantor import (
    QwenImageEditPlusPipeline,
    AutoencoderKLQwenImage,
)
from transformers import Qwen2_5_VLForConditionalGeneration
from .vrm import get_gpu_vram, get_affordable_precision
from PIL import Image

from nunchaku import NunchakuQwenImageTransformer2DModel # need nunchaku

def launch_image_edit_pluz_app(model_path,dtype):
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        model_path
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "callgg/qi-decoder",
        subfolder="text_encoder",
        # torch_dtype=torch.bfloat16
        dtype=dtype
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        "callgg/qi-decoder",
        subfolder="vae",
        torch_dtype=dtype
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "callgg/image-edit-plus",
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        torch_dtype=dtype
    )
    if get_gpu_vram() > 18:
        pipe.enable_model_cpu_offload()
    else:
        # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
        transformer.set_offload(
            True, use_pin_memory=False, num_blocks_on_gpu=1
        )  # increase num_blocks_on_gpu if you have more VRAM
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()
    # pipe.load_lora_weights("callgg/image-lite-lora", weight_name="lite.safetensors", adapter_name="lora")
    def generate_image(prompt, img1, img2, img3, steps, guidance):
        images = []
        for img in [img1, img2, img3]:
            if img is not None:
                if not isinstance(img, Image.Image):
                    img = Image.open(img)
                images.append(img.convert("RGB"))
        if not images:
            return None
        inputs = {
            "image": images,
            "prompt": prompt,
            "true_cfg_scale": guidance,
            "negative_prompt": " ",
            "num_inference_steps": steps,
            "num_images_per_prompt": 1,
        }
        with torch.inference_mode():
            output = pipe(**inputs)
            return output.images[0]
    sample_prompts = ['use image 2 as background of image 1',
                    'apply the image 2 costume to image 1 subject']
    sample_prompts = [[x] for x in sample_prompts]
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🖼️ Image Edit ++ - Multi Image Composer 🐷")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    img1 = gr.Image(label="Image 1", type="pil")
                    img2 = gr.Image(label="Image 2", type="pil")
                    img3 = gr.Image(label="Image 3", type="pil")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                generate_btn = gr.Button("Generate Image")
                steps = gr.Slider(1, 50, value=8, step=1, label="Inference Steps")
                guidance = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Guidance Scale")
            with gr.Column():
                output_image = gr.Image(label="Output", type="pil")
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, img1, img2, img3, steps, guidance],
            outputs=output_image,
        )
    block.launch()

import os
safetensors_files = [file for file in os.listdir() if file.endswith('.safetensors')]

if safetensors_files:
    print('Safetensors available. Select which one to use:')
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(safetensors_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = safetensors_files[choice_index]
        print(f'Model file: {selected_file} is selected!')
        input_path = selected_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        if device == "cuda":
            print(f"running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            prec = get_affordable_precision() # provide affordable precision hints
            print(f"affordable precision: {prec} (opt a wrong precision file will return error)")
        launch_image_edit_pluz_app(input_path,dtype)
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print('No safetensors are available in the current directory.')
    input('--- Press ENTER To Exit ---')

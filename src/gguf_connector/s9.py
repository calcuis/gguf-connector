
import torch # need torch, transformers, nunchaku and dequantor to work
import gradio as gr
from dequantor import (
    FluxKontextPipeline,
    QwenImageEditPlusPipeline,
    AutoencoderKLQwenImage,
)
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
    Qwen2_5_VLForConditionalGeneration,
)
from nunchaku import (
    NunchakuQwenImageTransformer2DModel,
    NunchakuFluxTransformer2dModel,
)
from .vrm import get_gpu_vram, get_affordable_precision
from .rg import repack_image

def launch_12b_sketch_app(model_path,dtype):
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        model_path
    )
    text_encoder = T5EncoderModel.from_pretrained(
        "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
        gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
        # torch_dtype=dtype
        dtype=dtype
    )
    tokenizer = T5TokenizerFast.from_pretrained(
        "callgg/kontext-decoder",
        subfolder="tokenizer_2",
        # torch_dtype=dtype
        dtype=dtype
    )
    pipe = FluxKontextPipeline.from_pretrained(
        "callgg/kontext-decoder",
        transformer=transformer,
        text_encoder_2=text_encoder,
        tokenizer_2=tokenizer,
        torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()
    def generate_image(image,prompt,neg_prompt,guidance_scale,num_steps):
        if image is None or prompt.strip() == "":
            return None
        else: image = repack_image(image)
        result = pipe(image=image,prompt=prompt,true_cfg_scale=guidance_scale,negative_prompt=neg_prompt,num_inference_steps=num_steps,
            ).images[0]
        return result
    sample_prompts = ['add a hat to the subject',
                    'convert to Ghibli style',
                    'turn this image into line style']
    sample_prompts = [[x] for x in sample_prompts]
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🖼️ Sketch (12b) 🐷")
        with gr.Row():
            with gr.Column():
                # input_image = gr.Image(type="pil", label="Input Image")
                input_image = gr.ImageEditor(
                    type="pil",
                    label="Draw here!",
                    image_mode="P", # color mode
                    brush=gr.Brush(colors=["#000000", "#FFFFFF"]), # black and white brush
                )
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                neg_prompt = gr.Textbox(label="Negative Prompt", value=" ", visible=False) # disable
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                num_steps = gr.Slider(minimum=1, maximum=100, value=15, step=1, label="Step")
                guidance = gr.Slider(minimum=0.0, maximum=10.0, value=1, step=0.1, label="Scale", visible=False)
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[input_image,prompt,neg_prompt,guidance,num_steps], outputs=output_image)
    block.launch()

def launch_20b_sketch_app(model_path,dtype):
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
        transformer.set_offload(
            True, use_pin_memory=False, num_blocks_on_gpu=1
        )
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()
    def generate_image(prompt, img1, img2, img3, steps, guidance):
        images = []
        for img in [img1, img2, img3]:
            if img is not None:
                images.append(repack_image(img))
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
    sample_prompts = ['turn the hand drawing into realistic object',
                    'color it',
                    'turn it into cartoon with color']
    sample_prompts = [[x] for x in sample_prompts]
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🖼️ Sketch (20b) 🐷")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    # img1 = gr.Image(label="Image 1", type="pil")
                    img1 = gr.ImageEditor(
                        type="pil",
                        label="Draw here!",
                        image_mode="P",
                        brush=gr.Brush(colors=["#000000", "#FFFFFF"]),
                    )
                    img2 = gr.Image(label="Image 2", type="pil", visible=False)
                    img3 = gr.Image(label="Image 3", type="pil", visible=False)
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                generate_btn = gr.Button("Transform Sketch")
                steps = gr.Slider(1, 50, value=4, step=1, label="Inference Steps", visible=False)
                guidance = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Guidance Scale", visible=False)
            with gr.Column():
                output_image = gr.Image(label="Output", type="pil")
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, img1, img2, img3, steps, guidance],
            outputs=output_image,
        )
    block.launch()

def launch_20b_sketch_dual_app(model_path,dtype):
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
        transformer.set_offload(
            True, use_pin_memory=False, num_blocks_on_gpu=1
        )
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()
    def generate_image(prompt, img1, img2, img3, steps, guidance):
        images = []
        for img in [img1, img2, img3]:
            if img is not None:
                images.append(repack_image(img))
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
    sample_prompts = ['turn the hand drawing into realistic object',
                    'color the black and white hand drawing']
    sample_prompts = [[x] for x in sample_prompts]
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🖼️ Sketch (dual mode) 🐷")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    img1 = gr.ImageEditor(
                        type="pil",
                        label="Draw here!",
                        # image_mode="L", # grayscale mode
                        image_mode="P", # color mode
                        brush=gr.Brush(colors=["#000000", "#FFFFFF"]),
                    )
                    img2 = gr.ImageEditor(
                        type="pil",
                        label="Draw here!",
                        image_mode="P",
                        brush=gr.Brush(colors=["#000000", "#FFFFFF"]),
                    )
                    img3 = gr.Image(label="Image 3", type="pil", visible=False)
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                generate_btn = gr.Button("Generate Image")
                steps = gr.Slider(1, 50, value=4, step=1, label="Inference Steps")
                guidance = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Guidance Scale", visible=False)
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
        prec = get_affordable_precision()
        print(f"machine precision: {prec} (note: opt a wrong precision file will return error)")
        if prec == "fp4":
            ask=input("Use 12b model instead (Y/n)? ")
            if ask.lower() == 'y':
                launch_12b_sketch_app(input_path,dtype)
            else:
                launch_20b_sketch_app(input_path,dtype)
        else:
            ask=input("Opt dual mode (Y/n)? ")
            if ask.lower() == 'y':
                launch_20b_sketch_dual_app(input_path,dtype)
            else:
                launch_20b_sketch_app(input_path,dtype)
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print('No safetensors are available in the current directory.')
    input('--- Press ENTER To Exit ---')

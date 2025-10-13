
import torch # need torch, transformers, lpips, nunchaku and dequantor to work
import gradio as gr
from PIL import Image
import numpy as np
import lpips
from dequantor import (
    StableDiffusion3Pipeline,
    GGUFQuantizationConfig,
    SD3Transformer2DModel,
    QwenImageEditPlusPipeline,
    AutoencoderKLQwenImage,
)
from transformers import (
    T5EncoderModel,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from nunchaku import (
    NunchakuQwenImageTransformer2DModel,
)
from .vrm import get_gpu_vram, get_affordable_precision
from .f4 import get_hf_cache_hub_path
from .quant3 import convert_gguf_to_safetensors
from .quant4 import add_metadata_to_safetensors

def launch_app(model_path1,model_path,dtype):
    # image recognition model
    MODEL_ID = "callgg/fastvlm-0.5b-bf16"
    IMAGE_TOKEN_INDEX = -200
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    def describe_image(img: Image.Image, prompt, num_tokens) -> str:
        if img is None:
            return "Please upload an image."
        messages = [{"role": "user", "content": f"<image>\n{prompt}."}]
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
                max_new_tokens=num_tokens
            )
        return tok.decode(out[0], skip_special_tokens=True)
    sample1_prompts = ['describe this image in detail',
                    'describe what you see in few words',
                    'tell me the difference']
    sample1_prompts = [[x] for x in sample1_prompts]
    # image generation model
    transformer1 = SD3Transformer2DModel.from_single_file(
        model_path1,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/sd3-decoder",
        subfolder="transformer_2"
    )
    text_encoder1 = T5EncoderModel.from_pretrained(
        "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
        gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
        dtype=dtype
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "callgg/sd3-decoder",
        transformer=transformer1,
        text_encoder_3=text_encoder1,
        torch_dtype=dtype
    )
    pipeline.enable_model_cpu_offload()
    # Inference function
    def generate_image2(prompt, num_steps, guidance):
        result = pipeline(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
        ).images[0]
        return result
    sample_prompts2 = ['a cat in a hat',
                    'a pig in a hat',
                    'a raccoon in a hat',
                    'a dog walking with joy']
    sample_prompts2 = [[x] for x in sample_prompts2]
    # image transformation model
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
    sample_prompts = ['merge it',
                    'remove background',
                    'use image 1 as background of image 2']
    sample_prompts = [[x] for x in sample_prompts]
    # image discrimination model
    def compare_images(img1,img2):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lpips_model = lpips.LPIPS(net='squeeze').to(device)
        if img1 is None or img2 is None:
            return "Please upload both images."
        img1_np = np.array(img1).astype(np.float32) / 255.0
        img2_np = np.array(img2).astype(np.float32) / 255.0
        # Convert to tensor in LPIPS format
        img1_tensor = lpips.im2tensor(img1_np).to(device)
        img2_tensor = lpips.im2tensor(img2_np).to(device)
        # Compute LPIPS distance
        with torch.no_grad():
            distance = lpips_model(img1_tensor, img2_tensor)
        score = distance.item()
        similarity = max(0.0, 1.0 - score*100)  # normalize to positive similarity
        result_text = (
            f"LPIPS Distance: {score:.4f}\n"
            f"Estimated Similarity: {similarity*100:.4f}%"
            # f"Estimated Similarity: {similarity:.4f} (1 = identical, 0 = dissimilar)"
        )
        return result_text
    # UI
    block = gr.Blocks(title="image studio").queue()
    with block:
        gr.Markdown("## Discriminator")
        with gr.Row():
            img1 = gr.Image(type="pil", label="Image 1")
            img2 = gr.Image(type="pil", label="Image 2")
        compare_btn = gr.Button("Discriminate")
        output_box = gr.Textbox(label="Statistics", lines=2)
        compare_btn.click(compare_images, inputs=[img1,img2], outputs=output_box)
        gr.Markdown("## Descriptor")
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Input Image")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample1_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                btn = gr.Button("Describe")
                num_tokens = gr.Slider(minimum=64, maximum=1024, value=128, step=1, label="Output Token")
            with gr.Column():
                output = gr.Textbox(label="Description", lines=5)
        btn.click(fn=describe_image, inputs=[img_input,prompt,num_tokens], outputs=output)
        gr.Markdown("## Generator")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts2, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                num_steps = gr.Slider(minimum=4, maximum=100, value=8, step=1, label="Step")
                guidance = gr.Slider(minimum=1.0, maximum=10.0, value=2.5, step=0.1, label="Scale")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image2, inputs=[prompt, num_steps, guidance], outputs=output_image)
        gr.Markdown("## Transformer")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    img1 = gr.Image(label="Image 1", type="pil")
                    img2 = gr.Image(label="Image 2", type="pil")
                    img3 = gr.Image(label="Image 3", type="pil")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                generate_btn = gr.Button("Transform")
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
import os
# part 1
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files:
    print("GGUF file(s) available. Select which one to use as image recognizor:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice2 = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice2)-1
        selected_model_file=gguf_files[choice_index]
        print(f"Model file: {selected_model_file} is selected as image recognizor!")
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
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")
# part 2
gguf_files2 = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files2:
    print('GGUF file(s) available. Select which one to use as image generator:')
    for index, file_name in enumerate(gguf_files2, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(gguf_files2)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = gguf_files2[choice_index]
        print(f'Model file: {selected_file} is selected as image generator!')
        input_path1 = selected_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        if device == "cuda":
            print(f"running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')
# part 3
safetensors_files = [file for file in os.listdir() if file.endswith('.safetensors')]
if safetensors_files:
    print('Safetensors available. Select which one to use as image transformer:')
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(safetensors_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = safetensors_files[choice_index]
        print(f'Model file: {selected_file} is selected as image transformer!')
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
        launch_app(input_path1,input_path,dtype)
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print('No safetensors are available in the current directory.')
    input('--- Press ENTER To Exit ---')

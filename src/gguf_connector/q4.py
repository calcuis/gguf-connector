import torch # need torch, transformers and dequantor to work
from dequantor import DiffusionPipeline, GGUFQuantizationConfig, QwenImageTransformer2DModel, AutoencoderKLQwenImage, QwenImageEditPipeline
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
import gradio as gr

# Multi-GPU detection and auto-mapping setup
_gpu_count = torch.cuda.device_count()
if _gpu_count > 1:
    _gpu_names = [torch.cuda.get_device_name(i) for i in range(_gpu_count)]
    print("Detected GPUs:", _gpu_names)
    device_map = "auto"
    max_memory = {}
    for i in range(_gpu_count):
        total_mb = torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)
        max_memory[i] = f"{int(total_mb)}MB"

    print("Using device_map:", device_map)
    print("Max memory per device:", max_memory)
else:
    device_map = None
    max_memory = None

def launch_qi_app(model_path,dtype):
    transformer = QwenImageTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/qi-decoder",
        subfolder="transformer",
        device_map=device_map,
        max_memory=max_memory
        )
    # text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "chatpig/qwen2.5-vl-7b-it-gguf",
    #     gguf_file="qwen2.5-vl-7b-it-q2_k.gguf",
    #     torch_dtype=dtype
    #     )
    # vae = AutoencoderKLQwenImage.from_pretrained(
    #     "callgg/qi-decoder",
    #     subfolder="vae",
    #     torch_dtype=dtype
    #     )
    pipe = DiffusionPipeline.from_pretrained(
        "callgg/qi-decoder",
        transformer=transformer,
        # text_encoder=text_encoder,
        # vae=vae,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory
    )
    pipe.enable_model_cpu_offload()
    positive_magic = {"en": "Ultra HD, 4K, cinematic composition."}
    negative_prompt = " "
    # Inference function
    def generate_image(prompt,num_steps,cfg_scale):
        result = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        height=1024,
        width=1024,
        # num_inference_steps=24,
        # true_cfg_scale=2.5,
        num_inference_steps=num_steps,
        true_cfg_scale=cfg_scale,
        generator=torch.Generator()
        ).images[0]
        return result
    # Lazy prompt
    sample_prompts = ['a bear in a hat',
                    'a cat walking in a cyber city',
                    'a pig holding a sign that says hello world']
    sample_prompts = [[x] for x in sample_prompts]
    # Gradio UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🐷 Qwen Image Generator")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                num_steps = gr.Slider(minimum=4, maximum=100, value=8, step=1, label="Step")
                cfg_scale = gr.Slider(minimum=0, maximum=10, value=1, step=0.5, label="Scale")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[prompt,num_steps,cfg_scale], outputs=output_image)
    block.launch(server_name="0.0.0.0")

def launch_qi_distill_app(model_path,dtype):
    transformer = QwenImageTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/qi-decoder",
        subfolder="transformer",
        device_map=device_map,
        max_memory=max_memory
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "callgg/qi-decoder",
        subfolder="text_encoder",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype
            ),
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory
    )
    text_encoder = text_encoder.to("cpu")
    # vae = AutoencoderKLQwenImage.from_pretrained(
    #     "callgg/qi-decoder",
    #     subfolder="vae",
    #     torch_dtype=dtype
    #     )
    pipe = DiffusionPipeline.from_pretrained(
        "callgg/qi-decoder",
        transformer=transformer,
        text_encoder=text_encoder,
        # vae=vae,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory
    )
    pipe.enable_model_cpu_offload()
    positive_magic = {"en": "Ultra HD, 4K, cinematic composition."}
    negative_prompt = " "
    # Inference function
    def generate_image(prompt,num_steps):
        result = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        # true_cfg_scale=2.5,
        generator=torch.Generator()
        ).images[0]
        return result
    sample_prompts = ['a pig holding a sign that says hello world',
                    'a bear walking in a cyber city',
                    'a frog in a hat']
    sample_prompts = [[x] for x in sample_prompts]
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🐷 Qwen Image Generator (distilled)")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                num_steps = gr.Slider(minimum=4, maximum=100, value=15, step=1, label="Step")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[prompt,num_steps], outputs=output_image)
    block.launch(server_name="0.0.0.0")

def launch_image_edit_app(model_path,dtype):
    transformer = QwenImageTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/image-edit-decoder",
        subfolder="transformer",
        device_map=device_map,
        max_memory=max_memory
        )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "callgg/qi-decoder",
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory
        )
    vae = AutoencoderKLQwenImage.from_pretrained(
        "callgg/qi-decoder",
        subfolder="vae",
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory
        )
    pipe = QwenImageEditPipeline.from_pretrained(
            "callgg/image-edit-decoder",
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory
        )
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights("callgg/image-lite-lora", weight_name="lite.safetensors", adapter_name="lora")
    def generate_image(image,prompt,neg_prompt,guidance_scale,num_steps):
        if image is None or prompt.strip() == "":
            return None
        result = pipe(image=image,prompt=prompt,true_cfg_scale=guidance_scale,negative_prompt=neg_prompt,num_inference_steps=num_steps,
            ).images[0]
        return result
    sample_prompts = ['add a hat to the subject',
                    'convert to Ghibli style',
                    'turn this image into line style']
    sample_prompts = [[x] for x in sample_prompts]
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🐷 Image Editor")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                neg_prompt = gr.Textbox(label="Negative Prompt", value=" ", visible=False) # disable
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                num_steps = gr.Slider(minimum=4, maximum=100, value=8, step=1, label="Step")
                guidance = gr.Slider(minimum=0.0, maximum=10.0, value=1, step=0.1, label="Scale")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[input_image,prompt,neg_prompt,guidance,num_steps], outputs=output_image)
    block.launch(server_name="0.0.0.0")


from transformers import T5EncoderModel # need transformers and dequantor to work
from dequantor import FluxPipeline, FluxKontextPipeline, GGUFQuantizationConfig, FluxTransformer2DModel
from PIL import Image
import gradio as gr

def launch_krea_app(model_path, dtype):
    transformer = FluxTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/krea-decoder",
        subfolder="transformer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
        gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
        torch_dtype=dtype
    )
    pipe = FluxPipeline.from_pretrained(
        "callgg/krea-decoder",
        transformer=transformer,
        text_encoder_2=text_encoder,
        torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()
    # Inference function
    def generate_image(prompt, num_steps, guidance):
        result = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
        ).images[0]
        return result
    # Lazy prompt
    sample_prompts = ['a cat in a hat',
                    'a dog walking in a cyber city with joy',
                    'a pig holding a sign that says hello world']
    sample_prompts = [[x] for x in sample_prompts]
    # Gradio UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🐷 Krea/Flux Image Generator")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                num_steps = gr.Slider(minimum=4, maximum=100, value=8, step=1, label="Step")
                guidance = gr.Slider(minimum=1.0, maximum=10.0, value=3.5, step=0.1, label="Guidance Scale")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[prompt,num_steps,guidance], outputs=output_image)
    block.launch()

def launch_kontext_app(model_path, dtype):
    transformer = FluxTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/kontext-decoder",
        subfolder="transformer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
        gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
        torch_dtype=dtype
    )
    pipe = FluxKontextPipeline.from_pretrained(
        "callgg/kontext-decoder",
        transformer=transformer,
        text_encoder_2=text_encoder,
        torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()
    # Inference function
    def generate_image(image: Image.Image, prompt: str, guidance_scale: float = 2.5):
        if image is None or prompt.strip() == "":
            return None
        result = pipe(image=image, prompt=prompt, guidance_scale=guidance_scale).images[0]
        return result
    # Lazy prompt
    sample_prompts = ['add a hat to the subject',
                    'convert to Ghibli style',
                    'turn this image into line style']
    sample_prompts = [[x] for x in sample_prompts]
    # Gradio UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🐷 Kontext Image Editor")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                guidance = gr.Slider(minimum=1.0, maximum=10.0, value=2.5, step=0.1, label="Guidance Scale")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[input_image, prompt, guidance], outputs=output_image)
    block.launch()

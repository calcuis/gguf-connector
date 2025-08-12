
import torch # need torch, transformers and dequantor to work
import gradio as gr
from dequantor import DiffusionPipeline, GGUFQuantizationConfig, QwenImageTransformer2DModel
# from dequantor import DiffusionPipeline, GGUFQuantizationConfig, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

def launch_qi_app(model_path,dtype):
    transformer = QwenImageTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/qi-decoder",
        subfolder="transformer"
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
        torch_dtype=dtype
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
                num_steps = gr.Slider(minimum=4, maximum=100, value=24, step=1, label="Step")
                cfg_scale = gr.Slider(minimum=0, maximum=10, value=2.5, step=0.5, label="Scale")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[prompt,num_steps,cfg_scale], outputs=output_image)
    block.launch()

def launch_qi_distill_app(model_path,dtype):
    transformer = QwenImageTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/qi-decoder",
        subfolder="transformer"
        )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "callgg/qi-decoder",
        subfolder="text_encoder",
        quantization_config=TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype
            ),
        torch_dtype=dtype,
    )
    text_encoder = text_encoder.to("cpu")
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
        torch_dtype=dtype
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
        # num_inference_steps=24,
        # true_cfg_scale=2.5,
        generator=torch.Generator()
        ).images[0]
        return result
    # Lazy prompt
    sample_prompts = ['a pig holding a sign that says hello world',
                    'a bear walking in a cyber city',
                    'a frog in a hat']
    sample_prompts = [[x] for x in sample_prompts]
    # Gradio UI
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
    block.launch()

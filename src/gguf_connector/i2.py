
import torch # optional (need torch, diffusers to work; pip install torch, diffusers)
import gradio as gr # optional (need gradio for lazy ui; pip install gradio)
from transformers import T5EncoderModel
from diffusers import PixArtSigmaPipeline

text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/ltxv-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    torch_dtype=torch.float16,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "callgg/pixart-decoder",
    text_encoder=text_encoder,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

def generate_image(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    image.save("output.png")
    return "output.png"

# Gradio UI

sample_prompts = [
    'close-up portrait of girl',
    'close-up portrait of anime pig',
    'close-up portrait of young lady',
]
sample_prompts = [[x] for x in sample_prompts]

block = gr.Blocks(title="gguf").queue()
with block:
    gr.Markdown("## 📷 Image 2")
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
        neg_prompt_input = gr.Textbox(label="Negative Prompt", value="blurry, cropped, ugly", visible=False) # disable
        quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt_input])
        quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt_input, show_progress=False, queue=False)
    with gr.Row():
        width_input = gr.Number(label="Width", value=1024)
        height_input = gr.Number(label="Height", value=1024)
        guidance_scale = gr.Number(label="Guidance Scale", value=3.5)
        num_inference_steps = gr.Number(label="Inference Steps", value=25)
    generate_btn = gr.Button("Generate")
    output_image = gr.Image(label="Generated Picture")

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input, neg_prompt_input,
            width_input, height_input,
            guidance_scale, num_inference_steps,
        ],
        outputs=output_image,
    )

block.launch()

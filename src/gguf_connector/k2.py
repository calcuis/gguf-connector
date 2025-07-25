
import torch # optional (need torch, diffusers to work; pip install torch, diffusers)
import gradio as gr # optional (need gradio for lazy ui; pip install gradio)
from transformers import T5EncoderModel
from diffusers import FluxKontextPipeline
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/kontext-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    torch_dtype=torch.bfloat16,
    )

pipe = FluxKontextPipeline.from_pretrained(
    "calcuis/kontext-gguf",
    text_encoder_2=text_encoder,
    torch_dtype=torch.bfloat16
    ).to(DEVICE)

def generate_image(image: Image.Image, prompt: str, guidance_scale: float = 2.5):
    if image is None or prompt.strip() == "":
        return None
    result = pipe(image=image, prompt=prompt, guidance_scale=guidance_scale).images[0]
    return result

# Gradio UI
block = gr.Blocks(title="gguf").queue()
with block:
    gr.Markdown("## 🧠 Kontext Image Editor")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            prompt = gr.Textbox(label="Prompt", placeholder="e.g., Add a hat to the cat")
            guidance = gr.Slider(minimum=1.0, maximum=10.0, value=2.5, step=0.1, label="Guidance Scale")
            submit_btn = gr.Button("Generate")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Output Image")
    submit_btn.click(fn=generate_image, inputs=[input_image, prompt, guidance], outputs=output_image)
block.launch()

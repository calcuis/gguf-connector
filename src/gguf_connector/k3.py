
import torch # optional (need torch, diffusers to work; pip install torch, diffusers)
import gradio as gr
from transformers import T5EncoderModel
from diffusers import FluxKontextPipeline, GGUFQuantizationConfig, FluxTransformer2DModel
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Detected device: {DEVICE}")

model_path = "https://huggingface.co/calcuis/kontext-gguf/blob/main/flux1-v2-kontext-dev-f32-q2_k.gguf"
transformer = FluxTransformer2DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="callgg/kontext-decoder",
    subfolder="transformer"
)

text_encoder = T5EncoderModel.from_pretrained(
    "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
    gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
    torch_dtype=torch.bfloat16
    )

pipe = FluxKontextPipeline.from_pretrained(
    "callgg/kontext-decoder",
    transformer=transformer,
    text_encoder_2=text_encoder,
    torch_dtype=torch.bfloat16
    )

ask=input(f"Use {DEVICE} instead of memory economy mode (Y/n)? ")
if ask.lower() == 'y':
    print(f"Using device: {DEVICE}")
    pipe.to(DEVICE)
else:
    print("Super low vram or no vram option activated!")
    pipe.enable_model_cpu_offload()

def generate_image(image: Image.Image, prompt: str, guidance_scale: float = 2.5):
    if image is None or prompt.strip() == "":
        return None
    result = pipe(image=image, prompt=prompt, guidance_scale=guidance_scale).images[0]
    return result

sample_prompts = ['add a hat to the subject',
                  'convert to Ghibli style',
                  'turn this image into line style']
sample_prompts = [[x] for x in sample_prompts]

# Gradio UI
block = gr.Blocks(title="gguf").queue()
with block:
    gr.Markdown("## 🧠 Kontext Image Editor")
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

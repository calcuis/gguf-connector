
import torch # need torch to work; pip install torch
import gradio as gr
from transformers import T5EncoderModel
from diffusers import FluxPipeline, GGUFQuantizationConfig, FluxTransformer2DModel

# Opt a model to dequant
ask=input("Opt 1 for 2-bit (default), 2 for 4-bit, or 3 for 8-bit (enter: 1-3)? ")
if ask == '3':
    print("8-bit model selected!")
    model_path = "https://huggingface.co/calcuis/krea-gguf/blob/main/flux1-krea-dev-q8_0.gguf"
elif ask == '2':
    print("4-bit model selected!")
    model_path = "https://huggingface.co/calcuis/krea-gguf/blob/main/flux1-krea-dev-q4_0.gguf"
else:
    print("2-bit model selected!")
    model_path = "https://huggingface.co/calcuis/krea-gguf/blob/main/flux1-krea-dev-q2_k.gguf"
# Launching
transformer = FluxTransformer2DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="callgg/krea-decoder",
    subfolder="transformer"
)
text_encoder = T5EncoderModel.from_pretrained(
    "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
    gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
    torch_dtype=torch.bfloat16
)
pipe = FluxPipeline.from_pretrained(
    "callgg/krea-decoder",
    transformer=transformer,
    text_encoder_2=text_encoder,
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
# Inference function
def generate_image(prompt):
    result = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=2.5,
    ).images[0]
    return result
# Gradio UI
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt", placeholder="e.g. a pig holding a sign that says hello world"),
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Krea 4 Image Generator 🐷",
    description="Enter a prompt and generate an image using Flux pipeline via gguf-connector"
)
interface.launch()

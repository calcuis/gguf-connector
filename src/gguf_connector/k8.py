
import torch # need torch to work; pip install torch
from transformers import T5EncoderModel # need transformers and dequantor to work
from dequantor import FluxKontextPipeline, GGUFQuantizationConfig, FluxTransformer2DModel
from PIL import Image
import gradio as gr

# Auto-detection logic
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device detected: {device}")
print(f"torch version: {torch.__version__}")
print(f"dtype using: {dtype}\n")

def launch_kontext_app(model_path):
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
        gr.Markdown("## 🐷 Kontext 8 Image Editor (semi-full quant supported)")
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

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print('GGUF file(s) available. Select which one to use:')
    for index, file_name in enumerate(gguf_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(gguf_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = gguf_files[choice_index]
        print(f'Model file: {selected_file} is selected!')
        input_path = selected_file
        launch_kontext_app(input_path)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

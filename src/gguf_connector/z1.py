
import torch # need torch to work; pip install torch
from dequantor import ZImagePipeline, GGUFQuantizationConfig
from dequantor.models import ZImageTransformer2DModel
import gradio as gr

def launch_z_app(model_path,dtype):
    transformer = ZImageTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/z-image-decoder",
        subfolder="transformer"
    )
    pipe = ZImagePipeline.from_pretrained(
        "callgg/z-image-decoder",
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()
    # Inference function
    def generate_image(prompt, guidance):
        result = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=guidance,
            guidance_scale=0.0,
        ).images[0]
        return result
    # Lazy prompt
    sample_prompts = ['a cat in a hat',
                    'a pig walking in a cyber city with joy',
                    'a frog holding a sign that says hello world']
    sample_prompts = [[x] for x in sample_prompts]
    # Gradio UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## Z Image Generator")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                guidance = gr.Slider(minimum=6, maximum=50, value=8, step=1, label="Step")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[prompt, guidance], outputs=output_image)
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        if device == "cuda":
            print(f"running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        launch_z_app(input_path,dtype)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

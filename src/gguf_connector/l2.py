
import torch # need torch and dequantor to work
from dequantor import Lumina2Pipeline, Lumina2Transformer2DModel, GGUFQuantizationConfig, AutoencoderKL
from transformers import Gemma2Model, GemmaTokenizerFast
import gradio as gr

def launch_lumina_app(model_path,dtype):
    transformer = Lumina2Transformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/lumina-decoder",
        subfolder="transformer"
        )
    text_encoder = Gemma2Model.from_pretrained(
        "callgg/lumina-decoder",
        subfolder="text_encoder",
        torch_dtype=dtype
        )
    tokenizer = GemmaTokenizerFast.from_pretrained(
        "callgg/lumina-decoder",
        subfolder="tokenizer",
        torch_dtype=dtype
        )
    vae = AutoencoderKL.from_pretrained(
        "callgg/kontext-decoder",
        subfolder="vae",
        torch_dtype=dtype
        )
    pipe = Lumina2Pipeline.from_pretrained(
            "callgg/lumina-decoder",
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            torch_dtype=dtype
        )
    pipe.enable_model_cpu_offload()
    def generate_image(prompt,guidance,cfg_scale,num_steps):
        result = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=guidance,
        num_inference_steps=num_steps,
        cfg_trunc_ratio=cfg_scale,
        cfg_normalization=True,
        generator=torch.Generator()
        ).images[0]
        return result
    sample_prompts = ['a pig holding a sign that says hello world',
                    'a dog in a hat',
                    'a racoon walking in a cyber city']
    sample_prompts = [[x] for x in sample_prompts]
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🐷 Lumina Image Generator")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                # neg_prompt = gr.Textbox(label="Negative Prompt", value=" ", visible=False) # disable
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
                num_steps = gr.Slider(minimum=4, maximum=100, value=8, step=1, label="Step") # 24 steps or higher recommended
                cfg_scale = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Scale")
                guidance = gr.Slider(minimum=0.0, maximum=10.0, value=2.5, step=0.1, label="Guidance")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[prompt,guidance,cfg_scale,num_steps], outputs=output_image)
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
        launch_lumina_app(input_path,dtype)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

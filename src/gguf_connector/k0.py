
import torch # need torch, transformers and dequantor to work
from transformers import T5EncoderModel, T5TokenizerFast
from dequantor import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
import gradio as gr

def launch_kx_image_edit_app(model_path,dtype):
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
    tokenizer = T5TokenizerFast.from_pretrained(
        "callgg/kontext-decoder",
        subfolder="tokenizer_2",
        torch_dtype=dtype
    )
    pipe = FluxKontextPipeline.from_pretrained(
        "callgg/kontext-decoder",
        transformer=transformer,
        text_encoder_2=text_encoder,
        tokenizer_2=tokenizer,
        torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()
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
        launch_kx_image_edit_app(input_path,dtype)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

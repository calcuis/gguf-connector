
import torch # need torch and dequantor to work
import gradio as gr
from dequantor import DiffusionPipeline, GGUFQuantizationConfig, QwenImageTransformer2DModel
# from dequantor import DiffusionPipeline, GGUFQuantizationConfig, QwenImageTransformer2DModel, AutoencoderKLQwenImage
# from transformers import Qwen2_5_VLForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device detected: {device}")
print(f"torch version: {torch.__version__}")
print(f"dtype using: {dtype}\n")

def launch_qi_app(model_path):
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
    def generate_image(prompt):
        result = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        height=1024,
        width=1024,
        num_inference_steps=24,
        true_cfg_scale=2.5,
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
        gr.Markdown("## 🐷 Qwen Image Generator")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
                submit_btn = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        submit_btn.click(fn=generate_image, inputs=[prompt], outputs=output_image)
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
        launch_qi_app(input_path)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

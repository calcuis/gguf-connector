
import torch
from transformers import T5EncoderModel
from dequantor import MochiPipeline, MochiTransformer3DModel, AutoencoderKLMochi, GGUFQuantizationConfig
from dequantor.utils import export_to_video
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device detected: {device}")
print(f"torch version: {torch.__version__}")
print(f"dtype using: {dtype}\n")

def launch_mochi_app(model_path):
    transformer = MochiTransformer3DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/mochi-decoder",
        subfolder="transformer"
        )
    text_encoder = T5EncoderModel.from_pretrained(
        "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
        gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
        torch_dtype=dtype,
        )
    vae = AutoencoderKLMochi.from_pretrained(
        "callgg/mochi-decoder",
        subfolder="vae",
        torch_dtype=dtype
        )
    pipe = MochiPipeline.from_pretrained(
        "callgg/mochi-decoder",
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=dtype
        )
    pipe.enable_model_cpu_offload()
    # Inference function
    def generate_video(prompt, negative_prompt, num_frames):
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=480,
            height=480,
            num_frames=num_frames,
            num_inference_steps=25
        ).frames[0]
        export_to_video(video, "output.mp4", fps=24)
        return "output.mp4"
    # Lazy prompt
    sample_prompts = [
        "A pinky pig moving quickly in a beautiful winter scenery nature trees sunset tracking camera.",
        "A pinky pig swimming in a beautiful lake scenery nature rock and hill sunset tracking camera."
    ]
    sample_prompts = [[x] for x in sample_prompts]
    # UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🎥 Mochi Video Generator")
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
                neg_prompt_input = gr.Textbox(label="Negative Prompt", value="blurry ugly bad", visible=False) # disable
                quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt_input])
                quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt_input, show_progress=False, queue=False)
                generate_btn = gr.Button("Generate")
                num_frames_input = gr.Slider(minimum=15, maximum=200, value=25, step=5, label="Length")
            with gr.Column():
                output_video = gr.Video(label="Generated Video")
        generate_btn.click(
            fn=generate_video,
            inputs=[prompt_input, neg_prompt_input,num_frames_input],
            outputs=output_video
        )
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
        launch_mochi_app(input_path)
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')

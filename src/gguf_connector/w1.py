
import torch # optional (need torch to work; pip install torch)
from transformers import UMT5EncoderModel
from dequantor import AutoencoderKLWan, WanPipeline, WanTransformer3DModel, WanVACEPipeline, WanVACETransformer3DModel, GGUFQuantizationConfig
from dequantor.utils import export_to_video
import gradio as gr

def launch_wan_app(model_path, dtype):
    transformer = WanTransformer3DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/t2v-decoder",
        subfolder="transformer"
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        "chatpig/umt5xxl-encoder-gguf",
        gguf_file="umt5xxl-encoder-q2_k.gguf",
        torch_dtype=dtype
    )
    vae = AutoencoderKLWan.from_pretrained(
        "callgg/t2v-decoder",
        subfolder="vae",
        torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        "callgg/t2v-decoder",
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae, torch_dtype=dtype
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
        export_to_video(video, "output.mp4", fps=16)
        return "output.mp4"
    # Lazy prompt
    sample_prompts = [
        'A cat and a dog baking a cake together in a kitchen.',
        'A pig moving quickly in a beautiful forest.',
    ]
    sample_prompts = [[x] for x in sample_prompts]
    # UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🎥 Wan Video Generator")
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

def launch_vace_app(model_path, dtype):
    transformer = WanVACETransformer3DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/wan-decoder",
        subfolder="transformer"
        )
    text_encoder = UMT5EncoderModel.from_pretrained(
        "chatpig/umt5xxl-encoder-gguf",
        gguf_file="umt5xxl-encoder-q2_k.gguf",
        torch_dtype=dtype
        )
    vae = AutoencoderKLWan.from_pretrained(
        "callgg/wan-decoder",
        subfolder="vae",
        torch_dtype=torch.float32
        )
    pipe = WanVACEPipeline.from_pretrained(
        "callgg/wan-decoder",
        transformer=transformer,
        text_encoder=text_encoder,
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
        export_to_video(video, "output.mp4", fps=16)
        return "output.mp4"
    # Lazy prompt
    sample_prompts = [
        'Fujifilm Portra 400H film still, slammed Nissan Skyline R33 GTR LM JGTC, 7-11 Tokyo, Midnight',
        'a pig moving quickly in a beautiful winter scenery nature trees sunset tracking camera',
    ]
    sample_prompts = [[x] for x in sample_prompts]
    # UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🎥 VACE Video Generator")
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


import torch # optional (need torch, dequantor to work; pip install torch, dequantor)
from transformers import T5EncoderModel
from dequantor import LTXPipeline, LTXConditionPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo, GGUFQuantizationConfig
from dequantor.utils import export_to_video
import gradio as gr

def launch_ltxv_app(model_path, dtype):
    transformer = LTXVideoTransformer3DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/ltxv0.9.6-decoder",
        subfolder="transformer"
        )
    text_encoder = T5EncoderModel.from_pretrained(
        "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
        gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
        torch_dtype=dtype,
        )
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "callgg/ltxv0.9.6-decoder",
        subfolder="vae",
        torch_dtype=torch.float32
        )
    pipe = LTXPipeline.from_pretrained(
        "callgg/ltxv0.9.6-decoder",
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
        "A drone quickly rises through a bank of morning fog, revealing a pristine alpine lake surrounded by snow-capped mountains. The camera glides forward over the glassy water, capturing perfect reflections of the peaks. As it continues, the perspective shifts to reveal a lone wooden cabin with a curl of smoke from its chimney, nestled among tall pines at the lake's edge. The shot tracks upward rapidly, transitioning from intimate to epic as the full mountain range comes into view, bathed in the golden light of sunrise breaking through scattered clouds."
    ]
    sample_prompts = [[x] for x in sample_prompts]
    # UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🎥 LTXV Video Generator (2B)")
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

def launch_ltxv_13b_app(model_path, dtype):
    transformer = LTXVideoTransformer3DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config="callgg/ltxv0.9.8-decoder",
        subfolder="transformer"
        )
    text_encoder = T5EncoderModel.from_pretrained(
        "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
        gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
        torch_dtype=dtype,
        )
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "callgg/ltxv0.9.8-decoder",
        subfolder="vae",
        torch_dtype=dtype
        )
    pipe = LTXConditionPipeline.from_pretrained(
        "callgg/ltxv0.9.8-decoder",
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
        "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region.",
        "A drone quickly rises through a bank of morning fog, revealing a pristine alpine lake surrounded by snow-capped mountains. The camera glides forward over the glassy water, capturing perfect reflections of the peaks. As it continues, the perspective shifts to reveal a lone wooden cabin with a curl of smoke from its chimney, nestled among tall pines at the lake's edge. The shot tracks upward rapidly, transitioning from intimate to epic as the full mountain range comes into view, bathed in the golden light of sunrise breaking through scattered clouds."
    ]
    sample_prompts = [[x] for x in sample_prompts]
    # UI
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🎥 LTXV Video Generator (13B)")
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

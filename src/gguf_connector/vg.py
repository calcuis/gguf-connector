
import torch # optional (need torch, diffusers to work; pip install torch, diffusers)
import gradio as gr # optional (need gradio for lazy ui; pip install gradio)
from transformers import T5EncoderModel
from diffusers import LTXPipeline, GGUFQuantizationConfig, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video

model_path = "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-2b-0.9.6-distilled-fp32-q8_0.gguf"
transformer = LTXVideoTransformer3DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/ltxv-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    torch_dtype=torch.bfloat16,
)
pipe = LTXPipeline.from_pretrained(
    "callgg/ltxv0.9.6-decoder",
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

def generate_video(prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps):
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    ).frames[0]
    export_to_video(video, "output.mp4", fps=fps)
    return "output.mp4"

# Gradio UI

sample_prompts = [
    "A drone quickly rises through a bank of morning fog, revealing a pristine alpine lake surrounded by snow-capped mountains. The camera glides forward over the glassy water, capturing perfect reflections of the peaks. As it continues, the perspective shifts to reveal a lone wooden cabin with a curl of smoke from its chimney, nestled among tall pines at the lake edge. The final shot tracks upward rapidly, transitioning from intimate to epic as the full mountain range comes into view, bathed in the golden light of sunrise breaking through scattered clouds.",
]
sample_prompts = [[x] for x in sample_prompts]

block = gr.Blocks(title="gguf").queue()
with block:
    gr.Markdown("## 🎥 Video Generator")
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
        neg_prompt_input = gr.Textbox(label="Negative Prompt", value="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly", visible=False) # disable
        quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt_input])
        quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt_input, show_progress=False, queue=False)
    with gr.Row():
        width_input = gr.Number(label="Width", value=512)
        height_input = gr.Number(label="Height", value=512)
        num_frames_input = gr.Number(label="Number of Frames", value=25)
        num_steps_input = gr.Number(label="Inference Steps", value=25)
        fps_input = gr.Number(label="FPS", value=24)
    generate_btn = gr.Button("Generate Video")
    output_video = gr.Video(label="Generated Video")

    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt_input, neg_prompt_input,
            width_input, height_input,
            num_frames_input, num_steps_input,
            fps_input
        ],
        outputs=output_video,
    )

block.launch()

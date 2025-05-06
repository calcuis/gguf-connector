
import torch # optional (need torch, diffusers to work; pip install torch, diffusers)
import gradio as gr # optional (need gradio for lazy ui; pip install gradio)
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers import GGUFQuantizationConfig, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

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
pipe = LTXConditionPipeline.from_pretrained(
    "callgg/ltxv0.9.6-decoder",
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
).to("cuda")

def generate_video(input_image, prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps):
    image = input_image.convert("RGB")
    condition1 = LTXVideoCondition(image=image, frame_index=0)
    generator = torch.Generator("cuda").manual_seed(0)
    video = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]
    export_to_video(video, "output.mp4", fps=fps)
    return "output.mp4"

# Gradio UI

sample_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
sample_prompts = [[x] for x in sample_prompts]

block = gr.Blocks(title="gguf").queue()
with block:
    gr.Markdown("## 🎥 Video Generator 2")
    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image")
    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
    negative_prompt = gr.Textbox(label="Negative Prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted", visible=False) # disable
    quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
    quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
    with gr.Row():
        width = gr.Number(label="Width", value=512)
        height = gr.Number(label="Height", value=512)
        num_frames = gr.Number(label="Num Frames", value=25)
        num_inference_steps = gr.Number(label="Num Inference Steps", value=25)
        fps = gr.Number(label="FPS", value=24)
    generate_btn = gr.Button("Generate")
    output_video = gr.Video(label="Generated Video")

    generate_btn.click(
        fn=generate_video,
        inputs=[input_image, prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps],
        outputs=output_video,
    )

block.launch()

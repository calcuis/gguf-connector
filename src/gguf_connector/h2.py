
import torch # need torch to work; pip install torch
import torchaudio # need torchaudio as well
import gradio as gr # need gradio and higgs to work; pip install higgs
from higgs.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from higgs.data_types import ChatMLSample, Message

MODEL_PATH = "callgg/higgs-decoder"
AUDIO_TOKENIZER_PATH = "callgg/higgs-encoder"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# System prompt
system_prompt = (
    "Generate audio following instruction.\n\n"
    "<|scene_desc_start|>\n"
    "Audio is recorded from a quiet room.\n"
    "<|scene_desc_end|>"
)
# Load model
serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=DEVICE)

def generate_audio(user_prompt, max_new_tokens, temperature, top_p, top_k):
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    # Save audio to file
    audio_path = "output.wav"
    waveform = torch.from_numpy(output.audio).unsqueeze(0)
    torchaudio.save(audio_path, waveform, output.sampling_rate)
    return audio_path
sample_prompts = [
    "[SPEAKER0] Hey Connector, why your appearance looks so stupid? [SPEAKER1] Oh, really? maybe I ate too much smart beans. [SPEAKER0] Wow. Awesome! [SPEAKER1] Do you want some?",
    "안녕히 주무셨어요?",
    "你好",
    "Herzlichen Glückwunsch",
    "Que tenga usted buenos días",
]
sample_prompts = [[x] for x in sample_prompts]
# Gradio UI
with gr.Blocks(title="gguf") as demo:
    gr.Markdown("## 🎧 Higgs Audio Generation")
    with gr.Row():
        user_input = gr.Textbox(label="Prompt", lines=4, placeholder="Enter your prompt here (or click Sample Prompt)", value="")
        quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[user_input])
        quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=user_input, show_progress=False, queue=False)
    with gr.Row():
        max_tokens = gr.Slider(100, 2048, value=1024, step=1, label="Max New Tokens")
        temperature = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Temperature")
    with gr.Row():
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k")
    with gr.Row():
        generate_btn = gr.Button("Generate Audio")
    output_audio = gr.Audio(label="Generated Audio", type="filepath")
    generate_btn.click(
        fn=generate_audio,
        inputs=[user_input, max_tokens, temperature, top_p, top_k],
        outputs=output_audio
    )
demo.launch()

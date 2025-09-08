
import torch # need torch to work; pip install torch
import gradio as gr # need gradio and yvoice; pip install yvoice
from yvoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from yvoice.processor.vibevoice_processor import VibeVoiceProcessor

sample_script = """Speaker 1: Hey, why you folks always act together like a wolf pack?
Speaker 2: Oh, really? We just hang out for good food and share the bills.
Speaker 1: Wow. Amazing. A pig pack then!
Speaker 2: You must be the smartest joke maker in this universe."""

def launch_vibevoice_app():
    model_id = "callgg/vibevoice-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device
    )
    processor = VibeVoiceProcessor.from_pretrained(model_id)
    def generate_voice(script, voice_samples, cfg_pace):
        if not script.strip():
            return None
        if not voice_samples:
            return None
        voice_sample_paths = [sample for sample in voice_samples]
        inputs = processor(
            text=[script],
            voice_samples=[voice_sample_paths],
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in inputs.items()}
        output = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            # cfg_scale=1.3,
            cfg_scale=cfg_pace,
            max_new_tokens=None,
        )
        generated_speech = output.speech_outputs[0]
        processor_sampling_rate = processor.audio_processor.sampling_rate
        output_path = "output.wav"
        processor.save_audio(generated_speech, output_path, sampling_rate=processor_sampling_rate)
        return output_path
    block = gr.Blocks(title="gguf").queue()
    with block:
        gr.Markdown("## 🎙️ VibeVoice Generator\nDrag & drop your voice samples and input a script to generate speech.")
        with gr.Row():
            script_input = gr.Textbox(
                lines=6,
                placeholder="Enter your script here...",
                label="Input Script",
                value=sample_script
            )
            voice_input = gr.File(
                file_types=[".wav"],
                label="Upload Voice Samples",
                file_count="multiple"
            )
        generate_btn = gr.Button("Generate Speech 🎵")
        cfg_scale = gr.Slider(0.0, 5, step=.1, label="CFG Scale/Pace", value=1.3)
        output_audio = gr.Audio(label="Generated Output", type="filepath")
        generate_btn.click(
            generate_voice,
            inputs=[script_input, voice_input, cfg_scale],
            outputs=output_audio,
        )
    block.launch()

from pathlib import Path
def get_hf_cache_hub_path():
    home_dir = Path.home()
    hf_cache_path = home_dir / ".cache" / "huggingface" / "hub" / "models--callgg--vibevoice-bf16" / "blobs" / "53a915ae1a937cde20531290877f23aee39a7cc21786ff3a783158ac443ae74d"
    return str(hf_cache_path)

import os
from .quant3 import convert_gguf_to_safetensors
from .quant4 import add_metadata_to_safetensors
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if gguf_files:
    print("\nGGUF file(s) available. Select which one to use:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice2 = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice2)-1
        selected_model_file=gguf_files[choice_index]
        print(f"Model file: {selected_model_file} is selected!")
        selected_file_path=selected_model_file
        model_path = get_hf_cache_hub_path()
        if DEVICE == "cuda":
            use_bf16 = True
        else:
            use_bf16 = False
        print(f"Prepare to dequantize: {selected_file_path}")
        convert_gguf_to_safetensors(selected_file_path, model_path, use_bf16)
        add_metadata_to_safetensors(model_path, {'format': 'pt'})
        launch_vibevoice_app()
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

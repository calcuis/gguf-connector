
filename = "o2.py"
content = """
import torch
from loguru import logger
from argparse import ArgumentParser
from pathlib import Path
from fishaudio.fish_speech.inference_engine import TTSInferenceEngine
from fishaudio.fish_speech.models.dac.inference import load_model as load_decoder_model
from fishaudio.fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fishaudio.fish_speech.utils.schema import ServeTTSRequest
from fishaudio.webui import build_app
from fishaudio.webui.inference import get_inference_wrapper
import os
if not os.path.isfile(os.path.join(os.path.dirname(__file__), "models/fish/config.json")):
    from huggingface_hub import snapshot_download
    save_dir = os.path.join(os.path.dirname(__file__), "models/fish")
    repo_id = "callgg/fish-decoder"
    cache_dir = save_dir + "/cache"
    snapshot_download(
        cache_dir=cache_dir,
        local_dir=save_dir,
        repo_id=repo_id,
        allow_patterns=["*.json", "*.tiktoken", "*.pth"],
        )
from gguf_connector.quant4 import convert_safetensors_to_pth
from gguf_connector.quant3 import convert_gguf_to_safetensors
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files:
    print("GGUF file(s) available. Select which one for codec:")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice)-1
        selected_file=gguf_files[choice_index]
        print(f"codec file: {selected_file} is selected!")
        input_path=selected_file
        use_bf16 = True
        out_path = f"{os.path.splitext(input_path)[0]}_bf16.safetensors"
        convert_gguf_to_safetensors(input_path, out_path, use_bf16)
        convert_safetensors_to_pth(out_path)
        output_path = os.path.splitext(out_path)[0] + ".pth"
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF/Safetensors are available in the current directory.")
    input("--- Press ENTER To Exit ---")
def parse_args(output_path):
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default=os.path.join(os.path.dirname(__file__), "models/fish"),
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default=output_path,
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="dark")
    return parser.parse_args()
args = parse_args(output_path)
args.precision = torch.half if args.half else torch.bfloat16
if torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
logger.info("Loading Llama model...")
llama_queue = launch_thread_safe_queue(
    checkpoint_path=args.llama_checkpoint_path,
    device=args.device,
    precision=args.precision,
    compile=args.compile,
)
logger.info("Loading VQ-GAN model...")
decoder_model = load_decoder_model(
    config_name=args.decoder_config_name,
    checkpoint_path=args.decoder_checkpoint_path,
    device=args.device,
)
logger.info("Decoder model loaded, warming up...")
# Create the inference engine
inference_engine = TTSInferenceEngine(
    llama_queue=llama_queue,
    decoder_model=decoder_model,
    compile=args.compile,
    precision=args.precision,
)
# Dry run to check if the model is loaded correctly and avoid the first-time latency
list(
    inference_engine.inference(
        ServeTTSRequest(
            text="Hello world.",
            references=[],
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.5,
            temperature=0.7,
            format="wav",
        )
    )
)
logger.info("Warming up done, launching the web UI...")
# Get the inference function with the immutable arguments
inference_fct = get_inference_wrapper(inference_engine)
app = build_app(inference_fct, args.theme)
app.launch(show_api=True)
"""

with open(filename, "w") as f:
    f.write(content)

import os
os.system(f"python {filename}")

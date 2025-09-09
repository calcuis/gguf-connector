
import torch # need torch to work; pip install torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import ImageGrab
import gradio as gr
import threading, time

MODEL_ID = "callgg/fastvlm-0.5b-bf16"
IMAGE_TOKEN_INDEX = -200  # placeholder token
# Load model + tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
# Global control flags
capturing = False
latest_caption = ""
def generate_caption(img: ImageGrab.Image) -> str:
    """Generate a caption for a given PIL image."""
    # Construct chat message
    messages = [{"role": "user", "content": "<image>\nDescribe this image in detail."}]
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    px = model.get_vision_tower().image_processor(
        images=img, return_tensors="pt"
    )["pixel_values"].to(model.device, dtype=model.dtype)
    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=px,
            max_new_tokens=64,
        )
    return tok.decode(out[0], skip_special_tokens=True)
def capture_loop():
    """Continuously grab screen and update caption."""
    global latest_caption, capturing
    while capturing:
        screen = ImageGrab.grab()
        latest_caption = generate_caption(screen)
        time.sleep(1)
def start_caption():
    """Start live captioning."""
    global capturing
    if not capturing:
        capturing = True
        threading.Thread(target=capture_loop, daemon=True).start()
    return "Live Caption started..."
def stop_caption():
    """Stop live captioning."""
    global capturing
    capturing = False
    return "Live Caption stopped."
def get_caption():
    """Fetch the latest caption."""
    # return latest_caption if latest_caption else "Waiting for first caption..."
    return latest_caption if latest_caption else ""
# Gradio UI
with gr.Blocks(title="Live Caption") as demo:
    gr.Markdown("## 🎥 Live Caption (Screen Describer)")
    with gr.Row():
        start_btn = gr.Button("▶ Start Captioning", variant="primary")
        stop_btn = gr.Button("⏹ Stop Captioning", variant="stop")
    output_box = gr.Textbox(label="Live Description", lines=5)
    # Buttons
    start_btn.click(start_caption, outputs=output_box)
    stop_btn.click(stop_caption, outputs=output_box)
    # Refresher
    timer = gr.Timer(1)
    timer.tick(get_caption, outputs=output_box)
demo.queue().launch()

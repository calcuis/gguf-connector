
import torch # need torch to work; pip install torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import gradio as gr
MODEL_ID = "callgg/fastvlm-0.5b-bf16"
IMAGE_TOKEN_INDEX = -200  # placeholder token
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
def describe_image(img: Image.Image) -> str:
    if img is None:
        return "Please upload an image."
    # Construct chat message
    messages = [{"role": "user", "content": "<image>\nDescribe this image in detail."}]
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)
    # Tokenize parts around the image token
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    # Insert image token
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    # Process image using vision tower
    px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=model.dtype)
    # Generate description
    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=px,
            max_new_tokens=128,
        )
    return tok.decode(out[0], skip_special_tokens=True)
# Gradio UI
block = gr.Blocks(title="gguf").queue()
with block:
    gr.Markdown("## 🖼️ Image Descriptor (f5)")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Input Image")
            btn = gr.Button("Describe Image", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Description", lines=5)
    btn.click(fn=describe_image, inputs=img_input, outputs=output)
block.launch()

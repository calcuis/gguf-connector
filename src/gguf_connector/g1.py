
import torch # optional (need torch to work; pip install torch)
import gradio as gr  # optional (need gradio for lazy ui; pip install gradio)
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

model_path = "callgg/solidity-decoder"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1400,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Gradio UI
interface = gr.Interface(
    fn=generate_code,
    inputs=gr.Textbox(lines=4, label="Enter Prompt", value="write a Solidity function to transfer tokens"),
    outputs=gr.Textbox(lines=20, label="Generated Solidity Code"),
    title="Smart Contract Generator",
    description="Enter a prompt to generate smart contract"
)
# Launch the app
interface.launch()

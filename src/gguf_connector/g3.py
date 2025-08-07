
from transformers import pipeline
import gradio as gr

model_id = "callgg/gpt-20b-8bit"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)
# Inference function
def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages, max_new_tokens=256)
    return outputs[0]["generated_text"][-1] if outputs else "No response generated."
# Gradio interface
with gr.Blocks(title="gguf") as demo:
    gr.Markdown("## 🚀 GPT-OSS-20B")
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Enter your prompt",
            placeholder="e.g., Explain quantum mechanics clearly and concisely.",
            lines=4
        )
    submit_button = gr.Button("Submit")
    output_text = gr.Textbox(label="Generated Response", lines=10)
    submit_button.click(fn=generate_response, inputs=prompt_input, outputs=output_text)
# Launch the app
demo.launch()

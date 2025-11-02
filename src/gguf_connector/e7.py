
filename = "backend.py"
content = """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io, os
from gguf_connector.quant3 import convert_gguf_to_safetensors
from gguf_connector.quant4 import add_metadata_to_safetensors
from gguf_connector.f4 import get_hf_cache_hub_path
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files:
    print('GGUF file(s) available. Select which one to use:')
    for index, file_name in enumerate(gguf_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(gguf_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = gguf_files[choice_index]
        print(f'Model file: {selected_file} is selected!')
        input_path = selected_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        selected = "0.5b"
        model_path = get_hf_cache_hub_path(selected)
        if device == "cuda":
            print(f"Running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            use_bf16 = True
        else:
            use_bf16 = False
        print(f"Prepare to dequantize: {input_path}")
        convert_gguf_to_safetensors(input_path, model_path, use_bf16)
        add_metadata_to_safetensors(model_path, {'format': 'pt'})
        MODEL_ID = "callgg/fastvlm-0.5b-bf16"
        IMAGE_TOKEN_INDEX = -200
        print("Loading model...")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded successfully.")
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        @app.post("/api/describe")
        async def describe_image_api(
            image: UploadFile = File(...),
            prompt: str = Form(...),
            num_tokens: int = Form(128),
        ):
            try:
                image_bytes = await image.read()
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = describe_image(img, prompt, num_tokens)
                return JSONResponse({"description": result})
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
        def describe_image(img: Image.Image, prompt, num_tokens) -> str:
            if img is None:
                return "Please upload an image."
            messages = [{"role": "user", "content": f"<image>\n{prompt}."}]
            rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            pre, post = rendered.split("<image>", 1)
            pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
            post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
            img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
            attention_mask = torch.ones_like(input_ids, device=model.device)
            px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
            px = px.to(model.device, dtype=model.dtype)
            with torch.no_grad():
                out = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    max_new_tokens=num_tokens,
                )
            return tok.decode(out[0], skip_special_tokens=True)
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')
"""

with open(filename, "w") as f:
    f.write(content)

import os
os.system("uvicorn backend:app --reload --port 8000")

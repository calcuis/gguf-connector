
filename = "backend.py"
content = """
import torch, base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from transformers import T5EncoderModel
from dequantor import FluxKontextPipeline, GGUFQuantizationConfig, FluxTransformer2DModel
import os
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
        if device == "cuda":
            print(f"running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        transformer = FluxTransformer2DModel.from_single_file(
            input_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
            config="callgg/kontext-decoder",
            subfolder="transformer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
            gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
            torch_dtype=dtype
        )
        pipe = FluxKontextPipeline.from_pretrained(
            "callgg/kontext-decoder",
            transformer=transformer,
            text_encoder_2=text_encoder,
            torch_dtype=dtype
        )
        pipe.enable_model_cpu_offload()
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        @app.post("/generate")
        async def generate_image(
            image: UploadFile,
            prompt: str = Form(...),
            num_steps: int = Form(...),
            guidance_scale: float = Form(...)
        ):
            pil_img = Image.open(BytesIO(await image.read())).convert("RGB")
            result = pipe(
                image=pil_img,
                prompt=prompt,
                num_inference_steps=num_steps,
                true_cfg_scale=guidance_scale
            ).images[0]
            buffer = BytesIO()
            result.save(buffer, format="PNG")
            buffer.seek(0)
            base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return JSONResponse(content={
                "image": base64_img,
                "format": "png",
                "message": "success"
            })
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

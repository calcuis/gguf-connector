
filename = "backend9.py"
content = """
import torch, base64
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from dequantor import (
    QwenImageEditPlusPipeline,
    AutoencoderKLQwenImage,
)
from transformers import Qwen2_5_VLForConditionalGeneration
from gguf_connector.vrm import get_gpu_vram, get_affordable_precision
from nunchaku import NunchakuQwenImageTransformer2DModel
import os
safetensors_files = [file for file in os.listdir() if file.endswith('.safetensors')]
if safetensors_files:
    print('Safetensors available. Select which one to use:')
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(safetensors_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = safetensors_files[choice_index]
        print(f'Model file: {selected_file} is selected!')
        input_path = selected_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Device detected: {device}")
        print(f"torch version: {torch.__version__}")
        print(f"dtype using: {dtype}")
        if device == "cuda":
            print(f"running with: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            prec = get_affordable_precision() # provide affordable precision hints
            print(f"affordable precision: {prec} (opt a wrong precision file will return error)")
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(input_path)
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "callgg/qi-decoder",
            subfolder="text_encoder",
            torch_dtype=dtype
        )
        vae = AutoencoderKLQwenImage.from_pretrained(
            "callgg/qi-decoder",
            subfolder="vae",
            torch_dtype=dtype
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "callgg/image-edit-plus",
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
            torch_dtype=dtype
        )
        if get_gpu_vram() > 18:
            pipe.enable_model_cpu_offload()
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        @app.post("/generate")
        async def generate_image_api(
            prompt: str = Form(...),
            steps: int = Form(...),
            guidance: float = Form(...),
            img1: UploadFile | None = File(None),
            img2: UploadFile | None = File(None),
            img3: UploadFile | None = File(None),
        ):
            images = []
            for img_file in [img1, img2, img3]:
                if img_file is not None:
                    img = Image.open(BytesIO(await img_file.read())).convert("RGB")
                    images.append(img)
            if not images:
                return JSONResponse({"error": "No images uploaded"}, status_code=400)
            with torch.inference_mode():
                output = pipe(
                    image=images,
                    prompt=prompt,
                    true_cfg_scale=guidance,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    num_images_per_prompt=1
                )
            buffer = BytesIO()
            output.images[0].save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return JSONResponse({"image": img_str})
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print('No safetensors are available in the current directory.')
    input('--- Press ENTER To Exit ---')
"""

with open(filename, "w") as f:
    f.write(content)

import os
os.system("uvicorn backend9:app --reload --port 8009")

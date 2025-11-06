
filename = "backend5.py"
content = """
import torch
import io, base64
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import T5EncoderModel
from dequantor import (
    LTXConditionPipeline,
    LTXVideoTransformer3DModel,
    AutoencoderKLLTXVideo,
    GGUFQuantizationConfig,
)
from dequantor.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from dequantor.utils import export_to_video
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
        transformer = LTXVideoTransformer3DModel.from_single_file(
            input_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
            config="callgg/ltxv0.9.6-decoder",
            subfolder="transformer",
        )
        text_encoder = T5EncoderModel.from_pretrained(
            "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
            gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
            torch_dtype=dtype,
        )
        vae = AutoencoderKLLTXVideo.from_pretrained(
            "callgg/ltxv0.9.6-decoder",
            subfolder="vae",
            torch_dtype=dtype,
        )
        pipe = LTXConditionPipeline.from_pretrained(
            "callgg/ltxv0.9.6-decoder",
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
            torch_dtype=dtype,
        )
        pipe.enable_model_cpu_offload()
        def generate_video(
            image: Image.Image,
            prompt: str,
            negative_prompt: str,
            width: int,
            height: int,
            num_frames: int,
            num_inference_steps: int,
            fps: int,
        ):
            condition1 = LTXVideoCondition(image=image, frame_index=0)
            video = pipe(
                conditions=[condition1],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
            ).frames[0]
            output_path = "output.mp4"
            export_to_video(video, output_path, fps=fps)
            return output_path
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        @app.post("/generate_video")
        async def generate_video_api(
            file: UploadFile = File(...),
            prompt: str = Form(...),
            negative_prompt: str = Form(" "),
            width: int = Form(480),
            height: int = Form(480),
            num_frames: int = Form(25),
            num_inference_steps: int = Form(15),
            fps: int = Form(24),
        ):
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            output_path = generate_video(
                image,
                prompt,
                negative_prompt,
                width,
                height,
                num_frames,
                num_inference_steps,
                fps,
            )
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
            return JSONResponse(
                content={"status": "success", "video_base64": video_b64, "mime": "video/mp4"}
            )
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')
"""

with open(filename, "w") as f:
    f.write(content)

import os
os.system("uvicorn backend5:app --reload --port 8005")

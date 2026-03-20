
filename = "backend.py"
content = """
import os
import shutil
import uuid
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import json
from multipart.multipart import parse_options_header
import multipart
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MAX_FILES = 1000000
class MultipartConfig:
    max_files = MAX_FILES
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
active_trainings = {}
class TrainingConfig(BaseModel):
    session_id: str
    codings_size: int = 100
    image_size: int = 24
    image_channels: int = 4
    batch_size: int = 16
    epochs: int = 50
@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    csv_path = os.path.join(session_dir, "attributes.csv")
    with open(csv_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"session_id": session_id, "filename": file.filename}
@app.post("/api/upload/images-batch/{session_id}")
async def upload_images_batch(session_id: str, files: List[UploadFile] = File(...)):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    images_dir = os.path.join(session_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    uploaded_files = []
    for file in files:
        if file.filename:
            file_path = os.path.join(images_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            uploaded_files.append(file.filename)
    return {"uploaded_count": len(uploaded_files), "batch_size": len(files)}
@app.post("/api/upload/images-zip/{session_id}")
async def upload_images_zip(session_id: str, file: UploadFile = File(...)):
    import zipfile
    import io
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    images_dir = os.path.join(session_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    content = await file.read()
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zip_file:
            file_list = zip_file.namelist()
            image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) and not f.startswith('__MACOSX')]
            for i, filename in enumerate(image_files):
                base_filename = os.path.basename(filename)
                if base_filename:
                    file_path = os.path.join(images_dir, base_filename)
                    with open(file_path, "wb") as f:
                        f.write(zip_file.read(filename))
                if (i + 1) % 1000 == 0:
                    print(f"Extracted {i + 1}/{len(image_files)} files...")
            return {"uploaded_count": len(image_files), "message": f"Successfully extracted {len(image_files)} images from ZIP"}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
@app.get("/api/generated-image/{session_id}/{filename}")
async def get_generated_image(session_id: str, filename: str):
    image_path = os.path.join(UPLOAD_DIR, session_id, "gen_images", filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/png")
@app.get("/api/download/model/{session_id}")
async def download_model(session_id: str):
    model_path = os.path.join(UPLOAD_DIR, session_id, "models", "generator_model.safetensors")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename="generator_model.safetensors"
    )
@app.websocket("/ws/train/{session_id}")
async def train_model(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        config_data = await websocket.receive_text()
        config = TrainingConfig(**json.loads(config_data))
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        if not os.path.exists(session_dir):
            await websocket.send_json({"type": "error", "message": "Session not found"})
            return
        active_trainings[session_id] = True
        await websocket.send_json({"type": "status", "message": "Initializing training..."})
        from gguf_build.trainer import train_gan
        async for progress in train_gan(
            session_dir=session_dir,
            codings_size=config.codings_size,
            image_size=config.image_size,
            image_channels=config.image_channels,
            batch_size=config.batch_size,
            epochs=config.epochs
        ):
            if session_id not in active_trainings:
                await websocket.send_json({"type": "status", "message": "Training cancelled"})
                break
            await websocket.send_json(progress)
        await websocket.send_json({"type": "complete", "message": "Training completed!"})
    except WebSocketDisconnect:
        if session_id in active_trainings:
            del active_trainings[session_id]
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if session_id in active_trainings:
            del active_trainings[session_id]
@app.delete("/api/session/{session_id}")
async def cleanup_session(session_id: str):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    return {"message": "Session cleaned up"}
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
"""

with open(filename, "w") as f:
    f.write(content)

import os
os.system("uvicorn backend:app --reload --port 8000")

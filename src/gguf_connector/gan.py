
filename = "backend.py"
content = """
import os
import shutil
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
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
    session_name: Optional[str] = None
    codings_size: int = 100
    image_size: int = 24
    image_channels: int = 4
    batch_size: int = 16
    epochs: int = 50
class SessionNameUpdate(BaseModel):
    session_name: str
def save_session_metadata(session_dir: str, metadata: Dict[str, Any]) -> None:
    metadata_path = os.path.join(session_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
def load_session_metadata(session_dir: str) -> Optional[Dict[str, Any]]:
    metadata_path = os.path.join(session_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)
def list_all_sessions() -> List[Dict[str, Any]]:
    sessions = []
    if not os.path.exists(UPLOAD_DIR):
        return sessions
    for session_id in os.listdir(UPLOAD_DIR):
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        if os.path.isdir(session_dir):
            metadata = load_session_metadata(session_dir)
            if metadata:
                sessions.append(metadata)
            else:
                sessions.append({
                    "session_id": session_id,
                    "session_name": None,
                    "status": "unknown",
                    "created_at": None
                })
    sessions.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return sessions
@app.get("/api/sessions")
async def get_sessions():
    sessions = list_all_sessions()
    return {"sessions": sessions}
@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    metadata = load_session_metadata(session_dir)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session metadata not found")
    return metadata
@app.put("/api/session/{session_id}/name")
async def update_session_name(session_id: str, update: SessionNameUpdate):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    metadata = load_session_metadata(session_dir)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session metadata not found")
    metadata["session_name"] = update.session_name
    save_session_metadata(session_dir, metadata)
    return {"message": "Session name updated", "session_name": update.session_name}
@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    csv_path = os.path.join(session_dir, "attributes.csv")
    with open(csv_path, "wb") as f:
        content = await file.read()
        f.write(content)
    metadata = {
        "session_id": session_id,
        "session_name": None,
        "status": "created",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
        "config": None,
        "results": None,
        "artifacts": {
            "csv_path": "attributes.csv"
        }
    }
    save_session_metadata(session_dir, metadata)
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
    start_time = datetime.utcnow()
    loss_history = []
    generated_images = []
    try:
        config_data = await websocket.receive_text()
        config = TrainingConfig(**json.loads(config_data))
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        if not os.path.exists(session_dir):
            await websocket.send_json({"type": "error", "message": "Session not found"})
            return
        metadata = load_session_metadata(session_dir)
        if not metadata:
            metadata = {
                "session_id": session_id,
                "session_name": config.session_name,
                "status": "in_progress",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "started_at": None,
                "completed_at": None,
                "config": None,
                "results": None,
                "artifacts": {}
            }
        metadata["status"] = "in_progress"
        metadata["started_at"] = start_time.isoformat() + "Z"
        metadata["session_name"] = config.session_name
        metadata["config"] = {
            "codings_size": config.codings_size,
            "image_size": config.image_size,
            "image_channels": config.image_channels,
            "batch_size": config.batch_size,
            "epochs": config.epochs
        }
        save_session_metadata(session_dir, metadata)
        active_trainings[session_id] = True
        await websocket.send_json({"type": "status", "message": "Initializing training..."})
        from gguf_gan.trainer import train_gan
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
            if progress.get("type") == "progress":
                loss_history.append({
                    "epoch": progress["epoch"],
                    "gen_loss": progress["gen_loss"],
                    "disc_loss": progress["disc_loss"]
                })
                if progress.get("image_url"):
                    image_filename = progress["image_url"].split("/")[-1]
                    generated_images.append(image_filename)
            await websocket.send_json(progress)
        end_time = datetime.utcnow()
        training_time = (end_time - start_time).total_seconds()
        metadata["status"] = "completed"
        metadata["completed_at"] = end_time.isoformat() + "Z"
        metadata["results"] = {
            "final_gen_loss": loss_history[-1]["gen_loss"] if loss_history else None,
            "final_disc_loss": loss_history[-1]["disc_loss"] if loss_history else None,
            "epochs_completed": len(loss_history),
            "total_training_time_seconds": training_time,
            "loss_history": loss_history
        }
        metadata["artifacts"] = {
            "model_path": "models/generator_model.safetensors",
            "generated_images_dir": "gen_images/",
            "csv_path": "attributes.csv",
            "images_dir": "images/",
            "generated_images": generated_images
        }
        save_session_metadata(session_dir, metadata)
        await websocket.send_json({"type": "complete", "message": "Training completed!"})
    except WebSocketDisconnect:
        if session_id in active_trainings:
            del active_trainings[session_id]
        metadata = load_session_metadata(session_dir)
        if metadata:
            metadata["status"] = "failed"
            metadata["error"] = "Connection disconnected"
            save_session_metadata(session_dir, metadata)
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        metadata = load_session_metadata(session_dir)
        if metadata:
            metadata["status"] = "failed"
            metadata["error"] = str(e)
            save_session_metadata(session_dir, metadata)
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

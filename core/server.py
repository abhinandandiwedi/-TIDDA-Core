import cv2
import base64
import time
import torch
from perception import PerceptionEngine
import asyncio
import os
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.message_schema import StartReconstructionRequest, JobResponse
from core.world_model import WorldModel
from ingest.video_ingest import ingest_video
from ingest.frame_sampler import sample_frames
from reconstruction.colmap_runner import run_colmap_pipeline

app = FastAPI(title="TIDDA Core API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static HTML frontend
app.mount("/static", StaticFiles(directory="html"), name="static")
# Serve the output models (like the PLY file)
output_dir = Path("models_served")
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/models", StaticFiles(directory=str(output_dir)), name="models")

world_model = WorldModel()

# ── Live Streaming & WebSocket Manager ──
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                self.active_connections.remove(connection)

ws_manager = ConnectionManager()
perception_engine = PerceptionEngine(model_name="yolov8n.pt", confidence_threshold=0.45)
_is_streaming = False

@app.websocket("/ws/ui")
async def websocket_ui(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

async def live_stream_task():
    global _is_streaming
    _is_streaming = True
    
    # 1. Hardware Verification
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available! YOLO cannot run on GPU. Stopping live stream.")
        _is_streaming = False
        return
        
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[SYSTEM] Starting Live Stream. YOLO Device: {gpu_name} (ROCm/CUDA)")
    
    # Initialize model explicitly
    perception_engine.initialize()
    
    video_path = "test_dataset/house_v1_demo.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video {video_path} not found.")
        _is_streaming = False
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps
    
    print(f"[SYSTEM] Video loaded: {video_path} @ {fps} FPS")

    frame_count = 0
    start_time = time.time()
    
    while _is_streaming and cap.isOpened():
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[SYSTEM] Video stream ended.")
            break
            
        frame_count += 1
        
        # Decode/process FPS calculation
        decode_time = time.time() - loop_start
        
        # YOLO inference
        inf_start = time.time()
        observations = perception_engine.process(frame, node_id="SIM_01", floor_id="F1")
        inference_latency = time.time() - inf_start
        
        # Format results
        bboxes = []
        classes = []
        confidences = []
        for obs in observations:
            if obs.bounding_box:
                bboxes.append(obs.bounding_box)
                classes.append(obs.class_name)
                confidences.append(obs.confidence)
                
        # Resize for preview to avoid flooding browser
        preview_h = 480
        aspect = frame.shape[1] / frame.shape[0]
        preview_w = int(preview_h * aspect)
        preview_frame = cv2.resize(frame, (preview_w, preview_h))
        
        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64_frame = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate End-to-end FPS
        end_to_end_time = time.time() - loop_start
        actual_fps = 1.0 / end_to_end_time if end_to_end_time > 0 else 0
        
        # Print metrics every 30 frames
        if frame_count % 30 == 0:
            print(f"[METRICS] Decode: {decode_time*1000:.1f}ms | YOLO: {inference_latency*1000:.1f}ms | End-to-end FPS: {actual_fps:.1f}")
            
        message = {
            "type": "live_frame",
            "frame": b64_frame,
            "bboxes": bboxes,
            "classes": classes,
            "confidences": confidences,
            "image_width": frame.shape[1],
            "image_height": frame.shape[0],
            "node_id": "SIM_01"
        }
        
        await ws_manager.broadcast(message)
        
        # Yield and throttle
        elapsed = time.time() - loop_start
        sleep_time = max(0, frame_delay - elapsed)
        await asyncio.sleep(sleep_time if sleep_time > 0 else 0.01)
        
    cap.release()
    _is_streaming = False

@app.post("/api/start_stream")
async def start_stream(background_tasks: BackgroundTasks):
    global _is_streaming
    if _is_streaming:
        return {"status": "Already streaming"}
    background_tasks.add_task(live_stream_task)
    return {"status": "Stream started"}



# In-memory job state
jobs = {}

def process_video_task(job_id: str, video_path: str):
    try:
        jobs[job_id] = "Extracting frames..."
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video {video_path} not found")
        
        # 1. Ingest and sample frames
        frames_dir = sample_frames(video_file)
        
        jobs[job_id] = "Running COLMAP sparse reconstruction..."
        # 2. Run COLMAP Sparse
        sparse_ply = run_colmap_pipeline(frames_dir)
        
        jobs[job_id] = "Running COLMAP dense reconstruction..."
        # 3. Run COLMAP Dense
        from reconstruction.dense_reconstruction import run_dense_reconstruction
        run_dense_reconstruction(frames_dir.parent)
        
        # Copy to central models directory
        import shutil
        fused_ply = frames_dir.parent / "dense" / "fused.ply"
        served_ply = output_dir / "fused.ply"
        shutil.copy(str(fused_ply), str(served_ply))
        
        # 4. Update world model
        world_model.update_point_cloud(f"/models/{fused_ply.name}")
        jobs[job_id] = f"Completed. Point cloud at /models/{fused_ply.name}"
    except Exception as e:
        jobs[job_id] = f"Error: {str(e)}"
        print(f"Job {job_id} failed: {e}")

@app.post("/api/reconstruct", response_model=JobResponse)
async def start_reconstruction(req: StartReconstructionRequest, background_tasks: BackgroundTasks):
    job_id = "job_1"  # Simple single job for now
    jobs[job_id] = "Starting..."
    background_tasks.add_task(process_video_task, job_id, req.video_path)
    return JobResponse(job_id=job_id, status="Started", message="Reconstruction job submitted")

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": jobs[job_id]}

if __name__ == "__main__":
    uvicorn.run("core.server:app", host="0.0.0.0", port=8001, reload=True)

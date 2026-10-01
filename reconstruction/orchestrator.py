import os
import sys
import time
import uuid
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from ingest.frame_sampler import sample_frames
from reconstruction.colmap_runner import run_colmap_pipeline
from reconstruction.dense_reconstruction import run_dense_reconstruction

logger = logging.getLogger("reconstruction.orchestrator")

# Directory setup relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure COLMAP and ROCm environment variables
DEFAULT_COLMAP = REPO_ROOT / "colmap_hip_build" / "build" / "src" / "colmap" / "exe" / "colmap"
if "COLMAP_EXEC" not in os.environ or not Path(os.environ["COLMAP_EXEC"]).exists():
    os.environ["COLMAP_EXEC"] = str(DEFAULT_COLMAP)

ROCM_LIB = "/opt/rocm/core-10.0/lib"
cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
if ROCM_LIB not in cur_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{ROCM_LIB}:{cur_ld}" if cur_ld else ROCM_LIB

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class ReconstructionOrchestrator:
    """Manages asynchronous reconstruction jobs and model serving updates."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or REPO_ROOT
        self.models_served_dir = self.repo_root / "models_served"
        self.models_served_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def validate_video_path(self, video_path_str: str) -> Path:
        """Validate that video path exists, is a file, and has a supported extension."""
        if not video_path_str or not video_path_str.strip():
            raise ValueError("video_path cannot be empty")

        raw_path = Path(video_path_str.strip())
        if raw_path.is_file():
            resolved = raw_path.resolve()
        elif (self.repo_root / raw_path).is_file():
            resolved = (self.repo_root / raw_path).resolve()
        else:
            raise FileNotFoundError(f"Video file not found: {video_path_str}")

        if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{resolved.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if resolved.stat().st_size == 0:
            raise ValueError(f"Video file is empty (0 bytes): {video_path_str}")

        return resolved

    def start_job(
        self,
        video_path_str: str,
        fps: float = 2.0,
        feature_method: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Validate input and register a new job in QUEUED status."""
        resolved_path = self.validate_video_path(video_path_str)
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        method = (feature_method or os.environ.get("TIDDA_FEATURE_METHOD", "sift")).lower().strip()

        with self._lock:
            job_record = {
                "job_id": job_id,
                "status": "QUEUED",
                "message": "Reconstruction job queued",
                "video_path": str(resolved_path),
                "fps": fps,
                "feature_method": method,
                "created_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "error": None,
                "output_model": None,
            }
            self.jobs[job_id] = job_record

        return job_id, job_record

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job record by ID."""
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all tracked jobs."""
        with self._lock:
            return [dict(j) for j in self.jobs.values()]

    def execute_job(self, job_id: str) -> None:
        """
        Execute the end-to-end reconstruction pipeline in a worker thread.
        Never throws; records errors into the job state so FastAPI remains healthy.
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                logger.error(f"Cannot execute unknown job: {job_id}")
                return
            job["status"] = "RUNNING"
            job["started_at"] = time.time()
            job["message"] = "Extracting frames..."

        video_path = Path(job["video_path"])
        fps = float(job.get("fps", 2.0))

        try:
            colmap_exec = os.environ.get("COLMAP_EXEC", str(DEFAULT_COLMAP))
            if not Path(colmap_exec).exists():
                raise FileNotFoundError(f"COLMAP binary not found at: {colmap_exec}")

            # 1. Ingest and sample frames (FFmpeg @ 2 FPS)
            logger.info(f"[{job_id}] Sampling frames at {fps} FPS from {video_path}...")
            frames_dir = sample_frames(video_path, fps=fps)
            workspace = frames_dir.parent

            # 2. Sparse SfM Reconstruction (SIFT or ALIKED+LightGlue)
            feature_method = self.jobs[job_id].get("feature_method", "sift")
            with self._lock:
                self.jobs[job_id]["message"] = f"Running COLMAP sparse reconstruction ({feature_method.upper()})..."
            logger.info(f"[{job_id}] Running COLMAP sparse reconstruction ({feature_method}) in {workspace}...")
            sparse_ply_path = run_colmap_pipeline(frames_dir, feature_method=feature_method)

            # 3. Dense Reconstruction (Undistort + HIP PatchMatch on AMD GPU + Stereo Fusion)
            with self._lock:
                self.jobs[job_id]["message"] = "Running COLMAP dense reconstruction (HIP PatchMatch on AMD GPU)..."
            logger.info(f"[{job_id}] Running COLMAP dense reconstruction...")
            run_dense_reconstruction(workspace)

            fused_ply = workspace / "dense" / "fused.ply"
            if not fused_ply.exists() or fused_ply.stat().st_size == 0:
                raise RuntimeError(f"Dense fused.ply was not generated at {fused_ply}")

            # 4. Atomic copy to models_served/fused.ply (Rule 11)
            target_ply = self.models_served_dir / "fused.ply"
            tmp_target = self.models_served_dir / "fused.ply.tmp"
            shutil.copy2(str(fused_ply), str(tmp_target))
            os.replace(str(tmp_target), str(target_ply))

            # Also maintain a copy in workspace/output/fused.ply
            output_dir = workspace / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(fused_ply), str(output_dir / "fused.ply"))

            # 5. Success
            with self._lock:
                self.jobs[job_id]["status"] = "COMPLETED"
                self.jobs[job_id]["message"] = "Completed. Point cloud served at /models/fused.ply"
                self.jobs[job_id]["output_model"] = "/models/fused.ply"
                self.jobs[job_id]["completed_at"] = time.time()

            logger.info(f"[{job_id}] Reconstruction completed successfully! Model served at /models/fused.ply")

        except Exception as exc:
            logger.error(f"[{job_id}] Reconstruction failed: {exc}", exc_info=True)
            with self._lock:
                self.jobs[job_id]["status"] = "FAILED"
                self.jobs[job_id]["message"] = f"Failed: {exc}"
                self.jobs[job_id]["error"] = str(exc)
                self.jobs[job_id]["completed_at"] = time.time()


# Global singleton instance
reconstruction_orchestrator = ReconstructionOrchestrator()

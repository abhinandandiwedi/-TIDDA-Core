import os
from pathlib import Path
import subprocess
import shutil
from typing import Optional

from reconstruction.features import extract_features
from reconstruction.matching import match_features
from reconstruction.pointcloud import convert_to_ply

# Set library path for ROCm dependencies
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.0/lib"
else:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/rocm/core-10.0/lib:{os.environ['LD_LIBRARY_PATH']}"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLMAP = str(REPO_ROOT / "colmap_hip_build" / "build" / "src" / "colmap" / "exe" / "colmap")
COLMAP_EXEC = os.environ.get("COLMAP_EXEC", DEFAULT_COLMAP)

def run_colmap_pipeline(frames_dir: Path, feature_method: Optional[str] = None) -> Path:
    method = (feature_method or os.environ.get("TIDDA_FEATURE_METHOD", "sift")).lower().strip()
    workspace = frames_dir.parent
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse" / "0"
    output_ply = workspace / "output" / "sparse.ply"

    # Cleanup previous run
    if database_path.exists():
        database_path.unlink()
    if (workspace / "sparse").exists():
        shutil.rmtree(workspace / "sparse")
    
    sparse_dir.mkdir(parents=True, exist_ok=True)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    print(f"--- 1. Feature Extraction (Method: {method.upper()}) ---")
    extract_method = "aliked" if "aliked" in method else "sift"
    extract_features(database_path, frames_dir, method=extract_method)
    
    print(f"--- 2. Feature Matching (Method: {method.upper()}) ---")
    match_method = "aliked_lightglue" if "aliked" in method or "lightglue" in method else "sift"
    match_features(database_path, method=match_method)

    print("--- 3. Sparse Mapping ---")
    cmd = [
        COLMAP_EXEC, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir.parent)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # COLMAP mapper might split the reconstruction into multiple models (0, 1, 2...)
    # We need to find the one with the most registered images (largest images.bin)
    models = [d for d in os.listdir(sparse_dir.parent) if (sparse_dir.parent / d).is_dir()]
    if not models:
        raise RuntimeError("Sparse reconstruction failed, no models generated.")
        
    largest_model = None
    max_size = -1
    for m in models:
        images_bin = sparse_dir.parent / m / "images.bin"
        if images_bin.exists():
            size = images_bin.stat().st_size
            if size > max_size:
                max_size = size
                largest_model = m
                
    if largest_model is None:
        raise RuntimeError("Sparse reconstruction failed, no images.bin found.")
        
    print(f"Selected largest model: {largest_model} (size: {max_size} bytes)")
    
    # If the largest model is not "0", rename it to "0" for downstream tasks
    if largest_model != "0":
        best_dir = workspace / "best_model_temp"
        shutil.move(str(sparse_dir.parent / largest_model), str(best_dir))
        shutil.rmtree(sparse_dir.parent)
        sparse_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(best_dir), str(sparse_dir))
    else:
        # Just clean up any other models
        for m in models:
            if m != "0":
                shutil.rmtree(sparse_dir.parent / m)

    print("--- 4. Point Cloud Export ---")
    # Verify the model exists
    if not (sparse_dir / "points3D.bin").exists() and not (sparse_dir / "points3D.txt").exists():
        raise RuntimeError("Sparse reconstruction failed, no points3D found.")
        
    convert_to_ply(sparse_dir, output_ply)
    
    return output_ply

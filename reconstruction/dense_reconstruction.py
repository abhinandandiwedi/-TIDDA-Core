import os
import subprocess
import shutil
from pathlib import Path

# Set library path for ROCm dependencies
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.0/lib"
else:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/rocm/core-10.0/lib:{os.environ['LD_LIBRARY_PATH']}"

def run_dense_reconstruction(workspace_dir: Path):
    """
    Runs the dense reconstruction pipeline using COLMAP (PatchMatch Stereo)
    Requires COLMAP compiled with HIP/CUDA for performance.
    """
    colmap_exec = os.environ.get("COLMAP_EXEC", "/home/abhinandan/-TIDDA-Core/colmap_hip_build/build/src/colmap/exe/colmap")
    
    dense_dir = workspace_dir / "dense"
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)
    
    database_path = workspace_dir / "database.db"
    image_path = workspace_dir / "frames"
    sparse_dir = workspace_dir / "sparse" / "0"
    
    # 1. Undistort images
    print("--- Dense 1. Image Undistorter ---")
    undistort_cmd = [
        colmap_exec, "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(sparse_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP"
    ]
    print(f"Running: {' '.join(undistort_cmd)}")
    subprocess.run(undistort_cmd, check=True)
    
    # 2. PatchMatch Stereo
    print("--- Dense 2. PatchMatch Stereo ---")
    patchmatch_cmd = [
        colmap_exec, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ]
    print(f"Running: {' '.join(patchmatch_cmd)}")
    subprocess.run(patchmatch_cmd, check=True)
    
    # 3. Stereo Fusion
    print("--- Dense 3. Stereo Fusion ---")
    dense_ply = dense_dir / "fused.ply"
    fusion_cmd = [
        colmap_exec, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_ply)
    ]
    print(f"Running: {' '.join(fusion_cmd)}")
    subprocess.run(fusion_cmd, check=True)
    
    print(f"Dense reconstruction completed! Mesh at: {dense_ply}")
    return dense_ply

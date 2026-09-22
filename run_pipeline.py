import sys
import os
from pathlib import Path
from ingest.frame_sampler import sample_frames
from reconstruction.colmap_runner import run_colmap_pipeline
from reconstruction.dense_reconstruction import run_dense_reconstruction
import shutil

# Make sure we use the verified path
COLMAP_EXEC = "/home/abhinandan/-TIDDA-Core/colmap_hip_build/build/src/colmap/exe/colmap"
os.environ["COLMAP_EXEC"] = COLMAP_EXEC

if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.0/lib"
else:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/rocm/core-10.0/lib:{os.environ['LD_LIBRARY_PATH']}"

def run():
    if not Path(COLMAP_EXEC).exists():
        print(f"ERROR: Real COLMAP binary not found at {COLMAP_EXEC}")
        sys.exit(1)

    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_dataset/test_video.mp4")
    if not video_path.exists():
        print(f"ERROR: Video {video_path} not found!")
        sys.exit(1)
        
    print(f"Extracting frames from {video_path}...")
    frames_dir = sample_frames(video_path, fps=2.0)
    workspace = frames_dir.parent
    
    print("Running COLMAP Sparse pipeline...")
    try:
        sparse_ply_path = run_colmap_pipeline(frames_dir)
    except Exception as e:
        print(f"ERROR: Sparse reconstruction failed: {e}")
        sys.exit(1)
        
    print("Running COLMAP Dense pipeline...")
    try:
        run_dense_reconstruction(workspace)
    except Exception as e:
        print(f"ERROR: Dense reconstruction failed: {e}")
        sys.exit(1)
        
    fused_ply = workspace / "dense" / "fused.ply"
    if not fused_ply.exists():
        print("ERROR: Dense fused.ply is missing!")
        sys.exit(1)

    # Copy to output directory for FastAPI server
    output_dir = workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    served_ply = output_dir / "fused.ply"
    shutil.copy(str(fused_ply), str(served_ply))
    
    print(f"Pipeline finished! Dense point cloud served at: /models/fused.ply")
    
if __name__ == "__main__":
    run()

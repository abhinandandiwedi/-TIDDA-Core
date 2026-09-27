from pathlib import Path
import os
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLMAP = str(REPO_ROOT / "colmap_hip_build" / "build" / "src" / "colmap" / "exe" / "colmap")
COLMAP_EXEC = os.environ.get("COLMAP_EXEC", DEFAULT_COLMAP)

def convert_to_ply(sparse_dir: Path, output_ply: Path):
    cmd = [
        COLMAP_EXEC, "model_converter",
        "--input_path", str(sparse_dir),
        "--output_path", str(output_ply),
        "--output_type", "PLY"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

from pathlib import Path
import os
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLMAP = str(REPO_ROOT / "colmap_hip_build" / "build" / "src" / "colmap" / "exe" / "colmap")
COLMAP_EXEC = os.environ.get("COLMAP_EXEC", DEFAULT_COLMAP)

def extract_features(database_path: Path, images_path: Path):
    cmd = [
        COLMAP_EXEC, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_path),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.num_threads", "4"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

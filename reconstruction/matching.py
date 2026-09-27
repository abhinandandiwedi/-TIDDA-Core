from pathlib import Path
import os
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLMAP = str(REPO_ROOT / "colmap_hip_build" / "build" / "src" / "colmap" / "exe" / "colmap")
COLMAP_EXEC = os.environ.get("COLMAP_EXEC", DEFAULT_COLMAP)

def match_features(database_path: Path):
    cmd = [
        COLMAP_EXEC, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.num_threads", "4"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

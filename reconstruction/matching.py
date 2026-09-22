from pathlib import Path
import os
import subprocess

COLMAP_EXEC = os.environ.get("COLMAP_EXEC", "/home/abhinandan/-TIDDA-Core/colmap_hip_build/build/src/colmap/exe/colmap")

def match_features(database_path: Path):
    cmd = [
        COLMAP_EXEC, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.num_threads", "4"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

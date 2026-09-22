import shutil
from pathlib import Path

def ingest_video(video_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / video_path.name
    shutil.copy(video_path, target_path)
    return target_path

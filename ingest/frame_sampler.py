from pathlib import Path
import subprocess

def sample_frames(video_path: Path, fps: float = 2.0) -> Path:
    workspace = Path(f"workspace_{video_path.stem}")
    frames_dir = workspace / "frames"
    if frames_dir.exists():
        import shutil
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Using ffmpeg to extract at 2 FPS
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-i", str(video_path), 
        "-vf", f"fps={fps}", str(frames_dir / "%06d.jpg")
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error:\n{e.stderr}")
        raise RuntimeError(f"Frame extraction failed: {e.stderr}")
    return frames_dir

from pathlib import Path

from .config import ReconstructionConfig
from .runner import CommandRunner


def extract_frames(config: ReconstructionConfig, runner: CommandRunner) -> Path:
    config.frames_dir.mkdir(parents=True, exist_ok=True)
    runner.require_tool(config.ffmpeg_executable)
    runner.run([config.ffmpeg_executable, "-hide_banner", "-y", "-i", str(config.input_path), "-vf", f"fps={config.frame_fps}", str(config.frames_dir / "%06d.jpg")])
    return config.frames_dir

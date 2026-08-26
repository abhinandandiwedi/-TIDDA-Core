from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float


def get_video_info(video_path: Path) -> VideoInfo:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Video metadata is incomplete or unsupported")
    return VideoInfo(fps, total_frames, width, height, total_frames / fps)

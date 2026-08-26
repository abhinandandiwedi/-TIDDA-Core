from pathlib import Path

from .video_processor import VideoInfo


def extract_frames(video_path: Path, output_dir: Path, every_n: int = 5,
                   blur_threshold: float = 40.0) -> list[Path]:
    import cv2

    if every_n < 1:
        raise ValueError("every_n must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    paths: list[Path] = []
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % every_n == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_threshold:
                    path = output_dir / f"frame_{frame_index:06d}.jpg"
                    if cv2.imwrite(str(path), frame):
                        paths.append(path)
            frame_index += 1
    finally:
        capture.release()
    if not paths:
        raise ValueError("No usable frames were extracted from the video")
    return paths

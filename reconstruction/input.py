from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def classify_input(path: Path) -> str:
    if path.is_dir():
        images = [item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise ValueError(f"image folder contains no supported images: {path}")
        return "images"
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"input must be an image folder or supported video: {path}")

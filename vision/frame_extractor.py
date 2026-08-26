from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from .models import Frame


class FrameExtractor:
    """Sequential video reader; OpenCV is imported only when extraction starts."""

    def __init__(self, source: str | int, sample_fps: float | None = None) -> None:
        self.source = source
        self.sample_fps = sample_fps

    def frames(self, start: int = 0, stop: int | None = None) -> Iterator[Frame]:
        try:
            import importlib
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise RuntimeError("FrameExtractor requires opencv-python") from exc
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise OSError(f"unable to open video source: {self.source}")
        native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(native_fps / self.sample_fps)) if self.sample_fps else 1
        index = 0
        emitted = 0
        try:
            while stop is None or emitted < stop - start:
                ok, image = capture.read()
                if not ok:
                    break
                if index >= start and (index - start) % stride == 0:
                    timestamp = datetime.now(timezone.utc)
                    yield Frame.create(index, image, timestamp)
                    emitted += 1
                index += 1
        finally:
            capture.release()

    @staticmethod
    def image_frame(image: Any, index: int = 0) -> Frame:
        return Frame.create(index, image)

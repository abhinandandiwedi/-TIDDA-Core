from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameQuality:
    valid: bool
    brightness: float
    sharpness: float
    reason: str = ""


def assess_frame(image, min_width: int = 160, min_height: int = 120) -> FrameQuality:
    if image is None or getattr(image, "ndim", 0) < 2:
        return FrameQuality(False, 0.0, 0.0, "empty image")
    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        return FrameQuality(False, 0.0, 0.0, "frame is too small")
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except ImportError:
        gray = image.mean(axis=2) if image.ndim == 3 else image
        brightness = float(gray.mean())
        sharpness = 0.0
    if brightness < 3 or brightness > 252:
        return FrameQuality(False, brightness, sharpness, "overexposed or underexposed")
    return FrameQuality(True, brightness, sharpness)

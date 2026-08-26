from .models import Frame


class DepthEstimator:
    def estimate(self, frame: Frame):
        raise NotImplementedError


class MonocularDepthEstimator(DepthEstimator):
    """Approximate relative depth from luminance; replace with MiDaS/Depth-Anything adapter for metric depth."""

    def __init__(self, scale_m: float = 10.0) -> None:
        self.scale_m = scale_m

    def estimate(self, frame: Frame):
        try:
            import importlib
            cv2 = importlib.import_module("cv2")
            np = importlib.import_module("numpy")
        except ImportError as exc:
            raise RuntimeError("monocular depth requires opencv-python and numpy") from exc
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        return np.clip((1.0 - normalized) * self.scale_m, 0.1, self.scale_m)


class StereoDepthEstimator(DepthEstimator):
    def __init__(self, left_camera_matrix, right_camera_matrix, baseline_m: float) -> None:
        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.baseline_m = baseline_m

    def estimate(self, frame: Frame):
        raise NotImplementedError("stereo depth requires synchronized left/right frames")

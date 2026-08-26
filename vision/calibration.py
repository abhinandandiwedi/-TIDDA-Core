from .models import CameraCalibration


def calibration_from_image_size(width: int, height: int, horizontal_fov_deg: float = 70.0) -> CameraCalibration:
    import math
    fx = (width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
    return CameraCalibration(fx, fx, width / 2, height / 2)


def calibrate_from_chessboard(*args, **kwargs) -> CameraCalibration:
    """Calibrate with cv2.calibrateCamera; callers provide OpenCV object/image points."""
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("chessboard calibration requires opencv-python and numpy") from exc
    object_points, image_points, image_size = args[:3]
    _, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    return CameraCalibration(float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2]), tuple(distortion.ravel().tolist()))

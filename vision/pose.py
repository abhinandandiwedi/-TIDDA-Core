from .models import CameraCalibration, Pose


def estimate_pose(object_points, image_points, calibration: CameraCalibration) -> Pose:
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("pose estimation requires opencv-python and numpy") from exc
    camera_matrix = np.array([[calibration.fx, 0, calibration.cx], [0, calibration.fy, calibration.cy], [0, 0, 1]], dtype=float)
    distortion = np.array(calibration.distortion or (0, 0, 0, 0, 0), dtype=float)
    ok, rotation, translation = cv2.solvePnP(np.asarray(object_points, dtype=float), np.asarray(image_points, dtype=float), camera_matrix, distortion)
    if not ok:
        raise ValueError("camera pose could not be estimated")
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    return Pose(tuple(rotation_matrix.ravel().tolist()), tuple(translation.ravel().tolist()))

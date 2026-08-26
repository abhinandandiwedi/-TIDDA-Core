from math import cos, radians

from .models import CameraCalibration, Pose


def pixel_to_ground(pixel: tuple[float, float], calibration: CameraCalibration, pose: Pose, ground_altitude_m: float = 0.0) -> tuple[float, float, float]:
    """Intersect a calibrated camera ray with a horizontal ground plane in local coordinates."""
    import numpy as np
    rotation = np.asarray(pose.rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(pose.translation, dtype=float).reshape(3)
    ray_camera = np.array([(pixel[0] - calibration.cx) / calibration.fx, (pixel[1] - calibration.cy) / calibration.fy, 1.0])
    ray_world = rotation.T @ ray_camera
    if abs(ray_world[2]) < 1e-9:
        raise ValueError("camera ray is parallel to ground")
    scale = (ground_altitude_m - translation[2]) / ray_world[2]
    if scale < 0:
        raise ValueError("pixel ray does not intersect ground in front of camera")
    return tuple((translation + scale * ray_world).tolist())


def local_to_gps(origin_lat: float, origin_lon: float, east_m: float, north_m: float, altitude_m: float = 0.0) -> tuple[float, float, float]:
    lat = origin_lat + north_m / 111_320.0
    lon = origin_lon + east_m / (111_320.0 * max(0.01, cos(radians(origin_lat))))
    return lat, lon, altitude_m

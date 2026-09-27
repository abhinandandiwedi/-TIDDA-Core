import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("reconstruction.localization_3d")


class Localization3D:
    """
    3D Localization Module.

    Responsibilities:
    - Load camera intrinsics from COLMAP (cameras.bin)
    - Load camera poses from COLMAP (images.bin)
    - Load dense depth maps from COLMAP PatchMatch (.geometric.bin)
    - Translate a 2D YOLO bounding box to a 3D world coordinate.
    """

    def __init__(self, sparse_dir: Path, dense_dir: Path):
        self.sparse_dir = sparse_dir
        self.dense_dir = dense_dir

        self.cameras = {}
        self.images = {}

        # We will attempt to load the sparse model files to cache intrinsics and poses.
        self._load_cameras()
        self._load_images()

    def _load_cameras(self):
        cameras_bin = self.sparse_dir / "0" / "cameras.bin"
        if not cameras_bin.exists():
            logger.warning(f"Cameras file not found: {cameras_bin}")
            return

        with open(cameras_bin, "rb") as fid:
            num_cameras = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_cameras):
                camera_properties = struct.unpack("<iiQQ", fid.read(24))
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                width = camera_properties[2]
                height = camera_properties[3]

                # Determine number of parameters based on model_id
                num_params = 8  # OPENCV default
                if model_id == 0: num_params = 3 # SIMPLE_PINHOLE
                elif model_id == 1: num_params = 4 # PINHOLE
                elif model_id == 2: num_params = 4 # SIMPLE_RADIAL
                elif model_id == 3: num_params = 5 # RADIAL
                elif model_id == 4: num_params = 8 # OPENCV

                params = struct.unpack("<" + "d" * num_params, fid.read(8 * num_params))

                self.cameras[camera_id] = {
                    "model_id": model_id,
                    "width": width,
                    "height": height,
                    "params": params
                }

    def _load_images(self):
        images_bin = self.sparse_dir / "0" / "images.bin"
        if not images_bin.exists():
            logger.warning(f"Images file not found: {images_bin}")
            return

        with open(images_bin, "rb") as fid:
            num_reg_images = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_reg_images):
                binary_image_properties = struct.unpack("<idddddddi", fid.read(64))
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]

                image_name = ""
                current_char = fid.read(1)
                while current_char != b"\x00" and current_char != b"":
                    image_name += current_char.decode("utf-8")
                    current_char = fid.read(1)

                num_points2D = struct.unpack("<Q", fid.read(8))[0]
                fid.read(num_points2D * 24)  # Skip 2D points (x, y, id)

                self.images[image_name] = {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "qvec": qvec,
                    "tvec": tvec
                }

    def _qvec2rotmat(self, qvec: np.ndarray) -> np.ndarray:
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
             2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
             2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
             1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
             2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
             2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
             1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
        ])

    def _read_depth_map(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fid:
                # Read shape from text header delimited by '&'
                header = b""
                num_delimiter = 0
                while num_delimiter < 3:
                    byte = fid.read(1)
                    if not byte: break
                    header += byte
                    if byte == b"&":
                        num_delimiter += 1

                parts = header.decode("utf-8").strip("&").split("&")
                width, height, channels = map(int, parts)

                array = np.fromfile(fid, np.float32)
            array = array.reshape((width, height, channels), order="F")
            return np.transpose(array, (1, 0, 2)).squeeze()
        except Exception as e:
            logger.error(f"Failed to read depth map {path}: {e}")
            return None

    def localize(self, frame_filename: str, bbox: list, class_name: str, confidence: float, track_id: str) -> Dict[str, Any]:
        """
        Localize a bounding box in 3D space.
        bbox format: [x1, y1, x2, y2]
        """
        if frame_filename not in self.images:
            return self._failed_state(track_id, class_name, confidence, "FRAME_NOT_REGISTERED")

        img_data = self.images[frame_filename]
        camera_id = img_data["camera_id"]

        if camera_id not in self.cameras:
            return self._failed_state(track_id, class_name, confidence, "CALIBRATION_MISSING")

        cam_data = self.cameras[camera_id]

        # Extrapolate intrinsics
        # Assuming OPENCV or PINHOLE, params are typically [fx, fy, cx, cy, ...]
        if len(cam_data["params"]) < 4:
            return self._failed_state(track_id, class_name, confidence, "CALIBRATION_MISSING")

        fx, fy, cx, cy = cam_data["params"][:4]

        # 3. PIXEL SELECTION
        # We explicitly choose the bottom-center pixel (u = center-X, v = bottom-Y).
        # This assumes ground-contact objects (e.g. people standing on the ground),
        # which provides a much more accurate world coordinate on the floor plane
        # compared to the physical center of the bounding box.
        x1, y1, x2, y2 = bbox
        u = int((x1 + x2) / 2.0)
        v = int(y2)

        # Ensure coordinates are within bounds
        width = cam_data["width"]
        height = cam_data["height"]
        u = max(0, min(u, width - 1))
        v = max(0, min(v, height - 1))

        # 4. DEPTH ASSOCIATION
        depth_map_path = self.dense_dir / "stereo" / "depth_maps" / f"{frame_filename}.geometric.bin"
        depth_map = self._read_depth_map(depth_map_path)

        if depth_map is None:
            return self._failed_state(track_id, class_name, confidence, "NO_DEPTH")

        # Due to some potential resizing between original frames and depth maps in COLMAP,
        # we scale the UV coordinates to the depth map"s resolution.
        depth_h, depth_w = depth_map.shape
        scale_x = depth_w / float(width)
        scale_y = depth_h / float(height)

        u_d = int(u * scale_x)
        v_d = int(v * scale_y)
        u_d = max(0, min(u_d, depth_w - 1))
        v_d = max(0, min(v_d, depth_h - 1))

        Z = float(depth_map[v_d, u_d])
        if Z <= 0.0 or np.isnan(Z) or np.isinf(Z):
            # Try a small neighborhood search if the exact pixel is invalid
            found = False
            for r in range(1, 5):
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        ny = np.clip(v_d + dy, 0, depth_h - 1)
                        nx = np.clip(u_d + dx, 0, depth_w - 1)
                        Z_neighbor = float(depth_map[ny, nx])
                        if Z_neighbor > 0.0 and not np.isnan(Z_neighbor) and not np.isinf(Z_neighbor):
                            Z = Z_neighbor
                            found = True
                            break
                    if found: break
                if found: break

            if not found:
                return self._failed_state(track_id, class_name, confidence, "INVALID_DEPTH")

        # 5. CAMERA INTRINSICS TRANSFORM (Pixel to Camera Coordinate)
        # Using the standard pinhole projection:
        # Xc = (u - cx) * Z / fx
        # Yc = (v - cy) * Z / fy
        # Zc = Z
        Xc = (u - cx) * Z / fx
        Yc = (v - cy) * Z / fy
        Zc = Z

        X_c = np.array([Xc, Yc, Zc])

        # 6. CAMERA POSE TRANSFORM (Camera to World Coordinate)
        # COLMAP poses transform world to camera: X_c = R * X_w + t
        # Thus: X_w = R^T * (X_c - t)
        qvec = img_data["qvec"]
        tvec = img_data["tvec"]
        R = self._qvec2rotmat(qvec)
        R_inv = R.T

        X_w = R_inv.dot(X_c - tvec)

        # Also compute camera optical center in world coordinates: t_c2w = -R^T * t
        C_w = -R_inv.dot(tvec)

        return {
            "track_id": track_id,
            "class": class_name,
            "confidence": float(confidence),
            "status": "LOCALIZED",
            "pixel": {
                "u": float(u),
                "v": float(v)
            },
            "depth_m": float(Z),
            "camera_position": {
                "x": float(C_w[0]),
                "y": float(C_w[1]),
                "z": float(C_w[2])
            },
            "world_position": {
                "x": float(X_w[0]),
                "y": float(X_w[1]),
                "z": float(X_w[2])
            }
        }

    def _failed_state(self, track_id: str, class_name: str, confidence: float, status: str) -> Dict[str, Any]:
        return {
            "track_id": track_id,
            "class": class_name,
            "confidence": float(confidence),
            "status": status,
            "pixel": None,
            "depth_m": None,
            "camera_position": None,
            "world_position": None
        }

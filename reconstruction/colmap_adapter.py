import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple
from mapping_models import Pose

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def qvec2rotmat(qvec):
    return [
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ]

def rotmat2euler(R):
    sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2][1], R[2][2])
        y = math.atan2(-R[2][0], sy)
        z = math.atan2(R[1][0], R[0][0])
    else:
        x = math.atan2(-R[1][2], R[1][1])
        y = math.atan2(-R[2][0], sy)
        z = 0
    return x, y, z  # roll, pitch, yaw

def colmap_to_pose(qvec: tuple, tvec: tuple) -> Pose:
    # W2C to C2W
    R = qvec2rotmat(qvec)
    
    # Transpose R (inverse for rotation matrices)
    R_inv = [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]]
    ]
    
    # t_c2w = -R_inv * tvec
    t_c2w = [
        -(R_inv[0][0]*tvec[0] + R_inv[0][1]*tvec[1] + R_inv[0][2]*tvec[2]),
        -(R_inv[1][0]*tvec[0] + R_inv[1][1]*tvec[1] + R_inv[1][2]*tvec[2]),
        -(R_inv[2][0]*tvec[0] + R_inv[2][1]*tvec[1] + R_inv[2][2]*tvec[2])
    ]
    
    roll, pitch, yaw = rotmat2euler(R_inv)
    
    return Pose(
        x=t_c2w[0], y=t_c2w[1], z=t_c2w[2],
        roll=roll, pitch=pitch, yaw=yaw
    )

def read_images_binary(path_to_model_file: Path) -> List[Pose]:
    poses = []
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2D * 24)  # Skip 2D points (x, y, id)
            
            poses.append((image_name, colmap_to_pose(qvec, tvec)))
            
    # Sort by image_name assuming they are frames like 0001.jpg
    poses.sort(key=lambda x: x[0])
    return [p[1] for p in poses]

def extract_poses(sparse_dir: Path) -> List[Pose]:
    images_bin = sparse_dir / "0" / "images.bin"
    if not images_bin.exists():
        raise FileNotFoundError(f"Missing {images_bin}")
    return read_images_binary(images_bin)

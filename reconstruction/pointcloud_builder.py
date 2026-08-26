from pathlib import Path


def build_point_cloud(frame_paths: list[Path], depths: list, output_path: Path,
                      sample_step: int = 4, voxel_size: float = 0.03):
    import cv2
    import numpy as np
    import open3d as o3d

    if len(frame_paths) != len(depths):
        raise ValueError("frame and depth counts must match")
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for frame_path, depth in zip(frame_paths, depths):
        image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
        if image is None or depth is None:
            continue
        height, width = depth.shape[:2]
        rgb = cv2.resize(image, (width, height))
        rows, cols = np.mgrid[0:height:sample_step, 0:width:sample_step]
        values = depth[rows, cols]
        valid = np.isfinite(values) & (values > 0.03)
        focal = max(width, height) * 0.9
        x = (cols - width / 2) * values / focal
        y = (rows - height / 2) * values / focal
        points.append(np.column_stack((x[valid], -y[valid], -values[valid])))
        colors.append(rgb[rows, cols][valid] / 255.0)
    if not points:
        raise ValueError("No valid 3D points were generated")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(points))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
    cloud = cloud.voxel_down_sample(voxel_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), cloud):
        raise OSError(f"Unable to write point cloud: {output_path}")
    return cloud

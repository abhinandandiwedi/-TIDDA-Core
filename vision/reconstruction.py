from .models import CameraCalibration, PointCloud


def depth_to_point_cloud(image, depth, calibration: CameraCalibration, stride: int = 4, max_points: int = 20_000, frame_index: int | None = None) -> PointCloud:
    import numpy as np
    height, width = depth.shape[:2]
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride].astype(np.float32)
    x = (cols.astype(np.float32) - calibration.cx) * z / calibration.fx
    y = (rows.astype(np.float32) - calibration.cy) * z / calibration.fy
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    colors = image[::stride, ::stride].reshape(-1, image.shape[2]) if image.ndim == 3 else None
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        if colors is not None: colors = colors[indices]
    return PointCloud(points, colors, frame_index=frame_index)


def point_cloud_to_mesh(cloud: PointCloud, width: int, height: int, stride: int = 4) -> PointCloud:
    import numpy as np
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    count = len(rows.ravel())
    if len(cloud.points) != count:
        return cloud
    faces = []
    grid_width = len(range(0, width, stride))
    grid_height = len(range(0, height, stride))
    for row in range(grid_height - 1):
        for col in range(grid_width - 1):
            index = row * grid_width + col
            faces.extend(((index, index + 1, index + grid_width),
                          (index + 1, index + grid_width + 1, index + grid_width)))
    return PointCloud(cloud.points, cloud.colors, np.asarray(faces, dtype=np.int32), cloud.frame_index)

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .depth_estimator import DepthEstimator
from .frame_extractor import extract_frames
from .pointcloud_builder import build_point_cloud
from .utils import estimate_camera_trajectory
from .video_processor import get_video_info


@dataclass
class ReconstructionResult:
    frame_paths: list[Path]
    pointcloud_path: Path
    trajectory_path: Path
    point_count: int
    elapsed_seconds: float
    device: str


def reconstruct(video_path: Path, workspace: Path, every_n: int = 5,
                blur_threshold: float = 40.0, voxel_size: float = 0.03,
                progress=None) -> ReconstructionResult:
    started = perf_counter()
    info = get_video_info(video_path)
    frames = extract_frames(video_path, workspace / "frames", every_n, blur_threshold)
    estimator = DepthEstimator()
    depths = []
    for index, frame in enumerate(frames):
        depths.append(estimator.estimate(frame))
        if progress:
            progress(index + 1, len(frames))
    output = workspace / "output"
    cloud = build_point_cloud(frames, depths, output / "pointcloud.ply", voxel_size=voxel_size)
    trajectory = output / "camera_trajectory.png"
    estimate_camera_trajectory(frames, trajectory)
    return ReconstructionResult(frames, output / "pointcloud.ply", trajectory,
                                len(cloud.points), perf_counter() - started,
                                str(estimator.device))

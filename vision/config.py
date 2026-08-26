from dataclasses import dataclass


@dataclass(slots=True)
class VisionConfig:
    sample_fps: float = 5.0
    min_frame_width: int = 160
    min_frame_height: int = 120
    detection_confidence: float = 0.45
    tracking_distance_px: float = 80.0
    depth_scale_m: float = 10.0
    point_stride: int = 4
    max_points: int = 20_000
    camera_height_m: float = 10.0

    def __post_init__(self) -> None:
        if self.sample_fps <= 0 or self.min_frame_width < 1 or self.min_frame_height < 1:
            raise ValueError("invalid frame configuration")
        if not 0 <= self.detection_confidence <= 1:
            raise ValueError("detection_confidence must be between 0 and 1")
        if self.tracking_distance_px <= 0 or self.depth_scale_m <= 0:
            raise ValueError("tracking and depth scales must be positive")
        if self.point_stride < 1 or self.max_points < 1:
            raise ValueError("point_stride and max_points must be positive")

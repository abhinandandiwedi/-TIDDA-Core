from datetime import datetime, timezone

from detection_fusion import Detection, FusionEngine

from .calibration import calibration_from_image_size
from .config import VisionConfig
from .depth import DepthEstimator, MonocularDepthEstimator
from .detector import Detector
from .models import Frame, VisionDetection
from .quality import assess_frame
from .reconstruction import depth_to_point_cloud
from .tracker import CentroidTracker


class VisionPipeline:
    """Runs quality, detection, tracking, depth, point cloud, and fusion for each frame."""

    def __init__(self, detector: Detector, fusion_engine: FusionEngine | None = None,
                 depth_estimator: DepthEstimator | None = None, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self.detector = detector
        self.tracker = CentroidTracker(self.config.tracking_distance_px)
        self.depth = depth_estimator or MonocularDepthEstimator(self.config.depth_scale_m)
        self.fusion = fusion_engine or FusionEngine()

    def process(self, frame: Frame, node_id: str, lat: float, lon: float) -> dict:
        quality = assess_frame(frame.image, self.config.min_frame_width, self.config.min_frame_height)
        if not quality.valid:
            return {"frame": frame, "quality": quality, "detections": [], "point_cloud": None, "entities": []}
        detections = self.tracker.update(self.detector.detect(frame))
        depth = self.depth.estimate(frame)
        calibration = calibration_from_image_size(frame.width, frame.height)
        cloud = depth_to_point_cloud(frame.image, depth, calibration, self.config.point_stride, self.config.max_points, frame.index)
        entities = []
        for detection in detections:
            entity = self.fusion.add_detection(Detection(node_id, lat, lon, detection.confidence,
                                                        detection.class_name, frame.timestamp,
                                                        sensor_type="camera", sensor_metadata={"track_id": detection.track_id,
                                                        "bbox": detection.bbox.__dict__ if hasattr(detection.bbox, "__dict__") else {"x1": detection.bbox.x1, "y1": detection.bbox.y1, "x2": detection.bbox.x2, "y2": detection.bbox.y2}}))
            if entity is not None:
                entities.append(entity)
        return {"frame": frame, "quality": quality, "detections": detections, "depth": depth,
                "point_cloud": cloud, "entities": entities}

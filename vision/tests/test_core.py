from datetime import datetime, timezone

import pytest

from detection_fusion import FusionEngine
from vision.calibration import calibration_from_image_size
from vision.models import BoundingBox, Frame, VisionDetection
from vision.quality import assess_frame
from vision.reconstruction import depth_to_point_cloud
from vision.tracker import CentroidTracker


def test_quality_and_point_cloud():
    np = pytest.importorskip("numpy")
    image = np.full((120, 160, 3), 100, dtype=np.uint8)
    frame = Frame.create(0, image, datetime.now(timezone.utc))
    assert assess_frame(image).valid
    cloud = depth_to_point_cloud(image, np.ones((120, 160), dtype=np.float32), calibration_from_image_size(160, 120), 8)
    assert len(cloud.points) > 0


def test_tracker_keeps_id_for_nearby_detection():
    tracker = CentroidTracker(50)
    first = tracker.update([VisionDetection("person", .8, BoundingBox(0, 0, 20, 20))])[0]
    second = tracker.update([VisionDetection("person", .9, BoundingBox(3, 2, 23, 22))])[0]
    assert first.track_id == second.track_id


def test_pipeline_fusion_can_be_consumed():
    assert FusionEngine().get_all_entities() == []

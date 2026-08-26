from datetime import datetime, timezone

import pytest

np = pytest.importorskip("numpy")

from vision import VisionPipeline
from vision.models import BoundingBox, Frame, VisionDetection


class FakeDetector:
    def detect(self, frame):
        return [VisionDetection("person", 0.9, BoundingBox(20, 20, 60, 90))]


def test_frame_to_fusion_and_point_cloud():
    image = np.full((120, 160, 3), 100, dtype=np.uint8)
    frame = Frame.create(3, image, datetime.now(timezone.utc))
    result = VisionPipeline(FakeDetector()).process(frame, "camera_01", 26.8467, 80.9462)
    assert result["quality"].valid
    assert result["detections"][0].track_id == 1
    assert result["point_cloud"].points.shape[1] == 3
    assert len(result["entities"]) == 1
    assert result["entities"][0].source_nodes == {"camera_01"}

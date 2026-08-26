from datetime import datetime, timezone
from detection_fusion import Detection, FusionEngine


def test_multi_sensor_fusion():
    engine = FusionEngine()
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    one = engine.add_detection(Detection("camera", 26.8467, 80.9462, .82, "vehicle", t))
    assert one is not None
    same = engine.add_detection(Detection("radar", 26.84671, 80.94621, .91, "vehicle", t))
    assert same is not None
    assert one.entity_id == same.entity_id
    assert set(same.source_nodes) == {"camera", "radar"}
    assert same.confidence > .91
    assert same.detection_count == 2

from datetime import datetime, timezone
from detection_fusion import Detection, FusionEngine


def test_serialization():
    entity = FusionEngine().add_detection(Detection("cam", 1, 2, .5, "vehicle", datetime.now(timezone.utc)))
    assert entity is not None
    assert entity.to_dict()["entity_id"] == entity.entity_id
    assert entity.to_json().startswith("{")

from datetime import datetime, timezone
from detection_fusion import Detection, FusionEngine


def test_threat_score_and_events():
    events = []
    engine = FusionEngine()
    engine.subscribe_to_event(events.append)
    entity = engine.add_detection(Detection("cam", 1, 1, .9, "drone", datetime.now(timezone.utc)))
    assert entity is not None
    assert 0 <= entity.threat_score <= 10
    assert entity.threat_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert any(event.event_type == "ENTITY_CREATED" for event in events)

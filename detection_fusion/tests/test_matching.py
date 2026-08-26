from datetime import datetime, timezone, timedelta
from detection_fusion import Detection, FusionConfig, FusionEngine


def detection(lat=26.8, lon=80.9, cls="vehicle", seconds=0):
    return Detection("cam", lat, lon, .8, cls, datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds))


def test_same_entity_and_rejections():
    engine = FusionEngine(FusionConfig(match_distance_meters=50))
    first = engine.add_detection(detection())
    assert engine.add_detection(detection(lon=80.9001, seconds=2)).entity_id == first.entity_id
    assert engine.add_detection(detection(cls="person", seconds=2)).entity_id != first.entity_id
    assert engine.add_detection(detection(lon=81, seconds=2)).entity_id != first.entity_id

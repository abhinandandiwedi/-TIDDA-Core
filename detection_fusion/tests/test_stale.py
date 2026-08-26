from datetime import datetime, timezone, timedelta
from detection_fusion import Detection, FusionConfig, FusionEngine, EntityState


def test_stale_expired_lifecycle():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    engine = FusionEngine(FusionConfig(stale_timeout_seconds=10, expire_timeout_seconds=60))
    entity = engine.add_detection(Detection("cam", 1, 1, .8, "vehicle", base))
    assert entity is not None
    engine.detect_stale_entities(base + timedelta(seconds=11))
    assert entity.status == EntityState.STALE
    engine.detect_stale_entities(base + timedelta(seconds=61))
    assert entity.status == EntityState.EXPIRED
    assert engine.cleanup_expired_entities() == [entity.entity_id]

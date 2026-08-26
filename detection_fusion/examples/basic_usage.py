from datetime import datetime, timedelta, timezone

from detection_fusion import Detection, FusionConfig, FusionEngine


base = datetime.now(timezone.utc)
engine = FusionEngine(FusionConfig(match_distance_meters=50))
engine.subscribe_to_event(lambda event: print(event.event_type, event.entity_id))

observations = [
    Detection("camera_01", 26.846700, 80.946200, 0.82, "vehicle", base),
    Detection("radar_01", 26.846710, 80.946210, 0.91, "vehicle", base + timedelta(seconds=2)),
    Detection("thermal_01", 26.846705, 80.946205, 0.87, "vehicle", base + timedelta(seconds=3)),
    Detection("camera_01", 26.850000, 80.950000, 0.80, "vehicle", base + timedelta(seconds=4)),
]
for observation in observations:
    engine.add_detection(observation)

print("FUSED ENTITIES")
for entity in engine.get_all_entities():
    print(entity.to_json())

engine.detect_stale_entities(base + timedelta(seconds=12))
print("STALE:", [entity.entity_id for entity in engine.get_stale_entities()])
engine.detect_stale_entities(base + timedelta(seconds=61))
print("EXPIRED:", [entity.entity_id for entity in engine.get_expired_entities()])

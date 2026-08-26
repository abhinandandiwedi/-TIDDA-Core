from detection_fusion import FusionEngine


def test_invalid_input_is_event_not_crash():
    engine = FusionEngine()
    assert engine.add_detection({"node_id": "", "lat": 0, "lon": 0, "confidence": 2, "class_name": "", "timestamp": 0}) is None
    assert engine.to_dict() == []

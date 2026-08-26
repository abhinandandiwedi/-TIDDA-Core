from datetime import datetime, timezone
import pytest

from detection_fusion import Detection
from detection_fusion.Exceptions import InvalidDetectionError


def valid():
    return Detection("cam", 26.8, 80.9, .8, "vehicle", datetime.now(timezone.utc))


def test_valid_detection():
    assert valid().node_id == "cam"


@pytest.mark.parametrize("field,value", [("lat", 100), ("lon", 200), ("confidence", 2)])
def test_invalid_detection(field, value):
    data = valid().__dict__ if hasattr(valid(), "__dict__") else {"node_id": "cam", "lat": 26.8, "lon": 80.9, "confidence": .8, "class_name": "vehicle", "timestamp": datetime.now(timezone.utc)}
    data[field] = value
    with pytest.raises(InvalidDetectionError):
        Detection(**data)

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Dict, Optional, cast

try:
    from .Exceptions import InvalidDetectionError
except ImportError:
    from Exceptions import InvalidDetectionError


@dataclass(frozen=True, slots=True)
class Detection:
    node_id: str
    lat: float
    lon: float
    confidence: float
    class_name: str
    timestamp: datetime | float | int
    altitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    sensor_type: Optional[str] = None
    sensor_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = self.timestamp
        if isinstance(timestamp, (int, float)):
            if not isfinite(float(timestamp)):
                raise InvalidDetectionError("timestamp must be finite")
            timestamp = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        elif isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
        else:
            raise InvalidDetectionError("timestamp must be datetime or Unix seconds")
        object.__setattr__(self, "timestamp", timestamp)
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise InvalidDetectionError("node_id is required")
        if not isfinite(float(self.lat)) or not -90 <= float(self.lat) <= 90:
            raise InvalidDetectionError("latitude must be between -90 and 90")
        if not isfinite(float(self.lon)) or not -180 <= float(self.lon) <= 180:
            raise InvalidDetectionError("longitude must be between -180 and 180")
        if not isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
            raise InvalidDetectionError("confidence must be between 0 and 1")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise InvalidDetectionError("class_name is required")
        for name in ("altitude", "speed", "heading"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or not isfinite(float(value))):
                raise InvalidDetectionError(f"{name} must be finite when provided")

    @property
    def time(self) -> datetime:
        return cast(datetime, self.timestamp)

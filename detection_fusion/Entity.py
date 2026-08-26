from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Set

try:
    from .Detection import Detection
    from .EntityState import EntityState
except ImportError:
    from Detection import Detection
    from EntityState import EntityState


@dataclass
class Entity:
    entity_id: str
    class_name: str
    latitude: float
    longitude: float
    confidence: float
    first_seen: datetime
    last_seen: datetime
    max_history_size: int = 100
    altitude: float | None = None
    speed: float | None = None
    heading: float | None = None
    detection_count: int = 1
    source_nodes: Set[str] = field(default_factory=set)
    threat_score: float = 0.0
    threat_level: str = "LOW"
    status: EntityState = EntityState.NEW
    history: Deque[dict] = field(default_factory=deque)
    class_history: Deque[dict] = field(default_factory=deque)
    observations: list[Detection] = field(default_factory=list)
    last_position: tuple[float, float] | None = None

    @classmethod
    def from_detection(cls, entity_id: str, detection: Detection, max_history_size: int):
        entity = cls(entity_id, detection.class_name, detection.lat, detection.lon,
                     detection.confidence, detection.time, detection.time,
                     max_history_size=max_history_size, status=EntityState.NEW)
        entity.record(detection)
        entity.status = EntityState.NEW
        return entity

    def record(self, detection: Detection) -> None:
        self.last_position = (self.latitude, self.longitude)
        self.latitude, self.longitude = detection.lat, detection.lon
        self.confidence = max(self.confidence, detection.confidence)
        self.last_seen = max(self.last_seen, detection.time)
        self.detection_count += 1 if self.observations else 0
        self.source_nodes.add(detection.node_id)
        self.observations.append(detection)
        self.observations = self.observations[-self.max_history_size:]
        self.history.append({"timestamp": detection.time.isoformat(), "lat": detection.lat,
                             "lon": detection.lon, "confidence": detection.confidence,
                             "class_name": detection.class_name, "node_id": detection.node_id})
        while len(self.history) > self.max_history_size:
            self.history.popleft()
        self.class_history.append({"class_name": detection.class_name,
                                   "confidence": detection.confidence,
                                   "timestamp": detection.time.isoformat()})
        while len(self.class_history) > self.max_history_size:
            self.class_history.popleft()
        class_scores: dict[str, float] = {}
        for observation in self.observations:
            class_scores[observation.class_name] = class_scores.get(observation.class_name, 0.0) + observation.confidence
        if class_scores:
            self.class_name = max(class_scores.items(), key=lambda item: item[1])[0]
        if detection.altitude is not None: self.altitude = detection.altitude
        if detection.speed is not None: self.speed = detection.speed
        if detection.heading is not None: self.heading = detection.heading
        self.status = EntityState.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {"entity_id": self.entity_id, "class_name": self.class_name,
                "latitude": self.latitude, "longitude": self.longitude,
                "confidence": self.confidence, "first_seen": self.first_seen.isoformat(),
                "last_seen": self.last_seen.isoformat(), "detection_count": self.detection_count,
                "source_nodes": sorted(self.source_nodes), "threat_score": self.threat_score,
                "threat_level": self.threat_level, "status": self.status.value,
                "altitude": self.altitude, "speed": self.speed, "heading": self.heading,
                "history": list(self.history), "class_history": list(self.class_history)}

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())

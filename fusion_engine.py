from dataclasses import dataclass, field
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from math import isfinite
from typing import Any, Optional


@dataclass
class Detection:
    node_id: str
    lat: float
    lon: float
    confidence: float
    class_name: str
    timestamp: datetime


@dataclass
class Entity:
    entity_id: str
    lat: float
    lon: float
    confidence: float
    class_name: str
    first_seen: datetime
    last_seen: datetime
    detections: list = field(default_factory=list)
    threat_score: float = 0.0


class FusionEngine:

    def __init__(
        self,
        distance_threshold_m: float = 100.0,
        time_window_s: float = 10.0,
        stale_after_s: float = 15.0
    ):
        if not isfinite(distance_threshold_m) or distance_threshold_m < 0:
            raise ValueError("distance_threshold_m must be non-negative")
        if not isfinite(time_window_s) or time_window_s < 0:
            raise ValueError("time_window_s must be non-negative")
        if not isfinite(stale_after_s) or stale_after_s < 0:
            raise ValueError("stale_after_s must be non-negative")

        self.distance_threshold_m = distance_threshold_m
        self.time_window_s = time_window_s
        self.stale_after_s = stale_after_s

        self.entities = {}
        self._next_entity_id = 1

    def add_detection(
        self,
        node_id: str,
        lat: float,
        lon: float,
        confidence: float,
        class_name: str,
        timestamp: datetime | float | int | None = None,
    ) -> dict[str, Any]:
        """Add a detection and return a JSON-friendly fusion event."""
        detection = Detection(
            node_id=str(node_id),
            lat=float(lat),
            lon=float(lon),
            confidence=float(confidence),
            class_name=str(class_name).strip(),
            timestamp=self._coerce_timestamp(timestamp),
        )
        self._validate_detection(detection)
        self.remove_stale_entities(detection.timestamp)

        existing_entity = self.find_match(detection)
        entity = self.process_detection(detection)
        return {
            "action": "updated" if existing_entity is not None else "created",
            "entity_id": entity.entity_id,
            "lat": entity.lat,
            "lon": entity.lon,
            "confidence": entity.confidence,
            "class_name": entity.class_name,
            "threat_score": entity.threat_score,
            "first_seen": entity.first_seen.isoformat(),
            "last_seen": entity.last_seen.isoformat(),
            "detection_count": len(entity.detections),
            "source_nodes": sorted({item.node_id for item in entity.detections}),
        }

    def snapshot(self) -> list[dict[str, Any]]:
        """Return the current entities in dashboard-safe form."""
        return [
            {
                "entity_id": entity.entity_id,
                "lat": entity.lat,
                "lon": entity.lon,
                "confidence": entity.confidence,
                "class_name": entity.class_name,
                "threat_score": entity.threat_score,
                "first_seen": entity.first_seen.isoformat(),
                "last_seen": entity.last_seen.isoformat(),
                "detection_count": len(entity.detections),
                "source_nodes": sorted({item.node_id for item in entity.detections}),
            }
            for entity in self.entities.values()
        ]

    # --------------------------------------------------
    # MAIN PROCESS
    # --------------------------------------------------

    def process_detection(self, detection: Detection) -> Entity:

        self._validate_detection(detection)

        entity = self.find_match(detection)

        if entity is not None:
            self.merge(entity, detection)
        else:
            entity = self.create_entity(detection)

        entity.threat_score = self.calculate_threat_score(entity)

        return entity

    # --------------------------------------------------
    # FIND MATCH
    # --------------------------------------------------

    def find_match(self, detection: Detection) -> Optional[Entity]:

        candidates = []

        for entity in self.entities.values():

            # Class must match
            if entity.class_name != detection.class_name:
                continue

            # Calculate time difference
            time_difference = abs(
                (detection.timestamp - entity.last_seen).total_seconds()
            )

            # Multiple people in one frame share node, GPS, and timestamp.
            # They must remain separate observations instead of collapsing.
            if (
                time_difference == 0
                and entity.detections
                and entity.detections[-1].node_id == detection.node_id
            ):
                continue

            # Outside time window
            if time_difference > self.time_window_s:
                continue

            # Calculate geographic distance
            distance = self.geo_distance(
                entity.lat,
                entity.lon,
                detection.lat,
                detection.lon
            )

            # Match found
            if distance <= self.distance_threshold_m:
                candidates.append((distance, entity.last_seen, entity.entity_id, entity))

        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    def merge(self, entity: Entity, detection: Detection):

        # Keep highest confidence
        entity.confidence = max(
            entity.confidence,
            detection.confidence
        )

        if detection.timestamp >= entity.last_seen:
            entity.lat = detection.lat
            entity.lon = detection.lon
            entity.last_seen = detection.timestamp

        entity.detections.append(detection)

    # --------------------------------------------------
    # CREATE NEW ENTITY
    # --------------------------------------------------

    def create_entity(self, detection: Detection) -> Entity:

        entity_id = f"ENTITY-{self._next_entity_id:04d}"

        self._next_entity_id += 1

        entity = Entity(
            entity_id=entity_id,
            lat=detection.lat,
            lon=detection.lon,
            confidence=detection.confidence,
            class_name=detection.class_name,
            first_seen=detection.timestamp,
            last_seen=detection.timestamp,
            detections=[detection]
        )

        self.entities[entity_id] = entity

        return entity

    # --------------------------------------------------
    # GEO DISTANCE
    # --------------------------------------------------

    @staticmethod
    def geo_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:

        earth_radius = 6371000.0

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)

        delta_lat = lat2_rad - lat1_rad
        delta_lon = radians(lon2 - lon1)

        a = (
            sin(delta_lat / 2) ** 2
            + cos(lat1_rad)
            * cos(lat2_rad)
            * sin(delta_lon / 2) ** 2
        )

        a = max(0.0, min(1.0, a))
        c = 2 * atan2(sqrt(a), sqrt(1.0 - a))

        return earth_radius * c

    # --------------------------------------------------
    # THREAT SCORE
    # --------------------------------------------------

    def calculate_threat_score(self, entity: Entity) -> float:

        score = entity.confidence * 10.0

        return max(0.0, min(10.0, score))

    # --------------------------------------------------
    # STALE ENTITY DETECTION
    # --------------------------------------------------

    def remove_stale_entities(
        self,
        current_time: Optional[datetime] = None
    ):

        if current_time is None:
            current_time = datetime.now()

        stale_entities = []

        for entity_id, entity in self.entities.items():

            age = (
                current_time - entity.last_seen
            ).total_seconds()

            if age > self.stale_after_s:
                stale_entities.append(entity_id)

        for entity_id in stale_entities:
            del self.entities[entity_id]

        return stale_entities

    @staticmethod
    def _coerce_timestamp(timestamp: datetime | float | int | None) -> datetime:
        if timestamp is None:
            return datetime.now()
        if isinstance(timestamp, datetime):
            return timestamp
        timestamp_value = float(timestamp)
        # Accept both Unix seconds and browser Date.now() milliseconds.
        if timestamp_value > 100_000_000_000:
            timestamp_value /= 1000.0
        return datetime.fromtimestamp(timestamp_value)

    @staticmethod
    def _validate_detection(detection: Detection) -> None:
        if not detection.node_id:
            raise ValueError("node_id must not be empty")
        if not detection.class_name:
            raise ValueError("class_name must not be empty")
        if not all(isfinite(value) for value in (detection.lat, detection.lon)):
            raise ValueError("latitude and longitude must be finite")
        if not -90.0 <= detection.lat <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= detection.lon <= 180.0:
            raise ValueError("longitude must be between -180 and 180")
        if not isfinite(detection.confidence) or not 0.0 <= detection.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
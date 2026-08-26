from datetime import datetime, timezone
from threading import RLock
from typing import Callable

from .ConfidenceFusion import fuse_confidence
from .Detection import Detection
from .Entity import Entity
from .EntityManager import EntityManager
from .EntityState import EntityState
from .EventManager import EventManager, FusionEvent
from .Exceptions import InvalidDetectionError
from .FusionConfig import FusionConfig
from .Logger import get_logger
from .Matcher import MatchResult, match_entity
from .PositionFusion import fuse_position
from .ThreatScorer import ThreatScorer


class FusionEngine:
    """Thread-safe, transport-independent multi-sensor entity tracker."""

    def __init__(self, config: FusionConfig | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.config = config or FusionConfig()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.entities = EntityManager()
        self.events = EventManager()
        self.scorer = ThreatScorer(self.config.class_threat_weights)
        self._lock = RLock()
        self._next_id = 1
        self._logger = get_logger()

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def add_detection(self, detection: Detection | dict) -> Entity | None:
        try:
            if isinstance(detection, dict):
                detection = Detection(**detection)
            if not isinstance(detection, Detection):
                raise InvalidDetectionError("detection must be Detection or dict")
        except (TypeError, ValueError, InvalidDetectionError) as exc:
            self.events.emit("INVALID_DETECTION", None, error=str(exc))
            self._logger.warning("Invalid detection rejected: %s", exc)
            return None
        with self._lock:
            self.detect_stale_entities(detection.time)
            result = self.find_matching_entity(detection)
            if result.entity is None:
                entity = self.create_entity(detection)
                self.events.emit("ENTITY_CREATED", entity.entity_id, source_node=detection.node_id)
            else:
                had_source = detection.node_id in result.entity.source_nodes
                entity = self.merge_detection(result.entity, detection)
                self.events.emit("ENTITY_MATCHED", entity.entity_id, source_node=detection.node_id,
                                 match_score=result.score)
                if not had_source:
                    self.events.emit("ENTITY_MERGED", entity.entity_id,
                                     source_node=detection.node_id)
                self.events.emit("ENTITY_UPDATED", entity.entity_id, source_node=detection.node_id)
            self._update_threat(entity)
            return entity

    def find_matching_entity(self, detection: Detection) -> MatchResult:
        with self._lock:
            candidates = [entity for entity in self.entities.get_all_entities()
                          if entity.status != EntityState.EXPIRED]
            return match_entity(detection, candidates, self.config)

    def create_entity(self, detection: Detection) -> Entity:
        entity_id = f"E{self._next_id:04d}"
        self._next_id += 1
        entity = Entity.from_detection(entity_id, detection, self.config.max_history_size)
        self.entities.create_entity(entity)
        return entity

    def merge_detection(self, entity: Entity, detection: Detection) -> Entity:
        with self._lock:
            entity.record(detection)
            lat, lon = fuse_position(entity.observations)
            entity.latitude, entity.longitude = lat, lon
            entity.confidence = fuse_confidence(entity.observations)
            return entity

    def _update_threat(self, entity: Entity) -> float:
        old_level = entity.threat_level
        entity.threat_score = self.calculate_threat_score(entity)
        entity.threat_level = self.scorer.level(entity.threat_score)
        if entity.threat_level != old_level:
            self.events.emit("THREAT_CHANGED", entity.entity_id,
                             threat_score=entity.threat_score,
                             threat_level=entity.threat_level)
        return entity.threat_score

    def calculate_threat_score(self, entity: Entity) -> float:
        return self.scorer.score(entity)

    def detect_stale_entities(self, current_time: datetime | None = None) -> list[Entity]:
        now = current_time or self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        changed = []
        with self._lock:
            candidates = [entity for entity in self.entities.get_all_entities()
                          if entity.status != EntityState.EXPIRED]
            for entity in candidates:
                age = (now - entity.last_seen).total_seconds()
                if age >= self.config.expire_timeout_seconds:
                    if entity.status != EntityState.EXPIRED:
                        entity.status = EntityState.EXPIRED
                        self.events.emit("ENTITY_EXPIRED", entity.entity_id)
                elif age >= self.config.stale_timeout_seconds:
                    if entity.status != EntityState.STALE:
                        entity.status = EntityState.STALE
                        changed.append(entity)
                        self.events.emit("ENTITY_STALE", entity.entity_id)
        return changed

    def cleanup_expired_entities(self) -> list[str]:
        with self._lock:
            expired = [e.entity_id for e in self.entities.get_expired_entities()]
            for entity_id in expired:
                self.entities.remove_entity(entity_id)
            return expired

    def get_entity(self, entity_id: str) -> Entity:
        return self.entities.get_entity(entity_id)

    def get_all_entities(self): return self.entities.get_all_entities()
    def get_active_entities(self): return self.entities.get_active_entities()
    def get_stale_entities(self): return self.entities.get_stale_entities()
    def get_expired_entities(self): return self.entities.get_expired_entities()
    def get_high_threat_entities(self):
        return [e for e in self.get_all_entities() if e.threat_score >= 6.0]

    def subscribe_to_event(self, callback: Callable[[FusionEvent], None]) -> None:
        self.events.subscribe(callback)

    def clear(self) -> None:
        with self._lock:
            self.entities.entities.clear()
            self._next_id = 1

    def to_dict(self) -> list[dict]:
        with self._lock:
            return [entity.to_dict() for entity in self.entities.get_all_entities()]

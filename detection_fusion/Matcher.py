from dataclasses import dataclass

from .Detection import Detection
from .Distance import calculate_distance
from .Entity import Entity
from .FusionConfig import FusionConfig


@dataclass(frozen=True, slots=True)
class MatchResult:
    entity: Entity | None
    score: float


def canonical_class(name: str, config: FusionConfig) -> str:
    value = name.strip().lower()
    return config.class_aliases.get(value, value)


def match_entity(detection: Detection, entities, config: FusionConfig) -> MatchResult:
    best = MatchResult(None, 0.0)
    detection_class = canonical_class(detection.class_name, config)
    total_weight = (config.distance_weight + config.time_weight +
                    config.class_weight + config.confidence_weight)
    for entity in entities:
        if canonical_class(entity.class_name, config) != detection_class:
            continue
        distance = calculate_distance(entity.latitude, entity.longitude, detection.lat, detection.lon)
        time_delta = abs((detection.time - entity.last_seen).total_seconds())
        if distance > config.match_distance_meters or time_delta > config.match_time_window_seconds:
            continue
        distance_score = 1.0 - distance / max(config.match_distance_meters, 1e-9)
        time_score = 1.0 - time_delta / max(config.match_time_window_seconds, 1e-9)
        confidence_score = 1.0 - abs(entity.confidence - detection.confidence)
        score = (config.distance_weight * distance_score + config.time_weight * time_score +
                 config.class_weight + config.confidence_weight * confidence_score) / total_weight
        if score > best.score:
            best = MatchResult(entity, score)
    return best

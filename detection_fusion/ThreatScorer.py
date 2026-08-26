from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class ThreatScorer:
    class_weights: Dict[str, float] = field(default_factory=dict)

    def score(self, entity) -> float:
        class_factor = self.class_weights.get(entity.class_name.lower(), 0.25)
        persistence = min(1.0, entity.detection_count / 5.0)
        sensor_support = min(1.0, len(entity.source_nodes) / 3.0)
        movement = min(1.0, (entity.speed or 0.0) / 20.0)
        raw = (0.45 * class_factor + 0.30 * entity.confidence +
               0.15 * persistence + 0.07 * sensor_support + 0.03 * movement)
        return round(max(0.0, min(10.0, raw * 10.0)), 3)

    @staticmethod
    def level(score: float) -> str:
        if score <= 2:
            return "LOW"
        if score <= 5:
            return "MEDIUM"
        if score <= 8:
            return "HIGH"
        return "CRITICAL"

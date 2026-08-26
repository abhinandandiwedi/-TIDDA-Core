from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class FusionConfig:
    match_distance_meters: float = 50.0
    match_time_window_seconds: float = 10.0
    stale_timeout_seconds: float = 10.0
    expire_timeout_seconds: float = 60.0
    max_history_size: int = 100
    distance_weight: float = 0.45
    time_weight: float = 0.25
    class_weight: float = 0.20
    confidence_weight: float = 0.10
    class_aliases: Dict[str, str] = field(default_factory=dict)
    class_threat_weights: Dict[str, float] = field(default_factory=lambda: {
        "person": 0.55,
        "vehicle": 0.45,
        "drone": 0.75,
    })

    def __post_init__(self) -> None:
        if self.match_distance_meters < 0 or self.match_time_window_seconds < 0:
            raise ValueError("matching thresholds must be non-negative")
        if self.stale_timeout_seconds < 0 or self.expire_timeout_seconds < 0:
            raise ValueError("lifecycle thresholds must be non-negative")
        if self.expire_timeout_seconds < self.stale_timeout_seconds:
            raise ValueError("expire timeout must be >= stale timeout")
        if self.max_history_size < 1:
            raise ValueError("max_history_size must be positive")
        weights = (self.distance_weight, self.time_weight, self.class_weight,
                   self.confidence_weight)
        if any(weight < 0 for weight in weights) or sum(weights) == 0:
            raise ValueError("matching weights must be non-negative and non-zero")

from typing import Iterable

from .Detection import Detection


def fuse_confidence(observations: Iterable[Detection]) -> float:
    """Combine independent confidence values with noisy-OR: 1 - product(1 - p)."""
    result = 0.0
    for item in observations:
        probability = max(0.0, min(1.0, float(item.confidence)))
        result = 1.0 - (1.0 - result) * (1.0 - probability)
    return result

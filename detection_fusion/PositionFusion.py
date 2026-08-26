from typing import Iterable

from .Detection import Detection


def fuse_position(observations: Iterable[Detection]) -> tuple[float, float]:
    """Confidence-weighted latitude/longitude; future filters can replace this component."""
    items = list(observations)
    if not items:
        raise ValueError("at least one observation is required")
    total = sum(max(0.001, float(item.confidence)) for item in items)
    lat = sum(item.lat * max(0.001, item.confidence) for item in items) / total
    lon = sum(item.lon * max(0.001, item.confidence) for item in items) / total
    return lat, lon

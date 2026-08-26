from math import hypot

from .models import BoundingBox, VisionDetection


class CentroidTracker:
    def __init__(self, max_distance_px: float = 80.0) -> None:
        self.max_distance_px = max_distance_px
        self._next_id = 1
        self._tracks: dict[int, tuple[float, float, str]] = {}

    def update(self, detections: list[VisionDetection]) -> list[VisionDetection]:
        available = set(self._tracks)
        updated: list[VisionDetection] = []
        for detection in detections:
            center = detection.bbox.center
            candidates = [(hypot(center[0] - old[0], center[1] - old[1]), track_id)
                          for track_id, old in self._tracks.items()
                          if track_id in available and old[2] == detection.class_name]
            match = min(candidates, default=(float("inf"), None))
            track_id = match[1] if match[0] <= self.max_distance_px else None
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            available.discard(track_id)
            self._tracks[track_id] = (center[0], center[1], detection.class_name)
            updated.append(VisionDetection(detection.class_name, detection.confidence, detection.bbox, track_id, detection.metadata))
        self._tracks = {track_id: value for track_id, value in self._tracks.items() if track_id not in available}
        return updated

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    timestamp: datetime
    image: Any
    width: int
    height: int

    @classmethod
    def create(cls, index: int, image: Any, timestamp: datetime | None = None):
        timestamp = timestamp or datetime.now(timezone.utc)
        height, width = image.shape[:2]
        return cls(index, timestamp, image, width, height)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class VisionDetection:
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Pose:
    rotation: tuple[float, ...]
    translation: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PointCloud:
    points: Any
    colors: Any | None = None
    faces: Any | None = None
    frame_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"points": self.points.tolist(), "colors": self.colors.tolist() if self.colors is not None else None,
                "faces": self.faces.tolist() if self.faces is not None else None, "frame_index": self.frame_index}

from typing import Protocol

from .models import BoundingBox, Frame, VisionDetection


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[VisionDetection]: ...


class YOLODetector:
    """Ultralytics adapter. The dependency and model load remain lazy."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.45) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model = None

    def _load(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLODetector requires ultralytics") from exc
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: Frame) -> list[VisionDetection]:
        results = self._load()(frame.image, conf=self.confidence, verbose=False)
        detections: list[VisionDetection] = []
        names = results[0].names
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            detections.append(VisionDetection(str(names[int(class_id)]), float(confidence),
                                               BoundingBox(*box)))
        return detections

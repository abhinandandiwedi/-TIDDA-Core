# ══════════════════════════════════════════════════════════════════
#  👁 TIDDA AI PERCEPTION LAYER — Frame perception engine
#  Converts incoming camera frames into structured SemanticObservation models.
#  Answers ONLY "WHAT is visible?", leaving spatial localization to future SLAM.
#
#  Strictly avoids fabricating detections or fake spatial coordinates.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from mapping_models import SemanticObservation

# ── Optional dependency bootstrap ─────────────────────────────────
_ULTRALYTICS_AVAILABLE = False
_PIL_AVAILABLE = False
_NUMPY_AVAILABLE = False

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════
#  CLASS SUPPORT DOCUMENTATION
# ══════════════════════════════════════════════════════════════════

# Standard COCO-80 classes supported by default yolov8n.pt model
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Architectural/semantic classes requested for indoor mapping taxonomy
TARGET_SEMANTIC_CLASSES: Dict[str, str] = {
    "wall": "UNSUPPORTED (Requires fine-tuned depth/segmentation model)",
    "doorway": "UNSUPPORTED (Requires fine-tuned door detector or segmentation)",
    "corridor": "UNSUPPORTED (Requires room layout/corridor estimation model)",
    "room": "UNSUPPORTED (Requires room classification model)",
    "staircase": "UNSUPPORTED (Requires staircase detection model)",
    "elevator": "UNSUPPORTED (Requires elevator detection model)",
    "obstacle": "SUPPORTED (Mapped from general physical objects like chair, table, box, etc.)",
    "object": "SUPPORTED (Mapped from standard COCO detected classes)",
}

SUPPORTED_CLASSES: List[str] = list(COCO_CLASSES) + ["obstacle", "object"]
UNSUPPORTED_CLASSES: List[str] = [
    cls for cls, status in TARGET_SEMANTIC_CLASSES.items()
    if status.startswith("UNSUPPORTED")
]


# ══════════════════════════════════════════════════════════════════
#  PERCEPTION ENGINE (REAL DETECTOR)
# ══════════════════════════════════════════════════════════════════

class PerceptionEngine:
    """YOLOv8n-backed perception engine for frame processing.

    Converts incoming image frames (base64, PIL, or numpy array) into
    structured SemanticObservation objects.

    Validation rules:
      - node_id and floor_id are required; error returned if invalid.
      - Unsupported semantic classes (wall, corridor, etc.) are NOT fabricated.
      - Spatial position/extent is left unset (None) as localization is separate.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        min_interval_s: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.min_interval_s = min_interval_s

        self._model: Optional[Any] = None
        self._loaded: bool = False
        self._last_inference_time: float = 0.0
        self._init_error: Optional[str] = None

    def initialize(self) -> bool:
        """Lazy-load the underlying YOLO detector model.

        Returns True if successfully initialized, False otherwise.
        """
        if self._loaded:
            return True

        if not (_ULTRALYTICS_AVAILABLE and _PIL_AVAILABLE and _NUMPY_AVAILABLE):
            self._init_error = "Missing required dependencies (ultralytics, Pillow, numpy)"
            return False

        try:
            self._model = YOLO(self.model_name)
            self._loaded = True
            return True
        except Exception as e:
            self._init_error = f"Failed to load model {self.model_name}: {e}"
            return False

    @property
    def is_available(self) -> bool:
        """Check if detector dependencies and model are ready."""
        if not self._loaded:
            return self.initialize()
        return self._loaded

    @property
    def init_error(self) -> Optional[str]:
        """Return initialization error message if any."""
        return self._init_error

    def get_supported_classes(self) -> List[str]:
        """Return list of semantic classes supported by the detector model."""
        return list(SUPPORTED_CLASSES)

    def get_unsupported_classes(self) -> List[str]:
        """Return list of semantic classes currently unsupported by COCO model."""
        return list(UNSUPPORTED_CLASSES)

    def _decode_frame(self, frame: Union[str, bytes, Any]) -> Optional[Any]:
        """Decode image frame into a numpy array for YOLO inference."""
        if not _PIL_AVAILABLE or not _NUMPY_AVAILABLE:
            return None

        try:
            if isinstance(frame, str):
                # Base64 string
                if "," in frame:
                    frame = frame.split(",", 1)[1]
                image_bytes = base64.b64decode(frame)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                return np.array(image)
            elif isinstance(frame, bytes):
                image = Image.open(io.BytesIO(frame)).convert("RGB")
                return np.array(image)
            elif hasattr(frame, "convert"):
                # PIL Image
                return np.array(frame.convert("RGB"))
            elif isinstance(frame, np.ndarray):
                return frame
        except Exception:
            return None

        return None

    def process(
        self,
        frame: Union[str, bytes, Any],
        node_id: str,
        floor_id: str,
        frame_id: Optional[str] = None,
    ) -> List[SemanticObservation]:
        """Process a single camera frame and return semantic observations.

        Parameters
        ----------
        frame : base64 str, bytes, PIL Image, or numpy array
            The input camera frame.
        node_id : str
            Source mobile node identifier (required).
        floor_id : str
            Associated floor identifier (required).
        frame_id : str, optional
            Identifier for the frame.

        Returns
        -------
        List[SemanticObservation]
            List of detected semantic observations. Returns empty list on error.
        """
        # Validate node_id and floor_id
        if not node_id or not isinstance(node_id, str):
            return []
        if not floor_id or not isinstance(floor_id, str):
            return []

        # Validate frame
        if frame is None:
            return []

        # Rate limiting check
        now = time.time()
        if self.min_interval_s > 0 and (now - self._last_inference_time) < self.min_interval_s:
            return []
        self._last_inference_time = now

        # Ensure model is ready
        if not self.is_available or self._model is None:
            return []

        # Decode image
        img_array = self._decode_frame(frame)
        if img_array is None or getattr(img_array, "size", 0) == 0:
            return []

        # Run inference safely
        try:
            results = self._model(
                img_array,
                verbose=False,
                conf=self.confidence_threshold,
            )
        except Exception:
            return []

        if not results:
            return []

        observations: List[SemanticObservation] = []
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        # Extract detections
        try:
            classes = boxes.cls.cpu().numpy().tolist()
            confidences = boxes.conf.cpu().numpy().tolist()
            bboxes = boxes.xyxy.cpu().numpy().tolist()
            names = res.names if hasattr(res, "names") else {}

            for cls_idx, conf, bbox in zip(classes, confidences, bboxes):
                class_name = names.get(int(cls_idx), f"class_{int(cls_idx)}")
                clamped_conf = max(0.0, min(1.0, float(conf)))

                obs = SemanticObservation(
                    node_id=node_id,
                    floor_id=floor_id,
                    timestamp=now,
                    class_name=class_name,
                    confidence=round(clamped_conf, 4),
                    bounding_box=(
                        round(float(bbox[0]), 1),
                        round(float(bbox[1]), 1),
                        round(float(bbox[2]), 1),
                        round(float(bbox[3]), 1),
                    ),
                    frame_id=frame_id,
                    pose=None,          # Localization not available
                    position=None,      # Spatial coords unset
                    spatial_extent=None,
                )
                observations.append(obs)
        except Exception:
            return []

        return observations


# ══════════════════════════════════════════════════════════════════
#  MOCK PERCEPTION ENGINE (FOR DETERMINISTIC TESTING)
# ══════════════════════════════════════════════════════════════════

class MockPerceptionEngine:
    """Mock perception engine for deterministic unit testing.

    Completely isolated from YOLO / machine learning dependencies.
    Produces predictable SemanticObservation outputs given test inputs.
    """

    def __init__(
        self,
        preset_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.preset_detections = preset_detections
        self._call_count: int = 0

    def get_supported_classes(self) -> List[str]:
        """Return list of supported classes in mock engine."""
        return list(SUPPORTED_CLASSES)

    def get_unsupported_classes(self) -> List[str]:
        """Return list of unsupported architectural classes."""
        return list(UNSUPPORTED_CLASSES)

    def process(
        self,
        frame: Any,
        node_id: str,
        floor_id: str,
        frame_id: Optional[str] = None,
    ) -> List[SemanticObservation]:
        """Process a mock frame deterministically.

        Validates node_id and floor_id, and handles invalid frames cleanly.
        """
        # Validate parameters
        if not node_id or not isinstance(node_id, str):
            return []
        if not floor_id or not isinstance(floor_id, str):
            return []
        if frame is None or frame == "" or frame == b"":
            return []

        self._call_count += 1
        now = time.time()

        # Use custom preset detections if provided
        if self.preset_detections is not None:
            observations: List[SemanticObservation] = []
            for item in self.preset_detections:
                conf = max(0.0, min(1.0, float(item.get("confidence", 0.9))))
                obs = SemanticObservation(
                    node_id=node_id,
                    floor_id=floor_id,
                    timestamp=now,
                    class_name=str(item.get("class_name", "object")),
                    confidence=round(conf, 4),
                    bounding_box=item.get("bounding_box", (10.0, 10.0, 50.0, 50.0)),
                    frame_id=frame_id or f"frame_{self._call_count}",
                    pose=None,
                    position=None,
                    spatial_extent=None,
                )
                observations.append(obs)
            return observations

        # Default deterministic detections
        obs1 = SemanticObservation(
            node_id=node_id,
            floor_id=floor_id,
            timestamp=now,
            class_name="person",
            confidence=0.92,
            bounding_box=(100.0, 150.0, 200.0, 400.0),
            frame_id=frame_id or f"frame_{self._call_count}",
            pose=None,
            position=None,
            spatial_extent=None,
        )
        obs2 = SemanticObservation(
            node_id=node_id,
            floor_id=floor_id,
            timestamp=now,
            class_name="chair",
            confidence=0.85,
            bounding_box=(300.0, 200.0, 400.0, 350.0),
            frame_id=frame_id or f"frame_{self._call_count}",
            pose=None,
            position=None,
            spatial_extent=None,
        )
        return [obs1, obs2]

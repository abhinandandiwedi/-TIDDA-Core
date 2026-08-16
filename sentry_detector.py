# ══════════════════════════════════════════════════════════════════
#  🛡 TIDDA SENTRY DETECTOR — YOLOv8n person detection for Sentry Mode
#  Wraps ultralytics YOLOv8n with rate-limiting and async support.
#  Completely isolated from drone physics — used by mobile nodes only.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import base64
import io
import sys
import time
from typing import Optional

# ── Dependency bootstrap (same pattern as main module) ────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[SYSTEM] ultralytics not found — installing...")
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ultralytics"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from ultralytics import YOLO

try:
    import numpy as np
except ImportError:
    print("[SYSTEM] numpy not found — installing...")
    import subprocess as _sp
    _sp.check_call(
        [sys.executable, "-m", "pip", "install", "numpy"],
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )
    import numpy as np

try:
    from PIL import Image
except ImportError:
    print("[SYSTEM] Pillow not found — installing...")
    import subprocess as _sp2
    _sp2.check_call(
        [sys.executable, "-m", "pip", "install", "Pillow"],
        stdout=_sp2.DEVNULL,
        stderr=_sp2.DEVNULL,
    )
    from PIL import Image


# ── Configuration ─────────────────────────────────────────────────
YOLO_MODEL: str = "yolov8n.pt"               # Auto-downloaded on first use (~6MB)
PERSON_CLASS_ID: int = 0                       # COCO class 0 = "person"
CONFIDENCE_THRESHOLD: float = 0.45             # Minimum confidence to trigger alert
MIN_INFERENCE_INTERVAL_S: float = 0.5          # Rate limit: max 2 FPS inference


class SentryDetector:
    """YOLOv8n-based person detector for sentry-mode phone nodes.

    Features:
    - Lazy model loading (only loads YOLO weights on first detection call)
    - Rate limiting to prevent inference backlog on CPU
    - Returns structured detection results for broadcasting

    Usage (synchronous — wrap in asyncio.to_thread for async):
        detector = SentryDetector()
        result = detector.detect_persons(base64_jpeg_string)
        if result:
            # Person detected! result has confidence, count, bbox
            broadcast_alert(result)
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        min_interval_s: float = MIN_INFERENCE_INTERVAL_S,
    ) -> None:
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._min_interval_s = min_interval_s

        self._model: Optional[YOLO] = None
        self._last_inference_time: float = 0.0
        self._loaded = False

    def _ensure_model(self) -> None:
        """Lazy-load the YOLO model on first use."""
        if not self._loaded:
            print(f"[SENTRY] Loading {self._model_name}...")
            self._model = YOLO(self._model_name)
            self._loaded = True
            print(f"[SENTRY] Model loaded — ready for inference")

    def detect_persons(self, jpeg_base64: str) -> Optional[dict]:
        """Run YOLOv8n inference on a base64-encoded JPEG frame.

        Parameters
        ----------
        jpeg_base64 : str
            Base64-encoded JPEG image data (without the data:image/... prefix).

        Returns
        -------
        dict or None
            If one or more persons detected above the confidence threshold:
            {
                "confidence": 0.87,       # highest person confidence
                "person_count": 2,        # number of persons detected
                "bboxes": [[x1,y1,x2,y2], ...],  # bounding boxes (pixels)
            }
            Returns None if no person detected or rate-limited.
        """
        # ── Rate limiter ──────────────────────────────────────────
        now = time.time()
        if now - self._last_inference_time < self._min_interval_s:
            return None  # Skip this frame — too soon
        self._last_inference_time = now

        # ── Lazy model load ───────────────────────────────────────
        self._ensure_model()
        assert self._model is not None

        # ── Decode JPEG ───────────────────────────────────────────
        try:
            jpeg_bytes = base64.b64decode(jpeg_base64)
            image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            img_array = np.array(image)
        except Exception as e:
            print(f"[SENTRY] Frame decode error: {e}")
            return None

        # ── Run inference ─────────────────────────────────────────
        try:
            results = self._model(
                img_array,
                verbose=False,
                classes=[PERSON_CLASS_ID],   # Filter to person class only
                conf=self._confidence_threshold,
            )
        except Exception as e:
            print(f"[SENTRY] Inference error: {e}")
            return None

        # ── Extract person detections ─────────────────────────────
        if not results or len(results) == 0:
            return None

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return None

        # Filter for person class (should already be filtered, but belt-and-suspenders)
        person_mask = boxes.cls == PERSON_CLASS_ID
        person_boxes = boxes[person_mask]

        if len(person_boxes) == 0:
            return None

        # Build response
        confidences = person_boxes.conf.cpu().numpy().tolist()
        bboxes = person_boxes.xyxy.cpu().numpy().tolist()

        return {
            "confidence": round(max(confidences), 3),
            "person_count": len(person_boxes),
            "bboxes": [[round(c, 1) for c in box] for box in bboxes],
        }

    @property
    def is_loaded(self) -> bool:
        """Whether the YOLO model has been loaded."""
        return self._loaded

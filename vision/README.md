# AI Drone Vision Pipeline

`vision/` provides the camera-to-fusion pipeline requested by TIDDA:

`FrameExtractor -> quality -> Detector -> CentroidTracker -> calibration/pose -> DepthEstimator -> PointCloud -> optional mesh -> geolocation -> fusion.FusionEngine -> dashboard payload`.

## Install

```powershell
pip install -r vision/requirements.txt
```

OpenCV, NumPy, and Ultralytics are optional at import time and required only by the corresponding adapters. `YOLODetector` lazily loads `yolov8n.pt`; `MonocularDepthEstimator` is an approximate relative-depth fallback for a normal single camera. It is not metric-accurate. Replace it with `Depth-Anything`/MiDaS or `StereoDepthEstimator` when calibrated depth is required.

## Example

```python
from vision import VisionConfig, VisionPipeline
from vision.detector import YOLODetector
from vision.frame_extractor import FrameExtractor

pipeline = VisionPipeline(YOLODetector("yolov8n.pt"), config=VisionConfig())
for frame in FrameExtractor("mission.mp4", sample_fps=5).frames():
    result = pipeline.process(frame, "drone-01", 26.8467, 80.9462)
    print(frame.index, len(result["detections"]), len(result["point_cloud"].points))
```

`dashboard.frame_payload(result)` produces a transport-neutral JSON payload for the existing WebSocket dashboard. Camera calibration uses a pinhole model; `estimate_pose()` wraps OpenCV `solvePnP`; `pixel_to_ground()` intersects a camera ray with a ground plane and `local_to_gps()` converts local ENU offsets to latitude/longitude.

Point clouds contain XYZ points and optional RGB colors. `point_cloud_to_mesh()` creates a grid mesh when the sampled depth is regular. The core does not depend on a database, MAVLink, WebSocket, or dashboard implementation.

Run tests with `python -m pytest vision/tests -q`. Tests requiring NumPy skip cleanly when the optional dependency is absent.

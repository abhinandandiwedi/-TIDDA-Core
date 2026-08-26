import numpy as np

from vision import VisionConfig, VisionPipeline
from vision.detector import YOLODetector
from vision.frame_extractor import FrameExtractor


pipeline = VisionPipeline(YOLODetector("yolov8n.pt"), config=VisionConfig(point_stride=8))
for frame in FrameExtractor("mission.mp4", sample_fps=2).frames(stop=10):
    result = pipeline.process(frame, "drone-01", 26.8467, 80.9462)
    print(frame.index, len(result["detections"]), len(result["point_cloud"].points))

from pydantic import BaseModel
from typing import Optional

class StartReconstructionRequest(BaseModel):
    video_path: str
    every_n: int = 5
    blur_threshold: float = 40.0
    feature_method: Optional[str] = "sift"

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class PointCloudResponse(BaseModel):
    url: str

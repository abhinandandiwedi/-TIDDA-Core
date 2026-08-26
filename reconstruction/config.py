from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReconstructionConfig:
    workspace: Path
    input_path: Path
    frame_fps: float = 2.0
    colmap_executable: str = "colmap"
    ffmpeg_executable: str = "ffmpeg"
    nerfstudio_executable: str = "ns-process-data"
    nstrain_executable: str = "ns-train"
    gaussian_repo: Path | None = None
    use_cuda: bool = True
    database_name: str = "database.db"
    camera_model: str = "OPENCV"
    matcher: str = "exhaustive_matcher"

    @property
    def frames_dir(self) -> Path: return self.workspace / "frames"
    @property
    def colmap_dir(self) -> Path: return self.workspace / "colmap"
    @property
    def sparse_dir(self) -> Path: return self.workspace / "sparse"
    @property
    def staged_input_dir(self) -> Path: return self.workspace / "input"
    @property
    def nerfstudio_dir(self) -> Path: return self.workspace / "nerfstudio"
    @property
    def output_dir(self) -> Path: return self.workspace / "output"

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"input does not exist: {self.input_path}")
        if self.frame_fps <= 0:
            raise ValueError("frame_fps must be positive")
        if self.matcher not in {"exhaustive_matcher", "sequential_matcher", "spatial_matcher"}:
            raise ValueError("unsupported COLMAP matcher")

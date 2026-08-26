import logging
import shutil

from .colmap import run_colmap
from .config import ReconstructionConfig
from .ffmpeg import extract_frames
from .gaussian import run_gaussian
from .gpu import check_gpu, require_cuda
from .input import classify_input
from .nerfstudio import prepare_nerfstudio, train_nerfstudio
from .runner import CommandRunner


class ReconstructionPipeline:
    def __init__(self, config: ReconstructionConfig, dry_run: bool = False) -> None:
        self.config = config
        self.runner = CommandRunner(dry_run=dry_run, logger=logging.getLogger("reconstruction"))
        self.logger = logging.getLogger("reconstruction")

    def prepare(self) -> dict:
        self.config.validate()
        self.config.workspace.mkdir(parents=True, exist_ok=True)
        kind = classify_input(self.config.input_path)
        if self.config.use_cuda:
            gpu = require_cuda()
        else:
            gpu = check_gpu()
        if kind == "video":
            frames = extract_frames(self.config, self.runner)
        else:
            self.config.frames_dir.mkdir(parents=True, exist_ok=True)
            for image in self.config.input_path.iterdir():
                if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    shutil.copy2(image, self.config.frames_dir / image.name)
            frames = self.config.frames_dir
        self.config.staged_input_dir.mkdir(parents=True, exist_ok=True)
        for image in frames.iterdir():
            if image.is_file():
                shutil.copy2(image, self.config.staged_input_dir / image.name)
        return {"input_type": kind, "frames": frames, "gpu": gpu}

    def run_colmap(self):
        return run_colmap(self.config, self.runner)

    def run_nerfstudio(self, train: bool = False):
        dataset = prepare_nerfstudio(self.config, self.runner)
        if train:
            train_nerfstudio(self.config, self.runner)
        return dataset

    def run_gaussian_splatting(self):
        return run_gaussian(self.config, self.runner)

    def run(self, method: str = "nerfstudio", train: bool = False) -> dict:
        state = self.prepare()
        sparse = self.run_colmap()
        state["sparse"] = sparse
        if method == "nerfstudio":
            state["dataset"] = self.run_nerfstudio(train=train)
        elif method == "gaussian":
            self.run_gaussian_splatting()
        else:
            raise ValueError("method must be nerfstudio or gaussian")
        return state

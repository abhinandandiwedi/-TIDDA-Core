from .config import ReconstructionConfig
from .runner import CommandRunner


def prepare_nerfstudio(config: ReconstructionConfig, runner: CommandRunner):
    runner.require_tool(config.nerfstudio_executable)
    config.nerfstudio_dir.parent.mkdir(parents=True, exist_ok=True)
    runner.run([config.nerfstudio_executable, "images", "--data", str(config.frames_dir), "--output-dir", str(config.nerfstudio_dir), "--matching-method", config.matcher.replace("_matcher", "")])
    return config.nerfstudio_dir


def train_nerfstudio(config: ReconstructionConfig, runner: CommandRunner, method: str = "nerfacto"):
    runner.require_tool(config.nstrain_executable)
    runner.run([config.nstrain_executable, method, "--data", str(config.nerfstudio_dir)])

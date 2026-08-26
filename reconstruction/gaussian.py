from pathlib import Path

from .config import ReconstructionConfig
from .runner import CommandRunner


def gaussian_commands(config: ReconstructionConfig) -> list[list[str]]:
    if config.gaussian_repo is None:
        raise ValueError("gaussian_repo must point to a cloned official gaussian-splatting repository")
    convert = config.gaussian_repo / "convert.py"
    train = config.gaussian_repo / "train.py"
    return [["python", str(convert), "-s", str(config.workspace)],
            ["python", str(train), "-s", str(config.workspace), "-m", str(config.output_dir / "gaussian_model")]]


def run_gaussian(config: ReconstructionConfig, runner: CommandRunner):
    for command in gaussian_commands(config):
        runner.run(command, cwd=config.gaussian_repo)

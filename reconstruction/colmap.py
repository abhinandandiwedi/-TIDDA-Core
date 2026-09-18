from pathlib import Path

from .config import ReconstructionConfig
from .runner import CommandRunner


def run_colmap(config: ReconstructionConfig, runner: CommandRunner) -> Path:
    runner.require_tool(config.colmap_executable)
    config.colmap_dir.mkdir(parents=True, exist_ok=True)
    config.sparse_dir.mkdir(parents=True, exist_ok=True)
    database = config.colmap_dir / config.database_name
    images = config.frames_dir
    feature_cmd = [config.colmap_executable, "feature_extractor", "--database_path", str(database), "--image_path", str(images), "--ImageReader.camera_model", config.camera_model]
    match_cmd = [config.colmap_executable, config.matcher, "--database_path", str(database)]
    if not config.use_cuda:
        feature_cmd.extend(["--FeatureExtraction.use_gpu", "0"])
        match_cmd.extend(["--FeatureMatching.use_gpu", "0"])
    
    runner.run(feature_cmd)
    runner.run(match_cmd)
    runner.run([config.colmap_executable, "mapper", "--database_path", str(database), "--image_path", str(images), "--output_path", str(config.sparse_dir)])
    return config.sparse_dir

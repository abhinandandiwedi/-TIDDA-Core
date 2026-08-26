from pathlib import Path

import pytest

from reconstruction.config import ReconstructionConfig
from reconstruction.gpu import GPUInfo
from reconstruction.input import classify_input
from reconstruction.pipeline import ReconstructionPipeline


def test_classify_image_folder(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"not decoded")
    assert classify_input(tmp_path) == "images"


def test_dry_run_cpu_image_pipeline(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"image")
    config = ReconstructionConfig(tmp_path / "work", tmp_path, use_cuda=False)
    pipeline = ReconstructionPipeline(config, dry_run=True)
    state = pipeline.prepare()
    assert state["input_type"] == "images"


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReconstructionConfig(tmp_path / "work", tmp_path / "missing").validate()

from dataclasses import dataclass
import shutil
import subprocess


@dataclass(frozen=True, slots=True)
class GPUInfo:
    nvidia_smi: bool
    torch_installed: bool
    cuda_available: bool
    torch_cuda_version: str | None
    driver_summary: str | None

    @property
    def ready(self) -> bool:
        return self.cuda_available and self.torch_installed


def check_gpu() -> GPUInfo:
    nvidia_smi = shutil.which("nvidia-smi") is not None
    driver_summary = None
    if nvidia_smi:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True)
            driver_summary = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            driver_summary = None
    try:
        import torch
    except ImportError:
        return GPUInfo(nvidia_smi, False, False, None, driver_summary)
    return GPUInfo(nvidia_smi, True, bool(torch.cuda.is_available()), torch.version.cuda, driver_summary)


def require_cuda() -> GPUInfo:
    info = check_gpu()
    if not info.ready:
        raise RuntimeError("CUDA is unavailable: install a CUDA-compatible PyTorch build and NVIDIA driver before GPU stages")
    return info

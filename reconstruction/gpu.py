from dataclasses import dataclass
import shutil
import subprocess


@dataclass(frozen=True, slots=True)
class GPUInfo:
    """Hardware-agnostic GPU information.

    Supports NVIDIA (CUDA), AMD (ROCm/HIP), and CPU-only environments.
    """
    nvidia_smi: bool
    rocm_smi: bool
    torch_installed: bool
    cuda_available: bool
    torch_cuda_version: str | None
    hip_version: str | None
    driver_summary: str | None
    accelerator: str  # "cuda", "rocm", or "cpu"

    @property
    def ready(self) -> bool:
        """True if any GPU accelerator (CUDA or ROCm/HIP) is available."""
        return self.torch_installed and self.cuda_available


def check_gpu() -> GPUInfo:
    """Detect available GPU acceleration — NVIDIA, AMD ROCm, or CPU fallback.

    PyTorch's ``torch.cuda.is_available()`` returns True for both CUDA and
    ROCm/HIP builds, so we use it as the unified accelerator check.  The
    ``nvidia-smi`` / ``rocm-smi`` probes are cosmetic metadata only.
    """
    # ── Probe system tools (cosmetic / driver info) ───────────────
    has_nvidia_smi = shutil.which("nvidia-smi") is not None
    has_rocm_smi = shutil.which("rocm-smi") is not None

    driver_summary: str | None = None
    if has_nvidia_smi:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            )
            driver_summary = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    elif has_rocm_smi:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, check=True,
            )
            driver_summary = result.stdout.strip()[:200]
        except (OSError, subprocess.CalledProcessError):
            pass

    # ── Probe PyTorch ─────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        return GPUInfo(
            nvidia_smi=has_nvidia_smi, rocm_smi=has_rocm_smi,
            torch_installed=False, cuda_available=False,
            torch_cuda_version=None, hip_version=None,
            driver_summary=driver_summary, accelerator="cpu",
        )

    cuda_available = bool(torch.cuda.is_available())
    torch_cuda_ver: str | None = getattr(torch.version, "cuda", None)
    hip_ver: str | None = getattr(torch.version, "hip", None)

    # Determine accelerator label
    if hip_ver:
        accel = "rocm"
    elif cuda_available:
        accel = "cuda"
    else:
        accel = "cpu"

    return GPUInfo(
        nvidia_smi=has_nvidia_smi, rocm_smi=has_rocm_smi,
        torch_installed=True, cuda_available=cuda_available,
        torch_cuda_version=torch_cuda_ver, hip_version=hip_ver,
        driver_summary=driver_summary, accelerator=accel,
    )


def require_gpu() -> GPUInfo:
    """Ensure a GPU accelerator is available, or raise RuntimeError."""
    info = check_gpu()
    if not info.ready:
        raise RuntimeError(
            "No GPU accelerator available.  Install a CUDA or ROCm-compatible "
            "PyTorch build and the appropriate driver before running GPU stages."
        )
    return info


# Backwards-compatible alias
require_cuda = require_gpu

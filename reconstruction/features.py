from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
import subprocess

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLMAP = str(REPO_ROOT / "colmap_hip_build" / "build" / "src" / "colmap" / "exe" / "colmap")
COLMAP_EXEC = os.environ.get("COLMAP_EXEC", DEFAULT_COLMAP)

# Ensure ROCm / HIP libraries are discovered
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.0/lib"
elif "/opt/rocm/core-10.0/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/rocm/core-10.0/lib:{os.environ['LD_LIBRARY_PATH']}"


@dataclass
class FeatureResult:
    keypoints: np.ndarray  # Shape (N, 2) float32 coordinates (x, y)
    descriptors: np.ndarray  # Shape (N, D) float32 or uint8
    image_size: Tuple[int, int]  # (width, height)
    feature_count: int
    method: str
    scores: Optional[np.ndarray] = None  # Shape (N,) float32
    scales: Optional[np.ndarray] = None  # Shape (N,) float32
    orientations: Optional[np.ndarray] = None  # Shape (N,) float32
    device: str = "cpu"


_ALIKED_EXTRACTORS: Dict[str, object] = {}


def get_torch_device(prefer_gpu: bool = True) -> "torch.device":
    import torch
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_aliked_model(device: str = "cuda", max_num_keypoints: int = 2048):
    """Lazy loader and cacher for ALIKED PyTorch model."""
    try:
        import torch
        from lightglue import ALIKED

        dev = torch.device(device)
        key = f"{dev.type}_{max_num_keypoints}"
        if key not in _ALIKED_EXTRACTORS:
            # Enable experimental ROCm attention optimization if running on AMD GPU
            if dev.type == "cuda" and getattr(torch.version, "hip", None):
                os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
            model = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(dev)
            _ALIKED_EXTRACTORS[key] = model
        return _ALIKED_EXTRACTORS[key]
    except Exception as e:
        logger.warning(f"Failed to initialize ALIKED model on {device}: {e}")
        return None


def extract_image_features(
    image: Union[np.ndarray, str, Path],
    method: str = "sift",
    max_num_keypoints: int = 2048,
    device: Optional[str] = None,
    prefer_gpu: bool = True,
) -> FeatureResult:
    """
    Modular feature extractor supporting SIFT and ALIKED.
    Handles numpy arrays (BGR / grayscale) and file paths.
    Gracefully falls back to SIFT if ALIKED is unavailable or fails.
    """
    method_normalized = method.lower().strip()

    # Load image if file path
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image at: {img_path}")
    elif isinstance(image, np.ndarray):
        img_bgr = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if img_bgr.ndim == 2:
        h, w = img_bgr.shape
        img_gray = img_bgr
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    elif img_bgr.ndim == 3:
        h, w = img_bgr.shape[:2]
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Invalid image dimensions: {img_bgr.shape}")

    # Check for empty / zero-size images
    if h == 0 or w == 0:
        return FeatureResult(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
            image_size=(0, 0),
            feature_count=0,
            method=method_normalized,
            device="cpu",
        )

    # ── ALIKED Extractor ──────────────────────────────────────────
    if method_normalized in ("aliked", "aliked_n16", "aliked_n16rot"):
        target_device = device or ("cuda" if prefer_gpu else "cpu")
        try:
            import torch
            from lightglue.utils import rbd

            if target_device == "cuda" and not torch.cuda.is_available():
                target_device = "cpu"

            model = get_aliked_model(device=target_device, max_num_keypoints=max_num_keypoints)
            if model is not None:
                # Prepare tensor (1, 3, H, W) normalized to [0, 1]
                tensor_img = (
                    torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                ).to(torch.device(target_device))

                with torch.no_grad():
                    feats = model.extract(tensor_img)
                    feats = rbd(feats)

                kpts = feats["keypoints"].detach().cpu().numpy().astype(np.float32)
                descs = feats["descriptors"].detach().cpu().numpy().astype(np.float32)
                scores = feats.get("keypoint_scores")
                scores_np = scores.detach().cpu().numpy().astype(np.float32) if scores is not None else None

                return FeatureResult(
                    keypoints=kpts,
                    descriptors=descs,
                    scores=scores_np,
                    image_size=(w, h),
                    feature_count=len(kpts),
                    method="aliked",
                    device=target_device,
                )
        except Exception as e:
            logger.warning(f"ALIKED extraction failed ({e}), falling back to SIFT.")

    # ── SIFT Extractor (Default / Fallback) ────────────────────────
    sift = cv2.SIFT_create(nfeatures=max_num_keypoints)
    kps, descs = sift.detectAndCompute(img_gray, None)

    if kps and len(kps) > 0:
        kpts = np.array([kp.pt for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)
        scales = np.array([kp.size for kp in kps], dtype=np.float32)
        orientations = np.array([kp.angle for kp in kps], dtype=np.float32)
        descs_out = descs.astype(np.float32) if descs is not None else np.empty((0, 128), dtype=np.float32)
    else:
        kpts = np.empty((0, 2), dtype=np.float32)
        descs_out = np.empty((0, 128), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        scales = np.empty((0,), dtype=np.float32)
        orientations = np.empty((0,), dtype=np.float32)

    return FeatureResult(
        keypoints=kpts,
        descriptors=descs_out,
        scores=scores,
        scales=scales,
        orientations=orientations,
        image_size=(w, h),
        feature_count=len(kpts),
        method="sift",
        device="cpu",
    )


def extract_colmap_features(
    database_path: Union[str, Path],
    images_path: Union[str, Path],
    method: str = "sift",
    camera_model: str = "OPENCV",
    single_camera: int = 1,
    num_threads: int = 8,
    use_gpu: int = 0,
):
    """
    Runs COLMAP CLI feature_extractor. Supports SIFT and ALIKED_N16ROT.
    """
    database_path = Path(database_path)
    images_path = Path(images_path)

    colmap_type = "SIFT"
    if method.lower() in ("aliked", "aliked_n16", "aliked_n16rot"):
        colmap_type = "ALIKED_N16ROT"

    cmd = [
        COLMAP_EXEC,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_path),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        str(single_camera),
        "--FeatureExtraction.type",
        colmap_type,
        "--FeatureExtraction.use_gpu",
        str(use_gpu),
        "--FeatureExtraction.num_threads",
        str(num_threads),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_features(
    target_or_database: Union[np.ndarray, str, Path],
    images_path: Optional[Union[str, Path]] = None,
    method: str = "sift",
    **kwargs,
) -> Union[FeatureResult, None]:
    """
    Universal entrypoint:
    - If called with (database_path, images_path): runs COLMAP feature_extractor
    - If called with (image, method=...): runs Python modular feature extraction
    """
    # Check if this is a COLMAP database call
    if images_path is not None or (
        isinstance(target_or_database, (str, Path))
        and str(target_or_database).endswith(".db")
    ):
        if images_path is None:
            raise ValueError("images_path required when database_path is provided")
        extract_colmap_features(target_or_database, images_path, method=method, **kwargs)
        return None

    # Image-level extraction
    return extract_image_features(target_or_database, method=method, **kwargs)

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
import subprocess

import cv2
import numpy as np

from .features import FeatureResult

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLMAP = str(REPO_ROOT / "colmap_hip_build" / "build" / "src" / "colmap" / "exe" / "colmap")
COLMAP_EXEC = os.environ.get("COLMAP_EXEC", DEFAULT_COLMAP)

if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.0/lib"
elif "/opt/rocm/core-10.0/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/rocm/core-10.0/lib:{os.environ['LD_LIBRARY_PATH']}"


@dataclass
class MatchResult:
    matches: np.ndarray  # Shape (M, 2) indices: [idx_in_a, idx_in_b]
    scores: np.ndarray  # Shape (M,) match confidence / scores
    raw_matches_count: int
    filtered_matches_count: int
    method: str
    device: str = "cpu"


@dataclass
class GeometryResult:
    raw_matches_count: int
    valid_matches_count: int
    geometric_inliers_count: int
    inlier_ratio: float
    inlier_matches: np.ndarray  # Shape (K, 2)
    inlier_mask: np.ndarray  # Shape (M,) boolean mask
    matrix: Optional[np.ndarray] = None  # Fundamental (3x3) or Essential (3x3) matrix
    method: str = "fundamental"
    is_valid: bool = False


_LIGHTGLUE_MATCHERS: Dict[str, object] = {}


def get_lightglue_matcher(feature_type: str = "aliked", device: str = "cuda"):
    """Lazy loader and cacher for LightGlue matcher."""
    try:
        import torch
        from lightglue import LightGlue

        dev = torch.device(device)
        key = f"{feature_type}_{dev.type}"
        if key not in _LIGHTGLUE_MATCHERS:
            if dev.type == "cuda" and getattr(torch.version, "hip", None):
                os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
            matcher = LightGlue(features=feature_type).eval().to(dev)
            _LIGHTGLUE_MATCHERS[key] = matcher
        return _LIGHTGLUE_MATCHERS[key]
    except Exception as e:
        logger.warning(f"Failed to load LightGlue matcher ({feature_type}) on {device}: {e}")
        return None


def match_image_features(
    features_a: FeatureResult,
    features_b: FeatureResult,
    method: str = "lightglue",
    ratio_threshold: float = 0.8,
    min_confidence: float = 0.1,
    device: Optional[str] = None,
    prefer_gpu: bool = True,
) -> MatchResult:
    """
    Modular matcher supporting LightGlue and standard SIFT (Lowe's ratio test).
    Robust to 0 or insufficient features.
    """
    method_normalized = method.lower().strip()

    # Edge cases: empty or insufficient features
    if (
        features_a is None
        or features_b is None
        or features_a.feature_count < 2
        or features_b.feature_count < 2
    ):
        return MatchResult(
            matches=np.empty((0, 2), dtype=np.int32),
            scores=np.empty((0,), dtype=np.float32),
            raw_matches_count=0,
            filtered_matches_count=0,
            method=method_normalized,
            device="cpu",
        )

    # ── LightGlue Matcher ──────────────────────────────────────────
    if "lightglue" in method_normalized or features_a.method == "aliked":
        target_device = device or ("cuda" if prefer_gpu else "cpu")
        try:
            import torch
            from lightglue.utils import rbd

            if target_device == "cuda" and not torch.cuda.is_available():
                target_device = "cpu"

            feat_name = "aliked" if features_a.method == "aliked" else "sift"
            matcher = get_lightglue_matcher(feature_type=feat_name, device=target_device)

            if matcher is not None:
                dev = torch.device(target_device)
                kpts0 = torch.from_numpy(features_a.keypoints).unsqueeze(0).to(dev)
                desc0 = torch.from_numpy(features_a.descriptors).unsqueeze(0).to(dev)
                kpts1 = torch.from_numpy(features_b.keypoints).unsqueeze(0).to(dev)
                desc1 = torch.from_numpy(features_b.descriptors).unsqueeze(0).to(dev)

                data = {
                    "image0": {
                        "keypoints": kpts0,
                        "descriptors": desc0,
                        "image_size": torch.tensor(features_a.image_size, device=dev).unsqueeze(0),
                    },
                    "image1": {
                        "keypoints": kpts1,
                        "descriptors": desc1,
                        "image_size": torch.tensor(features_b.image_size, device=dev).unsqueeze(0),
                    },
                }

                with torch.no_grad():
                    out = matcher(data)
                    out = rbd(out)

                raw_m = out["matches"].detach().cpu().numpy().astype(np.int32)
                raw_s = out["scores"].detach().cpu().numpy().astype(np.float32)

                # Filter by confidence if applicable
                valid_mask = raw_s >= min_confidence
                filtered_m = raw_m[valid_mask]
                filtered_s = raw_s[valid_mask]

                return MatchResult(
                    matches=filtered_m,
                    scores=filtered_s,
                    raw_matches_count=len(raw_m),
                    filtered_matches_count=len(filtered_m),
                    method="lightglue",
                    device=target_device,
                )
        except Exception as e:
            logger.warning(f"LightGlue matching failed ({e}), falling back to brute-force matching.")

    # ── SIFT / Brute-force Matcher (Fallback) ──────────────────────
    desc_a = features_a.descriptors.astype(np.float32)
    desc_b = features_b.descriptors.astype(np.float32)

    # Use cv2.NORM_L2
    bf = cv2.BFMatcher(cv2.NORM_L2)
    try:
        knn = bf.knnMatch(desc_a, desc_b, k=2)
    except Exception as e:
        logger.warning(f"knnMatch failed: {e}")
        return MatchResult(
            matches=np.empty((0, 2), dtype=np.int32),
            scores=np.empty((0,), dtype=np.float32),
            raw_matches_count=0,
            filtered_matches_count=0,
            method="sift_bf",
            device="cpu",
        )

    raw_count = len(knn)
    good_matches = []
    scores_list = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append([m.queryIdx, m.trainIdx])
                # Invert distance for a proxy confidence score in [0, 1]
                conf = 1.0 / (1.0 + m.distance)
                scores_list.append(conf)
        elif len(pair) == 1:
            m = pair[0]
            good_matches.append([m.queryIdx, m.trainIdx])
            scores_list.append(1.0 / (1.0 + m.distance))

    matches_arr = np.array(good_matches, dtype=np.int32) if good_matches else np.empty((0, 2), dtype=np.int32)
    scores_arr = np.array(scores_list, dtype=np.float32) if scores_list else np.empty((0,), dtype=np.float32)

    return MatchResult(
        matches=matches_arr,
        scores=scores_arr,
        raw_matches_count=raw_count,
        filtered_matches_count=len(matches_arr),
        method="sift_bf",
        device="cpu",
    )


def verify_geometry(
    features_a: FeatureResult,
    features_b: FeatureResult,
    match_result: MatchResult,
    method: str = "fundamental",
    ransac_threshold: float = 3.0,
    confidence: float = 0.99,
    min_inliers: int = 15,
    camera_matrix: Optional[np.ndarray] = None,
) -> GeometryResult:
    """
    Validates correspondences via 2D epipolar geometry (Fundamental or Essential matrix RANSAC).
    Computes exact inlier count and inlier ratio.
    """
    matches = match_result.matches
    num_matches = len(matches)

    if num_matches < 8:
        return GeometryResult(
            raw_matches_count=match_result.raw_matches_count,
            valid_matches_count=num_matches,
            geometric_inliers_count=0,
            inlier_ratio=0.0,
            inlier_matches=np.empty((0, 2), dtype=np.int32),
            inlier_mask=np.zeros(num_matches, dtype=bool),
            matrix=None,
            method=method,
            is_valid=False,
        )

    pts_a = features_a.keypoints[matches[:, 0]]
    pts_b = features_b.keypoints[matches[:, 1]]

    matrix = None
    mask = None

    if method == "essential" and camera_matrix is not None:
        matrix, mask = cv2.findEssentialMat(
            pts_a,
            pts_b,
            cameraMatrix=camera_matrix,
            method=cv2.RANSAC,
            prob=confidence,
            threshold=ransac_threshold,
        )
    else:
        matrix, mask = cv2.findFundamentalMat(
            pts_a,
            pts_b,
            cv2.FM_RANSAC,
            ransac_threshold,
            confidence,
        )

    if mask is not None:
        mask_bool = mask.ravel().astype(bool)
        inlier_count = int(mask_bool.sum())
        inlier_matches = matches[mask_bool]
        inlier_ratio = float(inlier_count / num_matches) if num_matches > 0 else 0.0
        is_valid = inlier_count >= min_inliers
    else:
        mask_bool = np.zeros(num_matches, dtype=bool)
        inlier_count = 0
        inlier_matches = np.empty((0, 2), dtype=np.int32)
        inlier_ratio = 0.0
        is_valid = False

    return GeometryResult(
        raw_matches_count=match_result.raw_matches_count,
        valid_matches_count=num_matches,
        geometric_inliers_count=inlier_count,
        inlier_ratio=inlier_ratio,
        inlier_matches=inlier_matches,
        inlier_mask=mask_bool,
        matrix=matrix,
        method=method,
        is_valid=is_valid,
    )


def match_colmap_database(
    database_path: Union[str, Path],
    method: str = "sift",
    use_gpu: int = 0,
    num_threads: int = 8,
):
    """
    Runs COLMAP exhaustive_matcher with SIFT_BRUTEFORCE or ALIKED_LIGHTGLUE.
    """
    database_path = Path(database_path)
    colmap_type = "SIFT_BRUTEFORCE"
    if method.lower() in ("aliked", "lightglue", "aliked_lightglue"):
        colmap_type = "ALIKED_LIGHTGLUE"

    cmd = [
        COLMAP_EXEC,
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.type",
        colmap_type,
        "--FeatureMatching.use_gpu",
        str(use_gpu),
        "--FeatureMatching.num_threads",
        str(num_threads),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def match_features(
    target_or_database: Union[FeatureResult, str, Path],
    features_b: Optional[FeatureResult] = None,
    method: str = "lightglue",
    **kwargs,
) -> Union[MatchResult, None]:
    """
    Universal entrypoint:
    - If called with (database_path, method=...): runs COLMAP exhaustive_matcher
    - If called with (features_a, features_b, method=...): runs Python modular feature matcher
    """
    if features_b is None or (
        isinstance(target_or_database, (str, Path))
        and str(target_or_database).endswith(".db")
    ):
        match_colmap_database(target_or_database, method=method, **kwargs)
        return None

    return match_image_features(target_or_database, features_b, method=method, **kwargs)

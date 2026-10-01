#!/usr/bin/env python3
"""
TIDDA-Core Phase 7 Test Suite: Advanced Perception & Feature Pipeline
Tests:
  A. SIFT extraction
  B. ALIKED extraction
  C. SIFT matching
  D. LightGlue matching
  E. 2D Geometric verification (RANSAC / Fundamental)
  F. No-feature failure handling (blank / black image)
  G. No-match failure handling (uncorrelated noise)
  H. CPU fallback
  I. ROCm / HIP execution
  J. Native vs Upscaled experiment
  K. Existing pipeline regression
"""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure project root in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reconstruction.features import (
    FeatureResult,
    extract_features,
    extract_image_features,
    get_aliked_model,
)
from reconstruction.matching import (
    GeometryResult,
    MatchResult,
    get_lightglue_matcher,
    match_features,
    match_image_features,
    verify_geometry,
)


def load_test_frames(workspace_path: Path):
    frames_dir = workspace_path / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames in {frames_dir}, found {len(frames)}")
    return frames[0], frames[1]


def test_a_sift_extraction(frame_path: Path):
    print("\n--- Test A: SIFT Extraction ---")
    res = extract_features(frame_path, method="sift")
    assert isinstance(res, FeatureResult), "Result must be a FeatureResult instance"
    assert res.method == "sift"
    assert res.feature_count > 0, "SIFT should detect features on real frame"
    assert res.keypoints.shape == (res.feature_count, 2)
    assert res.descriptors.shape[0] == res.feature_count
    assert res.descriptors.shape[1] == 128
    print(f"  [PASS] SIFT extracted {res.feature_count} keypoints from {frame_path.name}")
    return res


def test_b_aliked_extraction(frame_path: Path):
    print("\n--- Test B: ALIKED Extraction ---")
    res = extract_features(frame_path, method="aliked")
    assert isinstance(res, FeatureResult), "Result must be a FeatureResult instance"
    assert res.method == "aliked"
    assert res.feature_count > 0, "ALIKED should detect features on real frame"
    assert res.keypoints.shape == (res.feature_count, 2)
    assert res.descriptors.shape[0] == res.feature_count
    assert res.descriptors.shape[1] == 128
    print(f"  [PASS] ALIKED extracted {res.feature_count} keypoints from {frame_path.name} (device={res.device})")
    return res


def test_c_sift_matching(f0_sift: FeatureResult, f1_sift: FeatureResult):
    print("\n--- Test C: SIFT Matching ---")
    matches = match_features(f0_sift, f1_sift, method="sift")
    assert isinstance(matches, MatchResult), "Result must be a MatchResult instance"
    assert matches.method == "sift_bf"
    assert matches.filtered_matches_count > 0, "SIFT matching should produce correspondences"
    assert matches.matches.shape == (matches.filtered_matches_count, 2)
    assert len(matches.scores) == matches.filtered_matches_count
    print(f"  [PASS] SIFT matches: {matches.filtered_matches_count} good matches (from {matches.raw_matches_count} raw)")
    return matches


def test_d_lightglue_matching(f0_aliked: FeatureResult, f1_aliked: FeatureResult):
    print("\n--- Test D: LightGlue Matching ---")
    matches = match_features(f0_aliked, f1_aliked, method="lightglue")
    assert isinstance(matches, MatchResult), "Result must be a MatchResult instance"
    assert matches.method == "lightglue"
    assert matches.filtered_matches_count > 0, "LightGlue should produce correspondences"
    assert matches.matches.shape == (matches.filtered_matches_count, 2)
    assert len(matches.scores) == matches.filtered_matches_count
    print(f"  [PASS] LightGlue matches: {matches.filtered_matches_count} matches (device={matches.device})")
    return matches


def test_e_geometric_verification(f0: FeatureResult, f1: FeatureResult, matches: MatchResult):
    print("\n--- Test E: 2D Geometric Verification (RANSAC) ---")
    geom = verify_geometry(f0, f1, matches, method="fundamental", ransac_threshold=3.0)
    assert isinstance(geom, GeometryResult), "Result must be a GeometryResult instance"
    assert geom.is_valid, "Epipolar geometry should be valid for adjacent frames"
    assert geom.geometric_inliers_count > 15, f"Expected >15 inliers, got {geom.geometric_inliers_count}"
    assert 0.0 <= geom.inlier_ratio <= 1.0, f"Inlier ratio out of bounds: {geom.inlier_ratio}"
    assert geom.matrix is not None and geom.matrix.shape == (3, 3)
    print(f"  [PASS] Inliers: {geom.geometric_inliers_count}/{geom.valid_matches_count} ({geom.inlier_ratio*100:.2f}%)")
    return geom


def test_f_no_feature_failure():
    print("\n--- Test F: No-Feature Failure Handling ---")
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)

    # SIFT on blank
    res_sift = extract_image_features(black_img, method="sift")
    assert res_sift.feature_count == 0 or len(res_sift.keypoints) == res_sift.feature_count

    # ALIKED on blank
    res_aliked = extract_image_features(black_img, method="aliked")
    assert res_aliked.feature_count == 0 or len(res_aliked.keypoints) == res_aliked.feature_count

    # Match empty
    matches = match_image_features(res_sift, res_sift, method="sift")
    assert matches.filtered_matches_count == 0
    assert matches.matches.shape == (0, 2)

    geom = verify_geometry(res_sift, res_sift, matches)
    assert not geom.is_valid
    assert geom.geometric_inliers_count == 0
    print("  [PASS] Graceful handling of zero-feature inputs verified")


def test_g_no_match_failure():
    print("\n--- Test G: No-Match Failure Handling ---")
    # Two independent noise images
    rng = np.random.RandomState(42)
    noise_a = rng.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    rng_b = np.random.RandomState(99)
    noise_b = rng_b.randint(0, 256, (300, 300, 3), dtype=np.uint8)

    feats_a = extract_image_features(noise_a, method="sift")
    feats_b = extract_image_features(noise_b, method="sift")

    matches = match_image_features(feats_a, feats_b, method="sift")
    geom = verify_geometry(feats_a, feats_b, matches, min_inliers=50)
    assert not geom.is_valid, "Random noise should not produce valid epipolar geometry"
    print(f"  [PASS] Uncorrelated input rejected safely (inliers={geom.geometric_inliers_count}, valid={geom.is_valid})")


def test_h_cpu_fallback(frame_path: Path):
    print("\n--- Test H: CPU Fallback Execution ---")
    t0 = time.perf_counter()
    res = extract_image_features(frame_path, method="aliked", device="cpu", prefer_gpu=False)
    t_extract = time.perf_counter() - t0
    assert res.device == "cpu"
    assert res.feature_count > 0

    m = match_image_features(res, res, method="lightglue", device="cpu", prefer_gpu=False)
    assert m.device == "cpu"
    print(f"  [PASS] CPU fallback operational: ALIKED extracted {res.feature_count} kpts in {t_extract:.2f}s")


def test_i_rocm_execution(frame_path: Path):
    print("\n--- Test I: ROCm / HIP Execution ---")
    import torch
    hip_ver = getattr(torch.version, "hip", None)
    cuda_avail = torch.cuda.is_available()
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA/ROCm available: {cuda_avail}")
    print(f"  HIP version: {hip_ver}")
    if cuda_avail:
        dev_name = torch.cuda.get_device_name(0)
        print(f"  Device Name: {dev_name}")
        t0 = time.perf_counter()
        res = extract_image_features(frame_path, method="aliked", device="cuda")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        assert res.device == "cuda"
        assert res.feature_count > 0
        print(f"  [PASS] ROCm GPU execution verified: {res.feature_count} kpts in {t_gpu*1000:.1f}ms on {dev_name}")
    else:
        print("  [SKIP] ROCm GPU not detected; skipped GPU execution.")


def test_j_native_vs_upscale(frame0_path: Path, frame1_path: Path):
    print("\n--- Test J: Native vs Upscaling Experiment ---")
    img0 = cv2.imread(str(frame0_path))
    img1 = cv2.imread(str(frame1_path))
    h, w = img0.shape[:2]

    # Native
    t0 = time.perf_counter()
    f0_nat = extract_image_features(img0, method="aliked")
    f1_nat = extract_image_features(img1, method="aliked")
    m_nat = match_image_features(f0_nat, f1_nat, method="lightglue")
    g_nat = verify_geometry(f0_nat, f1_nat, m_nat)
    t_nat = time.perf_counter() - t0

    # 2x Upscaled
    t0 = time.perf_counter()
    img0_up = cv2.resize(img0, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    img1_up = cv2.resize(img1, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    f0_up = extract_image_features(img0_up, method="aliked")
    f1_up = extract_image_features(img1_up, method="aliked")
    m_up = match_image_features(f0_up, f1_up, method="lightglue")
    g_up = verify_geometry(f0_up, f1_up, m_up)
    t_up = time.perf_counter() - t0

    print(f"  Native ({w}x{h}): {f0_nat.feature_count} kpts, {m_nat.filtered_matches_count} matches, {g_nat.geometric_inliers_count} inliers ({g_nat.inlier_ratio*100:.1f}%), {t_nat*1000:.1f}ms")
    print(f"  2x Upscaled ({w*2}x{h*2}): {f0_up.feature_count} kpts, {m_up.filtered_matches_count} matches, {g_up.geometric_inliers_count} inliers ({g_up.inlier_ratio*100:.1f}%), {t_up*1000:.1f}ms")
    print(f"  [PASS] Native vs Upscale measured successfully")


def test_k_existing_pipeline_regression():
    print("\n--- Test K: Existing Pipeline Regression ---")
    from reconstruction.colmap_runner import run_colmap_pipeline
    import inspect

    sig = inspect.signature(run_colmap_pipeline)
    assert "frames_dir" in sig.parameters
    assert "feature_method" in sig.parameters

    from reconstruction.features import extract_colmap_features
    from reconstruction.matching import match_colmap_database
    assert callable(extract_colmap_features)
    assert callable(match_colmap_database)

    # Check that calling extract_features and match_features with colmap signature works
    sig_feat = inspect.signature(extract_features)
    assert "target_or_database" in sig_feat.parameters
    assert "images_path" in sig_feat.parameters

    print("  [PASS] Existing reconstruction API backwards-compatibility verified")


def run_all_tests(workspace: str = "workspace_house_v1_demo"):
    print("==================================================")
    print(" TIDDA-CORE PHASE 7 PERCEPTION / FEATURE TEST SUITE")
    print("==================================================")
    ws_path = Path(workspace)
    f0, f1 = load_test_frames(ws_path)
    print(f"Loaded test frames:\n  Frame 0: {f0}\n  Frame 1: {f1}")

    f0_sift = test_a_sift_extraction(f0)
    f1_sift = test_a_sift_extraction(f1)
    f0_aliked = test_b_aliked_extraction(f0)
    f1_aliked = test_b_aliked_extraction(f1)

    m_sift = test_c_sift_matching(f0_sift, f1_sift)
    m_aliked = test_d_lightglue_matching(f0_aliked, f1_aliked)

    test_e_geometric_verification(f0_sift, f1_sift, m_sift)
    test_e_geometric_verification(f0_aliked, f1_aliked, m_aliked)

    test_f_no_feature_failure()
    test_g_no_match_failure()
    test_h_cpu_fallback(f0)
    test_i_rocm_execution(f0)
    test_j_native_vs_upscale(f0, f1)
    test_k_existing_pipeline_regression()

    print("\n" + "=" * 50)
    print("✅ ALL PHASE 7 FEATURE & PERCEPTION TESTS PASSED")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIDDA Phase 7 Feature Tests")
    parser.add_argument("--workspace", type=str, default="workspace_house_v1_demo", help="Path to workspace with frames/")
    args = parser.parse_args()
    run_all_tests(args.workspace)

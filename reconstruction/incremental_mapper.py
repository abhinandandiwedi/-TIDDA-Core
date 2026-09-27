# ══════════════════════════════════════════════════════════════════
#  🗺 TIDDA PHASE 4 — Incremental 3D Mapper
#  Real-time incremental mapping from live phone camera frames.
#
#  Pipeline:
#    ingest_frame → keyframe_select → feature_track → pose_estimate
#    → depth_estimate → integrate → temporal_fuse → map_snapshot
#
#  Uses real ORB features, real Essential matrix pose estimation,
#  real depth (monocular DPT or COLMAP dense), and voxel fusion.
#
#  Does NOT replace COLMAP. Does NOT fake depth or poses.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import base64
import io
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("reconstruction.incremental_mapper")


# ══════════════════════════════════════════════════════════════════
#  TRACKING STATUS ENUM
# ══════════════════════════════════════════════════════════════════

class TrackingStatus:
    POSE_OK = "POSE_OK"
    INSUFFICIENT_FEATURES = "INSUFFICIENT_FEATURES"
    MOTION_ESTIMATION_FAILED = "MOTION_ESTIMATION_FAILED"
    TRACK_LOST = "TRACK_LOST"
    DEPTH_UNAVAILABLE = "DEPTH_UNAVAILABLE"
    INVALID_FRAME = "INVALID_FRAME"
    NO_KEYFRAME = "NO_KEYFRAME"
    MAPPER_RESET = "MAPPER_RESET"
    FRAME_SKIPPED = "FRAME_SKIPPED"


# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class MapPoint:
    """A single 3D point in the incremental map."""
    point_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rgb: np.ndarray = field(default_factory=lambda: np.array([128, 128, 128], dtype=np.uint8))
    observation_count: int = 1
    source_frame: str = ""
    confidence: float = 1.0
    last_seen: float = field(default_factory=time.time)

    def fuse(self, other_xyz: np.ndarray, other_rgb: np.ndarray) -> None:
        """Running-average fusion with another observation of the same point."""
        n = self.observation_count
        self.xyz = (self.xyz * n + other_xyz) / (n + 1)
        self.rgb = ((self.rgb.astype(np.float64) * n + other_rgb.astype(np.float64)) / (n + 1)).astype(np.uint8)
        self.observation_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        self.last_seen = time.time()


@dataclass
class CameraKeyframe:
    """A keyframe with its estimated camera pose."""
    frame_id: str = ""
    timestamp: float = 0.0
    source_node: str = ""
    # 4x4 world-from-camera transform (identity = origin)
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    # Camera intrinsic matrix 3x3
    intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))
    num_tracked_features: int = 0
    tracking_status: str = TrackingStatus.NO_KEYFRAME
    # Grayscale image used for tracking (kept in memory for matching)
    gray_image: Optional[np.ndarray] = None
    # ORB keypoints and descriptors
    keypoints: Optional[Any] = None
    descriptors: Optional[np.ndarray] = None


@dataclass
class MapperStats:
    """Live statistics for the mapping pipeline."""
    frames_received: int = 0
    frames_processed: int = 0
    keyframes_selected: int = 0
    pose_ok_count: int = 0
    pose_fail_count: int = 0
    points_inserted: int = 0
    points_fused: int = 0
    total_map_points: int = 0
    avg_processing_ms: float = 0.0
    max_processing_ms: float = 0.0
    last_update_time: float = 0.0
    tracking_failures: int = 0
    _processing_times: list = field(default_factory=list)

    def record_processing_time(self, ms: float) -> None:
        self._processing_times.append(ms)
        if len(self._processing_times) > 200:
            self._processing_times = self._processing_times[-200:]
        self.avg_processing_ms = sum(self._processing_times) / len(self._processing_times)
        self.max_processing_ms = max(self.max_processing_ms, ms)
        self.last_update_time = time.time()

    def to_dict(self) -> dict:
        return {
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "keyframes_selected": self.keyframes_selected,
            "pose_ok": self.pose_ok_count,
            "pose_fail": self.pose_fail_count,
            "points_inserted": self.points_inserted,
            "points_fused": self.points_fused,
            "total_map_points": self.total_map_points,
            "avg_processing_ms": round(self.avg_processing_ms, 1),
            "max_processing_ms": round(self.max_processing_ms, 1),
            "last_update_time": self.last_update_time,
            "tracking_failures": self.tracking_failures,
            "consecutive_failures": getattr(self, "consecutive_failures", 0),
            "successful_recoveries": getattr(self, "successful_recoveries", 0),
        }


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

@dataclass
class MapperConfig:
    """Configurable parameters for the incremental mapper."""
    # Frame sampling
    max_processing_fps: float = 3.0              # Max frames processed per second
    keyframe_min_interval_s: float = 0.3          # Min time between keyframes
    keyframe_motion_threshold: float = 15.0       # Min pixel displacement for keyframe
    keyframe_feature_threshold: int = 30          # Min features required for keyframe

    # Feature detection (Phase 4.5)
    orb_num_features: int = 1000                  # ORB features to detect
    orb_scale_factor: float = 1.2
    orb_levels: int = 8
    match_ratio_threshold: float = 0.80           # Lowe's ratio test threshold
    match_cross_check: bool = True
    min_matches_for_pose: int = 15                # Min good matches for Essential matrix
    
    # RANSAC parameters
    ransac_threshold: float = 2.0                 # RANSAC reprojection error threshold
    ransac_probability: float = 0.999

    # Depth estimation
    default_focal_ratio: float = 0.9              # focal = max(w,h) * ratio when unknown
    depth_sample_step: int = 8                    # Pixel sampling step for depth→3D
    min_valid_depth: float = 0.05                 # Min depth in meters
    max_valid_depth: float = 50.0                 # Max depth in meters

    # Map limits
    max_map_points: int = 500_000                 # Hard cap on total map points
    voxel_size: float = 0.02                      # Voxel grid cell size (meters)
    voxel_fusion_radius: float = 0.03             # Radius for point fusion (meters)

    # Queue
    max_queue_size: int = 5                       # Bounded frame queue


# ══════════════════════════════════════════════════════════════════
#  VOXEL GRID — Spatial index for temporal fusion
# ══════════════════════════════════════════════════════════════════

class VoxelGrid:
    """Simple hash-based voxel grid for fast nearest-point lookup and fusion."""

    def __init__(self, voxel_size: float = 0.02, fusion_radius: float = 0.03):
        self.voxel_size = voxel_size
        self.fusion_radius = fusion_radius
        # key: (vx, vy, vz) → MapPoint
        self._grid: Dict[Tuple[int, int, int], MapPoint] = {}
        self._lock = threading.Lock()

    def _voxel_key(self, xyz: np.ndarray) -> Tuple[int, int, int]:
        return (
            int(np.floor(xyz[0] / self.voxel_size)),
            int(np.floor(xyz[1] / self.voxel_size)),
            int(np.floor(xyz[2] / self.voxel_size)),
        )

    def insert_or_fuse(self, xyz: np.ndarray, rgb: np.ndarray, source_frame: str) -> Tuple[bool, MapPoint]:
        """Insert a new point or fuse with an existing one in the same voxel.

        Returns (is_new, point).
        """
        key = self._voxel_key(xyz)
        with self._lock:
            if key in self._grid:
                existing = self._grid[key]
                existing.fuse(xyz, rgb)
                return False, existing
            else:
                pt = MapPoint(
                    xyz=xyz.copy(),
                    rgb=rgb.copy(),
                    source_frame=source_frame,
                )
                self._grid[key] = pt
                return True, pt

    def get_all_points(self) -> List[MapPoint]:
        with self._lock:
            return list(self._grid.values())

    def get_points_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Nx3 positions, Nx3 colors) arrays."""
        with self._lock:
            if not self._grid:
                return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
            pts = list(self._grid.values())
        positions = np.array([p.xyz for p in pts])
        colors = np.array([p.rgb for p in pts])
        return positions, colors

    def count(self) -> int:
        with self._lock:
            return len(self._grid)

    def clear(self) -> None:
        with self._lock:
            self._grid.clear()


# ══════════════════════════════════════════════════════════════════
#  INCREMENTAL MAPPER
# ══════════════════════════════════════════════════════════════════

class IncrementalMapper:
    """Real-time incremental 3D mapper from live phone camera frames.

    Thread-safe — designed to be called from an async context via
    asyncio.to_thread() or from a background worker thread.

    Pipeline per frame:
      1. Decode JPEG → grayscale
      2. Keyframe selection (time + motion threshold)
      3. ORB feature detection + BFMatcher matching
      4. Essential matrix + recoverPose → relative camera motion
      5. Depth estimation (monocular DPT or projected reconstruction)
      6. 3D point back-projection with valid depth + pose
      7. Voxel grid insertion/fusion
    """

    def __init__(self, config: Optional[MapperConfig] = None):
        self.config = config or MapperConfig()
        self.stats = MapperStats()
        self._lock = threading.Lock()

        # ORB detector (CPU-friendly, no GPU needed)
        self._orb = cv2.ORB_create(
            nfeatures=self.config.orb_num_features,
            scaleFactor=self.config.orb_scale_factor,
            nlevels=self.config.orb_levels,
        )
        # Brute-force matcher with Hamming distance for ORB
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=self.config.match_cross_check)

        # Map state
        self._voxel_grid = VoxelGrid(
            voxel_size=self.config.voxel_size,
            fusion_radius=self.config.voxel_fusion_radius,
        )
        self._keyframes: List[CameraKeyframe] = []
        self._current_pose: np.ndarray = np.eye(4)  # World-from-camera

        # Frame processing state
        self._last_process_time: float = 0.0
        self._last_keyframe_time: float = 0.0
        self._frame_counter: int = 0

        # Previous keyframe data for tracking
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kps: Optional[Any] = None
        self._prev_descs: Optional[np.ndarray] = None

        # Depth estimator (lazy-loaded)
        self._depth_estimator = None
        self._depth_estimator_failed = False

        # Mapping active flag
        self._active = False

        # New points buffer for incremental dashboard updates
        self._new_points_buffer: deque = deque(maxlen=5000)
        self._new_points_lock = threading.Lock()

        logger.info("IncrementalMapper initialized with config: max_fps=%.1f, voxel=%.3f",
                     self.config.max_processing_fps, self.config.voxel_size)

    # ── Public API ────────────────────────────────────────────────

    def start(self) -> None:
        """Activate mapping."""
        self._active = True
        logger.info("Incremental mapping STARTED")

    def stop(self) -> None:
        """Deactivate mapping (preserves map state)."""
        self._active = False
        logger.info("Incremental mapping STOPPED")

    @property
    def is_active(self) -> bool:
        return self._active

    def ingest_frame(
        self,
        frame_b64: str,
        node_id: str,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process a single camera frame (base64 JPEG).

        Returns a status dict with tracking info.
        This is the MAIN ENTRY POINT called from the WebSocket handler.
        """
        ts = timestamp or time.time()
        self._frame_counter += 1
        frame_id = f"{node_id}_{self._frame_counter}"

        with self._lock:
            self.stats.frames_received += 1

        if not self._active:
            return self._status_result(frame_id, node_id, ts, TrackingStatus.FRAME_SKIPPED,
                                       "Mapper not active")

        # ── Rate limiting ─────────────────────────────────────────
        min_interval = 1.0 / max(0.1, self.config.max_processing_fps)
        now = time.time()
        if (now - self._last_process_time) < min_interval:
            return self._status_result(frame_id, node_id, ts, TrackingStatus.FRAME_SKIPPED,
                                       "Rate limited")

        proc_start = time.time()

        # ── Decode JPEG ───────────────────────────────────────────
        try:
            img_bgr = self._decode_frame(frame_b64)
        except Exception as e:
            logger.warning("Frame decode failed: %s", e)
            return self._status_result(frame_id, node_id, ts, TrackingStatus.INVALID_FRAME,
                                       str(e))

        if img_bgr is None or img_bgr.size == 0:
            return self._status_result(frame_id, node_id, ts, TrackingStatus.INVALID_FRAME,
                                       "Empty frame")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # ── Feature detection ─────────────────────────────────────
        kps, descs = self._orb.detectAndCompute(gray, None)
        num_features = len(kps) if kps is not None else 0

        if num_features < self.config.keyframe_feature_threshold:
            with self._lock:
                self.stats.frames_processed += 1
                self.stats.tracking_failures += 1
            return self._status_result(frame_id, node_id, ts,
                                       TrackingStatus.INSUFFICIENT_FEATURES,
                                       f"Only {num_features} features detected")

        # ── Keyframe selection ────────────────────────────────────
        is_keyframe = self._select_keyframe(gray, kps, ts)
        if not is_keyframe:
            self._last_process_time = time.time()
            with self._lock:
                self.stats.frames_processed += 1
            return self._status_result(frame_id, node_id, ts, TrackingStatus.NO_KEYFRAME,
                                       "Below keyframe threshold")

        # ── Pose estimation ───────────────────────────────────────
        tracking_status = TrackingStatus.POSE_OK
        relative_R = np.eye(3)
        relative_t = np.zeros((3, 1))
        matched_pts1 = None
        matched_pts2 = None
        pose_mask = None

        pose_quality = 0.0
        pose_source = "NONE"

        if self._prev_descs is not None and descs is not None:
            pose_result = self._estimate_motion(
                self._prev_kps, self._prev_descs,
                kps, descs, w, h
            )
            if pose_result is not None:
                relative_R, relative_t, num_inliers, matched_pts1, matched_pts2, pose_mask, pose_quality = pose_result
                tracking_status = TrackingStatus.POSE_OK
                pose_source = "ESSENTIAL_MATRIX"
                self.consecutive_failures = 0
            else:
                tracking_status = TrackingStatus.MOTION_ESTIMATION_FAILED
                pose_source = "NONE"
                self.consecutive_failures = getattr(self, "consecutive_failures", 0) + 1
                with self._lock:
                    self.stats.pose_fail_count += 1
                    self.stats.tracking_failures += 1
        else:
            # First keyframe — no previous to match against
            tracking_status = TrackingStatus.POSE_OK
            pose_source = "INITIALIZATION"
            pose_quality = 1.0

        # ── Update cumulative pose ────────────────────────────────
        if tracking_status == TrackingStatus.POSE_OK:
            # Build 4x4 transform from relative R, t
            rel_transform = np.eye(4)
            rel_transform[:3, :3] = relative_R
            rel_transform[:3, 3] = relative_t.ravel()
            self._current_pose = self._current_pose @ rel_transform

            with self._lock:
                self.stats.pose_ok_count += 1

        # ── 3D Point Generation ───────────────────────────────────
        # Primary: feature triangulation from two-view geometry
        # This uses the matched inlier feature points + the relative pose
        # to triangulate real 3D points. No depth model needed.
        if (tracking_status == TrackingStatus.POSE_OK
                and matched_pts1 is not None
                and matched_pts2 is not None
                and pose_mask is not None):
            points_new, points_fused = self._triangulate_and_integrate(
                img_bgr, matched_pts1, matched_pts2, pose_mask,
                relative_R, relative_t, self._current_pose,
                w, h, frame_id
            )
            with self._lock:
                self.stats.points_inserted += points_new
                self.stats.points_fused += points_fused
                self.stats.total_map_points = self._voxel_grid.count()

        # ── Store keyframe ────────────────────────────────────────
        kf = CameraKeyframe(
            frame_id=frame_id,
            timestamp=ts,
            source_node=node_id,
            pose=self._current_pose.copy(),
            intrinsics=self._make_intrinsics(w, h),
            num_tracked_features=num_features,
            tracking_status=tracking_status,
        )
        self._keyframes.append(kf)
        with self._lock:
            self.stats.keyframes_selected += 1

        # ── Update previous frame data for next match ─────────────
        self._prev_gray = gray
        self._prev_kps = kps
        self._prev_descs = descs
        self._last_keyframe_time = ts
        self._last_process_time = time.time()

        with self._lock:
            self.stats.frames_processed += 1

        proc_ms = (time.time() - proc_start) * 1000.0
        self.stats.record_processing_time(proc_ms)

        logger.info(
            "Frame %s: status=%s features=%d points_total=%d proc=%.1fms",
            frame_id, tracking_status, num_features,
            self._voxel_grid.count(), proc_ms
        )

        return self._status_result(
            frame_id, node_id, ts, tracking_status,
            f"features={num_features}",
            points_added=self.stats.points_inserted,
            points_total=self._voxel_grid.count(),
            pose_source=pose_source,
            pose_quality=pose_quality,
        )

    def get_map_snapshot(self) -> Dict[str, Any]:
        """Return a compact map state snapshot for API/WebSocket."""
        positions, colors = self._voxel_grid.get_points_array()
        keyframe_poses = []
        for kf in self._keyframes[-50:]:  # Last 50 keyframes
            pos = kf.pose[:3, 3].tolist()
            keyframe_poses.append({
                "frame_id": kf.frame_id,
                "position": pos,
                "timestamp": kf.timestamp,
                "source_node": kf.source_node,
                "tracking_status": kf.tracking_status,
            })

        return {
            "active": self._active,
            "stats": self.stats.to_dict(),
            "keyframe_count": len(self._keyframes),
            "keyframes": keyframe_poses,
            "map_point_count": self._voxel_grid.count(),
            "current_pose": self._current_pose[:3, 3].tolist() if self._current_pose is not None else None,
        }

    def get_new_points(self, max_points: int = 2000) -> List[Dict[str, Any]]:
        """Drain the new-points buffer for incremental dashboard updates.

        Returns a list of {x, y, z, r, g, b} dicts.
        """
        result = []
        with self._new_points_lock:
            count = 0
            while self._new_points_buffer and count < max_points:
                result.append(self._new_points_buffer.popleft())
                count += 1
        return result

    def get_full_pointcloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Nx3 positions, Nx3 colors) for the full map."""
        return self._voxel_grid.get_points_array()

    def reset(self) -> None:
        """Reset all mapping state."""
        self._voxel_grid.clear()
        self._keyframes.clear()
        self._current_pose = np.eye(4)
        self._prev_gray = None
        self._prev_kps = None
        self._prev_descs = None
        self._last_process_time = 0.0
        self._last_keyframe_time = 0.0
        self._frame_counter = 0
        with self._new_points_lock:
            self._new_points_buffer.clear()
        self.stats = MapperStats()
        logger.info("IncrementalMapper RESET")

    # ── Internal Methods ──────────────────────────────────────────

    def _decode_frame(self, frame_b64: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG string to a BGR numpy array."""
        if not frame_b64:
            return None
        # Handle data URI prefix
        if "," in frame_b64[:100]:
            frame_b64 = frame_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(frame_b64)
        except Exception:
            return None
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def _select_keyframe(
        self, gray: np.ndarray, kps: Any, timestamp: float
    ) -> bool:
        """Determine if the current frame should be a keyframe.

        Criteria:
          1. Minimum time interval since last keyframe
          2. Sufficient visual motion (median keypoint displacement)
        """
        # First frame is always a keyframe
        if self._prev_gray is None:
            return True

        # Time threshold
        if (timestamp - self._last_keyframe_time) < self.config.keyframe_min_interval_s:
            return False

        # Motion threshold using optical flow on a subset of points
        try:
            prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8
            )
            if prev_pts is None or len(prev_pts) < 10:
                return True  # Can't measure motion → accept as keyframe

            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, prev_pts, None
            )
            if new_pts is None:
                return True

            valid = status.ravel() == 1
            if valid.sum() < 5:
                return True

            displacement = np.sqrt(
                ((new_pts[valid] - prev_pts[valid]) ** 2).sum(axis=-1)
            ).mean()

            return displacement >= self.config.keyframe_motion_threshold
        except Exception:
            return True

    def _estimate_motion(
        self,
        prev_kps: Any,
        prev_descs: np.ndarray,
        curr_kps: Any,
        curr_descs: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, float]]:
        """Estimate relative camera motion using ORB matching + Essential matrix.

        Returns (R, t, num_inliers, pts1, pts2, pose_mask, pose_quality) or None if estimation fails.
        """
        if prev_descs is None or curr_descs is None:
            logger.info("  [MOTION] Missing descriptors")
            return None
        if len(prev_descs) < self.config.min_matches_for_pose:
            logger.info(f"  [MOTION] Too few prev_descs: {len(prev_descs)}")
            return None
        if len(curr_descs) < self.config.min_matches_for_pose:
            logger.info(f"  [MOTION] Too few curr_descs: {len(curr_descs)}")
            return None

        try:
            if self.config.match_cross_check:
                matches = self._bf_matcher.match(prev_descs, curr_descs)
                # Sort by distance and filter
                matches = sorted(matches, key=lambda x: x.distance)
                # Keep top 20% or a minimum number
                keep = max(self.config.min_matches_for_pose * 2, int(len(matches) * 0.2))
                good_matches = matches[:keep]
            else:
                matches = self._bf_matcher.knnMatch(prev_descs, curr_descs, k=2)
                good_matches = []
                for m_pair in matches:
                    if len(m_pair) == 2:
                        m, n = m_pair
                        if m.distance < self.config.match_ratio_threshold * n.distance:
                            good_matches.append(m)
        except Exception as e:
            logger.info(f"  [MOTION] Matching error: {e}")
            return None

        if len(good_matches) < self.config.min_matches_for_pose:
            logger.info(f"  [MOTION] Too few good matches: {len(good_matches)}")
            return None

        # Extract matched point coordinates
        pts1 = np.float32([prev_kps[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([curr_kps[m.trainIdx].pt for m in good_matches])

        # Camera intrinsics (approximate)
        focal = max(width, height) * self.config.default_focal_ratio
        pp = (width / 2.0, height / 2.0)

        # Find Essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, focal=focal, pp=pp,
            method=cv2.RANSAC, 
            prob=self.config.ransac_probability, 
            threshold=self.config.ransac_threshold
        )

        if E is None or mask is None:
            logger.info("  [MOTION] Essential matrix estimation failed")
            return None

        inliers = int(mask.sum())
        if inliers < 8:
            logger.info(f"  [MOTION] Too few E inliers: {inliers}")
            return None

        # Recover pose from Essential matrix
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=pp, mask=mask)

        if R is None or t is None:
            logger.info("  [MOTION] recoverPose failed")
            return None

        # Validate rotation matrix (det should be ~1)
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.01:
            logger.info(f"  [MOTION] Invalid rotation det: {det}")
            return None

        # Compute pose quality (0.0 to 1.0)
        # Based on inlier ratio and total inliers
        inlier_ratio = inliers / max(1, len(good_matches))
        inlier_score = min(1.0, inliers / 100.0) # 100 inliers is "perfect"
        pose_quality = (inlier_ratio * 0.4) + (inlier_score * 0.6)

        return R, t, inliers, pts1, pts2, pose_mask, round(pose_quality, 3)

    def _estimate_depth(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth map for the current frame.

        Uses DPT monocular depth if available.
        Returns a 2D numpy array of depth values, or None.
        """
        if self._depth_estimator_failed:
            return None

        if self._depth_estimator is None:
            try:
                from reconstruction.depth_estimator import DepthEstimator
                self._depth_estimator = DepthEstimator()
                logger.info("DPT depth estimator loaded successfully")
            except Exception as e:
                logger.warning("DPT depth estimator unavailable: %s. "
                               "Will use feature-based triangulation only.", e)
                self._depth_estimator_failed = True
                return None

        try:
            # DepthEstimator expects a file path; we'll use PIL image instead
            from PIL import Image as PILImage
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(img_rgb)

            # Use the estimator's internals directly to avoid disk I/O
            import torch
            inputs = self._depth_estimator.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self._depth_estimator.device) for k, v in inputs.items()}
            with torch.inference_mode():
                prediction = self._depth_estimator.model(**inputs).predicted_depth

            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(img_bgr.shape[0], img_bgr.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            # Normalize to relative depth (0-1 range, NOT metric)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max - depth_min > 0:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                return None

            # Scale relative depth to approximate metric range
            # DPT produces inverse depth; larger values = closer
            # We invert and scale to a reasonable range (0.1 to 10 meters)
            depth = 1.0 / (depth + 0.01)  # Invert
            depth = depth / depth.max() * 8.0  # Scale to ~8m max
            depth = np.clip(depth, self.config.min_valid_depth, self.config.max_valid_depth)

            return depth.astype(np.float32)

        except Exception as e:
            logger.warning("Depth estimation failed for frame: %s", e)
            return None

    def _triangulate_and_integrate(
        self,
        img_bgr: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        pose_mask: np.ndarray,
        relative_R: np.ndarray,
        relative_t: np.ndarray,
        current_pose: np.ndarray,
        width: int,
        height: int,
        frame_id: str,
    ) -> Tuple[int, int]:
        """Triangulate 3D points from matched features and integrate into map."""
        if self._voxel_grid.count() >= self.config.max_map_points:
            return 0, 0

        if len(self._keyframes) == 0:
            return 0, 0

        prev_pose = self._keyframes[-1].pose
        K = self._make_intrinsics(width, height)

        # World-to-Camera transforms
        inv_prev = np.linalg.inv(prev_pose)
        inv_curr = np.linalg.inv(current_pose)

        P1 = K @ inv_prev[:3, :]
        P2 = K @ inv_curr[:3, :]

        # Filter points by mask
        good_pts1 = pts1[pose_mask.ravel() == 1].T
        good_pts2 = pts2[pose_mask.ravel() == 1].T

        if good_pts1.shape[1] == 0:
            return 0, 0

        # Triangulate points (returns 4xN array in homogeneous coords)
        pts4d = cv2.triangulatePoints(P1, P2, good_pts1, good_pts2)
        pts3d_world = pts4d[:3, :] / (pts4d[3, :] + 1e-6)

        new_count = 0
        fuse_count = 0

        for i in range(pts3d_world.shape[1]):
            pt = pts3d_world[:, i]

            # Simple depth/distance check
            pt_cam = inv_curr @ np.append(pt, 1.0)
            if pt_cam[2] < self.config.min_valid_depth or pt_cam[2] > self.config.max_valid_depth:
                continue

            c, r = int(good_pts2[0, i]), int(good_pts2[1, i])
            if 0 <= r < height and 0 <= c < width:
                bgr = img_bgr[r, c]
                rgb = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)
            else:
                rgb = np.array([128, 128, 128], dtype=np.uint8)

            is_new, _ = self._voxel_grid.insert_or_fuse(pt, rgb, frame_id)
            if is_new:
                new_count += 1
                with self._new_points_lock:
                    self._new_points_buffer.append({
                        "x": float(pt[0]),
                        "y": float(pt[1]),
                        "z": float(pt[2]),
                        "r": int(rgb[0]),
                        "g": int(rgb[1]),
                        "b": int(rgb[2]),
                    })
            else:
                fuse_count += 1

            if self._voxel_grid.count() >= self.config.max_map_points:
                break

        return new_count, fuse_count

    def _integrate_frame(
        self,
        img_bgr: np.ndarray,
        depth_map: np.ndarray,
        pose: np.ndarray,
        width: int,
        height: int,
        frame_id: str,
    ) -> Tuple[int, int]:
        """Integrate depth + pose into the voxel grid map.

        Returns (num_new_points, num_fused_points).
        """
        # Check map size limit
        if self._voxel_grid.count() >= self.config.max_map_points:
            return 0, 0

        step = self.config.depth_sample_step
        focal = max(width, height) * self.config.default_focal_ratio
        cx = width / 2.0
        cy = height / 2.0

        # Camera-to-world transform
        R_cw = pose[:3, :3]
        t_cw = pose[:3, 3]

        new_count = 0
        fuse_count = 0

        # Sample grid of pixels
        rows = np.arange(0, height, step)
        cols = np.arange(0, width, step)

        for r in rows:
            for c in cols:
                z = float(depth_map[r, c])
                if z < self.config.min_valid_depth or z > self.config.max_valid_depth:
                    continue
                if np.isnan(z) or np.isinf(z):
                    continue

                # Back-project pixel to camera coordinates
                x_cam = (c - cx) * z / focal
                y_cam = (r - cy) * z / focal
                z_cam = z

                # Transform to world coordinates
                pt_cam = np.array([x_cam, y_cam, z_cam])
                pt_world = R_cw @ pt_cam + t_cw

                # Get color
                bgr = img_bgr[r, c]
                rgb = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)

                # Insert or fuse into voxel grid
                is_new, _ = self._voxel_grid.insert_or_fuse(pt_world, rgb, frame_id)
                if is_new:
                    new_count += 1
                    # Buffer for incremental dashboard update
                    with self._new_points_lock:
                        self._new_points_buffer.append({
                            "x": float(pt_world[0]),
                            "y": float(pt_world[1]),
                            "z": float(pt_world[2]),
                            "r": int(rgb[0]),
                            "g": int(rgb[1]),
                            "b": int(rgb[2]),
                        })
                else:
                    fuse_count += 1

                # Check map limit mid-integration
                if self._voxel_grid.count() >= self.config.max_map_points:
                    break
            if self._voxel_grid.count() >= self.config.max_map_points:
                break

        return new_count, fuse_count

    def _make_intrinsics(self, width: int, height: int) -> np.ndarray:
        """Create approximate camera intrinsic matrix."""
        focal = max(width, height) * self.config.default_focal_ratio
        K = np.array([
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1],
        ])
        return K

    def _status_result(
        self,
        frame_id: str,
        node_id: str,
        timestamp: float,
        status: str,
        message: str = "",
        points_added: int = 0,
        points_total: int = 0,
        pose_source: str = "NONE",
        pose_quality: float = 0.0,
    ) -> Dict[str, Any]:
        """Build a standard status result dict."""
        return {
            "type": "MAP_UPDATE",
            "frame_id": frame_id,
            "node_id": node_id,
            "timestamp": timestamp,
            "tracking_status": status,
            "pose_source": pose_source,
            "pose_quality": pose_quality,
            "message": message,
            "points_added": points_added,
            "points_total": points_total,
            "keyframe_count": len(self._keyframes),
            "stats": self.stats.to_dict(),
        }

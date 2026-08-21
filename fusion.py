# ══════════════════════════════════════════════════════════════════
#  🧩 TIDDA MULTI-NODE FUSION ENGINE — Multi-map association & merge
#  Evaluates observation and map evidence across mobile nodes to determine
#  whether local maps represent the same physical space.
#
#  Strict rules:
#    - Never blindly merge.
#    - Never mutate source local maps.
#    - Different floors MUST be KEEP_SEPARATE.
#    - Missing evidence signals renormalize weights; never fabricate data.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import copy
import enum
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from mapping_models import (
    Landmark,
    LocalMapState,
    Pose,
    SemanticObservation,
)


# ══════════════════════════════════════════════════════════════════
#  FUSION ENUMS & CONSTANTS
# ══════════════════════════════════════════════════════════════════

class AssociationDecision(enum.Enum):
    """Decision outcome for multi-node map association."""
    MERGE = "MERGE"
    KEEP_SEPARATE = "KEEP_SEPARATE"


DEFAULT_SIGNAL_WEIGHTS: Dict[str, float] = {
    "visual_similarity": 0.20,
    "landmark_similarity": 0.20,
    "pose_consistency": 0.15,
    "trajectory_consistency": 0.10,
    "semantic_similarity": 0.15,
    "orientation_consistency": 0.10,
    "gps_consistency": 0.10,
}

DEFAULT_MERGE_THRESHOLD: float = 0.80

# Geo constant for GPS distance
METERS_PER_DEG_LAT: float = 111_320.0


# ══════════════════════════════════════════════════════════════════
#  FUSION DATA MODELS
# ══════════════════════════════════════════════════════════════════

@dataclass
class MapAssociation:
    """Association evaluation between two mobile node maps."""
    node_a: str
    node_b: str
    floor_a: str
    floor_b: str
    confidence: float                  # [0.0, 1.0]
    decision: str                      # "MERGE" or "KEEP_SEPARATE"
    evidence: Dict[str, Optional[float]]  # per-signal evidence scores (None if unavailable)
    relative_transform: Optional[Dict[str, float]] = None  # e.g. {"dx": 0.5, "dy": 0.2, "dyaw": 0.05}
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """JSON-serializable snapshot of MapAssociation."""
        return {
            "node_a": self.node_a,
            "node_b": self.node_b,
            "floor_a": self.floor_a,
            "floor_b": self.floor_b,
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "evidence": {
                k: round(v, 4) if v is not None else None
                for k, v in self.evidence.items()
            },
            "relative_transform": self.relative_transform,
            "timestamp": self.timestamp,
        }


@dataclass
class FusionResult:
    """Result of multi-node map fusion across participating nodes."""
    participating_nodes: List[str]
    associations: List[MapAssociation]
    merge_decisions: Dict[str, str]        # f"{node_a}:{node_b}" -> decision
    confidence: float                      # Overall fusion confidence [0.0, 1.0]
    aligned_observations: List[SemanticObservation]  # Preserves source node ownership
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """JSON-serializable snapshot of FusionResult."""
        return {
            "participating_nodes": list(self.participating_nodes),
            "associations": [assoc.to_dict() for assoc in self.associations],
            "merge_decisions": dict(self.merge_decisions),
            "confidence": round(self.confidence, 4),
            "aligned_observations": [
                {
                    "observation_id": obs.observation_id,
                    "node_id": obs.node_id,
                    "floor_id": obs.floor_id,
                    "class_name": obs.class_name,
                    "confidence": obs.confidence,
                    "timestamp": obs.timestamp,
                }
                for obs in self.aligned_observations
            ],
            "warnings": list(self.warnings),
            "timestamp": self.timestamp,
        }


# ══════════════════════════════════════════════════════════════════
#  EVIDENCE SIGNAL COMPUTATION HELPERS
# ══════════════════════════════════════════════════════════════════

def compute_angular_diff_deg(angle1_deg: float, angle2_deg: float) -> float:
    """Compute minimal angular difference handling wrap-around (e.g. 359° vs 1° = 2°)."""
    diff = abs(angle1_deg - angle2_deg) % 360.0
    return min(diff, 360.0 - diff)


def compute_angular_diff_rad(rad1: float, rad2: float) -> float:
    """Compute minimal angular difference in radians handling wrap-around."""
    diff = abs(rad1 - rad2) % (2.0 * math.pi)
    return min(diff, 2.0 * math.pi - diff)


def compute_gps_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Equirectangular approximation of distance between two GPS coordinates in meters."""
    dlat = (lat2 - lat1) * METERS_PER_DEG_LAT
    avg_lat_rad = math.radians((lat1 + lat2) / 2.0)
    dlon = (lon2 - lon1) * (METERS_PER_DEG_LAT * math.cos(avg_lat_rad))
    return math.sqrt(dlat * dlat + dlon * dlon)


# ══════════════════════════════════════════════════════════════════
#  FUSION ENGINE
# ══════════════════════════════════════════════════════════════════

class FusionEngine:
    """Multi-node map association and fusion engine.

    Evaluates 7 evidence signals between map pairs:
      1. visual_similarity (Unavailable — YOLOv8n has no visual descriptors)
      2. landmark_similarity (Based on matching landmark positions/descriptors)
      3. pose_consistency (Distance between latest local poses)
      4. trajectory_consistency (Sequence alignment of local trajectories)
      5. semantic_similarity (Jaccard similarity of observed class sets)
      6. orientation_consistency (Minimal angular yaw difference with wrap-around)
      7. gps_consistency (Haversine/Equirectangular distance between GPS coords)

    Guarantees:
      - Never mutates source local maps.
      - Maps on different floors MUST be KEEP_SEPARATE.
      - Missing signals renormalize remaining weights cleanly.
      - Final confidence is strictly clamped to [0.0, 1.0].
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
        max_pose_distance_m: float = 5.0,
        max_gps_distance_m: float = 20.0,
        max_angle_deg: float = 45.0,
    ) -> None:
        self.weights = dict(weights or DEFAULT_SIGNAL_WEIGHTS)
        self.merge_threshold = merge_threshold
        self.max_pose_distance_m = max_pose_distance_m
        self.max_gps_distance_m = max_gps_distance_m
        self.max_angle_deg = max_angle_deg

    # ── Signal Computations ───────────────────────────────────────

    def calc_visual_similarity(self, map_a: LocalMapState, map_b: LocalMapState) -> Optional[float]:
        """Visual similarity signal.

        DOCUMENTED LIMITATION: The current YOLOv8n detector does not provide
        visual feature descriptors. Returns None (unavailable).
        """
        return None

    def calc_landmark_similarity(self, map_a: LocalMapState, map_b: LocalMapState) -> Optional[float]:
        """Landmark similarity signal based on landmark positions and descriptors."""
        if not map_a.landmarks or not map_b.landmarks:
            return None

        # Compare positions of landmarks
        matches = 0
        total = max(len(map_a.landmarks), len(map_b.landmarks))

        for lm_a in map_a.landmarks:
            for lm_b in map_b.landmarks:
                # Check descriptor match or position proximity (<= 2.0 meters)
                pos_a = lm_a.position
                pos_b = lm_b.position
                dist = math.sqrt(
                    (pos_a[0] - pos_b[0]) ** 2 +
                    (pos_a[1] - pos_b[1]) ** 2 +
                    (pos_a[2] - pos_b[2]) ** 2
                )
                descriptor_match = (
                    lm_a.descriptor and lm_b.descriptor and
                    lm_a.descriptor == lm_b.descriptor
                )
                if dist <= 2.0 or descriptor_match:
                    matches += 1
                    break

        score = matches / total
        return max(0.0, min(1.0, float(score)))

    def calc_pose_consistency(self, map_a: LocalMapState, map_b: LocalMapState) -> Optional[float]:
        """Pose consistency signal comparing latest local (x, y, z) poses."""
        pose_a = map_a.trajectory[-1] if map_a.trajectory else None
        pose_b = map_b.trajectory[-1] if map_b.trajectory else None

        if pose_a is None or pose_b is None:
            return None

        dist = math.sqrt(
            (pose_a.x - pose_b.x) ** 2 +
            (pose_a.y - pose_b.y) ** 2 +
            (pose_a.z - pose_b.z) ** 2
        )
        score = 1.0 - (dist / self.max_pose_distance_m)
        return max(0.0, min(1.0, float(score)))

    def calc_trajectory_consistency(self, map_a: LocalMapState, map_b: LocalMapState) -> Optional[float]:
        """Trajectory consistency signal comparing trajectory pose sequences."""
        if not map_a.trajectory or not map_b.trajectory:
            return None

        # Calculate average distance between corresponding trajectory points
        min_len = min(len(map_a.trajectory), len(map_b.trajectory))
        if min_len == 0:
            return None

        total_dist = 0.0
        for p1, p2 in zip(map_a.trajectory[-min_len:], map_b.trajectory[-min_len:]):
            total_dist += math.sqrt(
                (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
            )
        avg_dist = total_dist / min_len
        score = 1.0 - (avg_dist / self.max_pose_distance_m)
        return max(0.0, min(1.0, float(score)))

    def calc_semantic_similarity(self, map_a: LocalMapState, map_b: LocalMapState) -> Optional[float]:
        """Semantic similarity signal using Jaccard index on observation class names."""
        classes_a = {obs.class_name for obs in map_a.observations if obs.class_name}
        classes_b = {obs.class_name for obs in map_b.observations if obs.class_name}

        if not classes_a or not classes_b:
            return None

        intersection = classes_a.intersection(classes_b)
        union = classes_a.union(classes_b)

        if not union:
            return None

        score = len(intersection) / len(union)
        return max(0.0, min(1.0, float(score)))

    def calc_orientation_consistency(self, map_a: LocalMapState, map_b: LocalMapState) -> Optional[float]:
        """Orientation consistency signal comparing yaw angles with wrap-around."""
        pose_a = map_a.trajectory[-1] if map_a.trajectory else None
        pose_b = map_b.trajectory[-1] if map_b.trajectory else None

        if pose_a is None or pose_b is None:
            return None

        # Determine unit consistency: if both <= 2*pi, convert both from radians to degrees
        if max(abs(pose_a.yaw), abs(pose_b.yaw)) <= (2.0 * math.pi):
            yaw_a_deg = math.degrees(pose_a.yaw)
            yaw_b_deg = math.degrees(pose_b.yaw)
        else:
            yaw_a_deg = pose_a.yaw
            yaw_b_deg = pose_b.yaw

        angle_diff_deg = compute_angular_diff_deg(yaw_a_deg, yaw_b_deg)
        score = 1.0 - (angle_diff_deg / self.max_angle_deg)
        return max(0.0, min(1.0, float(score)))

    def calc_gps_consistency(self, map_a: LocalMapState, map_b: LocalMapState) -> Optional[float]:
        """GPS consistency signal comparing latitude/longitude coordinates."""
        pose_a = map_a.trajectory[-1] if map_a.trajectory else None
        pose_b = map_b.trajectory[-1] if map_b.trajectory else None

        if (
            pose_a is None or pose_b is None or
            pose_a.latitude is None or pose_a.longitude is None or
            pose_b.latitude is None or pose_b.longitude is None
        ):
            return None

        dist_m = compute_gps_distance_m(
            pose_a.latitude, pose_a.longitude,
            pose_b.latitude, pose_b.longitude
        )
        score = 1.0 - (dist_m / self.max_gps_distance_m)
        return max(0.0, min(1.0, float(score)))

    # ── Evaluation & Association ─────────────────────────────────

    def evaluate_association(self, map_a: LocalMapState, map_b: LocalMapState) -> MapAssociation:
        """Evaluate association between two local maps.

        Computes evidence signals, renormalizes weights for available signals,
        and enforces decision rules (e.g. floor mismatch -> KEEP_SEPARATE).
        """
        evidence: Dict[str, Optional[float]] = {
            "visual_similarity": self.calc_visual_similarity(map_a, map_b),
            "landmark_similarity": self.calc_landmark_similarity(map_a, map_b),
            "pose_consistency": self.calc_pose_consistency(map_a, map_b),
            "trajectory_consistency": self.calc_trajectory_consistency(map_a, map_b),
            "semantic_similarity": self.calc_semantic_similarity(map_a, map_b),
            "orientation_consistency": self.calc_orientation_consistency(map_a, map_b),
            "gps_consistency": self.calc_gps_consistency(map_a, map_b),
        }

        # Available signals (excluding None)
        available_signals = {
            sig: score for sig, score in evidence.items() if score is not None
        }

        # Renormalize weights
        if not available_signals:
            confidence = 0.0
        else:
            total_weight = sum(
                self.weights.get(sig, 0.0) for sig in available_signals
            )
            if total_weight > 0:
                weighted_sum = sum(
                    score * self.weights.get(sig, 0.0)
                    for sig, score in available_signals.items()
                )
                confidence = weighted_sum / total_weight
            else:
                confidence = 0.0

        confidence = max(0.0, min(1.0, float(confidence)))

        # Floor constraint: different floors MUST be KEEP_SEPARATE
        if map_a.floor_id != map_b.floor_id:
            decision = AssociationDecision.KEEP_SEPARATE.value
        elif confidence >= self.merge_threshold:
            decision = AssociationDecision.MERGE.value
        else:
            decision = AssociationDecision.KEEP_SEPARATE.value

        # Calculate relative transform if pose information available
        relative_transform = None
        pose_a = map_a.trajectory[-1] if map_a.trajectory else None
        pose_b = map_b.trajectory[-1] if map_b.trajectory else None
        if pose_a is not None and pose_b is not None:
            relative_transform = {
                "dx": round(pose_b.x - pose_a.x, 4),
                "dy": round(pose_b.y - pose_a.y, 4),
                "dz": round(pose_b.z - pose_a.z, 4),
                "dyaw": round(compute_angular_diff_rad(pose_a.yaw, pose_b.yaw), 4),
            }

        return MapAssociation(
            node_a=map_a.node_id,
            node_b=map_b.node_id,
            floor_a=map_a.floor_id,
            floor_b=map_b.floor_id,
            confidence=confidence,
            decision=decision,
            evidence=evidence,
            relative_transform=relative_transform,
        )

    # ── Map Fusion ────────────────────────────────────────────────

    def fuse_maps(self, maps: List[LocalMapState]) -> FusionResult:
        """Evaluate association across all pairs of maps and produce FusionResult.

        Does NOT mutate source maps. Preserves source node ownership.
        """
        participating_nodes = [m.node_id for m in maps]
        associations: List[MapAssociation] = []
        merge_decisions: Dict[str, str] = {}
        aligned_observations: List[SemanticObservation] = []
        warnings: List[str] = []

        if len(maps) < 2:
            if maps:
                aligned_observations = [copy.deepcopy(obs) for obs in maps[0].observations]
            return FusionResult(
                participating_nodes=participating_nodes,
                associations=[],
                merge_decisions={},
                confidence=1.0 if maps else 0.0,
                aligned_observations=aligned_observations,
                warnings=["Fewer than 2 maps provided for fusion."],
            )

        total_conf = 0.0
        pair_count = 0

        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                assoc = self.evaluate_association(maps[i], maps[j])
                associations.append(assoc)
                pair_key = f"{assoc.node_a}:{assoc.node_b}"
                merge_decisions[pair_key] = assoc.decision
                total_conf += assoc.confidence
                pair_count += 1

                if assoc.floor_a != assoc.floor_b:
                    warnings.append(
                        f"Cross-floor pair {assoc.node_a} ({assoc.floor_a}) and "
                        f"{assoc.node_b} ({assoc.floor_b}) forced KEEP_SEPARATE."
                    )

        avg_confidence = (total_conf / pair_count) if pair_count > 0 else 0.0

        # Collect observations from maps that were MERGE eligible (or all participating)
        # Note: Copying preserves original node_id source attribution
        for m in maps:
            for obs in m.observations:
                obs_copy = copy.deepcopy(obs)

                # Ensure source attribution is explicitly retained
                if not obs_copy.node_id:
                    obs_copy.node_id = m.node_id
                aligned_observations.append(obs_copy)

        return FusionResult(
            participating_nodes=participating_nodes,
            associations=associations,
            merge_decisions=merge_decisions,
            confidence=max(0.0, min(1.0, float(avg_confidence))),
            aligned_observations=aligned_observations,
            warnings=warnings,
        )

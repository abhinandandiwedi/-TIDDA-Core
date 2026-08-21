# ══════════════════════════════════════════════════════════════════
#  🏛 TIDDA SHARED WORLD MODEL — Building-wide mapping repository
#  Persistent backend representation of the reconstructed building.
#  Maintains multi-floor structure (Floor 1 .. Floor N) and ingests
#  validated local maps, observations, landmarks, and fusion results.
#
#  Strict rules:
#    - Preserves source attribution (node_id, observation_id, landmark_id).
#    - Never mutates source LocalMapState objects.
#    - Never merges observations across different floors.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Set

from mapping_models import (
    FloorState,
    FloorStatus,
    Landmark,
    LocalMapState,
    Pose,
    SemanticObservation,
)
from fusion import FusionResult, MapAssociation, AssociationDecision


# ══════════════════════════════════════════════════════════════════
#  SHARED WORLD MODEL CLASS
# ══════════════════════════════════════════════════════════════════

class WorldModel:
    """Persistent backend representation of a reconstructed building.

    Structure:
      Building
       ├── Floor 1 (FloorState)
       ├── Floor 2 (FloorState)
       └── Floor N (FloorState)

    Maintains shared observations, shared landmarks, multi-floor states,
    node attribution, and fusion audit history.
    """

    def __init__(self, building_id: str = "BUILDING-01") -> None:
        self.building_id: str = building_id
        self.floors: Dict[str, FloorState] = {}
        self.participating_nodes: Set[str] = set()
        self.shared_landmarks: List[Landmark] = []
        self.shared_observations: List[SemanticObservation] = []
        self.fusion_history: List[Dict[str, Any]] = []
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

    # ── Floor Management ──────────────────────────────────────────

    def get_or_create_floor(self, floor_id: str) -> FloorState:
        """Retrieve existing FloorState or create a new one for floor_id."""
        if floor_id not in self.floors:
            self.floors[floor_id] = FloorState(
                floor_id=floor_id,
                status=FloorStatus.NOT_STARTED,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self.updated_at = time.time()
        return self.floors[floor_id]

    def get_floor(self, floor_id: str) -> Optional[FloorState]:
        """Return FloorState for floor_id, or None if not found."""
        return self.floors.get(floor_id)

    def list_floors(self) -> List[FloorState]:
        """Return list of all floors in the building."""
        return [self.floors[fid] for fid in sorted(self.floors)]

    def get_active_floors(self) -> List[FloorState]:
        """Return list of all currently active floors."""
        return [
            f for f in self.floors.values()
            if f.status == FloorStatus.ACTIVE
        ]

    def activate_floor(self, floor_id: str) -> bool:
        """Activate a floor state."""
        floor = self.get_or_create_floor(floor_id)
        floor.status = FloorStatus.ACTIVE
        floor.updated_at = time.time()
        self.updated_at = time.time()
        return True

    def get_node_floor(self, node_id: str) -> Optional[str]:
        """Return the floor_id a node is currently contributing to, or None."""
        for floor_id, floor in self.floors.items():
            if node_id in floor.node_ids:
                return floor_id
        return None

    def get_floor_nodes(self, floor_id: str) -> List[str]:
        """Return all node_ids associated with a floor."""
        floor = self.floors.get(floor_id)
        if floor is None:
            return []
        return list(floor.node_ids)

    # ── Ingestion API ─────────────────────────────────────────────

    def add_observation(self, observation: SemanticObservation) -> bool:
        """Ingest a validated SemanticObservation into the world model.

        Preserves all observation fields including node_id, floor_id,
        observation_id, timestamp, class_name, and confidence.
        """
        if not observation.node_id or not observation.floor_id:
            return False

        # Make copy to ensure no external mutation
        obs_copy = copy.deepcopy(observation)

        floor = self.get_or_create_floor(obs_copy.floor_id)

        # Add node attribution if missing from floor node list
        if obs_copy.node_id not in floor.node_ids:
            floor.node_ids.append(obs_copy.node_id)
        self.participating_nodes.add(obs_copy.node_id)

        # Append observation to floor state
        floor.observations.append(obs_copy)
        floor.observation_count = len(floor.observations)
        floor.updated_at = time.time()

        # Add to global shared observations
        self.shared_observations.append(obs_copy)
        self.updated_at = time.time()
        return True

    def add_landmark(self, landmark: Landmark, floor_id: Optional[str] = None) -> bool:
        """Ingest a validated Landmark into the world model.

        Preserves landmark_id, source node_id, floor_id, position, and confidence.
        """
        if not landmark.node_id:
            return False

        target_floor_id = floor_id or getattr(landmark, "reference_frame_id", None)
        if not target_floor_id:
            # Fallback to node's floor or DEFAULT
            target_floor_id = self.get_node_floor(landmark.node_id) or "FLOOR-1"

        lm_copy = copy.deepcopy(landmark)
        if not lm_copy.reference_frame_id:
            lm_copy.reference_frame_id = target_floor_id

        floor = self.get_or_create_floor(target_floor_id)

        if lm_copy.node_id not in floor.node_ids:
            floor.node_ids.append(lm_copy.node_id)
        self.participating_nodes.add(lm_copy.node_id)

        floor.landmarks.append(lm_copy)
        floor.landmark_count = len(floor.landmarks)
        floor.updated_at = time.time()

        self.shared_landmarks.append(lm_copy)
        self.updated_at = time.time()
        return True

    # ── Fusion Result Ingestion ───────────────────────────────────

    def apply_fusion_result(self, result: FusionResult) -> bool:
        """Ingest and record a validated FusionResult from FusionEngine.

        Rules:
          1. Record result in fusion_history.
          2. Apply ONLY MERGE decisions (and only if floor_a == floor_b).
          3. Keep KEEP_SEPARATE observations & landmarks separate.
          4. Never mutate source LocalMapState objects.
          5. Never merge observations across different floors.
          6. Store shared landmark provenance.
        """
        if result is None:
            return False

        now = time.time()

        # 1. Record fusion history entry
        history_record = {
            "timestamp": result.timestamp or now,
            "participating_nodes": list(result.participating_nodes),
            "associations_count": len(result.associations),
            "merge_decisions": dict(result.merge_decisions),
            "confidence": result.confidence,
            "warnings": list(result.warnings),
        }
        self.fusion_history.append(history_record)

        # Record participating nodes
        for nid in result.participating_nodes:
            self.participating_nodes.add(nid)

        # 2. Process associations
        for assoc in result.associations:
            # Enforce strict floor constraint: different floors NEVER merge!
            if assoc.floor_a != assoc.floor_b:
                continue

            if assoc.decision == AssociationDecision.MERGE.value:
                floor = self.get_or_create_floor(assoc.floor_a)
                floor.status = FloorStatus.ACTIVE

                if assoc.node_a not in floor.node_ids:
                    floor.node_ids.append(assoc.node_a)
                if assoc.node_b not in floor.node_ids:
                    floor.node_ids.append(assoc.node_b)

                # Update floor confidence from merge confidence
                floor.confidence = max(floor.confidence, assoc.confidence)
                floor.updated_at = now

        # 3. Ingest aligned observations (copying guarantees non-mutation of source maps)
        for obs in result.aligned_observations:
            if obs.floor_id and obs.node_id:
                # Add observation copy while retaining original node_id source attribution
                self.add_observation(obs)

        self.updated_at = now
        return True

    # ── Coverage & Confidence ─────────────────────────────────────

    def set_floor_coverage(self, floor_id: str, coverage: float) -> bool:
        """Set floor coverage metric clamped to [0.0, 1.0]."""
        floor = self.get_or_create_floor(floor_id)
        floor.coverage = max(0.0, min(1.0, float(coverage)))
        floor.updated_at = time.time()
        self.updated_at = time.time()
        return True

    def set_floor_confidence(self, floor_id: str, confidence: float) -> bool:
        """Set floor confidence metric clamped to [0.0, 1.0]."""
        floor = self.get_or_create_floor(floor_id)
        floor.confidence = max(0.0, min(1.0, float(confidence)))
        floor.updated_at = time.time()
        self.updated_at = time.time()
        return True

    # ── Queries ───────────────────────────────────────────────────

    def get_observations(self, floor_id: str) -> List[SemanticObservation]:
        """Return all observations for a given floor_id."""
        floor = self.floors.get(floor_id)
        if floor is None:
            return []
        return list(floor.observations)

    def get_landmarks(self, floor_id: str) -> List[Landmark]:
        """Return all landmarks for a given floor_id."""
        floor = self.floors.get(floor_id)
        if floor is None:
            return []
        return list(floor.landmarks)

    def get_fusion_history(self) -> List[dict]:
        """Return fusion operation audit history."""
        return list(self.fusion_history)

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the entire world model."""
        floors_dict = {}
        for fid, floor in self.floors.items():
            floors_dict[fid] = {
                "floor_id": floor.floor_id,
                "status": floor.status.value,
                "node_ids": list(floor.node_ids),
                "coverage": floor.coverage,
                "confidence": floor.confidence,
                "observation_count": len(floor.observations),
                "landmark_count": len(floor.landmarks),
                "created_at": floor.created_at,
                "updated_at": floor.updated_at,
            }

        return {
            "building_id": self.building_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "participating_nodes": list(sorted(self.participating_nodes)),
            "floors": floors_dict,
            "shared_observations": [
                {
                    "observation_id": obs.observation_id,
                    "node_id": obs.node_id,
                    "floor_id": obs.floor_id,
                    "class_name": obs.class_name,
                    "confidence": obs.confidence,
                    "timestamp": obs.timestamp,
                }
                for obs in self.shared_observations
            ],
            "shared_landmarks": [
                {
                    "landmark_id": lm.landmark_id,
                    "node_id": lm.node_id,
                    "position": list(lm.position),
                    "confidence": lm.confidence,
                    "reference_frame_id": lm.reference_frame_id,
                }
                for lm in self.shared_landmarks
            ],
            "fusion_history": self.get_fusion_history(),
        }

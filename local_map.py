# ══════════════════════════════════════════════════════════════════
#  🗺 TIDDA LOCAL MAP ENGINE — Per-node local mapping manager
#  Manages independent local map state for each scanning phone node.
#  Preserves source node ownership and floor association.
#
#  This module is ONLY state management — no AI, no SLAM, no fusion.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from mapping_models import (
    Landmark,
    LocalMapState,
    Pose,
    SemanticObservation,
)
from floor_manager import FloorManager


# ══════════════════════════════════════════════════════════════════
#  SERIALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════

def pose_to_dict(pose: Pose) -> dict:
    """Serialize a Pose instance to a dict."""
    return {
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "yaw": pose.yaw,
        "pitch": pose.pitch,
        "roll": pose.roll,
        "timestamp": pose.timestamp,
        "latitude": pose.latitude,
        "longitude": pose.longitude,
        "altitude": pose.altitude,
    }


def observation_to_dict(obs: SemanticObservation) -> dict:
    """Serialize a SemanticObservation instance to a dict."""
    return {
        "observation_id": obs.observation_id,
        "node_id": obs.node_id,
        "timestamp": obs.timestamp,
        "floor_id": obs.floor_id,
        "class_name": obs.class_name,
        "confidence": obs.confidence,
        "bounding_box": list(obs.bounding_box) if obs.bounding_box is not None else None,
        "pose": pose_to_dict(obs.pose) if obs.pose is not None else None,
        "position": list(obs.position) if obs.position is not None else None,
        "spatial_extent": list(obs.spatial_extent) if obs.spatial_extent is not None else None,
        "frame_id": obs.frame_id,
    }


def landmark_to_dict(lm: Landmark) -> dict:
    """Serialize a Landmark instance to a dict."""
    return {
        "landmark_id": lm.landmark_id,
        "node_id": lm.node_id,
        "position": list(lm.position) if lm.position is not None else [0.0, 0.0, 0.0],
        "confidence": lm.confidence,
        "observation_count": lm.observation_count,
        "descriptor": lm.descriptor,
        "reference_frame_id": lm.reference_frame_id,
    }


def local_map_to_dict(map_state: LocalMapState) -> dict:
    """Serialize a LocalMapState instance to a dict."""
    return {
        "node_id": map_state.node_id,
        "floor_id": map_state.floor_id,
        "trajectory": [pose_to_dict(p) for p in map_state.trajectory],
        "observations": [observation_to_dict(o) for o in map_state.observations],
        "landmarks": [landmark_to_dict(l) for l in map_state.landmarks],
        "coverage": map_state.coverage,
        "mapping_confidence": map_state.mapping_confidence,
        "last_update": map_state.last_update,
    }


# ══════════════════════════════════════════════════════════════════
#  LOCAL MAP MANAGER
# ══════════════════════════════════════════════════════════════════

class LocalMapManager:
    """Manages independent local map states for phone nodes.

    Each scanning mobile node owns exactly one local map tied to one floor.
    Coverage is represented as a float in [0.0, 1.0].
    Mapping confidence is represented as a float in [0.0, 1.0].
    """

    def __init__(self, floor_manager: Optional[FloorManager] = None) -> None:
        self._maps: Dict[str, LocalMapState] = {}
        self._floor_manager = floor_manager

    # ── Map Lifecycle ─────────────────────────────────────────────

    def create_map(self, node_id: str, floor_id: str) -> LocalMapState:
        """Create or update a local map for a given node_id and floor_id.

        If a map already exists for node_id, updates its floor_id.
        Optionally updates floor_manager node assignment if provided.
        """
        if node_id in self._maps:
            local_map = self._maps[node_id]
            local_map.floor_id = floor_id
            local_map.last_update = time.time()
        else:
            local_map = LocalMapState(
                node_id=node_id,
                floor_id=floor_id,
                trajectory=[],
                landmarks=[],
                observations=[],
                mapping_confidence=0.0,
                coverage=0.0,
                last_update=time.time(),
            )
            self._maps[node_id] = local_map

        if self._floor_manager is not None:
            if self._floor_manager.get_floor(floor_id) is None:
                self._floor_manager.create_floor(floor_id)
            self._floor_manager.assign_node(node_id, floor_id)

        return local_map

    def get_map(self, node_id: str) -> Optional[LocalMapState]:
        """Return the LocalMapState for a node_id, or None if not found."""
        return self._maps.get(node_id)

    def remove_map(self, node_id: str) -> bool:
        """Remove the local map for node_id.

        Returns True if the map existed and was removed, False otherwise.
        """
        if node_id in self._maps:
            del self._maps[node_id]
            if self._floor_manager is not None:
                self._floor_manager.remove_node(node_id)
            return True
        return False

    def list_maps(self) -> List[LocalMapState]:
        """Return a list of all current local maps."""
        return list(self._maps.values())

    # ── Map Updates ───────────────────────────────────────────────

    def update_pose(self, node_id: str, pose: Pose) -> bool:
        """Append a pose to the node's trajectory and update last_update timestamp.

        Returns True if successful, False if map does not exist.
        """
        local_map = self._maps.get(node_id)
        if local_map is None:
            return False

        local_map.trajectory.append(pose)
        local_map.last_update = time.time()
        return True

    def add_observation(self, node_id: str, observation: SemanticObservation) -> bool:
        """Add a semantic observation to the node's local map.

        Validation rules:
          1. Reject if map does not exist for node_id.
          2. Reject if observation.node_id is provided and != node_id.
          3. Reject if observation.floor_id is provided and != local_map.floor_id.

        Sets missing node_id/floor_id fields on observation if left blank.
        Returns True if added successfully, False if rejected.
        """
        local_map = self._maps.get(node_id)
        if local_map is None:
            return False

        # Validate node_id
        if observation.node_id and observation.node_id != node_id:
            return False

        # Validate floor_id
        if observation.floor_id and observation.floor_id != local_map.floor_id:
            return False

        # Populate missing identifiers if empty
        if not observation.node_id:
            observation.node_id = node_id
        if not observation.floor_id:
            observation.floor_id = local_map.floor_id

        local_map.observations.append(observation)
        local_map.last_update = time.time()
        return True

    def add_landmark(self, node_id: str, landmark: Landmark) -> bool:
        """Add a landmark to the node's local map.

        Validation rules:
          1. Reject if map does not exist for node_id.
          2. Reject if landmark.node_id is provided and != node_id.

        Sets missing node_id on landmark if left blank.
        Returns True if added successfully, False if rejected.
        """
        local_map = self._maps.get(node_id)
        if local_map is None:
            return False

        if landmark.node_id and landmark.node_id != node_id:
            return False

        if not landmark.node_id:
            landmark.node_id = node_id

        local_map.landmarks.append(landmark)
        local_map.last_update = time.time()
        return True

    # ── Coverage & Confidence Controls ───────────────────────────

    def set_coverage(self, node_id: str, coverage: float) -> bool:
        """Set local map coverage metric clamped to [0.0, 1.0].

        Returns True if successful, False if map does not exist.
        """
        local_map = self._maps.get(node_id)
        if local_map is None:
            return False

        clamped = max(0.0, min(1.0, float(coverage)))
        local_map.coverage = clamped
        local_map.last_update = time.time()
        return True

    def set_confidence(self, node_id: str, confidence: float) -> bool:
        """Set mapping confidence metric clamped to [0.0, 1.0].

        Returns True if successful, False if map does not exist.
        """
        local_map = self._maps.get(node_id)
        if local_map is None:
            return False

        clamped = max(0.0, min(1.0, float(confidence)))
        local_map.mapping_confidence = clamped
        local_map.last_update = time.time()
        return True

    # ── Read-only Accessors ───────────────────────────────────────

    def get_trajectory(self, node_id: str) -> List[Pose]:
        """Return the trajectory (list of Pose) for node_id, or [] if map not found."""
        local_map = self._maps.get(node_id)
        if local_map is None:
            return []
        return list(local_map.trajectory)

    def get_observations(self, node_id: str) -> List[SemanticObservation]:
        """Return observations for node_id, or [] if map not found."""
        local_map = self._maps.get(node_id)
        if local_map is None:
            return []
        return list(local_map.observations)

    def get_landmarks(self, node_id: str) -> List[Landmark]:
        """Return landmarks for node_id, or [] if map not found."""
        local_map = self._maps.get(node_id)
        if local_map is None:
            return []
        return list(local_map.landmarks)

    def get_state(self, node_id: str) -> Optional[dict]:
        """Return a serialized dict representation of node_id's local map, or None."""
        local_map = self._maps.get(node_id)
        if local_map is None:
            return None
        return local_map_to_dict(local_map)

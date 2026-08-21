# ══════════════════════════════════════════════════════════════════
#  🏢 TIDDA FLOOR MANAGER — Multi-floor state management
#  Manages multiple simultaneously active floors.
#  No current_floor singleton — all floors coexist independently.
#
#  This module is ONLY state management — no AI, no SLAM, no mapping.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
from typing import Dict, List, Optional

from mapping_models import FloorState, FloorStatus


class FloorManager:
    """Manages multiple floors for cooperative indoor mapping.

    Key design constraints:
      - Multiple floors can be ACTIVE simultaneously.
      - No global current_floor variable.
      - A node belongs to at most one floor at a time.
      - Reassigning a node removes it from the old floor.
      - Removing a node does NOT delete the floor.

    Usage:
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-1")
        mgr.assign_node("PHONE-01", "FLOOR-1")
    """

    def __init__(self) -> None:
        self._floors: Dict[str, FloorState] = {}
        # Reverse index: node_id → floor_id (for O(1) lookup)
        self._node_floor: Dict[str, str] = {}

    # ── Floor lifecycle ──────────────────────────────────────────

    def create_floor(self, floor_id: str) -> FloorState:
        """Create a new floor in NOT_STARTED status.

        If the floor already exists, returns the existing floor.
        """
        if floor_id in self._floors:
            return self._floors[floor_id]

        floor = FloorState(
            floor_id=floor_id,
            status=FloorStatus.NOT_STARTED,
        )
        self._floors[floor_id] = floor
        return floor

    def remove_floor(self, floor_id: str) -> bool:
        """Remove a floor and unassign all its nodes.

        Returns True if the floor existed, False otherwise.
        """
        floor = self._floors.pop(floor_id, None)
        if floor is None:
            return False

        # Unassign all nodes that belonged to this floor
        for node_id in list(floor.node_ids):
            self._node_floor.pop(node_id, None)

        return True

    def get_floor(self, floor_id: str) -> Optional[FloorState]:
        """Return the FloorState for a given floor_id, or None."""
        return self._floors.get(floor_id)

    def list_floors(self) -> List[FloorState]:
        """Return all floors sorted by floor_id."""
        return [self._floors[fid] for fid in sorted(self._floors)]

    # ── Floor status transitions ─────────────────────────────────

    def activate_floor(self, floor_id: str) -> bool:
        """Transition a floor to ACTIVE status.

        The floor must exist. Multiple floors can be ACTIVE simultaneously.
        Returns True if successful.
        """
        floor = self._floors.get(floor_id)
        if floor is None:
            return False
        floor.status = FloorStatus.ACTIVE
        floor.updated_at = time.time()
        return True

    def pause_floor(self, floor_id: str) -> bool:
        """Transition a floor to PAUSED status.

        Returns True if the floor exists.
        """
        floor = self._floors.get(floor_id)
        if floor is None:
            return False
        floor.status = FloorStatus.PAUSED
        floor.updated_at = time.time()
        return True

    def complete_floor(self, floor_id: str) -> bool:
        """Transition a floor to COMPLETE status.

        Returns True if the floor exists.
        """
        floor = self._floors.get(floor_id)
        if floor is None:
            return False
        floor.status = FloorStatus.COMPLETE
        floor.updated_at = time.time()
        return True

    # ── Node assignment ──────────────────────────────────────────

    def assign_node(self, node_id: str, floor_id: str) -> bool:
        """Assign a node to a floor.

        If the node is already assigned to a different floor,
        it is automatically removed from the old floor first.
        A node belongs to at most ONE floor at a time.

        Returns True if the floor exists, False otherwise.
        """
        floor = self._floors.get(floor_id)
        if floor is None:
            return False

        # Remove from old floor if reassigning
        old_floor_id = self._node_floor.get(node_id)
        if old_floor_id is not None and old_floor_id != floor_id:
            old_floor = self._floors.get(old_floor_id)
            if old_floor is not None and node_id in old_floor.node_ids:
                old_floor.node_ids.remove(node_id)
                old_floor.updated_at = time.time()

        # Assign to new floor
        if node_id not in floor.node_ids:
            floor.node_ids.append(node_id)
        floor.updated_at = time.time()
        self._node_floor[node_id] = floor_id
        return True

    def remove_node(self, node_id: str, floor_id: Optional[str] = None) -> bool:
        """Remove a node from its assigned floor.

        If floor_id is provided, only remove from that specific floor.
        If floor_id is None, remove from whichever floor currently owns it.

        The floor itself remains — only the node is removed.
        Returns True if the node was found and removed.
        """
        if floor_id is None:
            floor_id = self._node_floor.get(node_id)

        if floor_id is None:
            return False

        floor = self._floors.get(floor_id)
        if floor is None:
            return False

        if node_id in floor.node_ids:
            floor.node_ids.remove(node_id)
            floor.updated_at = time.time()
            self._node_floor.pop(node_id, None)
            return True

        return False

    # ── Read-only accessors ──────────────────────────────────────

    def get_nodes(self, floor_id: str) -> List[str]:
        """Return all node_ids assigned to a floor."""
        floor = self._floors.get(floor_id)
        if floor is None:
            return []
        return list(floor.node_ids)

    def get_active_floors(self) -> List[FloorState]:
        """Return all floors with ACTIVE status."""
        return [
            f for f in self._floors.values()
            if f.status == FloorStatus.ACTIVE
        ]

    def get_node_floor(self, node_id: str) -> Optional[str]:
        """Return the floor_id a node is assigned to, or None."""
        return self._node_floor.get(node_id)

    # ── Serialization ────────────────────────────────────────────

    def to_state(self) -> dict:
        """Return a JSON-serializable snapshot of all floor states.

        Example output:
            {
                "floors": {
                    "FLOOR-1": {
                        "floor_id": "FLOOR-1",
                        "status": "ACTIVE",
                        "node_ids": ["PHONE-01", "PHONE-02"],
                        "coverage": 0.0,
                        "confidence": 0.0,
                        "observation_count": 0,
                        "landmark_count": 0,
                        "created_at": 1692700000.0,
                        "updated_at": 1692700000.0
                    }
                },
                "node_assignments": {
                    "PHONE-01": "FLOOR-1",
                    "PHONE-02": "FLOOR-1"
                }
            }
        """
        floors_out = {}
        for fid, floor in self._floors.items():
            floors_out[fid] = {
                "floor_id": floor.floor_id,
                "status": floor.status.value,
                "node_ids": list(floor.node_ids),
                "coverage": floor.coverage,
                "confidence": floor.confidence,
                "observation_count": floor.observation_count,
                "landmark_count": floor.landmark_count,
                "created_at": floor.created_at,
                "updated_at": floor.updated_at,
            }

        return {
            "floors": floors_out,
            "node_assignments": dict(self._node_floor),
        }

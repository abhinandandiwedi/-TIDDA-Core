# ══════════════════════════════════════════════════════════════════
#  🔍 TIDDA SCAN SESSION — Scan state management for mobile nodes
#  Controls whether connected mobile nodes are actively scanning.
#
#  Key distinction:
#    CONNECTED ≠ SCANNING
#    A phone can be connected while IDLE.
#    A phone enters SCANNING only after an explicit scan-start action.
#
#  This module is ONLY state management — no AI, no SLAM, no mapping.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ══════════════════════════════════════════════════════════════════
#  NODE SCAN STATE ENUM
# ══════════════════════════════════════════════════════════════════

class NodeScanState(enum.Enum):
    """Scan state of an individual mobile node.

    This is independent of connection state.
    A connected node starts IDLE and must be explicitly transitioned.

    State transitions:
        IDLE → SCANNING         (start_node_scan)
        SCANNING → IDLE         (stop_node_scan)
        SCANNING → PAUSED       (pause_node_scan)
        PAUSED → SCANNING       (resume_node_scan)
        any → DISCONNECTED      (node disconnect / timeout)
    """
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    PAUSED = "PAUSED"
    DISCONNECTED = "DISCONNECTED"


# ══════════════════════════════════════════════════════════════════
#  SCAN SESSION MANAGER
# ══════════════════════════════════════════════════════════════════

class ScanSession:
    """Manages scan state for a cooperative mapping session.

    Tracks which connected nodes are actively scanning vs idle.
    Multiple nodes can scan simultaneously.
    Stopping one node does not affect other scanning nodes.
    Stopping the session stops ALL currently scanning nodes.

    Usage:
        session = ScanSession()
        session.connect_node("PHONE-01")       # node joins → IDLE
        session.start_session()                 # session becomes active
        session.start_node_scan("PHONE-01")     # IDLE → SCANNING
        session.stop_node_scan("PHONE-01")      # SCANNING → IDLE
        session.stop_session()                  # stops all, session inactive
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or f"SCAN-{uuid.uuid4().hex[:8].upper()}"
        self._active: bool = False
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None

        # Node tracking
        self._connected_nodes: Set[str] = set()
        self._node_states: Dict[str, NodeScanState] = {}

    # ── Session lifecycle ────────────────────────────────────────

    def start_session(self) -> None:
        """Activate the scan session.

        Starting a session does NOT automatically start scanning on any node.
        Individual nodes must be started explicitly.
        """
        self._active = True
        self._started_at = time.time()
        self._stopped_at = None

    def stop_session(self) -> None:
        """Deactivate the session and stop ALL currently scanning nodes.

        All SCANNING and PAUSED nodes are moved to IDLE.
        """
        # Stop all scanning/paused nodes
        for node_id in list(self._node_states.keys()):
            if self._node_states[node_id] in (
                NodeScanState.SCANNING,
                NodeScanState.PAUSED,
            ):
                self._node_states[node_id] = NodeScanState.IDLE

        self._active = False
        self._stopped_at = time.time()

    @property
    def active(self) -> bool:
        """Whether the session is currently active."""
        return self._active

    # ── Node connection management ───────────────────────────────

    def connect_node(self, node_id: str) -> None:
        """Register a node as connected.  Always starts IDLE.

        If a node reconnects, it is reset to IDLE regardless of
        its previous state.
        """
        self._connected_nodes.add(node_id)
        self._node_states[node_id] = NodeScanState.IDLE

    def disconnect_node(self, node_id: str) -> None:
        """Mark a node as disconnected.

        The node is removed from connected_nodes and scanning_node_ids.
        Its state is set to DISCONNECTED.
        """
        self._connected_nodes.discard(node_id)
        self._node_states[node_id] = NodeScanState.DISCONNECTED

    # ── Individual node scan control ─────────────────────────────

    def start_node_scan(self, node_id: str) -> bool:
        """Transition a node from IDLE → SCANNING.

        Returns True if the transition succeeded, False otherwise.
        A node must be connected and IDLE to start scanning.
        """
        state = self._node_states.get(node_id)
        if state == NodeScanState.IDLE:
            self._node_states[node_id] = NodeScanState.SCANNING
            return True
        return False

    def stop_node_scan(self, node_id: str) -> bool:
        """Transition a node from SCANNING → IDLE.

        Returns True if the transition succeeded, False otherwise.
        """
        state = self._node_states.get(node_id)
        if state in (NodeScanState.SCANNING, NodeScanState.PAUSED):
            self._node_states[node_id] = NodeScanState.IDLE
            return True
        return False

    def pause_node_scan(self, node_id: str) -> bool:
        """Transition a node from SCANNING → PAUSED.

        Returns True if the transition succeeded, False otherwise.
        """
        state = self._node_states.get(node_id)
        if state == NodeScanState.SCANNING:
            self._node_states[node_id] = NodeScanState.PAUSED
            return True
        return False

    def resume_node_scan(self, node_id: str) -> bool:
        """Transition a node from PAUSED → SCANNING.

        Returns True if the transition succeeded, False otherwise.
        """
        state = self._node_states.get(node_id)
        if state == NodeScanState.PAUSED:
            self._node_states[node_id] = NodeScanState.SCANNING
            return True
        return False

    # ── Read-only accessors ──────────────────────────────────────

    @property
    def connected_node_ids(self) -> List[str]:
        """List of currently connected node IDs."""
        return sorted(self._connected_nodes)

    @property
    def scanning_node_ids(self) -> List[str]:
        """List of node IDs currently in SCANNING state."""
        return sorted(
            nid for nid, state in self._node_states.items()
            if state == NodeScanState.SCANNING
        )

    def get_node_state(self, node_id: str) -> Optional[NodeScanState]:
        """Return the scan state of a specific node, or None if unknown."""
        return self._node_states.get(node_id)

    # ── Serializable state output ────────────────────────────────

    def to_state(self) -> dict:
        """Return a JSON-serializable snapshot of the entire session state.

        Example output:
            {
                "session_id": "SCAN-A1B2C3D4",
                "active": true,
                "started_at": 1692700000.0,
                "stopped_at": null,
                "connected_node_ids": ["PHONE-01", "PHONE-02"],
                "scanning_node_ids": ["PHONE-01"],
                "node_states": {
                    "PHONE-01": "SCANNING",
                    "PHONE-02": "IDLE"
                }
            }
        """
        # Only include non-DISCONNECTED nodes in node_states output
        visible_states = {
            nid: state.value
            for nid, state in self._node_states.items()
            if state != NodeScanState.DISCONNECTED
        }

        return {
            "session_id": self.session_id,
            "active": self._active,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "connected_node_ids": self.connected_node_ids,
            "scanning_node_ids": self.scanning_node_ids,
            "node_states": visible_states,
        }

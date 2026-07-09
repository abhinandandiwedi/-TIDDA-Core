# ══════════════════════════════════════════════════════════════════
#  📱 TIDDA MOBILE NODE — Android phone node subsystem
#  Completely isolated from the swarm physics simulation.
#  Phones register here; DroneUnit / physics_loop never touch them.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

NODE_TIMEOUT_S: float = 15.0   # prune threshold — exported to main module


@dataclass
class MobileNode:
    """Real phone node — holds live telemetry from an Android handset.

    All fields are updated via apply_telemetry() from incoming JSON.
    The physics simulation never touches this dataclass.
    """

    node_id: str
    lat: float = 0.0
    lon: float = 0.0
    altitude_m: float = 0.0
    battery_pct: float = 100.0
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    camera_active: bool = False
    network_type: str = "UNKNOWN"        # e.g. "LTE", "5G", "WIFI"
    signal_strength: int = 0             # dBm or RSSI integer
    status: str = "SCANNING"
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # ── Partial-update ingestion ─────────────────────────────────

    def apply_telemetry(self, payload: dict) -> None:
        """Update fields from a JSON payload; unknown/missing keys are silently skipped.

        Designed for flaky mobile connections — a packet that only contains
        battery_pct and GPS is still applied without resetting other fields.
        """
        if "lat" in payload:
            self.lat = float(payload["lat"])
        if "lon" in payload or "lng" in payload:
            self.lon = float(payload.get("lon", payload.get("lng", self.lon)))
        if "altitude_m" in payload:
            self.altitude_m = float(payload["altitude_m"])
        if "battery_pct" in payload:
            raw = float(payload["battery_pct"])
            self.battery_pct = max(0.0, min(100.0, raw))   # clamp [0, 100]
        if "heading_deg" in payload:
            self.heading_deg = float(payload["heading_deg"])
        if "speed_mps" in payload:
            self.speed_mps = float(payload["speed_mps"])
        if "camera_active" in payload:
            self.camera_active = bool(payload["camera_active"])
        if "network_type" in payload:
            self.network_type = str(payload["network_type"])
        if "signal_strength" in payload:
            self.signal_strength = int(payload["signal_strength"])
        if "status" in payload:
            self.status = str(payload["status"])
        self.last_seen = time.time()

    # ── Heartbeat-only update ────────────────────────────────────

    def touch(self) -> None:
        """Bump last_seen without changing any telemetry field."""
        self.last_seen = time.time()

    # ── Staleness check ──────────────────────────────────────────

    def is_stale(self, timeout_s: float = NODE_TIMEOUT_S) -> bool:
        """Return True if no update has been received within timeout_s seconds."""
        return (time.time() - self.last_seen) > timeout_s

    # ── Telemetry serialisation ──────────────────────────────────

    def to_telemetry(self) -> dict:
        """Return a dict matching DroneUnit.to_telemetry() key names exactly.

        The extra "node_type": "mobile" key lets the dashboard optionally
        style phone nodes differently without requiring any frontend change
        (it simply ignores unknown keys).
        """
        return {
            "drone_id":    self.node_id,
            "lat":         round(self.lat, 6),
            "lng":         round(self.lon, 6),
            "altitude_m":  round(self.altitude_m, 1),
            "battery_pct": round(self.battery_pct),
            "status":      self.status,
            "timestamp":   time.time(),
            # ── mobile-specific extras ──
            "node_type":       "mobile",
            "heading_deg":     round(self.heading_deg, 1),
            "speed_mps":       round(self.speed_mps, 2),
            "camera_active":   self.camera_active,
            "network_type":    self.network_type,
            "signal_strength": self.signal_strength,
        }


# ══════════════════════════════════════════════════════════════════
#  REGISTRY — thread-safe (single-threaded asyncio) store
# ══════════════════════════════════════════════════════════════════

class MobileNodeRegistry:
    """In-memory store of all currently connected Android nodes."""

    def __init__(self) -> None:
        self._nodes: Dict[str, MobileNode] = {}

    # ── Lifecycle ────────────────────────────────────────────────

    def register(self, node_id: str) -> MobileNode:
        """Create or refresh a node entry; returns the node."""
        if node_id not in self._nodes:
            self._nodes[node_id] = MobileNode(node_id=node_id)
        else:
            self._nodes[node_id].touch()  # re-registered after reconnect
        return self._nodes[node_id]

    def remove(self, node_id: str) -> None:
        """Remove a node (called on clean WS disconnect)."""
        self._nodes.pop(node_id, None)

    # ── Update helpers ───────────────────────────────────────────

    def update_telemetry(self, node_id: str, payload: dict) -> bool:
        """Apply a telemetry payload to a known node.

        Returns False if node_id is not registered (safety guard).
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False
        node.apply_telemetry(payload)
        return True

    def heartbeat(self, node_id: str) -> bool:
        """Bump last_seen for a registered node.

        Returns False if node_id is not registered.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False
        node.touch()
        return True

    # ── Watchdog ─────────────────────────────────────────────────

    def prune_stale(self, timeout_s: float = NODE_TIMEOUT_S) -> List[str]:
        """Remove nodes that haven't sent any message within timeout_s.

        Returns list of removed node_ids so the caller can broadcast
        node_offline events.
        """
        stale_ids = [
            nid for nid, node in self._nodes.items()
            if node.is_stale(timeout_s)
        ]
        for nid in stale_ids:
            del self._nodes[nid]
        return stale_ids

    # ── Read ─────────────────────────────────────────────────────

    def all_telemetry(self) -> List[dict]:
        """Return serialised telemetry for every registered node."""
        return [node.to_telemetry() for node in self._nodes.values()]

    def count(self) -> int:
        """Current number of registered mobile nodes."""
        return len(self._nodes)

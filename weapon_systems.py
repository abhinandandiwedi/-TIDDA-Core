"""
🎯 TIDDA V2.0 — JADC2 Weapon Systems Module
Joint All-Domain Command & Control — Kill Web Integration

Asset Types: AUTO_TURRET, LOITERING_MUNITION, MORTAR_SYSTEM, RECON_DRONE, ATTACK_DRONE
Kill Chain:  Find → Fix → Track → Target → Engage → Assess (F2T2EA)

All state access is guarded by asyncio.Lock for single-threaded async safety.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Module logger ─────────────────────────────────────────────────

log_jadc2 = logging.getLogger("JADC2")


# ══════════════════════════════════════════════════════════════════
#  ASSET TYPE ENUM
# ══════════════════════════════════════════════════════════════════

class AssetType(enum.Enum):
    """Classification of assets in the unified kill web."""
    RECON_DRONE = "RECON_DRONE"
    ATTACK_DRONE = "ATTACK_DRONE"
    AUTO_TURRET = "AUTO_TURRET"
    LOITERING_MUNITION = "LOITERING_MUNITION"
    MORTAR_SYSTEM = "MORTAR_SYSTEM"


# ══════════════════════════════════════════════════════════════════
#  WEAPON STATUS ENUM
# ══════════════════════════════════════════════════════════════════

class WeaponStatus(enum.Enum):
    """Operational status of a weapon system."""
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    ENGAGED = "ENGAGED"
    RELOADING = "RELOADING"
    EXPENDED = "EXPENDED"
    OFFLINE = "OFFLINE"


# ══════════════════════════════════════════════════════════════════
#  ENGAGEMENT STATUS ENUM
# ══════════════════════════════════════════════════════════════════

class EngagementPhase(enum.Enum):
    """F2T2EA kill chain phases."""
    FIND = "FIND"
    FIX = "FIX"
    TRACK = "TRACK"
    TARGET = "TARGET"
    ENGAGE = "ENGAGE"
    ASSESS = "ASSESS"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


# ══════════════════════════════════════════════════════════════════
#  WEAPON STATE — Single effector unit
# ══════════════════════════════════════════════════════════════════

@dataclass
class WeaponState:
    """State of a single weapon system / effector."""
    weapon_id: str
    asset_type: AssetType
    lat: float
    lng: float
    status: WeaponStatus = WeaponStatus.IDLE
    ammo_count: int = -1            # -1 = unlimited (turrets)
    threat_radius_m: float = 500.0  # engagement envelope radius
    assigned_target: Optional[str] = None  # engagement ID if tracking
    last_fired: float = 0.0         # epoch timestamp
    reload_time_s: float = 5.0      # seconds to reload after firing
    kills: int = 0                  # total confirmed engagements

    def to_telemetry(self) -> dict:
        """JSON-serializable telemetry for GCS broadcast."""
        return {
            "type": "weapon_telemetry",
            "weapon_id": self.weapon_id,
            "asset_type": self.asset_type.value,
            "lat": round(self.lat, 6),
            "lng": round(self.lng, 6),
            "status": self.status.value,
            "ammo_count": self.ammo_count,
            "threat_radius_m": self.threat_radius_m,
            "assigned_target": self.assigned_target,
            "kills": self.kills,
            "timestamp": time.time(),
        }


# ══════════════════════════════════════════════════════════════════
#  ENGAGEMENT RECORD — Active kill chain instance
# ══════════════════════════════════════════════════════════════════

@dataclass
class Engagement:
    """A single active engagement through the F2T2EA kill chain."""
    engagement_id: str
    threat_lat: float
    threat_lng: float
    phase: EngagementPhase = EngagementPhase.FIND
    assigned_weapon_id: Optional[str] = None
    detecting_drone_id: Optional[str] = None
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: str = ""  # "NEUTRALIZED", "MISSED", "ABORTED"


# ══════════════════════════════════════════════════════════════════
#  WEAPON REGISTRY — Dynamic spawn/destroy/manage
# ══════════════════════════════════════════════════════════════════

# Type aliases for registry
WeaponId = str
WeaponDict = Dict[WeaponId, WeaponState]


class WeaponRegistry:
    """Dynamic registry of all weapon systems in the kill web.

    Supports runtime spawn/destroy — no hardcoded weapon IDs.
    """

    # Default weapon profiles
    PROFILES: Dict[AssetType, Dict[str, Any]] = {
        AssetType.AUTO_TURRET: {
            "ammo_count": -1,        # unlimited
            "threat_radius_m": 800.0,
            "reload_time_s": 2.0,
        },
        AssetType.LOITERING_MUNITION: {
            "ammo_count": 1,         # one-shot expendable
            "threat_radius_m": 2000.0,
            "reload_time_s": 0.0,    # N/A — expended after use
        },
        AssetType.MORTAR_SYSTEM: {
            "ammo_count": 12,
            "threat_radius_m": 1500.0,
            "reload_time_s": 8.0,
        },
    }

    def __init__(self) -> None:
        self.weapons: WeaponDict = {}
        self._id_counters: Dict[AssetType, int] = {}

    def spawn(
        self,
        asset_type: AssetType,
        lat: float,
        lng: float,
        weapon_id: Optional[str] = None,
    ) -> WeaponState:
        """Spawn a new weapon system and register it.

        If weapon_id is None, auto-generates one (e.g., TURRET-01, MUNITION-02).
        """
        if weapon_id is None:
            # Auto-generate sequential ID
            prefix_map = {
                AssetType.AUTO_TURRET: "TURRET",
                AssetType.LOITERING_MUNITION: "MUNITION",
                AssetType.MORTAR_SYSTEM: "MORTAR",
                AssetType.RECON_DRONE: "RECON",
                AssetType.ATTACK_DRONE: "ATTACK",
            }
            prefix = prefix_map.get(asset_type, "WEAPON")
            counter = self._id_counters.get(asset_type, 0) + 1
            self._id_counters[asset_type] = counter
            weapon_id = f"{prefix}-{counter:02d}"

        # Apply default profile for this type
        profile = self.PROFILES.get(asset_type, {})

        weapon = WeaponState(
            weapon_id=weapon_id,
            asset_type=asset_type,
            lat=lat,
            lng=lng,
            ammo_count=profile.get("ammo_count", -1),
            threat_radius_m=profile.get("threat_radius_m", 500.0),
            reload_time_s=profile.get("reload_time_s", 5.0),
        )

        self.weapons[weapon_id] = weapon
        log_jadc2.info(
            "🎯 WEAPON SPAWNED: %s [%s] at (%.4f, %.4f) — radius: %.0fm, ammo: %s",
            weapon_id, asset_type.value, lat, lng,
            weapon.threat_radius_m,
            "∞" if weapon.ammo_count == -1 else str(weapon.ammo_count),
        )
        return weapon

    def destroy(self, weapon_id: WeaponId) -> bool:
        """Remove a weapon system from the registry."""
        if weapon_id in self.weapons:
            weapon = self.weapons.pop(weapon_id)
            log_jadc2.info(
                "💥 WEAPON DESTROYED: %s [%s] — kills: %d",
                weapon_id, weapon.asset_type.value, weapon.kills,
            )
            return True
        return False

    def get(self, weapon_id: WeaponId) -> Optional[WeaponState]:
        """Retrieve weapon state by ID."""
        return self.weapons.get(weapon_id)

    def all_active(self) -> List[WeaponState]:
        """Return all weapons not OFFLINE or EXPENDED."""
        return [
            w for w in self.weapons.values()
            if w.status not in (WeaponStatus.OFFLINE, WeaponStatus.EXPENDED)
        ]

    def count(self) -> int:
        """Total weapons in registry."""
        return len(self.weapons)


# ══════════════════════════════════════════════════════════════════
#  GEO HELPER — Haversine distance (meters)
# ══════════════════════════════════════════════════════════════════

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two lat/lng points."""
    R = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    return 2.0 * R * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════
#  THREAT ROUTER — Optimal effector selection
# ══════════════════════════════════════════════════════════════════

class ThreatRouter:
    """Given a threat coordinate, find the optimal weapon to engage.

    Scoring: proximity (40%) + readiness (35%) + ammo (25%)
    Only considers weapons with status IDLE or TRACKING and within range.
    """

    WEIGHT_PROXIMITY: float = 0.40
    WEIGHT_READINESS: float = 0.35
    WEIGHT_AMMO: float = 0.25

    def __init__(self, registry: WeaponRegistry) -> None:
        self.registry = registry

    def find_best_effector(
        self,
        threat_lat: float,
        threat_lng: float,
    ) -> List[Tuple[WeaponState, float]]:
        """Return ranked list of (weapon, score) for a threat coordinate.

        Score is 0.0–1.0 (higher = better engagement option).
        Only includes weapons within their threat_radius_m.
        """
        candidates: List[Tuple[WeaponState, float, float]] = []

        for weapon in self.registry.all_active():
            if weapon.status not in (WeaponStatus.IDLE, WeaponStatus.TRACKING):
                continue

            dist_m = _haversine_m(
                weapon.lat, weapon.lng, threat_lat, threat_lng,
            )

            if dist_m > weapon.threat_radius_m:
                continue  # out of range

            candidates.append((weapon, dist_m, weapon.threat_radius_m))

        if not candidates:
            return []

        # Score each candidate
        scored: List[Tuple[WeaponState, float]] = []
        max_range = max(c[2] for c in candidates)

        for weapon, dist_m, max_r in candidates:
            # Proximity score: closer = better (normalized 0–1)
            proximity_score = 1.0 - (dist_m / max_r) if max_r > 0 else 1.0

            # Readiness score: IDLE > TRACKING
            readiness_score = 1.0 if weapon.status == WeaponStatus.IDLE else 0.6

            # Ammo score: unlimited = 1.0, else normalize
            if weapon.ammo_count == -1:
                ammo_score = 1.0
            elif weapon.ammo_count == 0:
                ammo_score = 0.0
            else:
                ammo_score = min(1.0, weapon.ammo_count / 10.0)

            total = (
                self.WEIGHT_PROXIMITY * proximity_score
                + self.WEIGHT_READINESS * readiness_score
                + self.WEIGHT_AMMO * ammo_score
            )

            scored.append((weapon, round(total, 3)))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        log_jadc2.info(
            "🎯 THREAT ROUTING: (%.4f, %.4f) — %d effectors in range, "
            "best: %s (score: %.3f)",
            threat_lat, threat_lng, len(scored),
            scored[0][0].weapon_id if scored else "NONE",
            scored[0][1] if scored else 0.0,
        )

        return scored


# ══════════════════════════════════════════════════════════════════
#  KILL CHAIN MANAGER — F2T2EA Orchestration
# ══════════════════════════════════════════════════════════════════

class KillChainManager:
    """Orchestrates the Find → Fix → Track → Target → Engage → Assess loop.

    Each engagement progresses through phases automatically.
    Broadcasts status updates via the alert callback.
    """

    # Phase durations (simulated, in seconds)
    PHASE_DURATIONS: Dict[EngagementPhase, float] = {
        EngagementPhase.FIND: 0.0,     # instant (drone already detected)
        EngagementPhase.FIX: 2.0,      # confirm coordinates
        EngagementPhase.TRACK: 3.0,    # maintain lock
        EngagementPhase.TARGET: 1.0,   # assign effector
        EngagementPhase.ENGAGE: 2.0,   # fire
        EngagementPhase.ASSESS: 3.0,   # BDA (battle damage assessment)
    }

    def __init__(
        self,
        registry: WeaponRegistry,
        router: ThreatRouter,
        lock: asyncio.Lock,
    ) -> None:
        self.registry = registry
        self.router = router
        self.lock = lock
        self.engagements: Dict[str, Engagement] = {}
        self._engagement_counter: int = 0
        self._alerts: List[dict] = []

    def create_engagement(
        self,
        threat_lat: float,
        threat_lng: float,
        detecting_drone_id: Optional[str] = None,
        confidence: float = 0.9,
    ) -> Optional[Engagement]:
        """Initiate a new kill chain engagement.

        Returns the Engagement if effectors are available, None otherwise.
        """
        # Check for available effectors
        options = self.router.find_best_effector(threat_lat, threat_lng)
        if not options:
            log_jadc2.warning(
                "⚠ NO EFFECTORS IN RANGE for threat at (%.4f, %.4f)",
                threat_lat, threat_lng,
            )
            return None

        self._engagement_counter += 1
        eng_id = f"ENG-{self._engagement_counter:04d}"

        best_weapon, score = options[0]

        engagement = Engagement(
            engagement_id=eng_id,
            threat_lat=threat_lat,
            threat_lng=threat_lng,
            phase=EngagementPhase.FIND,
            assigned_weapon_id=best_weapon.weapon_id,
            detecting_drone_id=detecting_drone_id,
            confidence=confidence,
        )

        self.engagements[eng_id] = engagement

        # Update weapon status
        best_weapon.status = WeaponStatus.TRACKING
        best_weapon.assigned_target = eng_id

        log_jadc2.warning(
            "🔥 KILL CHAIN INITIATED: %s — threat at (%.4f, %.4f) | "
            "weapon: %s [%s] | confidence: %.0f%% | score: %.3f",
            eng_id, threat_lat, threat_lng,
            best_weapon.weapon_id, best_weapon.asset_type.value,
            confidence * 100, score,
        )

        self._alerts.append({
            "type": "jadc2_alert",
            "alert": "KILL_CHAIN_INITIATED",
            "engagement_id": eng_id,
            "threat_lat": round(threat_lat, 6),
            "threat_lng": round(threat_lng, 6),
            "weapon_id": best_weapon.weapon_id,
            "weapon_type": best_weapon.asset_type.value,
            "confidence": round(confidence, 2),
            "score": score,
            "timestamp": time.time(),
        })

        return engagement

    async def tick(self) -> None:
        """Advance all active engagements through their kill chain phases."""
        async with self.lock:
            now = time.time()
            completed: List[str] = []

            for eng_id, eng in self.engagements.items():
                if eng.phase in (EngagementPhase.COMPLETE, EngagementPhase.ABORTED):
                    continue

                phase_dur = self.PHASE_DURATIONS.get(eng.phase, 2.0)
                elapsed = now - eng.created_at

                # Progress through phases based on elapsed time
                cumulative = 0.0
                target_phase = EngagementPhase.FIND
                for phase in [
                    EngagementPhase.FIND,
                    EngagementPhase.FIX,
                    EngagementPhase.TRACK,
                    EngagementPhase.TARGET,
                    EngagementPhase.ENGAGE,
                    EngagementPhase.ASSESS,
                ]:
                    cumulative += self.PHASE_DURATIONS.get(phase, 2.0)
                    if elapsed < cumulative:
                        target_phase = phase
                        break
                else:
                    target_phase = EngagementPhase.COMPLETE

                # Phase transition
                if target_phase != eng.phase:
                    old_phase = eng.phase
                    eng.phase = target_phase

                    log_jadc2.info(
                        "⚡ %s: %s → %s | weapon: %s",
                        eng_id, old_phase.value, target_phase.value,
                        eng.assigned_weapon_id,
                    )

                    # Handle ENGAGE phase — fire the weapon
                    if target_phase == EngagementPhase.ENGAGE:
                        self._execute_fire(eng)

                    # Handle COMPLETE phase
                    if target_phase == EngagementPhase.COMPLETE:
                        eng.completed_at = now
                        eng.result = "NEUTRALIZED"
                        self._complete_engagement(eng)
                        completed.append(eng_id)

                    self._alerts.append({
                        "type": "jadc2_alert",
                        "alert": "PHASE_TRANSITION",
                        "engagement_id": eng_id,
                        "old_phase": old_phase.value,
                        "new_phase": target_phase.value,
                        "weapon_id": eng.assigned_weapon_id,
                        "timestamp": time.time(),
                    })

    def _execute_fire(self, eng: Engagement) -> None:
        """Fire the assigned weapon."""
        if not eng.assigned_weapon_id:
            return

        weapon = self.registry.get(eng.assigned_weapon_id)
        if not weapon:
            return

        weapon.status = WeaponStatus.ENGAGED
        weapon.last_fired = time.time()

        # Decrement ammo (unless unlimited)
        if weapon.ammo_count > 0:
            weapon.ammo_count -= 1

        log_jadc2.warning(
            "💥 WEAPON FIRED: %s [%s] → %s | ammo remaining: %s",
            weapon.weapon_id, weapon.asset_type.value,
            eng.engagement_id,
            "∞" if weapon.ammo_count == -1 else str(weapon.ammo_count),
        )

    def _complete_engagement(self, eng: Engagement) -> None:
        """Finalize a completed engagement — update weapon status."""
        if not eng.assigned_weapon_id:
            return

        weapon = self.registry.get(eng.assigned_weapon_id)
        if not weapon:
            return

        weapon.kills += 1
        weapon.assigned_target = None

        # Check if weapon is expended (loitering munitions)
        if weapon.ammo_count == 0:
            weapon.status = WeaponStatus.EXPENDED
            log_jadc2.warning(
                "⚠ WEAPON EXPENDED: %s [%s] — %d total kills",
                weapon.weapon_id, weapon.asset_type.value, weapon.kills,
            )
        else:
            # Enter reload cycle, then back to IDLE
            weapon.status = WeaponStatus.RELOADING
            log_jadc2.info(
                "🔄 %s RELOADING (%.1fs) — kills: %d, ammo: %s",
                weapon.weapon_id, weapon.reload_time_s, weapon.kills,
                "∞" if weapon.ammo_count == -1 else str(weapon.ammo_count),
            )

        log_jadc2.warning(
            "✅ ENGAGEMENT COMPLETE: %s — result: %s | weapon: %s",
            eng.engagement_id, eng.result, eng.assigned_weapon_id,
        )

        self._alerts.append({
            "type": "jadc2_alert",
            "alert": "ENGAGEMENT_COMPLETE",
            "engagement_id": eng.engagement_id,
            "result": eng.result,
            "weapon_id": eng.assigned_weapon_id,
            "timestamp": time.time(),
        })

    def abort_engagement(self, engagement_id: str) -> bool:
        """Abort an active engagement."""
        eng = self.engagements.get(engagement_id)
        if not eng or eng.phase in (EngagementPhase.COMPLETE, EngagementPhase.ABORTED):
            return False

        eng.phase = EngagementPhase.ABORTED
        eng.completed_at = time.time()
        eng.result = "ABORTED"

        # Release weapon
        if eng.assigned_weapon_id:
            weapon = self.registry.get(eng.assigned_weapon_id)
            if weapon:
                weapon.status = WeaponStatus.IDLE
                weapon.assigned_target = None

        log_jadc2.warning(
            "🛑 ENGAGEMENT ABORTED: %s | weapon %s released",
            engagement_id, eng.assigned_weapon_id,
        )
        return True

    def pop_alerts(self) -> List[dict]:
        """Return and clear pending alerts for WebSocket broadcast."""
        alerts = list(self._alerts)
        self._alerts.clear()
        return alerts

    async def reload_check(self) -> None:
        """Check and transition weapons from RELOADING → IDLE."""
        async with self.lock:
            now = time.time()
            for weapon in self.registry.weapons.values():
                if weapon.status == WeaponStatus.RELOADING:
                    if now - weapon.last_fired >= weapon.reload_time_s:
                        weapon.status = WeaponStatus.IDLE
                        log_jadc2.info(
                            "✅ %s RELOAD COMPLETE — ready to engage",
                            weapon.weapon_id,
                        )

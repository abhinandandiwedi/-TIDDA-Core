"""
🦗 TIDDA V2.0 Swarm Logic Module
Contains: SwarmBrain (AI autonomy), FlightPhysics (flight model),
          TacticalEventEngine (dynamic tactical events),
          WakeTriggerEngine (Perch & Strike event system),
          GridPlanner (N-node area decomposition)

All classes use asyncio.Lock for single-threaded async concurrency.
V2.0: Fully N-node compatible — zero hardcoded drone IDs.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════
#  TYPE ALIASES
# ══════════════════════════════════════════════════════════════════

DroneId = str
DroneState = Dict[str, Any]
DroneRegistry = Dict[DroneId, DroneState]

# ── Module loggers ────────────────────────────────────────────────

log_sim = logging.getLogger("SIM")
log_brain = logging.getLogger("SWARM-BRAIN")


# ══════════════════════════════════════════════════════════════════
#  SWARM BRAIN AI — AUTONOMY + MANET MESH
# ══════════════════════════════════════════════════════════════════

class SwarmBrain:
    """Autonomous decision engine running as an asyncio task.

    Monitors the shared DRONES registry every second and enforces:
      - Battery safety:  force RTB when bat < 20%  (HIGHEST PRIORITY)
      - MANET mesh relay: switch to MESH_RELAY on signal loss
      - Tactical coordination:
          • Threat Response — SCANNING drones converge toward THREAT drone
          • Mesh Healing   — nearest SCANNING drone closes gap to MESH_RELAY drone
      - Stealth-Perch:
          • hide() — navigate to nearest hiding spot, cut motors, go dark
          • Silent Alert — detect threats while perched without moving
      - Wake Triggers (V2.0):
          • Acoustic Spike — sound threshold → wake PERCHED drone
          • Motion Event   — VSLAM/YOLO detection → wake PERCHED drone
          • Manual Callout — squad API request → wake PERCHED drone
    """

    BATTERY_RTB_THRESHOLD: float = 20.0
    SIGNAL_RELAY_THRESHOLD: float = 30.0

    # How much GPS to nudge per tick (degrees ≈ ~5-10m per step)
    CONVERGENCE_RATE: float = 0.0002
    MESH_HEAL_RATE: float = 0.00015

    # Stealth-Perch parameters
    STEALTH_MOVE_RATE: float = 0.0003    # Faster transit to hiding spot
    PERCH_ARRIVAL_THRESHOLD: float = 0.0002  # Close enough = arrived
    PERCH_BATTERY_DRAIN: float = 0.01    # Near-zero drain while perched
    SILENT_ALERT_CHANCE: float = 0.08    # 8% chance per tick of detecting threat

    # Pre-defined hiding spots on the tactical map
    HIDING_SPOTS: Dict[str, Dict[str, float]] = {
        "POLE_A":     {"lat": 28.6232, "lng": 77.4541, "alt": 4.0},
        "ROOF_EDGE":  {"lat": 28.6238, "lng": 77.4552, "alt": 6.0},
        "SHADOW_B":   {"lat": 28.6244, "lng": 77.4545, "alt": 3.0},
        "ALLEY_D":    {"lat": 28.6228, "lng": 77.4548, "alt": 2.5},
        "OVERHANG_E": {"lat": 28.6241, "lng": 77.4538, "alt": 5.0},
    }

    def __init__(self, drones: DroneRegistry, lock: asyncio.Lock) -> None:
        self.drones = drones
        self.lock = lock
        # Track which drones are heading to which hiding spot
        self._stealth_targets: Dict[DroneId, str] = {}
        # V2.0: Wake trigger engine for Perch & Strike
        self._wake_engine = WakeTriggerEngine(drones)
        # V2.0: Pending wake queue from passive detection
        self._wake_queue: List[Dict[str, Any]] = []

    async def decision_loop(self) -> None:
        """Continuous AI decision loop — runs forever."""
        log_brain.info(
            "SwarmBrain online — autonomy + MANET mesh + "
            "tactical coordination + stealth-perch",
        )
        while True:
            try:
                async with self.lock:
                    # ── Pass 1: Safety (highest priority) ─────────
                    for drone_id, state in self.drones.items():
                        self._check_battery_safety(drone_id, state)

                    # ── Pass 2: Mesh signal checks ────────────────
                    for drone_id, state in self.drones.items():
                        self._check_mesh_signal(drone_id, state)

                    # ── Pass 3: Tactical coordination ─────────────
                    self._tactical_threat_response()
                    self._tactical_mesh_healing()

                    # ── Pass 4: Stealth movement ──────────────────
                    self._process_stealth_movement()

                    # ── Pass 5: Perched detection ─────────────────
                    self._perch_detection_sweep()

                    # ── Pass 6: Wake triggers (V2.0) ──────────────
                    self._process_wake_triggers()

            except Exception:
                log_brain.exception("Error in decision loop tick")
            await asyncio.sleep(1)

    # ── Safety ────────────────────────────────────────────────────

    def _check_battery_safety(self, drone_id: DroneId, state: DroneState) -> None:
        """Force RTB if battery is critically low."""
        if (
            state["bat"] < self.BATTERY_RTB_THRESHOLD
            and state["status"] not in ("RTB", "LANDED")
        ):
            # Cancel any stealth mission — safety overrides
            self._stealth_targets.pop(drone_id, None)
            state["status"] = "RTB"
            log_brain.warning(
                "%s battery critical (%.1f%%) — autonomous RTB triggered",
                drone_id, state["bat"],
            )

    # ── Mesh signal ───────────────────────────────────────────────

    def _check_mesh_signal(self, drone_id: DroneId, state: DroneState) -> None:
        """Switch to MESH_RELAY when signal strength drops."""
        signal: float = state.get("signal_strength", 100.0)
        if (
            signal < self.SIGNAL_RELAY_THRESHOLD
            and state["status"] not in ("RTB", "LANDED", "PERCHED", "STEALTH_MOVING")
        ):
            state["status"] = "MESH_RELAY"
            log_brain.warning(
                "%s signal weak (%.0f) — switching to MESH_RELAY",
                drone_id, signal,
            )

    # ── Tactical: Threat Response ─────────────────────────────────

    def _tactical_threat_response(self) -> None:
        """If any drone has THREAT status, converge SCANNING drones toward it.

        Drones in RTB/LANDED are never moved (battery safety takes priority).
        """
        threat_drones = [
            (did, s) for did, s in self.drones.items()
            if s["status"] == "THREAT"
        ]
        if not threat_drones:
            return

        available = [
            (did, s) for did, s in self.drones.items()
            if s["status"] == "SCANNING"
        ]
        if not available:
            return

        for threat_id, threat_state in threat_drones:
            t_lat: float = threat_state["lat"]
            t_lng: float = threat_state["lng"]

            for helper_id, helper_state in available:
                h_lat: float = helper_state["lat"]
                h_lng: float = helper_state["lng"]

                # Vector toward the threat drone
                dlat = t_lat - h_lat
                dlng = t_lng - h_lng
                dist = (dlat ** 2 + dlng ** 2) ** 0.5

                if dist < 0.0001:
                    continue  # Already co-located

                # Normalize and apply convergence step
                step_lat = (dlat / dist) * self.CONVERGENCE_RATE
                step_lng = (dlng / dist) * self.CONVERGENCE_RATE

                helper_state["lat"] = round(h_lat + step_lat, 6)
                helper_state["lng"] = round(h_lng + step_lng, 6)

                log_brain.info(
                    "⚔ THREAT RESPONSE: %s converging toward %s "
                    "(Δ%.4f°, Δ%.4f°)",
                    helper_id, threat_id, step_lat, step_lng,
                )

    # ── Tactical: Mesh Healing ────────────────────────────────────

    def _tactical_mesh_healing(self) -> None:
        """If a drone is in MESH_RELAY, move the nearest SCANNING drone closer.

        Drones in RTB/LANDED are never moved (battery safety takes priority).
        """
        relay_drones = [
            (did, s) for did, s in self.drones.items()
            if s["status"] == "MESH_RELAY"
        ]
        if not relay_drones:
            return

        available = [
            (did, s) for did, s in self.drones.items()
            if s["status"] == "SCANNING"
        ]
        if not available:
            return

        for relay_id, relay_state in relay_drones:
            r_lat: float = relay_state["lat"]
            r_lng: float = relay_state["lng"]

            # Find the nearest SCANNING drone
            nearest_id: str = ""
            nearest_state: DroneState = {}
            nearest_dist: float = float("inf")

            for cand_id, cand_state in available:
                dlat = r_lat - cand_state["lat"]
                dlng = r_lng - cand_state["lng"]
                dist = (dlat ** 2 + dlng ** 2) ** 0.5
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = cand_id
                    nearest_state = cand_state

            if not nearest_id or nearest_dist < 0.0001:
                continue  # No candidate or already co-located

            # Move the nearest drone toward the relay drone
            dlat = r_lat - nearest_state["lat"]
            dlng = r_lng - nearest_state["lng"]
            step_lat = (dlat / nearest_dist) * self.MESH_HEAL_RATE
            step_lng = (dlng / nearest_dist) * self.MESH_HEAL_RATE

            nearest_state["lat"] = round(nearest_state["lat"] + step_lat, 6)
            nearest_state["lng"] = round(nearest_state["lng"] + step_lng, 6)

            log_brain.info(
                "🔗 MESH HEAL: %s moving toward %s to strengthen link "
                "(dist=%.5f°, step=%.4f°)",
                nearest_id, relay_id, nearest_dist, self.MESH_HEAL_RATE,
            )

    # ══════════════════════════════════════════════════════════════
    #  STEALTH-PERCH CAPABILITY
    # ══════════════════════════════════════════════════════════════

    def hide(self, drone_id: DroneId) -> bool:
        """Command a drone to navigate to the nearest hiding spot and perch.

        Returns True if the hide command was accepted, False if the drone
        is in a state that prevents hiding (RTB, LANDED, already perched).
        """
        state = self.drones.get(drone_id)
        if not state:
            log_brain.warning("HIDE: unknown drone %s", drone_id)
            return False

        if state["status"] in ("RTB", "LANDED", "PERCHED", "STEALTH_MOVING"):
            log_brain.info(
                "HIDE: %s rejected — current status %s",
                drone_id, state["status"],
            )
            return False

        # Find the nearest hiding spot
        nearest_spot: str = ""
        nearest_dist: float = float("inf")

        for spot_name, spot_coords in self.HIDING_SPOTS.items():
            dlat = spot_coords["lat"] - state["lat"]
            dlng = spot_coords["lng"] - state["lng"]
            dist = (dlat ** 2 + dlng ** 2) ** 0.5
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_spot = spot_name

        if not nearest_spot:
            return False

        # Begin stealth transit
        self._stealth_targets[drone_id] = nearest_spot
        state["status"] = "STEALTH_MOVING"

        spot = self.HIDING_SPOTS[nearest_spot]
        log_brain.warning(
            "🫥 STEALTH: %s moving to hiding spot %s "
            "(%.4f°N, %.4f°E, %.1fm) — dist=%.5f°",
            drone_id, nearest_spot,
            spot["lat"], spot["lng"], spot["alt"], nearest_dist,
        )
        return True

    def _process_stealth_movement(self) -> None:
        """Move STEALTH_MOVING drones toward their hiding spots."""
        completed: List[DroneId] = []

        for drone_id, spot_name in self._stealth_targets.items():
            state = self.drones.get(drone_id)
            if not state or state["status"] != "STEALTH_MOVING":
                completed.append(drone_id)
                continue

            spot = self.HIDING_SPOTS.get(spot_name)
            if not spot:
                completed.append(drone_id)
                continue

            # Calculate vector to hiding spot
            dlat = spot["lat"] - state["lat"]
            dlng = spot["lng"] - state["lng"]
            dist = (dlat ** 2 + dlng ** 2) ** 0.5

            if dist < self.PERCH_ARRIVAL_THRESHOLD:
                # Arrived — perch and cut motors
                state["lat"] = spot["lat"]
                state["lng"] = spot["lng"]
                state["alt"] = 0.0
                state["status"] = "PERCHED"
                completed.append(drone_id)
                log_brain.warning(
                    "🪺 PERCHED: %s hidden at %s — motors cut, "
                    "drain=%.2f%%/tick",
                    drone_id, spot_name, self.PERCH_BATTERY_DRAIN,
                )
            else:
                # Move toward spot
                step_lat = (dlat / dist) * self.STEALTH_MOVE_RATE
                step_lng = (dlng / dist) * self.STEALTH_MOVE_RATE
                state["lat"] = round(state["lat"] + step_lat, 6)
                state["lng"] = round(state["lng"] + step_lng, 6)

                # Also descend toward target altitude
                alt_diff = spot["alt"] - state["alt"]
                if abs(alt_diff) > 0.3:
                    state["alt"] = round(
                        state["alt"] + alt_diff * 0.15, 1,
                    )

                # Minimal drain during stealth transit
                state["bat"] = round(
                    max(0.0, state["bat"] - self.PERCH_BATTERY_DRAIN * 3), 1,
                )

        # Clean up completed targets
        for drone_id in completed:
            self._stealth_targets.pop(drone_id, None)

        # Apply near-zero drain to all PERCHED drones
        for drone_id, state in self.drones.items():
            if state["status"] == "PERCHED":
                state["bat"] = round(
                    max(0.0, state["bat"] - self.PERCH_BATTERY_DRAIN), 1,
                )

    def _perch_detection_sweep(self) -> None:
        """Perched drones passively detect threats via acoustic/thermal mock.

        If a threat is detected, a SILENT ALERT is broadcast to GCS.
        V2.0: High-confidence detections are queued as wake triggers.
        """
        for drone_id, state in self.drones.items():
            if state["status"] != "PERCHED":
                continue

            # Simulate passive sensor detection
            if random.random() < self.SILENT_ALERT_CHANCE:
                detection_type = random.choice(["ACOUSTIC", "THERMAL", "RF_EMISSION"])
                confidence = round(random.uniform(0.6, 0.98), 2)

                # Store the alert in the drone state for GCS to pick up
                state["silent_alert"] = {
                    "type": detection_type,
                    "confidence": confidence,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                log_brain.warning(
                    "🔇 SILENT ALERT from %s [PERCHED at %.4f°N, %.4f°E] — "
                    "%s detection (confidence: %.0f%%) — drone holding position",
                    drone_id,
                    state["lat"], state["lng"],
                    detection_type,
                    confidence * 100,
                )

                # V2.0: Queue high-confidence detections as wake triggers
                if confidence >= 0.75:
                    trigger_type = (
                        WakeTriggerType.ACOUSTIC_SPIKE
                        if detection_type == "ACOUSTIC"
                        else WakeTriggerType.MOTION_EVENT
                    )
                    self._wake_queue.append({
                        "drone_id": drone_id,
                        "trigger_type": trigger_type,
                        "confidence": confidence,
                        "lat": state["lat"],
                        "lng": state["lng"],
                    })

    def _process_wake_triggers(self) -> None:
        """V2.0: Process pending wake triggers from passive detection."""
        for wake_event in self._wake_queue:
            self._wake_engine.trigger_wake(
                drone_id=wake_event["drone_id"],
                trigger_type=wake_event["trigger_type"],
                source_lat=wake_event["lat"],
                source_lng=wake_event["lng"],
            )
        self._wake_queue.clear()

    def wake(self, drone_id: DroneId, trigger_type: "WakeTriggerType") -> bool:
        """V2.0: External API to wake a specific perched drone.

        Used by the C2 server for MANUAL_CALLOUT wake triggers.
        """
        state = self.drones.get(drone_id)
        if not state:
            return False
        return self._wake_engine.trigger_wake(
            drone_id=drone_id,
            trigger_type=trigger_type,
        )



# ══════════════════════════════════════════════════════════════════
#  FLIGHT PHYSICS — altitude, GPS drift, battery drain
# ══════════════════════════════════════════════════════════════════

class FlightPhysics:
    """Pure flight-model calculations. Stateless — operates on drone dicts."""

    NORMAL_DRAIN: float = 0.1
    WEATHER_DRAIN: float = 0.4
    EVASION_DRAIN: float = 0.25
    CLIMB_DRAIN: float = 0.08
    LANDING_DRAIN: float = 0.05

    @staticmethod
    def apply_climb(
        state: DroneState,
        target_alt: float,
        climb_steps: int = 20,
    ) -> None:
        """Increment altitude toward *target_alt* and drain battery."""
        if state["alt"] < target_alt:
            state["alt"] = round(
                min(state["alt"] + target_alt / climb_steps, target_alt), 1,
            )
            state["status"] = "CLIMBING" if state["alt"] < target_alt else "SCANNING"
        state["bat"] = round(max(0.0, state["bat"] - FlightPhysics.CLIMB_DRAIN), 1)
        FlightPhysics._gps_drift(state, magnitude=0.0001)

    @staticmethod
    def apply_hover(state: DroneState, target_alt: float) -> None:
        """Normal patrol hover: gentle altitude jitter + GPS drift."""
        state["alt"] = round(target_alt + random.uniform(-0.3, 0.3), 1)
        state["bat"] = round(max(0.0, state["bat"] - FlightPhysics.NORMAL_DRAIN), 1)
        FlightPhysics._gps_drift(state, magnitude=0.00005)

    @staticmethod
    def apply_weather_hold(state: DroneState, target_alt: float) -> None:
        """Turbulent weather: heavy altitude jitter, GPS scatter, 4× drain."""
        turbulence = random.uniform(-2.0, 2.0)
        state["alt"] = round(max(1.0, target_alt + turbulence), 1)
        state["status"] = "WEATHER_HOLD"
        state["bat"] = round(max(0.0, state["bat"] - FlightPhysics.WEATHER_DRAIN), 1)
        FlightPhysics._gps_drift(state, magnitude=0.0003)

    @staticmethod
    def apply_evasion(
        state: DroneState,
        target_alt: float,
        phase: int,
    ) -> bool:
        """Evasion maneuver phases. Returns True when evasion is complete."""
        state["status"] = "EVADING"

        if phase <= 3:
            # Aggressive dive
            state["alt"] = round(max(1.5, state["alt"] - 3.0), 1)
            FlightPhysics._lateral_dodge(state, magnitude=0.0005)
        elif phase <= 6:
            # Low-altitude jink
            state["alt"] = round(
                max(1.5, state["alt"] + random.uniform(-0.5, 0.3)), 1,
            )
            FlightPhysics._gps_drift(state, magnitude=0.0002)
        elif phase <= 10:
            # Recovery climb
            recover_rate = (target_alt - state["alt"]) / max(1, 10 - phase)
            state["alt"] = round(
                min(target_alt, state["alt"] + max(recover_rate, 0.5)), 1,
            )
        else:
            # Evasion complete
            state["alt"] = round(target_alt + random.uniform(-0.3, 0.3), 1)
            state["status"] = "SCANNING"
            return True

        state["bat"] = round(max(0.0, state["bat"] - FlightPhysics.EVASION_DRAIN), 1)
        return False

    @staticmethod
    def apply_landing(state: DroneState, target_alt: float) -> None:
        """Descend toward ground and mark LANDED when alt ≤ 0.2m."""
        if state["status"] == "LANDED":
            return
        state["alt"] = round(max(0.0, state["alt"] - target_alt / 14), 1)
        state["status"] = "RTB"
        state["bat"] = round(max(0.0, state["bat"] - FlightPhysics.LANDING_DRAIN), 1)
        if state["alt"] <= 0.2:
            state["alt"] = 0.0
            state["status"] = "LANDED"

    # ── Internal helpers ──

    @staticmethod
    def _gps_drift(state: DroneState, magnitude: float) -> None:
        state["lat"] += random.uniform(-magnitude, magnitude)
        state["lng"] += random.uniform(-magnitude, magnitude)

    @staticmethod
    def _lateral_dodge(state: DroneState, magnitude: float) -> None:
        state["lat"] += random.choice([-1, 1]) * magnitude
        state["lng"] += random.choice([-1, 1]) * magnitude


# ══════════════════════════════════════════════════════════════════
#  TACTICAL EVENT ENGINE — weather, evasion, threat triggers
# ══════════════════════════════════════════════════════════════════

class TacticalEventEngine:
    """Manages dynamic tactical events during the scan phase.

    All state mutations happen under the shared asyncio.Lock.
    """

    WEATHER_CHANCE: float = 0.15
    EVASION_CHANCE: float = 0.10
    WEATHER_MIN_TICK: int = 5
    EVASION_MIN_TICK: int = 10
    FIXED_THREAT_TICK: int = 20
    FIXED_LOWBAT_TICK: int = 40
    LOW_BAT_THRESHOLD: float = 55.0

    def __init__(
        self,
        drones: DroneRegistry,
        target_alt: Dict[DroneId, int],
        lock: asyncio.Lock,
    ) -> None:
        self.drones = drones
        self.target_alt = target_alt
        self.lock = lock

        # Event trackers
        self.weather: Dict[str, Any] = {
            "active": False, "drone": None, "ticks_left": 0,
        }
        self.evasion: Dict[str, Any] = {
            "active": False, "drone": None, "phase": 0, "original_alt": 0.0,
        }

    # ── Per-tick processing ───────────────────────────────────────

    async def tick(self, step: int) -> None:
        """Run one simulation tick: physics + event triggers."""
        async with self.lock:
            for drone_id in self.drones:
                self._process_drone(drone_id, step)

        await self._tick_weather()
        self._maybe_trigger_weather(step)
        self._maybe_trigger_evasion(step)
        self._fixed_events(step)

    def _process_drone(self, drone_id: DroneId, step: int) -> None:
        """Apply physics to a single drone based on its active event."""
        state = self.drones[drone_id]
        tgt = self.target_alt.get(drone_id, 10)  # V2.0: default alt for dynamic nodes

        # Default to SCANNING if not in a special state
        # V2.0: Added PERCHED, STEALTH_MOVING, ARMED, LANDED to exclusion list
        if state["status"] not in (
            "WEATHER_HOLD", "EVADING", "THREAT", "RTB",
            "PERCHED", "STEALTH_MOVING", "ARMED", "LANDED",
        ):
            state["status"] = "SCANNING"

        if self.weather["active"] and self.weather["drone"] == drone_id:
            FlightPhysics.apply_weather_hold(state, tgt)

        elif self.evasion["active"] and self.evasion["drone"] == drone_id:
            done = FlightPhysics.apply_evasion(state, tgt, self.evasion["phase"])
            self.evasion["phase"] += 1
            if done:
                self.evasion["active"] = False
                log_sim.info("✅ %s evasion complete — resuming patrol", drone_id)
        else:
            FlightPhysics.apply_hover(state, tgt)

    # ── Weather lifecycle ─────────────────────────────────────────

    async def _tick_weather(self) -> None:
        if not self.weather["active"]:
            return
        self.weather["ticks_left"] -= 1
        if self.weather["ticks_left"] <= 0:
            w_drone: DroneId = self.weather["drone"]
            self.weather["active"] = False
            async with self.lock:
                if self.drones[w_drone]["status"] == "WEATHER_HOLD":
                    self.drones[w_drone]["status"] = "SCANNING"
                    self.drones[w_drone]["alt"] = round(
                        self.target_alt[w_drone], 1,
                    )
            log_sim.info("🌤️  Weather cleared for %s — normal ops", w_drone)

    def _maybe_trigger_weather(self, step: int) -> None:
        if (
            not self.weather["active"]
            and not self.evasion["active"]
            and step > self.WEATHER_MIN_TICK
            and random.random() < self.WEATHER_CHANCE
        ):
            candidates = self._scanning_drones()
            if candidates:
                target = random.choice(candidates)
                duration = random.randint(6, 12)
                self.weather = {
                    "active": True, "drone": target, "ticks_left": duration,
                }
                log_sim.warning(
                    "🌩️  WEATHER ANOMALY — %s | turbulence %d ticks "
                    "(4× drain, ±2m jitter)",
                    target, duration,
                )

    # ── Evasion lifecycle ─────────────────────────────────────────

    def _maybe_trigger_evasion(self, step: int) -> None:
        if (
            not self.evasion["active"]
            and not self.weather["active"]
            and step > self.EVASION_MIN_TICK
            and random.random() < self.EVASION_CHANCE
        ):
            candidates = self._scanning_drones()
            if candidates:
                target = random.choice(candidates)
                orig_alt = self.drones[target]["alt"]
                self.evasion = {
                    "active": True, "drone": target,
                    "phase": 0, "original_alt": orig_alt,
                }
                log_sim.warning(
                    "🚨 DYNAMIC THREAT EVASION — %s (alt %.1fm) "
                    "| Dive → Jink → Recover",
                    target, orig_alt,
                )

    # ── Fixed / legacy events ─────────────────────────────────────

    def _fixed_events(self, step: int) -> None:
        if step == self.FIXED_THREAT_TICK:
            t_drone = random.choice(list(self.drones.keys()))
            self.drones[t_drone]["status"] = "THREAT"
            log_sim.warning("⚠️  THREAT detected — %s", t_drone)

        if step == self.FIXED_LOWBAT_TICK:
            low_bat = [
                d for d in self.drones
                if self.drones[d]["bat"] < self.LOW_BAT_THRESHOLD
            ]
            if low_bat:
                rtb_drone = min(low_bat, key=lambda x: self.drones[x]["bat"])
                self.drones[rtb_drone]["status"] = "RTB"
                log_sim.info(
                    "%s low battery (%.1f%%) → RTB",
                    rtb_drone, self.drones[rtb_drone]["bat"],
                )

    # ── Utility ───────────────────────────────────────────────────

    def _scanning_drones(self) -> List[DroneId]:
        return [d for d in self.drones if self.drones[d]["status"] == "SCANNING"]


# ══════════════════════════════════════════════════════════════════
#  V2.0 — WAKE TRIGGER ENGINE (Perch & Strike)
# ══════════════════════════════════════════════════════════════════

class WakeTriggerType(enum.Enum):
    """Types of events that can wake a PERCHED drone."""
    ACOUSTIC_SPIKE = "ACOUSTIC_SPIKE"    # Sound threshold breached
    MOTION_EVENT = "MOTION_EVENT"        # VSLAM/YOLO motion detection
    MANUAL_CALLOUT = "MANUAL_CALLOUT"    # Squad requests overwatch via API


class WakeTriggerEngine:
    """Manages event-driven wake transitions for PERCHED sentinel drones.

    A drone in PERCHED mode (0.8W power) must instantly transition to
    ARMED or SCANNING when a wake trigger fires.

    Trigger Types:
      ACOUSTIC_SPIKE — db > threshold → PERCHED → SCANNING
      MOTION_EVENT   — VSLAM/YOLO confidence > threshold → PERCHED → ARMED
      MANUAL_CALLOUT — squad API request → PERCHED → ARMED
    """

    # Configurable thresholds
    ACOUSTIC_DB_THRESHOLD: float = 75.0
    MOTION_CONFIDENCE_THRESHOLD: float = 0.7

    # Wake transition targets
    WAKE_TARGET: Dict[WakeTriggerType, str] = {
        WakeTriggerType.ACOUSTIC_SPIKE: "SCANNING",
        WakeTriggerType.MOTION_EVENT: "ARMED",
        WakeTriggerType.MANUAL_CALLOUT: "ARMED",
    }

    # Power draw in PERCHED mode (watts) — for display only
    PERCH_POWER_W: float = 0.8

    def __init__(self, drones: DroneRegistry) -> None:
        self.drones = drones
        # Log of recent wake events
        self.wake_log: List[Dict[str, Any]] = []

    def trigger_wake(
        self,
        drone_id: DroneId,
        trigger_type: WakeTriggerType,
        source_lat: Optional[float] = None,
        source_lng: Optional[float] = None,
        db_level: float = 0.0,
        confidence: float = 0.0,
    ) -> bool:
        """Attempt to wake a PERCHED drone.

        Returns True if the drone was woken, False otherwise.
        """
        state = self.drones.get(drone_id)
        if not state:
            log_brain.warning("WAKE: unknown drone %s", drone_id)
            return False

        if state["status"] != "PERCHED":
            log_brain.info(
                "WAKE: %s rejected — status is %s (not PERCHED)",
                drone_id, state["status"],
            )
            return False

        # Determine target state based on trigger type
        target_status = self.WAKE_TARGET.get(trigger_type, "SCANNING")

        # Apply the wake transition
        state["status"] = target_status

        # Store wake event for GCS
        wake_event = {
            "drone_id": drone_id,
            "trigger_type": trigger_type.value,
            "target_status": target_status,
            "source_lat": source_lat,
            "source_lng": source_lng,
            "db_level": db_level,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        state["wake_event"] = wake_event
        self.wake_log.append(wake_event)

        # Keep log bounded
        if len(self.wake_log) > 100:
            self.wake_log = self.wake_log[-50:]

        log_brain.warning(
            "⚡ WAKE TRIGGER: %s | %s → %s | trigger: %s"
            " | source: (%.4f, %.4f)",
            drone_id, "PERCHED", target_status,
            trigger_type.value,
            source_lat or 0.0, source_lng or 0.0,
        )

        return True

    def inject_area_event(
        self,
        trigger_type: WakeTriggerType,
        event_lat: float,
        event_lng: float,
        radius_m: float = 200.0,
        db_level: float = 85.0,
        confidence: float = 0.9,
    ) -> List[DroneId]:
        """Inject an area-wide event that wakes ALL perched drones within radius.

        Used by the C2 server for inject_event commands.
        Returns list of drone IDs that were woken.
        """
        woken: List[DroneId] = []

        for drone_id, state in self.drones.items():
            if state["status"] != "PERCHED":
                continue

            # Check if drone is within event radius
            dlat = (event_lat - state["lat"]) * 111_320.0
            dlng = (event_lng - state["lng"]) * 111_320.0 * math.cos(
                math.radians(event_lat),
            )
            dist_m = math.hypot(dlat, dlng)

            if dist_m <= radius_m:
                if self.trigger_wake(
                    drone_id, trigger_type,
                    source_lat=event_lat,
                    source_lng=event_lng,
                    db_level=db_level,
                    confidence=confidence,
                ):
                    woken.append(drone_id)

        if woken:
            log_brain.warning(
                "⚡ AREA WAKE: %s at (%.4f, %.4f) radius %.0fm — "
                "woke %d drones: %s",
                trigger_type.value, event_lat, event_lng, radius_m,
                len(woken), ", ".join(woken),
            )

        return woken


# ══════════════════════════════════════════════════════════════════
#  V2.0 — GRID PLANNER (N-Node Area Decomposition)
# ══════════════════════════════════════════════════════════════════

class GridPlanner:
    """Divide a scan area mathematically by N available drones.

    Given a rectangular area (defined by SW/NE corners) and N drones,
    computes optimal waypoint assignments using centroid-based grid
    partitioning.

    Usage:
        planner = GridPlanner()
        waypoints = planner.divide_area(
            sw_lat=28.620, sw_lng=77.450,
            ne_lat=28.630, ne_lng=77.460,
            n_drones=8,
        )
        # Returns: {'TIDDA-01': [(lat, lng), ...], 'TIDDA-02': [...], ...}
    """

    @staticmethod
    def divide_area(
        sw_lat: float,
        sw_lng: float,
        ne_lat: float,
        ne_lng: float,
        n_drones: int,
    ) -> List[Dict[str, Any]]:
        """Divide a rectangular area into N sub-sectors with centroids.

        Returns a list of N dicts, each containing:
          - sector_id: int (0-indexed)
          - centroid_lat, centroid_lng: center of the sector
          - bounds: {sw_lat, sw_lng, ne_lat, ne_lng}
          - waypoints: list of (lat, lng) patrol points within sector
        """
        if n_drones <= 0:
            return []

        # Compute optimal grid dimensions (rows × cols) closest to N
        cols = max(1, round(math.sqrt(n_drones * (ne_lng - sw_lng) / max(0.0001, ne_lat - sw_lat))))
        rows = max(1, math.ceil(n_drones / cols))

        # Recalculate cols to avoid excess sectors
        while rows * cols < n_drones:
            cols += 1

        lat_step = (ne_lat - sw_lat) / rows
        lng_step = (ne_lng - sw_lng) / cols

        sectors: List[Dict[str, Any]] = []
        sector_id = 0

        for r in range(rows):
            for c in range(cols):
                if sector_id >= n_drones:
                    break

                s_sw_lat = sw_lat + r * lat_step
                s_sw_lng = sw_lng + c * lng_step
                s_ne_lat = s_sw_lat + lat_step
                s_ne_lng = s_sw_lng + lng_step

                centroid_lat = (s_sw_lat + s_ne_lat) / 2
                centroid_lng = (s_sw_lng + s_ne_lng) / 2

                # Generate patrol waypoints within sector (lawnmower pattern)
                waypoints = GridPlanner._lawnmower_pattern(
                    s_sw_lat, s_sw_lng, s_ne_lat, s_ne_lng, passes=3,
                )

                sectors.append({
                    "sector_id": sector_id,
                    "centroid_lat": round(centroid_lat, 6),
                    "centroid_lng": round(centroid_lng, 6),
                    "bounds": {
                        "sw_lat": round(s_sw_lat, 6),
                        "sw_lng": round(s_sw_lng, 6),
                        "ne_lat": round(s_ne_lat, 6),
                        "ne_lng": round(s_ne_lng, 6),
                    },
                    "waypoints": waypoints,
                })
                sector_id += 1

        log_brain.info(
            "📐 GRID PLAN: area (%.4f,%.4f)→(%.4f,%.4f) divided into "
            "%d sectors for %d drones (%d×%d grid)",
            sw_lat, sw_lng, ne_lat, ne_lng,
            len(sectors), n_drones, rows, cols,
        )

        return sectors

    @staticmethod
    def _lawnmower_pattern(
        sw_lat: float,
        sw_lng: float,
        ne_lat: float,
        ne_lng: float,
        passes: int = 3,
    ) -> List[Tuple[float, float]]:
        """Generate a lawnmower search pattern within a rectangular sector."""
        waypoints: List[Tuple[float, float]] = []
        lat_step = (ne_lat - sw_lat) / max(1, passes)

        for i in range(passes + 1):
            lat = sw_lat + i * lat_step
            if i % 2 == 0:
                # West to East
                waypoints.append((round(lat, 6), round(sw_lng, 6)))
                waypoints.append((round(lat, 6), round(ne_lng, 6)))
            else:
                # East to West
                waypoints.append((round(lat, 6), round(ne_lng, 6)))
                waypoints.append((round(lat, 6), round(sw_lng, 6)))

        return waypoints

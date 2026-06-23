"""
🦗 TIDDA V2.0 C2 SERVER — Infinite Scaling Architecture
Port: 8000 | Path: /ws/ui
Run:  python tidda_c2.py

Architecture:
  - swarm_logic.py     → SwarmBrain, FlightPhysics, TacticalEventEngine,
                         WakeTriggerEngine, GridPlanner
  - weapon_systems.py  → WeaponRegistry, ThreatRouter, KillChainManager
  - tidda_c2.py        → Server, WebSocket handler, async mission sim

V2.0: Dynamic N-node registry — zero hardcoded drone IDs.
      Event-driven wake system (Perch & Strike).
      JADC2 effector/weapon integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set

# ── Dependency check ──────────────────────────────────────────────
try:
    import websockets
    import websockets.exceptions
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets
    import websockets.exceptions

# ── Local imports ─────────────────────────────────────────────────
from swarm_logic import (
    DroneId,
    DroneRegistry,
    FlightPhysics,
    GridPlanner,
    SwarmBrain,
    TacticalEventEngine,
    WakeTriggerType,
)
from weapon_systems import (
    AssetType,
    KillChainManager,
    ThreatRouter,
    WeaponRegistry,
)

# ══════════════════════════════════════════════════════════════════
#  LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════════

LOG_FORMAT = (
    "%(asctime)s │ %(levelname)-7s │ %(name)-12s │ %(message)s"
)
DATE_FORMAT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_main = logging.getLogger("MAIN")
log_sim = logging.getLogger("SIM")
log_ws = logging.getLogger("WEBSOCKET")
log_cmd = logging.getLogger("CMD")


# ══════════════════════════════════════════════════════════════════
#  V2.0 — DYNAMIC SWARM REGISTRY (N-Node)
# ══════════════════════════════════════════════════════════════════

# Home base coordinates
HOME_LAT: float = 28.6235
HOME_LNG: float = 77.4543
DEFAULT_ALT: int = 10

# Dynamic registries — populated at runtime via spawn functions
DRONES: DroneRegistry = {}
TARGET_ALT: Dict[DroneId, int] = {}

# asyncio.Lock — guards ALL reads/writes to DRONES across async tasks
drone_lock = asyncio.Lock()

# Track connected WebSocket clients for broadcast
connected_clients: Set[websockets.WebSocketServerProtocol] = set()

# V2.0: Weapon systems
weapon_registry = WeaponRegistry()
threat_router = ThreatRouter(weapon_registry)
kill_chain: Optional[KillChainManager] = None  # Initialized in main()

# V2.0: SwarmBrain reference for wake commands
_brain_ref: Optional[SwarmBrain] = None


# ══════════════════════════════════════════════════════════════════
#  V2.0 — DYNAMIC SPAWN / DESTROY
# ══════════════════════════════════════════════════════════════════

_drone_counter: int = 0


def spawn_drone(
    drone_id: Optional[str] = None,
    lat: float = HOME_LAT,
    lng: float = HOME_LNG,
    bat: float = 85.0,
    target_alt: int = DEFAULT_ALT,
) -> str:
    """Spawn a new drone into the registry at runtime.

    If drone_id is None, auto-generates sequential ID (TIDDA-01, TIDDA-02, ...).
    Returns the drone_id.
    """
    global _drone_counter

    if drone_id is None:
        _drone_counter += 1
        drone_id = f"TIDDA-{_drone_counter:02d}"
    else:
        # Track highest ID for auto-generation
        try:
            num = int(drone_id.split("-")[-1])
            _drone_counter = max(_drone_counter, num)
        except (ValueError, IndexError):
            pass

    DRONES[drone_id] = {
        "lat": lat,
        "lng": lng,
        "alt": 0.0,
        "bat": round(bat, 1),
        "status": "STANDBY",
    }
    TARGET_ALT[drone_id] = target_alt

    log_main.info(
        "🚀 SPAWN: %s at (%.4f, %.4f) bat=%.1f%% alt_target=%dm",
        drone_id, lat, lng, bat, target_alt,
    )
    return drone_id


def destroy_drone(drone_id: str) -> bool:
    """Remove a drone from the registry at runtime."""
    if drone_id in DRONES:
        DRONES.pop(drone_id)
        TARGET_ALT.pop(drone_id, None)
        log_main.info("💀 DESTROY: %s removed from registry", drone_id)
        return True
    return False


def auto_spawn_swarm(
    n: int = 4,
    home_lat: float = HOME_LAT,
    home_lng: float = HOME_LNG,
) -> List[str]:
    """Spawn N drones with computed offsets using golden-angle spiral.

    Returns list of spawned drone IDs.
    """
    ids: List[str] = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ~2.399 radians

    for i in range(n):
        # Golden-angle spiral for even distribution
        radius = 0.0003 * math.sqrt(i + 1)  # ~30-50m offsets
        angle = i * golden_angle
        dlat = radius * math.cos(angle)
        dlng = radius * math.sin(angle)

        bat = round(random.uniform(65.0, 95.0), 1)
        alt = random.choice([8, 10, 12, 15])

        drone_id = spawn_drone(
            lat=home_lat + dlat,
            lng=home_lng + dlng,
            bat=bat,
            target_alt=alt,
        )
        ids.append(drone_id)

    log_main.info(
        "✅ SWARM SPAWNED: %d drones at (%.4f, %.4f) — IDs: %s",
        n, home_lat, home_lng, ", ".join(ids),
    )
    return ids


def spawn_default_weapons() -> None:
    """Spawn default weapon systems alongside the drone swarm."""
    weapon_registry.spawn(
        AssetType.AUTO_TURRET,
        lat=HOME_LAT + 0.001,
        lng=HOME_LNG - 0.001,
    )
    weapon_registry.spawn(
        AssetType.LOITERING_MUNITION,
        lat=HOME_LAT - 0.002,
        lng=HOME_LNG + 0.002,
    )
    weapon_registry.spawn(
        AssetType.MORTAR_SYSTEM,
        lat=HOME_LAT + 0.003,
        lng=HOME_LNG + 0.001,
    )
    log_main.info(
        "🎯 WEAPONS SPAWNED: %d effectors in kill web",
        weapon_registry.count(),
    )


# ══════════════════════════════════════════════════════════════════
#  ASYNC MISSION SIMULATION
# ══════════════════════════════════════════════════════════════════

async def run_mission_loop(
    drones: DroneRegistry,
    target_alt: Dict[DroneId, int],
    lock: asyncio.Lock,
) -> None:
    """Full mission lifecycle: ARM → CLIMB → SCAN → LAND → RESET (loops forever)."""
    while True:
        try:
            await _execute_mission(drones, target_alt, lock)
        except Exception:
            log_sim.exception("Mission loop crashed — restarting in 5s")
            await asyncio.sleep(5)


async def _execute_mission(
    drones: DroneRegistry,
    target_alt: Dict[DroneId, int],
    lock: asyncio.Lock,
) -> None:
    """Single mission execution — fully async, N-node compatible."""
    await asyncio.sleep(2)
    log_sim.info("Mission starting... (%d drones)", len(drones))

    # ── Phase 1: Arm ──────────────────────────────────────────────
    async with lock:
        drone_ids = list(drones.keys())
    for drone_id in drone_ids:
        async with lock:
            if drone_id in drones:
                drones[drone_id]["status"] = "ARMED"
        await asyncio.sleep(0.5)
        log_sim.info("%s ARMED", drone_id)

    # ── Phase 2: Climb ────────────────────────────────────────────
    log_sim.info("Takeoff — climbing to patrol altitude...")
    for _ in range(30):
        async with lock:
            for drone_id in list(drones.keys()):
                tgt = target_alt.get(drone_id, DEFAULT_ALT)
                FlightPhysics.apply_climb(drones[drone_id], tgt)
        await asyncio.sleep(0.5)

    # ── Phase 3: Scan + Tactical Events ───────────────────────────
    log_sim.info("Scanning zones — tactical events armed...")
    tac_engine = TacticalEventEngine(drones, target_alt, lock)
    scan_ticks: int = 60

    for step in range(scan_ticks):
        try:
            # Inject dynamic signal strength for MANET mesh testing
            async with lock:
                for drone_id in list(drones.keys()):
                    if drone_id in drones:
                        drones[drone_id]["signal_strength"] = random.randint(20, 100)
            await tac_engine.tick(step)
        except Exception:
            log_sim.exception("Tactical tick %d failed — continuing", step)
        await asyncio.sleep(0.8)

    # ── Phase 4: Land ─────────────────────────────────────────────
    log_sim.info("Landing sequence initiated...")
    for _ in range(20):
        async with lock:
            for drone_id in list(drones.keys()):
                tgt = target_alt.get(drone_id, DEFAULT_ALT)
                FlightPhysics.apply_landing(drones[drone_id], tgt)
        await asyncio.sleep(0.5)

    log_sim.info("✅ All drones landed — mission complete")

    # ── Reset for next mission ────────────────────────────────────
    await asyncio.sleep(10)
    log_sim.info("Resetting for next mission...")
    async with lock:
        for drone_id in list(drones.keys()):
            drones[drone_id].update({
                "alt": 0.0,
                "bat": round(random.uniform(65.0, 90.0), 1),
                "status": "STANDBY",
            })


# ══════════════════════════════════════════════════════════════════
#  V2.0 — COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════

def handle_command(raw: str) -> Optional[dict]:
    """Parse and execute a JSON command from the GCS dashboard.

    V2.0 commands:
      spawn_node     — add a drone to the swarm at runtime
      destroy_node   — remove a drone from the swarm
      wake_drone     — wake a PERCHED drone via manual callout
      inject_event   — inject acoustic/motion event at coordinates
      scan_area      — divide area by N drones for grid scan
      spawn_weapon   — add a weapon system
      destroy_weapon — remove a weapon system
      engage_target  — initiate kill chain on threat coordinates

    Returns a response dict if applicable, None otherwise.
    """
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if msg.get("type") != "command":
        return None

    action = msg.get("action", "")

    # ── SPAWN NODE ────────────────────────────────────────────────
    if action == "spawn_node":
        lat = msg.get("lat", HOME_LAT + random.uniform(-0.003, 0.003))
        lng = msg.get("lng", HOME_LNG + random.uniform(-0.003, 0.003))
        bat = msg.get("battery", random.uniform(70.0, 95.0))
        alt = msg.get("target_alt", DEFAULT_ALT)
        drone_id = msg.get("drone_id")
        new_id = spawn_drone(drone_id=drone_id, lat=lat, lng=lng, bat=bat, target_alt=alt)
        log_cmd.info("✅ Spawned %s via GCS command", new_id)
        return {"type": "response", "action": "spawn_node", "drone_id": new_id, "status": "ok"}

    # ── DESTROY NODE ──────────────────────────────────────────────
    if action == "destroy_node":
        drone_id = msg.get("target_unit", "")
        if destroy_drone(drone_id):
            log_cmd.info("✅ Destroyed %s via GCS command", drone_id)
            return {"type": "response", "action": "destroy_node", "drone_id": drone_id, "status": "ok"}
        log_cmd.warning("❌ Destroy failed — unknown drone: %s", drone_id)
        return {"type": "response", "action": "destroy_node", "drone_id": drone_id, "status": "error"}

    # ── WAKE DRONE (V2.0 Perch & Strike) ──────────────────────────
    if action == "wake_drone":
        drone_id = msg.get("target_unit", "")
        trigger = msg.get("trigger", "MANUAL_CALLOUT")
        try:
            trigger_type = WakeTriggerType(trigger)
        except ValueError:
            trigger_type = WakeTriggerType.MANUAL_CALLOUT

        if _brain_ref:
            success = _brain_ref.wake(drone_id, trigger_type)
            log_cmd.info("WAKE %s: %s (%s)", drone_id, "OK" if success else "FAILED", trigger)
            return {"type": "response", "action": "wake_drone", "drone_id": drone_id,
                    "status": "ok" if success else "error"}

    # ── INJECT EVENT (V2.0 area wake) ─────────────────────────────
    if action == "inject_event":
        event_type = msg.get("event_type", "ACOUSTIC_SPIKE")
        lat = msg.get("lat", HOME_LAT)
        lng = msg.get("lng", HOME_LNG)
        radius = msg.get("radius_m", 200.0)
        db_level = msg.get("db_level", 85.0)

        try:
            trigger_type = WakeTriggerType(event_type)
        except ValueError:
            trigger_type = WakeTriggerType.ACOUSTIC_SPIKE

        if _brain_ref:
            woken = _brain_ref._wake_engine.inject_area_event(
                trigger_type, lat, lng, radius, db_level,
            )
            return {"type": "response", "action": "inject_event",
                    "woken_drones": woken, "status": "ok"}

    # ── SCAN AREA (V2.0 grid decomposition) ───────────────────────
    if action == "scan_area":
        sw_lat = msg.get("sw_lat", HOME_LAT - 0.005)
        sw_lng = msg.get("sw_lng", HOME_LNG - 0.005)
        ne_lat = msg.get("ne_lat", HOME_LAT + 0.005)
        ne_lng = msg.get("ne_lng", HOME_LNG + 0.005)
        n = len(DRONES)

        sectors = GridPlanner.divide_area(sw_lat, sw_lng, ne_lat, ne_lng, n)
        log_cmd.info("📐 Grid planned: %d sectors for %d drones", len(sectors), n)
        return {"type": "response", "action": "scan_area", "sectors": sectors, "status": "ok"}

    # ── SPAWN WEAPON (V2.0 JADC2) ────────────────────────────────
    if action == "spawn_weapon":
        asset_type_str = msg.get("asset_type", "AUTO_TURRET")
        lat = msg.get("lat", HOME_LAT)
        lng = msg.get("lng", HOME_LNG)
        try:
            asset_type = AssetType(asset_type_str)
        except ValueError:
            asset_type = AssetType.AUTO_TURRET
        weapon = weapon_registry.spawn(asset_type, lat, lng)
        return {"type": "response", "action": "spawn_weapon",
                "weapon_id": weapon.weapon_id, "status": "ok"}

    # ── DESTROY WEAPON ────────────────────────────────────────────
    if action == "destroy_weapon":
        weapon_id = msg.get("weapon_id", "")
        success = weapon_registry.destroy(weapon_id)
        return {"type": "response", "action": "destroy_weapon",
                "weapon_id": weapon_id, "status": "ok" if success else "error"}

    # ── ENGAGE TARGET (V2.0 kill chain) ───────────────────────────
    if action == "engage_target":
        lat = msg.get("lat", HOME_LAT)
        lng = msg.get("lng", HOME_LNG)
        confidence = msg.get("confidence", 0.9)
        drone_id = msg.get("detecting_drone", None)

        if kill_chain:
            eng = kill_chain.create_engagement(lat, lng, drone_id, confidence)
            if eng:
                return {"type": "response", "action": "engage_target",
                        "engagement_id": eng.engagement_id, "status": "ok"}
            return {"type": "response", "action": "engage_target",
                    "status": "error", "reason": "no_effectors_in_range"}

    log_cmd.warning("Unknown command action: %s", action)
    return None


# ══════════════════════════════════════════════════════════════════
#  WEBSOCKET HANDLER — with stale connection cleanup
# ══════════════════════════════════════════════════════════════════

async def ws_handler(
    websocket: websockets.WebSocketServerProtocol,
    path: Optional[str] = None,
) -> None:
    """Stream live telemetry to a single GCS client.

    V2.0: Dynamically broadcasts N drones + M weapons.
    Processes incoming commands for spawn/destroy/wake/engage.
    """
    client_addr = getattr(websocket, "remote_address", "unknown")
    connected_clients.add(websocket)
    log_ws.info("GCS connected: %s  (total clients: %d)", client_addr, len(connected_clients))

    try:
        # Spawn two tasks: one for sending telemetry, one for receiving commands
        send_task = asyncio.create_task(_telemetry_sender(websocket, client_addr))
        recv_task = asyncio.create_task(_command_receiver(websocket, client_addr))

        # Wait for either to finish (client disconnect)
        done, pending = await asyncio.wait(
            {send_task, recv_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except Exception:
        log_ws.exception("Unexpected error in WS handler for %s", client_addr)
    finally:
        connected_clients.discard(websocket)
        log_ws.info(
            "Cleaned up client %s  (remaining: %d)",
            client_addr, len(connected_clients),
        )


async def _telemetry_sender(
    websocket: websockets.WebSocketServerProtocol,
    client_addr: str,
) -> None:
    """Send telemetry for all drones + weapons at 2 Hz."""
    try:
        while True:
            # Snapshot drones under lock
            async with drone_lock:
                snapshot: DroneRegistry = {k: dict(v) for k, v in DRONES.items()}

            # Send drone telemetry
            for drone_id, state in snapshot.items():
                msg = json.dumps({
                    "drone_id":    drone_id,
                    "altitude_m":  state["alt"],
                    "battery_pct": round(state["bat"]),
                    "lat":         round(state["lat"], 6),
                    "lng":         round(state["lng"], 6),
                    "status":      state["status"],
                    "timestamp":   time.time(),
                })
                await websocket.send(msg)

            # V2.0: Send weapon telemetry
            for weapon in weapon_registry.weapons.values():
                await websocket.send(json.dumps(weapon.to_telemetry()))

            # V2.0: Send JADC2 alerts
            if kill_chain:
                for alert in kill_chain.pop_alerts():
                    await websocket.send(json.dumps(alert))

            await asyncio.sleep(0.5)

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        log_ws.exception("Telemetry sender error for %s", client_addr)


async def _command_receiver(
    websocket: websockets.WebSocketServerProtocol,
    client_addr: str,
) -> None:
    """Listen for incoming commands from GCS."""
    try:
        async for message in websocket:
            if isinstance(message, str) and message.strip():
                async with drone_lock:
                    response = handle_command(message)
                if response:
                    try:
                        await websocket.send(json.dumps(response))
                    except websockets.exceptions.ConnectionClosed:
                        return
    except websockets.exceptions.ConnectionClosed:
        pass


# ══════════════════════════════════════════════════════════════════
#  V2.0 — KILL CHAIN TICK LOOP
# ══════════════════════════════════════════════════════════════════

async def kill_chain_loop() -> None:
    """Tick the kill chain manager + weapon reload checks at 1 Hz."""
    while True:
        try:
            if kill_chain:
                await kill_chain.tick()
                await kill_chain.reload_check()
        except Exception:
            log_main.exception("Kill chain tick error")
        await asyncio.sleep(1.0)


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

async def main() -> None:
    """Start all async tasks: simulation, SwarmBrain, JADC2, and WebSocket server."""
    global kill_chain, _brain_ref

    log_main.info("=" * 56)
    log_main.info("  🦗 TIDDA V2.0 C2 SERVER — Infinite Scaling Architecture")
    log_main.info("  WebSocket: ws://localhost:8000/ws/ui")
    log_main.info("  Pillars: N-Node | Perch & Strike | JADC2 Kill Web")
    log_main.info("=" * 56)

    # V2.0: Dynamic swarm spawn (default 4, can be changed)
    auto_spawn_swarm(n=4, home_lat=HOME_LAT, home_lng=HOME_LNG)

    # V2.0: Spawn default weapons
    spawn_default_weapons()

    # V2.0: Initialize kill chain manager
    kill_chain = KillChainManager(weapon_registry, threat_router, drone_lock)

    # Mission simulation — async task
    asyncio.create_task(
        run_mission_loop(DRONES, TARGET_ALT, drone_lock),
    )

    # SwarmBrain AI — async task
    brain = SwarmBrain(drones=DRONES, lock=drone_lock)
    _brain_ref = brain
    asyncio.create_task(brain.decision_loop())

    # V2.0: Kill chain tick loop
    asyncio.create_task(kill_chain_loop())

    log_main.info("Server starting on ws://localhost:8000/ws/ui")
    log_main.info("Open GCS HTML in browser now!")
    log_main.info("Swarm: %d drones | Weapons: %d effectors", len(DRONES), weapon_registry.count())

    # WebSocket server — runs forever
    async with websockets.serve(ws_handler, "localhost", 8000):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log_main.info("Server shut down by operator (Ctrl+C)")
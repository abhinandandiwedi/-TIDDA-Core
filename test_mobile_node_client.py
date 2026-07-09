# ══════════════════════════════════════════════════════════════════
#  🧪 TEST TOOL — TIDDA Mobile Node Client Simulator
#  Purpose: Validates the mobile-node integration WITHOUT a real phone.
#           Run this AFTER the main server is already started.
#  Usage:   python test_mobile_node_client.py
#
#  This script is a developer test tool ONLY and is NOT part of the
#  production system. Do not ship it to a phone.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import asyncio
import json
import math
import random
import sys
import time

# Reuse the server's HOME_LAT/HOME_LON values as the simulation origin.
HOME_LAT: float = -35.3633
HOME_LON: float = 149.1652
WS_URL: str = "ws://localhost:8000/ws/ui"
NODE_ID: str = "PHONE-01"
TICK_INTERVAL_S: float = 2.0   # how often we send a telemetry packet
HEARTBEAT_EVERY: int = 3       # send a heartbeat on every 3rd tick instead


try:
    import websockets
except ImportError:
    print("[TEST] websockets not found — install with:  pip install websockets")
    sys.exit(1)


def _jitter(base: float, scale: float = 0.0005) -> float:
    """Return base ± random offset for simulating GPS drift."""
    return base + random.uniform(-scale, scale)


def _build_telemetry(tick: int, battery: float) -> dict:
    """Construct a realistic mobile telemetry payload."""
    heading = (tick * 15) % 360          # rotate 15° per tick
    speed = round(random.uniform(0.0, 3.5), 2)
    signal = random.randint(-110, -60)   # typical RSSI range
    return {
        "type":            "telemetry",
        "lat":             _jitter(HOME_LAT),
        "lon":             _jitter(HOME_LON),
        "altitude_m":      round(random.uniform(0.0, 5.0), 1),
        "battery_pct":     round(battery, 1),
        "heading_deg":     round(heading, 1),
        "speed_mps":       speed,
        "camera_active":   (tick % 4 != 0),    # camera off every 4th tick
        "network_type":    random.choice(["LTE", "5G", "WIFI"]),
        "signal_strength": signal,
        "status":          "SCANNING",
    }


async def run_phone_client() -> None:
    """Connect as a mobile node, send telemetry/heartbeats, then disconnect."""
    print(f"[TEST] Connecting to {WS_URL} as {NODE_ID} …")

    async with websockets.connect(WS_URL) as ws:
        # 1. Register as a mobile node first.
        reg_msg = json.dumps({"type": "node_register", "node_id": NODE_ID})
        await ws.send(reg_msg)
        print(f"[TEST] Sent node_register for {NODE_ID}")

        battery = 95.0
        tick = 0
        total_ticks = 10   # run for 10 ticks (~20 s) then exit cleanly

        while tick < total_ticks:
            await asyncio.sleep(TICK_INTERVAL_S)
            tick += 1
            battery = max(0.0, battery - random.uniform(0.3, 1.2))

            if tick % HEARTBEAT_EVERY == 0:
                # Send a heartbeat (keepalive only — no telemetry fields)
                hb = json.dumps({"type": "heartbeat", "node_id": NODE_ID})
                await ws.send(hb)
                print(f"[TEST] tick={tick:02d}  ♥ heartbeat sent")
            else:
                telem = _build_telemetry(tick, battery)
                await ws.send(json.dumps(telem))
                print(
                    f"[TEST] tick={tick:02d}  📡 telemetry  "
                    f"bat={telem['battery_pct']}%  "
                    f"lat={telem['lat']:.5f}  lon={telem['lon']:.5f}"
                )

        print(f"[TEST] {total_ticks} ticks complete — disconnecting cleanly.")


async def run_dashboard_listener(results: list) -> None:
    """Open a second WS connection as a GCS dashboard and capture messages.

    Records any message whose drone_id == NODE_ID so the caller can assert
    the schema is correct.
    """
    print(f"[TEST] Dashboard listener connecting to {WS_URL} …")
    try:
        async with websockets.connect(WS_URL) as ws:
            deadline = time.monotonic() + 15.0   # wait up to 15 s
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("drone_id") == NODE_ID:
                    results.append(msg)
                    print(f"[TEST] ✅ Dashboard received PHONE-01 packet: {msg}")
                    return   # got what we needed
                if time.monotonic() > deadline:
                    print("[TEST] ⏱ Timeout waiting for PHONE-01 packet.")
                    return
    except Exception as e:
        print(f"[TEST] Dashboard listener error: {e}")


async def main() -> None:
    results: list = []

    # Run phone client and dashboard listener concurrently.
    await asyncio.gather(
        run_phone_client(),
        run_dashboard_listener(results),
        return_exceptions=True,
    )

    # ── Assertions ────────────────────────────────────────────────
    print("\n[TEST] ══ SCHEMA VERIFICATION ══")
    if not results:
        print("[TEST] ❌ FAIL — Dashboard never received a PHONE-01 telemetry packet.")
        sys.exit(1)

    pkt = results[0]
    required_keys = {"drone_id", "lat", "lng", "altitude_m", "battery_pct",
                     "status", "timestamp", "node_type"}
    missing = required_keys - pkt.keys()
    if missing:
        print(f"[TEST] ❌ FAIL — Missing keys in packet: {missing}")
        sys.exit(1)

    assert pkt["drone_id"] == NODE_ID,     f"Wrong drone_id: {pkt['drone_id']}"
    assert pkt["node_type"] == "mobile",   f"Wrong node_type: {pkt['node_type']}"
    assert 0 <= pkt["battery_pct"] <= 100, f"battery_pct out of range: {pkt['battery_pct']}"

    print(f"[TEST] ✅ PASS — All required keys present and values valid.")
    print(f"[TEST] Packet dump: {json.dumps(pkt, indent=2)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted.")

# ──────────────────────────────────────────────────────────
# TIDDA MASTER — Unified Takeoff + Live C2 Telemetry Stream
# One script, zero manual switching, zero failsafes.
# ──────────────────────────────────────────────────────────

# Python 3.10+ collections fix (must be FIRST)
import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from dronekit import connect, VehicleMode
from pymavlink import mavutil
import asyncio
import websockets
import json
import time

# ──────── CONFIG ────────
SITL_CONNECTION   = "tcp:127.0.0.1:5760"
C2_WEBSOCKET_URI  = "ws://localhost:8000/ws/telemetry"
TAKEOFF_ALT       = 15        # meters — target altitude for takeoff command
ALT_CONFIRM       = 3.0       # meters — altitude to confirm takeoff success
TELEMETRY_HZ      = 0.5       # seconds — telemetry push interval

# ArduCopter mode numbers
GUIDED_MODE_NUM   = 4


def set_mode_via_mavlink(vehicle, mode_num):
    """Send SET_MODE directly via MAVLink — bypasses DroneKit's broken mode setter."""
    msg = vehicle.message_factory.set_mode_encode(
        0,                                  # target system (0 = broadcast)
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_num                            # custom_mode: 4 = GUIDED
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def arm_via_mavlink(vehicle):
    """Send ARM command directly via MAVLink."""
    msg = vehicle.message_factory.command_long_encode(
        0, 0,                                           # target system, component
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,   # command
        0,                                               # confirmation
        1,                                               # param1: 1 = arm
        0, 0, 0, 0, 0, 0                                # params 2-7 unused
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


# ═══════════════════════════════════════════════════════════
#  PHASE 1 — CONNECT & WAIT FOR FULL INITIALIZATION
# ═══════════════════════════════════════════════════════════

print("[TIDDA MASTER] Connecting to SITL drone...")
vehicle = connect(SITL_CONNECTION, wait_ready=True)
print(f"[+] Connected! Firmware: {vehicle.version}")
print(f"[+] Vehicle mode: {vehicle.mode.name}")

# Disable all arming checks
print("[PRE-ARM] Disabling arming checks...")
vehicle.parameters['ARMING_CHECK'] = 0
time.sleep(2)

# Wait for GPS 3D fix
print("[BOOT] Waiting for GPS 3D fix...")
for i in range(60):
    gps_fix = vehicle.gps_0.fix_type
    sats = vehicle.gps_0.satellites_visible
    if gps_fix is not None and gps_fix >= 3:
        print(f"[+] GPS 3D fix! (type={gps_fix}, sats={sats})")
        break
    print(f"   GPS: fix={gps_fix}, sats={sats}...")
    time.sleep(2)

# Wait for armable
print("[BOOT] Waiting for vehicle to become armable...")
for i in range(30):
    if vehicle.is_armable:
        print("[+] Vehicle is armable!")
        break
    print(f"   -> Not armable yet... ({i*2}s)")
    time.sleep(2)

# Let everything fully settle
print("[BOOT] Waiting 25s for full EKF/IMU convergence...")
for remaining in range(25, 0, -1):
    print(f"   Settling: {remaining}s...")
    time.sleep(1)
print("[+] System settled!")


# ═══════════════════════════════════════════════════════════
#  PHASE 2 — SET GUIDED VIA RAW MAVLINK + ARM + TAKEOFF
# ═══════════════════════════════════════════════════════════

# Use raw MAVLink to set GUIDED — DroneKit's VehicleMode() doesn't work
# with ArduCopter 3.3 in this SITL
print("[CMD] Setting GUIDED mode via MAVLink (mode=4)...")
for attempt in range(30):
    set_mode_via_mavlink(vehicle, GUIDED_MODE_NUM)
    time.sleep(2)
    current_mode = vehicle.mode.name
    if current_mode == "GUIDED":
        print(f"[+] GUIDED confirmed on attempt {attempt+1}!")
        break
    if attempt % 5 == 4:
        print(f"   -> Attempt {attempt+1}: mode is {current_mode}")
else:
    print(f"[!] Mode still {vehicle.mode.name} after 30 attempts")
    print("[!] Attempting arm+takeoff in current mode anyway...")

print(f"[+] Current mode: {vehicle.mode.name}")

# Arm via MAVLink
print("[CMD] Arming via MAVLink...")
arm_via_mavlink(vehicle)
for attempt in range(20):
    time.sleep(1)
    if vehicle.armed:
        print(f"[+] ARMED on attempt {attempt+1}!")
        break
    if attempt % 3 == 2:
        arm_via_mavlink(vehicle)
        print(f"   -> Re-sending arm command... attempt {attempt+1}")

if not vehicle.armed:
    print("[FAIL] Could not arm! Exiting.")
    vehicle.close()
    sys.exit(1)

# Force GUIDED one more time right before takeoff
set_mode_via_mavlink(vehicle, GUIDED_MODE_NUM)
time.sleep(1)
print(f"[+] Mode at takeoff: {vehicle.mode.name}")

# TAKEOFF
print(f"[CMD] Commanding takeoff to {TAKEOFF_ALT}m!")
vehicle.simple_takeoff(TAKEOFF_ALT)

# Keep forcing GUIDED + re-sending takeoff for the first few seconds
for i in range(5):
    time.sleep(1)
    if vehicle.mode.name != "GUIDED":
        set_mode_via_mavlink(vehicle, GUIDED_MODE_NUM)
        time.sleep(0.5)
        vehicle.simple_takeoff(TAKEOFF_ALT)
        print(f"   -> Re-forced GUIDED + takeoff (pass {i+1})")

# Monitor altitude
print(f"[MONITOR] Waiting to cross {ALT_CONFIRM}m...")
timeout = 60
start = time.time()
while True:
    current_alt = vehicle.location.global_relative_frame.alt or 0.0
    mode = vehicle.mode.name
    armed = vehicle.armed

    print(f"   ALT: {current_alt:.1f}m | Mode: {mode} | Armed: {armed}")

    if current_alt >= ALT_CONFIRM:
        break

    # Self-heal: if disarmed, re-arm + re-takeoff
    if not armed:
        print("   [!] Disarmed! Re-arming + retaking off...")
        set_mode_via_mavlink(vehicle, GUIDED_MODE_NUM)
        time.sleep(2)
        arm_via_mavlink(vehicle)
        time.sleep(3)
        if vehicle.armed:
            set_mode_via_mavlink(vehicle, GUIDED_MODE_NUM)
            time.sleep(1)
            vehicle.simple_takeoff(TAKEOFF_ALT)

    if time.time() - start > timeout:
        print(f"[WARN] Timeout after {timeout}s. Proceeding to telemetry.")
        break
    time.sleep(1)

if current_alt >= ALT_CONFIRM:
    print(f"[+] TAKEOFF SUCCESSFUL! Altitude: {current_alt:.1f}m")
else:
    print(f"[!] Altitude: {current_alt:.1f}m — continuing to telemetry.")

print("[TELEMETRY] Switching to C2 Telemetry Stream...")


# ═══════════════════════════════════════════════════════════
#  PHASE 3 — LIVE TELEMETRY TO C2 DASHBOARD
# ═══════════════════════════════════════════════════════════

async def stream_telemetry(vehicle):
    """Push live DroneKit telemetry to TIDDA C2 backend via WebSocket."""
    retry_delay = 3

    while True:
        try:
            async with websockets.connect(C2_WEBSOCKET_URI) as ws:
                print(f"[OK] Connected to TIDDA C2 at {C2_WEBSOCKET_URI}")

                while True:
                    battery_level = vehicle.battery.level if vehicle.battery.level is not None else 100

                    # Custom Swarm AI Failsafe Check
                    if battery_level < 20 and vehicle.mode.name != "RTL":
                        print(f"⚠️ CRITICAL BATTERY ({battery_level}%): Swarm Brain triggering Emergency RTB...")
                        vehicle.mode = VehicleMode("RTL")

                    payload = {
                        "drone_id":    "TIDDA-01",
                        "status":      "SCANNING",
                        "lat":         vehicle.location.global_frame.lat,
                        "lng":         vehicle.location.global_frame.lon,
                        "altitude_m":  vehicle.location.global_relative_frame.alt,
                        "battery_pct": battery_level if battery_level is not None else 100
                    }

                    await ws.send(json.dumps(payload))
                    print(
                        f"[TX] TIDDA-01 | "
                        f"LAT: {payload['lat']:.6f} | "
                        f"LNG: {payload['lng']:.6f} | "
                        f"ALT: {payload['altitude_m']:.1f}m | "
                        f"BAT: {payload['battery_pct']}%"
                    )
                    await asyncio.sleep(TELEMETRY_HZ)

        except ConnectionRefusedError:
            print(f"[WARN] C2 not reachable. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[WARN] WS dropped ({e}). Reconnecting...")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            print(f"[ERROR] {e}. Retrying...")
            await asyncio.sleep(retry_delay)


try:
    asyncio.run(stream_telemetry(vehicle))
except KeyboardInterrupt:
    print("\n[TIDDA MASTER] Stopped.")
finally:
    print("[SHUTDOWN] Closing vehicle...")
    vehicle.close()
    print("[SHUTDOWN] Done.")

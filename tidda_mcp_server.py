#!/usr/bin/env python3
"""
TIDDA MCP Server — MAVLink Telemetry + Git Operations
Exposes drone flight data and version control as MCP tools.

Run:  python tidda_mcp_server.py
Dev:  mcp dev tidda_mcp_server.py
"""

import subprocess
import os
import json
from mcp.server.fastmcp import FastMCP

# ── Server Setup ──────────────────────────────────────────────
mcp = FastMCP(
    "TIDDA MCP",
    description="TIDDA drone telemetry (MAVLink) and Git operations"
)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MAVLINK_CONN = "tcp:127.0.0.1:5760"

# ── Lazy MAVLink connection (only connects when first needed) ─
_vehicle = None

def _get_vehicle():
    """Connect to MAVLink vehicle on first call, reuse afterward."""
    global _vehicle
    if _vehicle is None:
        from pymavlink import mavutil
        _vehicle = mavutil.mavlink_connection(MAVLINK_CONN)
        _vehicle.wait_heartbeat(timeout=10)
        _vehicle.mav.request_data_stream_send(
            _vehicle.target_system, _vehicle.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 2, 1
        )
    return _vehicle


# ══════════════════════════════════════════════════════════════
#  TOOL A — Drone Telemetry (read / write via MAVLink)
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def drone_telemetry(action: str = "read", command: str = "", value: str = "") -> str:
    """
    Read or write drone telemetry via MAVLink.

    action: "read"  → returns current GPS, altitude, battery, heading
            "write" → sends a command to the drone

    command (write only): "arm", "disarm", "takeoff", "mode", "waypoint"
    value   (write only): depends on command —
        takeoff: altitude in meters (e.g. "10")
        mode:    flight mode name  (e.g. "GUIDED", "LOITER", "RTL")
        waypoint: "lat,lng,alt"    (e.g. "-35.363,149.165,20")
    """
    try:
        vehicle = _get_vehicle()
    except Exception as e:
        return json.dumps({"error": f"MAVLink connection failed: {e}"})

    from pymavlink import mavutil

    # ── READ ──────────────────────────────────────────────────
    if action == "read":
        telemetry = {
            "drone_id": "TIDDA-01",
            "lat": 0.0, "lng": 0.0,
            "altitude_m": 0.0,
            "battery_pct": -1,
            "heading_deg": 0,
            "mode": vehicle.flightmode,
            "armed": vehicle.motors_armed(),
        }

        # Pull latest GPS
        gps = vehicle.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=3)
        if gps:
            telemetry["lat"] = gps.lat / 1e7
            telemetry["lng"] = gps.lon / 1e7
            telemetry["altitude_m"] = round(gps.relative_alt / 1000.0, 1)
            telemetry["heading_deg"] = gps.hdg / 100

        # Pull latest battery
        bat = vehicle.recv_match(type="SYS_STATUS", blocking=True, timeout=2)
        if bat:
            telemetry["battery_pct"] = bat.battery_remaining if bat.battery_remaining > 0 else 100

        return json.dumps(telemetry, indent=2)

    # ── WRITE ─────────────────────────────────────────────────
    if action == "write":
        if command == "arm":
            vehicle.arducopter_arm()
            vehicle.motors_armed_wait()
            return json.dumps({"result": "Armed"})

        elif command == "disarm":
            vehicle.arducopter_disarm()
            return json.dumps({"result": "Disarmed"})

        elif command == "takeoff":
            alt = float(value) if value else 10.0
            vehicle.mav.command_long_send(
                vehicle.target_system, vehicle.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, alt
            )
            return json.dumps({"result": f"Takeoff to {alt}m"})

        elif command == "mode":
            mode_name = value.upper() if value else "GUIDED"
            mode_id = vehicle.mode_mapping().get(mode_name)
            if mode_id is None:
                return json.dumps({"error": f"Unknown mode: {mode_name}"})
            vehicle.set_mode(mode_id)
            return json.dumps({"result": f"Mode set to {mode_name}"})

        elif command == "waypoint":
            parts = value.split(",")
            if len(parts) != 3:
                return json.dumps({"error": "waypoint value must be 'lat,lng,alt'"})
            lat, lng, alt = float(parts[0]), float(parts[1]), float(parts[2])
            vehicle.mav.mission_item_send(
                vehicle.target_system, vehicle.target_component,
                0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                2, 0, 0, 0, 0, 0, lat, lng, alt
            )
            return json.dumps({"result": f"Waypoint set: {lat}, {lng}, {alt}m"})

        return json.dumps({"error": f"Unknown command: {command}"})

    return json.dumps({"error": f"Unknown action: {action}. Use 'read' or 'write'."})


# ══════════════════════════════════════════════════════════════
#  TOOL B — Git Status & Commit
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def git_status() -> str:
    """Check git status of the TIDDA-Core repository."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "-b"],
        cwd=REPO_DIR, capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

    branch = ""
    changed = []
    for line in lines:
        if line.startswith("##"):
            branch = line[3:]
        else:
            changed.append(line)

    return json.dumps({
        "branch": branch,
        "changed_files": len(changed),
        "files": changed,
        "clean": len(changed) == 0,
    }, indent=2)


@mcp.tool()
def git_commit(message: str) -> str:
    """Stage all changes and commit to the local TIDDA-Core repository.

    message: The commit message.
    """
    if not message.strip():
        return json.dumps({"error": "Commit message cannot be empty"})

    # git add -A
    add = subprocess.run(
        ["git", "add", "-A"],
        cwd=REPO_DIR, capture_output=True, text=True
    )
    if add.returncode != 0:
        return json.dumps({"error": f"git add failed: {add.stderr.strip()}"})

    # git commit
    commit = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=REPO_DIR, capture_output=True, text=True
    )
    if commit.returncode != 0:
        # Could be "nothing to commit"
        return json.dumps({"error": commit.stdout.strip() or commit.stderr.strip()})

    return json.dumps({
        "result": "Committed",
        "message": message,
        "output": commit.stdout.strip()
    }, indent=2)


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()

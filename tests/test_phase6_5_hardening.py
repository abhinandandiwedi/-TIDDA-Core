import argparse
import asyncio
import json
import time
import requests
import os
import shutil
import glob
import math
from pathlib import Path

API_URL = "http://localhost:8000"

def get_frames(workspace, count=50):
    frames_dir = Path("workspace_house_v1_demo") / "frames"
    if not frames_dir.exists():
        frames_dir = Path(workspace) / "frames"
        if not frames_dir.exists():
            return []
    frames = sorted(glob.glob(str(frames_dir / "*.jpg")))
    return frames[:count]

async def send_to_node(node_id, frame_paths):
    import websockets
    import base64
    async with websockets.connect(f"ws://localhost:8000/ws/mobile") as ws:
        await ws.send(json.dumps({"type": "node_register", "node_id": node_id, "mode": "mapping"}))
        await asyncio.sleep(0.1)
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            payload = json.dumps({
                "type": "camera_frame",
                "frame": b64,
                "timestamp": time.time(),
                "node_id": node_id
            })
            await ws.send(payload)
            await asyncio.sleep(0.15)
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

def test_hardening(workspace):
    print("=== Phase 6.5 Persistence Hardening Test ===")
    world_dir = Path("workspaces/default/world")

    def reset_world():
        if world_dir.exists():
            shutil.rmtree(world_dir)
        world_dir.mkdir(parents=True, exist_ok=True)
        requests.post(f"{API_URL}/api/map/reset")
        time.sleep(1)

    reset_world()

    # ---------------------------------------------------------
    # MULTI-CYCLE & CONTINUE MAPPING (Goals 2, 3, 9, 10)
    # ---------------------------------------------------------
    print("\n--- MULTI-CYCLE & CONTINUE MAPPING TEST ---")
    frames = get_frames(workspace, count=60)
    assert len(frames) >= 40, "Need at least 40 frames"

    print("Cycle 1: Creating Map")
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-A"})
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-B"})
    requests.post(f"{API_URL}/api/map/start")

    asyncio.run(send_to_node("NODE-A", frames[0:30]))
    asyncio.run(send_to_node("NODE-B", frames[10:40]))
    time.sleep(1)

    align1 = requests.post(f"{API_URL}/api/mapping/align", json={"source_id": "NODE-B", "target_id": "NODE-A"}).json()
    time.sleep(1)
    status1 = requests.get(f"{API_URL}/api/world/status").json()

    assert status1["status"] == "ONLINE"
    assert status1["map_version"] > 0
    assert status1["total_points"] > 0
    assert status1["active_nodes"] >= 2

    v1 = status1["map_version"]
    pts1 = status1["total_points"]
    nodes1 = status1["active_nodes"]
    world_id = status1.get("world_id")

    print(f"Cycle 1 saved: v{v1}, {pts1} points, {nodes1} nodes")

    print("Cycle 2: Reloading & Continuing")
    # Simulate reload
    requests.post(f"{API_URL}/api/world/reload")
    status2 = requests.get(f"{API_URL}/api/world/status").json()

    assert status2["map_version"] == v1, f"Expected v{v1}, got v{status2.get('map_version')}"
    assert status2["total_points"] == pts1
    assert status2["active_nodes"] == nodes1
    assert status2.get("world_id") == world_id

    # Continue mapping (Goal 9)
    # We send frames from the beginning to ensure they have enough overlap to successfully map and align
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-A"})
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-B"})
    requests.post(f"{API_URL}/api/map/start")
    asyncio.run(send_to_node("NODE-A", frames[0:30]))
    asyncio.run(send_to_node("NODE-B", frames[10:40]))
    time.sleep(2)

    align2 = requests.post(f"{API_URL}/api/mapping/align", json={"source_id": "NODE-B", "target_id": "NODE-A"}).json()
    time.sleep(1)
    status2b = requests.get(f"{API_URL}/api/world/status").json()

    assert status2b["map_version"] > v1
    assert status2b["total_points"] >= pts1

    v2 = status2b["map_version"]
    pts2 = status2b["total_points"]
    print(f"Cycle 2 saved: v{v2}, {pts2} points")

    print("Cycle 3: Reloading again")
    requests.post(f"{API_URL}/api/world/reload")
    status3 = requests.get(f"{API_URL}/api/world/status").json()

    assert status3["map_version"] == v2
    assert status3["total_points"] == pts2
    assert status3.get("world_id") == world_id

    # ---------------------------------------------------------
    # DUPLICATE SAVE (Goal 8)
    # ---------------------------------------------------------
    print("\n--- DUPLICATE SAVE TEST ---")
    req_save1 = requests.post(f"{API_URL}/api/world/save").json()
    assert req_save1["status"] == "OK"
    v_save = req_save1["version"]

    # Save again
    req_save2 = requests.post(f"{API_URL}/api/world/save").json()
    assert req_save2["status"] == "OK"
    assert req_save2["version"] == v_save, "Version should not increment on manual duplicate save without mapping"

    status_dup = requests.get(f"{API_URL}/api/world/status").json()
    assert status_dup["map_version"] == v_save
    assert status_dup["total_points"] == pts2
    print("Duplicate save verified.")

    # ---------------------------------------------------------
    # LARGE MAP / STRESS & TIMING (Goal 4)
    # ---------------------------------------------------------
    print("\n--- PERFORMANCE TIMING TEST ---")
    start_time = time.time()
    requests.post(f"{API_URL}/api/world/save")
    save_time = time.time() - start_time

    start_time = time.time()
    requests.post(f"{API_URL}/api/world/reload")
    reload_time = time.time() - start_time

    ply_size = (world_dir / "global_map.ply").stat().st_size if (world_dir / "global_map.ply").exists() else 0
    json_size = (world_dir / "world_state.json").stat().st_size if (world_dir / "world_state.json").exists() else 0

    print(f"Save Time:   {save_time*1000:.2f} ms")
    print(f"Reload Time: {reload_time*1000:.2f} ms")
    print(f"PLY Size:    {ply_size/1024:.2f} KB")
    print(f"JSON Size:   {json_size/1024:.2f} KB")

    # ---------------------------------------------------------
    # MISSING FILE TESTS (Goal 6)
    # ---------------------------------------------------------
    print("\n--- MISSING FILE TESTS ---")
    # A. Missing JSON
    json_path = world_dir / "world_state.json"
    ply_path = world_dir / "global_map.ply"

    # backup
    shutil.copy(json_path, Path("workspaces/default/backup.json"))
    shutil.copy(ply_path, Path("workspaces/default/backup.ply"))

    json_path.unlink()
    requests.post(f"{API_URL}/api/world/reload")
    status_m1 = requests.get(f"{API_URL}/api/world/status").json()
    assert status_m1.get("status") == "OFFLINE" or status_m1.get("persistence") == "NONE"
    print("Missing JSON handled gracefully.")

    shutil.copy(Path("workspaces/default/backup.json"), json_path)
    ply_path.unlink()
    res_reload = requests.post(f"{API_URL}/api/world/reload").json()
    status_m2 = requests.get(f"{API_URL}/api/world/status").json()
    # It might load metadata but fail PLY, meaning RECOVERY or 0 points
    print("Missing PLY reload response:", res_reload)
    assert res_reload.get("status") in ["RECOVERY", "ERROR"], "Should fail or recover when PLY is missing"
    print("Missing PLY handled gracefully.")

    json_path.unlink()
    requests.post(f"{API_URL}/api/world/reload")
    status_m3 = requests.get(f"{API_URL}/api/world/status").json()
    assert status_m3.get("status") == "OFFLINE" or status_m3.get("persistence") == "NONE"

    shutil.rmtree(world_dir)
    requests.post(f"{API_URL}/api/world/reload")
    status_m4 = requests.get(f"{API_URL}/api/world/status").json()
    assert status_m4.get("status") == "OFFLINE" or status_m4.get("persistence") == "NONE"
    print("Missing directory handled gracefully.")

    # ---------------------------------------------------------
    # CORRUPTION TESTS (Goal 7)
    # ---------------------------------------------------------
    print("\n--- CORRUPTION TESTS ---")
    world_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        f.write("{ invalid json")

    requests.post(f"{API_URL}/api/world/reload")
    status_c1 = requests.get(f"{API_URL}/api/world/status").json()
    assert status_c1.get("status") == "OFFLINE" or status_c1.get("persistence") == "NONE"
    print("Malformed JSON handled gracefully.")

    shutil.copy(Path("workspaces/default/backup.json"), json_path)
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 10\ninvalid property")

    requests.post(f"{API_URL}/api/world/reload")
    status_c2 = requests.get(f"{API_URL}/api/world/status").json()
    print("Malformed PLY status:", status_c2)

    print("\n✅ PHASE 6.5 HARDENING: COMPLETE")

    Path("workspaces/default/backup.json").unlink(missing_ok=True)
    Path("workspaces/default/backup.ply").unlink(missing_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="workspaces/default")
    args = parser.parse_args()
    test_hardening(args.workspace)

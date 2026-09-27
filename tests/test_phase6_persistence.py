import argparse
import asyncio
import json
import time
import requests
import os
import shutil
import glob
from pathlib import Path

API_URL = "http://localhost:8000"

def get_frames(workspace, count=20):
    frames_dir = Path(workspace) / "frames"
    if not frames_dir.exists():
        return []
    frames = sorted(glob.glob(str(frames_dir / "*.jpg")))
    return frames[:count]

async def send_to_node(node_id, frame_paths):
    import websockets
    async with websockets.connect(f"ws://localhost:8000/ws/mobile") as ws:
        await ws.send(json.dumps({"type": "node_register", "node_id": node_id, "mode": "mapping"}))
        await asyncio.sleep(0.1)
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                import base64
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

def test_persistence(workspace):
    print("=== Phase 6 Persistence Test ===")
    
    # Setup clean workspace for persistence
    world_dir = Path("workspaces/default/world")
    if world_dir.exists():
        shutil.rmtree(world_dir)
    world_dir.mkdir(parents=True, exist_ok=True)
        
    requests.post(f"{API_URL}/api/map/reset")
    time.sleep(1)
    
    # A. Fresh world (or reset world)
    print("\n--- Test A: Fresh World ---")
    status = requests.get(f"{API_URL}/api/world/status").json()
    print("Initial status:", status)
    if status.get("status") == "ONLINE":
        assert status["map_version"] == 0, "Map version should be 0 on reset"
        assert status["total_points"] == 0, "Points should be 0 on reset"
    else:
        assert status.get("status") == "OFFLINE", "Should be offline initially"
    
    # B. Generate some map data
    print("\n--- Test B: Create & Save World ---")
    frames = get_frames(workspace, count=50)
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-A"})
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-B"})
    requests.post(f"{API_URL}/api/map/start")
    
    asyncio.run(send_to_node("NODE-A", frames[0:30]))
    asyncio.run(send_to_node("NODE-B", frames[10:40]))
    
    time.sleep(1)
    
    # Align nodes
    align = requests.post(f"{API_URL}/api/mapping/align", json={"source_id": "NODE-B", "target_id": "NODE-A"}).json()
    print("Alignment Result:", align.get("status"))
    
    # Ensure it's saved (alignment triggers auto-save now)
    time.sleep(1)
    status = requests.get(f"{API_URL}/api/world/status").json()
    print("Status after fusion:", status)
    assert status["status"] == "ONLINE"
    assert status["map_version"] > 0
    assert status["total_points"] > 0
    assert status["active_nodes"] > 0
    assert (world_dir / "world_state.json").exists()
    assert (world_dir / "global_map.ply").exists()
    
    v1 = status["map_version"]
    
    # C. Reload Simulation
    print("\n--- Test C: Destroy Runtime & Reload ---")
    
    # We won't use /api/map/reset here because reset explicitly wipes the disk state too!
    # Instead, we will directly reload from disk and check if it gets the version correctly.
    requests.post(f"{API_URL}/api/world/reload")
    status = requests.get(f"{API_URL}/api/world/status").json()
    print("Status after reload:", status)
    assert status["map_version"] == v1
    
    # D. Versioning check
    print("\n--- Test D: Version increment ---")
    align2 = requests.post(f"{API_URL}/api/mapping/align", json={"source_id": "NODE-B", "target_id": "NODE-A"}).json()
    time.sleep(1)
    status2 = requests.get(f"{API_URL}/api/world/status").json()
    print("Status after second alignment:", status2)
    assert status2["map_version"] > v1
    
    # E. Corruption handling
    print("\n--- Test E: Corruption handling ---")
    with open(world_dir / "world_state.json", "w") as f:
        f.write("{ invalid json")
    
    res = requests.post(f"{API_URL}/api/world/reload").json()
    status_c = requests.get(f"{API_URL}/api/world/status").json()
    print("Status after corruption:", status_c)
    assert status_c["persistence"] == "NONE"
    
    print("\n✅ PHASE 6 PERSISTENCE: COMPLETE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="workspaces/default")
    args = parser.parse_args()
    test_persistence(args.workspace)

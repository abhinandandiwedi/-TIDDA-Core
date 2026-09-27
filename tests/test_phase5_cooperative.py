import argparse
import base64
import requests
import time
import os
import glob
from pathlib import Path

API_URL = "http://localhost:8000"

def get_frames(workspace, count=20):
    frames_dir = Path(workspace) / "frames"
    if not frames_dir.exists():
        print(f"Error: {frames_dir} not found.")
        return []

    frames = sorted(glob.glob(str(frames_dir / "*.jpg")))
    if not frames:
        print("No frames found.")
        return []

    return frames[:count]

def test_cooperative_mapping(workspace):
    print("=== Phase 5 Cooperative Mapping Test ===")

    frames = get_frames(workspace, count=50)
    if not frames:
        return

    print(f"Found {len(frames)} frames for test.")

    # 1. Register Nodes
    print("Registering Node A and Node B...")
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-A"})
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-B"})

    # Start mapping!
    requests.post(f"{API_URL}/api/map/start")

    # We will simulate Node A mapping the first 30 frames
    # and Node B mapping frames 10 to 40 (so there's an overlap of 20 frames)

    print("Sending frames to Node A (0 to 30)...")
    import websockets
    import asyncio
    import json

    async def send_to_node(node_id, frame_paths):
        async with websockets.connect(f"ws://localhost:8000/ws/mobile") as ws:
            # Register Node over WS
            await ws.send(json.dumps({
                "type": "node_register",
                "node_id": node_id,
                "mode": "mapping"
            }))
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
                # Wait briefly to allow processing
                await asyncio.sleep(0.15)

                # Receive responses
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass

    async def run_simulation():
        await send_to_node("NODE-A", frames[0:30])
        print("Sending frames to Node B (10 to 40)...")
        await send_to_node("NODE-B", frames[10:40])

    asyncio.run(run_simulation())

    time.sleep(1) # Let backend settle

    # 3. Check Local Maps
    print("Fetching nodes state...")
    r = requests.get(f"{API_URL}/api/mapping/nodes")
    print(r.json())

    # 4. Global Map Before
    r = requests.get(f"{API_URL}/api/mapping/global")
    state_before = r.json()
    print("Global map state before fusion:", state_before)

    # 5. Execute Alignment!
    print("Executing Cooperative Alignment (NODE-B -> NODE-A)...")
    r = requests.post(f"{API_URL}/api/mapping/align", json={
        "source_id": "NODE-B",
        "target_id": "NODE-A"
    })

    res = r.json()
    print("Alignment Result:")
    if isinstance(res, dict) and 'status' in res:
        for k, v in res.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        for k, v in res.items():
            print(f"  {k}: {v}")

    # 6. Global Map After
    r = requests.get(f"{API_URL}/api/mapping/global")
    state_after = r.json()
    print("Global map state after fusion:", state_after)

    if res.get("status") == "ALIGNMENT_OK":
        print("\n✅ PHASE 5 CLASSIFICATION: COMPLETE")
    else:
        print(f"\n❌ PHASE 5 CLASSIFICATION: INCOMPLETE (Alignment Failed: {res.get('status')})")

    print("\n=== Failure Benchmarks ===")

    # Empty map failure
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-EMPTY"})
    r_empty = requests.post(f"{API_URL}/api/mapping/align", json={"source_id": "NODE-EMPTY", "target_id": "NODE-A"})
    print("Empty Map Alignment Result:", r_empty.json())

    # Duplicate node failure
    r_dup = requests.post(f"{API_URL}/api/mapping/align", json={"source_id": "NODE-A", "target_id": "NODE-A"})
    print("Duplicate Node Alignment Result:", r_dup.json())

    # Unrelated Maps (Node-C with non-overlapping frames)
    print("\n=== 3-Node Stress Test & Unrelated Maps ===")
    requests.post(f"{API_URL}/api/mapping/nodes/register", json={"node_id": "NODE-C"})
    async def run_node_c():
        await send_to_node("NODE-C", frames[45:50]) # Very few frames, tiny map
    asyncio.run(run_node_c())

    time.sleep(1)

    # Try align Node C to Node A (will likely fail due to insufficient correspondences)
    r_c_to_a = requests.post(f"{API_URL}/api/mapping/align", json={"source_id": "NODE-C", "target_id": "NODE-A"})
    print("Tiny/Unrelated Map (C -> A) Alignment Result:", r_c_to_a.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="/home/abhinandan/-TIDDA-Core/workspaces/test_workspace")
    args = parser.parse_args()
    test_cooperative_mapping(args.workspace)

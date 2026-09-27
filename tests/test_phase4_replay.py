#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════
#  🧪 TIDDA PHASE 4 — End-to-End Frame Replay Test
#
#  Replays real JPEG frames from a demo workspace through the
#  /ws/mobile WebSocket path, exercising the full incremental
#  mapping pipeline:
#
#    Connect → Register → Start Map → Send Frames → Observe Updates
#
#  Usage:
#    python3 tests/test_phase4_replay.py [--host HOST] [--port PORT]
#                                        [--frames-dir DIR] [--fps FPS]
#
#  Default: replays workspace_house_v1_demo/frames/ at ~5 FPS
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import argparse
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


async def run_test(
    host: str = "127.0.0.1",
    port: int = 8000,
    frames_dir: str = "",
    target_fps: float = 5.0,
    max_frames: int = 0,
    skip_map_start: bool = False,
):
    """Run the frame replay test through the WebSocket mobile path."""
    import websockets

    if not frames_dir:
        frames_dir = str(REPO_ROOT / "workspace_house_v1_demo" / "frames")

    frames_path = Path(frames_dir)
    if not frames_path.exists():
        print(f"[ERROR] Frames directory not found: {frames_path}")
        return

    # Collect sorted JPEG frames
    frame_files = sorted(frames_path.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_path.glob("*.jpeg"))
    if not frame_files:
        frame_files = sorted(frames_path.glob("*.png"))
    if not frame_files:
        print(f"[ERROR] No image files found in: {frames_path}")
        return

    if max_frames > 0:
        frame_files = frame_files[:max_frames]

    total_frames = len(frame_files)
    print(f"[TEST] Found {total_frames} frames in {frames_path}")
    print(f"[TEST] Target FPS: {target_fps}")
    print(f"[TEST] Connecting to ws://{host}:{port}/ws/mobile")

    # Statistics
    stats = {
        "frames_sent": 0,
        "map_updates": 0,
        "map_points_msgs": 0,
        "total_points_received": 0,
        "tracking_statuses": {},
        "errors": [],
        "start_time": 0,
        "end_time": 0,
        "last_points_total": 0,
        "last_keyframe_count": 0,
        "max_keyframe_count": 0,
        "last_stats": {},
    }

    ws_url = f"ws://{host}:{port}/ws/mobile"

    try:
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws_cam, \
                   websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws_dash:
            print("[TEST] Connected to WebSockets (Camera & Dashboard)")

            # 1. Register ws_cam as mobile node
            register_msg = json.dumps({
                "type": "node_register",
                "node_id": "TEST-REPLAY-NODE",
                "mode": "mapping",
            })
            await ws_cam.send(register_msg)
            print("[TEST] Registered ws_cam as TEST-REPLAY-NODE (mode=mapping)")

            # Wait briefly for registration to be processed
            await asyncio.sleep(0.5)

            # 2. Start incremental mapping (via API)
            if not skip_map_start:
                start_msg = json.dumps({"type": "map_start"})
                await ws_dash.send(start_msg)
                print("[TEST] Sent map_start command via ws_dash")
                await asyncio.sleep(0.3)

            # 3. Background listener for MAP_UPDATE and MAP_POINTS messages on ws_dash
            received_done = asyncio.Event()

            async def listen_for_updates():
                try:
                    async for msg_text in ws_dash:
                        try:
                            msg = json.loads(msg_text)
                        except json.JSONDecodeError:
                            continue

                        msg_type = msg.get("type", "")

                        if msg_type == "MAP_UPDATE":
                            stats["map_updates"] += 1
                            status = msg.get("tracking_status", "UNKNOWN")
                            stats["tracking_statuses"][status] = stats["tracking_statuses"].get(status, 0) + 1
                            stats["last_points_total"] = msg.get("points_total", 0)
                            stats["last_keyframe_count"] = msg.get("keyframe_count", 0)
                            stats["max_keyframe_count"] = max(stats["max_keyframe_count"], stats["last_keyframe_count"])
                            if "stats" in msg:
                                stats["last_stats"] = msg["stats"]

                            if stats["map_updates"] % 10 == 0 or status == "POSE_OK":
                                print(f"  [MAP_UPDATE] status={status} pts={msg.get('points_total', 0)} "
                                      f"kf={msg.get('keyframe_count', 0)}")

                        elif msg_type == "MAP_POINTS":
                            pts_count = len(msg.get("points", []))
                            stats["map_points_msgs"] += 1
                            stats["total_points_received"] += pts_count
                            if stats["map_points_msgs"] % 5 == 0:
                                print(f"  [MAP_POINTS] +{pts_count} points (total received: {stats['total_points_received']})")

                        elif msg_type == "MAP_STATE":
                            print(f"  [MAP_STATE] active={msg.get('active')} points={msg.get('map_point_count', 0)} "
                                  f"keyframes={msg.get('keyframe_count', 0)}")

                except Exception as e:
                    if "closed" not in str(e).lower():
                        stats["errors"].append(f"Listener error: {e}")

            listener_task = asyncio.create_task(listen_for_updates())

            # 4. Send frames
            print(f"\n[TEST] Sending {total_frames} frames...")
            stats["start_time"] = time.time()
            frame_interval = 1.0 / target_fps

            for i, frame_file in enumerate(frame_files):
                try:
                    with open(frame_file, "rb") as f:
                        frame_bytes = f.read()

                    frame_b64 = base64.b64encode(frame_bytes).decode("ascii")

                    frame_msg = json.dumps({
                        "type": "camera_frame",
                        "frame": frame_b64,
                        "timestamp": time.time(),
                    })

                    await ws_cam.send(frame_msg)
                    stats["frames_sent"] += 1

                    if (i + 1) % 20 == 0:
                        elapsed = time.time() - stats["start_time"]
                        actual_fps = stats["frames_sent"] / max(elapsed, 0.001)
                        print(f"  [SEND] Frame {i+1}/{total_frames} "
                              f"({actual_fps:.1f} FPS actual) "
                              f"pts={stats['last_points_total']} kf={stats['last_keyframe_count']}")

                except Exception as e:
                    stats["errors"].append(f"Frame {i}: {e}")

                await asyncio.sleep(frame_interval)

            stats["end_time"] = time.time()

            # 5. Wait for processing to catch up
            print("\n[TEST] Waiting for processing to complete...")
            await asyncio.sleep(3.0)

            # 6. Request final map state
            await ws_dash.send(json.dumps({"type": "map_state"}))
            await asyncio.sleep(1.0)

            # 7. Test failure cases
            print("\n[TEST] === FAILURE TESTS ===")

            # Test: invalid JPEG
            print("  [FAIL-TEST] Sending invalid JPEG...")
            await ws_cam.send(json.dumps({
                "type": "camera_frame",
                "frame": "not_a_valid_base64_jpeg",
                "timestamp": time.time(),
            }))
            await asyncio.sleep(0.5)

            # Test: empty frame
            print("  [FAIL-TEST] Sending empty frame...")
            await ws_cam.send(json.dumps({
                "type": "camera_frame",
                "frame": "",
                "timestamp": time.time(),
            }))
            await asyncio.sleep(0.5)

            # Test: tiny image (insufficient features)
            print("  [FAIL-TEST] Sending tiny image (insufficient features)...")
            # Create a 10x10 solid gray image
            import struct
            tiny_data = b'\xff\xd8\xff\xe0' + b'\x00' * 100  # Invalid but small JPEG-like
            tiny_b64 = base64.b64encode(tiny_data).decode("ascii")
            await ws_cam.send(json.dumps({
                "type": "camera_frame",
                "frame": tiny_b64,
                "timestamp": time.time(),
            }))
            await asyncio.sleep(0.5)

            # Test: mapper reset
            print("  [FAIL-TEST] Sending map_reset...")
            await ws_dash.send(json.dumps({"type": "map_reset"}))
            await asyncio.sleep(1.0)

            # Cancel listener
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print(f"  Make sure the backend is running: python3 tidda_lightweight_swarm.py")
        return

    # ── Print Report ──────────────────────────────────────────────
    elapsed = stats["end_time"] - stats["start_time"]
    print("\n" + "=" * 60)
    print("  PHASE 4 — INCREMENTAL MAPPING TEST REPORT")
    print("=" * 60)
    print(f"  Frames directory:     {frames_dir}")
    print(f"  Total frames sent:    {stats['frames_sent']}")
    print(f"  Elapsed time:         {elapsed:.1f}s")
    print(f"  Send FPS:             {stats['frames_sent'] / max(elapsed, 0.001):.1f}")
    print(f"  MAP_UPDATE msgs:      {stats['map_updates']}")
    print(f"  MAP_POINTS msgs:      {stats['map_points_msgs']}")
    print(f"  Total points received:{stats['total_points_received']}")
    print(f"  Final points total:   {stats['last_points_total']}")
    print(f"  Final keyframe count: {stats['last_keyframe_count']}")
    print()
    print("  Tracking Status Distribution:")
    for status, count in sorted(stats["tracking_statuses"].items()):
        print(f"    {status}: {count}")
    print()
    if stats["last_stats"]:
        s = stats["last_stats"]
        print("  Mapper Internal Stats:")
        print(f"    Frames received:    {s.get('frames_received', 0)}")
        print(f"    Frames processed:   {s.get('frames_processed', 0)}")
        print(f"    Keyframes selected: {s.get('keyframes_selected', 0)}")
        print(f"    Pose OK:            {s.get('pose_ok', 0)}")
        print(f"    Pose fail:          {s.get('pose_fail', 0)}")
        print(f"    Points inserted:    {s.get('points_inserted', 0)}")
        print(f"    Points fused:       {s.get('points_fused', 0)}")
        print(f"    Total map points:   {s.get('total_map_points', 0)}")
        print(f"    Avg processing ms:  {s.get('avg_processing_ms', 0):.1f}")
        print(f"    Max processing ms:  {s.get('max_processing_ms', 0):.1f}")
        print(f"    Tracking failures:  {s.get('tracking_failures', 0)}")
    print()
    if stats["errors"]:
        print(f"  Errors ({len(stats['errors'])}):")
        for err in stats["errors"][:10]:
            print(f"    - {err}")
    else:
        print("  Errors: NONE")
    print()

    # Validation
    passed = True
    checks = []

    if stats["frames_sent"] >= total_frames:
        checks.append(("Frames sent", "PASS"))
    else:
        checks.append(("Frames sent", "FAIL"))
        passed = False

    if stats["map_updates"] > 0:
        checks.append(("MAP_UPDATE received", "PASS"))
    else:
        checks.append(("MAP_UPDATE received", "FAIL"))
        passed = False

    if stats["max_keyframe_count"] > 0:
        checks.append(("Keyframes selected", "PASS"))
    else:
        checks.append(("Keyframes selected", "FAIL"))
        passed = False

    pose_ok = stats["tracking_statuses"].get("POSE_OK", 0)
    if pose_ok > 0:
        checks.append(("Pose estimation", "PASS"))
    else:
        checks.append(("Pose estimation", "WARN - no POSE_OK"))

    print("  Validation Checks:")
    for name, result in checks:
        icon = "✓" if result == "PASS" else ("⚠" if "WARN" in result else "✗")
        print(f"    {icon} {name}: {result}")

    print()
    print("  OVERALL:", "PASS ✓" if passed else "FAIL ✗")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 frame replay test")
    parser.add_argument("--host", default="127.0.0.1", help="Backend host")
    parser.add_argument("--port", type=int, default=8000, help="Backend port")
    parser.add_argument("--frames-dir", default="", help="Frames directory")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    parser.add_argument("--skip-map-start", action="store_true", help="Skip sending map_start")
    args = parser.parse_args()

    try:
        import websockets
    except ImportError:
        print("[SYSTEM] Installing websockets...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"],
                              stdout=subprocess.DEVNULL)
        import websockets

    asyncio.run(run_test(
        host=args.host,
        port=args.port,
        frames_dir=args.frames_dir,
        target_fps=args.fps,
        max_frames=args.max_frames,
        skip_map_start=args.skip_map_start,
    ))

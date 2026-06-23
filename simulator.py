import asyncio
import websockets
import json
import random

async def simulate_drone():
    uri = "ws://localhost:8000/ws/telemetry"
    try:
        async with websockets.connect(uri) as websocket:
            print("[OK] Fake Drone Connected to TIDDA Backend!")
            while True:
                telemetry = {
                    "drone_id": "TIDDA-01",
                    "lat": round(26.8467 + random.uniform(-0.005, 0.005), 6),
                    "lng": round(80.9462 + random.uniform(-0.005, 0.005), 6),
                    "altitude_m": random.randint(40, 120),
                    "battery_pct": random.randint(75, 100)
                }
                await websocket.send(json.dumps(telemetry))
                print(f"[TX] Sent: {telemetry}")
                await asyncio.sleep(1)
    except ConnectionRefusedError:
        print("[ERROR] Backend is not running. Start the C2 server first!")

if __name__ == "__main__":
    asyncio.run(simulate_drone())
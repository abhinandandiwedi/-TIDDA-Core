import asyncio
import websockets
import json
from pymavlink import mavutil

# ArduPilot SITL default TCP port
MAVLINK_CONNECTION_STRING = 'tcp:127.0.0.1:5760'
C2_SERVER_URI = "ws://localhost:8000/ws/telemetry"

async def bridge_mavlink_to_c2():
    print(f"[+] Connecting to MAVLink SITL at {MAVLINK_CONNECTION_STRING}...")
    vehicle = mavutil.mavlink_connection(MAVLINK_CONNECTION_STRING)
    vehicle.wait_heartbeat()
    print("[+] Heartbeat received! Drone is alive.")

    # Request specific data streams (Position & Battery)
    vehicle.mav.request_data_stream_send(
        vehicle.target_system, vehicle.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 2, 1
    )

    try:
        async with websockets.connect(C2_SERVER_URI) as ws:
            print("[OK] Connected to TIDDA C2 WebSocket!")
            
            # Persistent state — survives across loop iterations so GPS + Battery
            # data from separate MAVLink packets are merged into one payload.
            drone_state = {
                "drone_id": "TIDDA-01",
                "status": "SCANNING",
                "lat": 28.6235,
                "lng": 77.4543,
                "altitude_m": 0.0,
                "battery_pct": 100
            }

            while True:
                msg = vehicle.recv_match(type=['GLOBAL_POSITION_INT', 'SYS_STATUS'], blocking=True)
                if not msg:
                    continue

                # Extract GPS & Altitude
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    drone_state["lat"] = msg.lat / 1e7
                    drone_state["lng"] = msg.lon / 1e7
                    drone_state["altitude_m"] = msg.relative_alt / 1000.0  # mm to meters

                # Extract Battery (guard against SITL startup returning -1/0)
                elif msg.get_type() == 'SYS_STATUS':
                    drone_state["battery_pct"] = msg.battery_remaining if msg.battery_remaining > 0 else 100

                # Always send the combined state — every field is guaranteed present
                await ws.send(json.dumps(drone_state))
                print(f"[TX] {drone_state['drone_id']} | ALT: {drone_state['altitude_m']}m | BAT: {drone_state['battery_pct']}%")

                await asyncio.sleep(0.5)  # Throttle to 2Hz

    except ConnectionRefusedError:
        print("[ERROR] FastAPI C2 Backend is not running. Start it first!")

if __name__ == "__main__":
    asyncio.run(bridge_mavlink_to_c2())
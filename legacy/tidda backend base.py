# tidda backend base.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio

app = FastAPI(title="TIDDA Backend C2", version="1.0.0")

# Allow browser connections from any origin (file:// or localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.ui_connections: list[WebSocket] = []      # HTML dashboards (listen-only)
        self.sender_connections: list[WebSocket] = []   # Simulators / drones (send data)

    async def connect_ui(self, websocket: WebSocket):
        """Accept a UI listener (dashboard) -- it only receives broadcasts."""
        await websocket.accept()
        self.ui_connections.append(websocket)
        print(f"[+] UI Client connected. Total UI clients: {len(self.ui_connections)}")

    async def connect_sender(self, websocket: WebSocket):
        """Accept a data sender (simulator/drone) -- it pushes telemetry."""
        await websocket.accept()
        self.sender_connections.append(websocket)
        print(f"[+] Sender connected. Total senders: {len(self.sender_connections)}")

    def disconnect_ui(self, websocket: WebSocket):
        if websocket in self.ui_connections:
            self.ui_connections.remove(websocket)
        print(f"[-] UI Client disconnected. Total UI clients: {len(self.ui_connections)}")

    def disconnect_sender(self, websocket: WebSocket):
        if websocket in self.sender_connections:
            self.sender_connections.remove(websocket)
        print(f"[-] Sender disconnected. Total senders: {len(self.sender_connections)}")

    async def broadcast_telemetry(self, message: dict):
        """Send telemetry data to all connected UI dashboards."""
        disconnected = []
        for connection in self.ui_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        # Clean up any broken connections
        for conn in disconnected:
            self.disconnect_ui(conn)


manager = ConnectionManager()


@app.get("/")
async def root():
    return {"message": "TIDDA C2 Server is Online."}


# -- Endpoint for SIMULATOR / DRONE (sends telemetry) --
@app.websocket("/ws/telemetry")
async def telemetry_endpoint(websocket: WebSocket):
    await manager.connect_sender(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            telemetry_data = json.loads(data)
            print(f"[RX] Received Telemetry: {telemetry_data}")
            # Broadcast to all UI dashboards
            await manager.broadcast_telemetry(telemetry_data)
    except WebSocketDisconnect:
        manager.disconnect_sender(websocket)
        print("Drone Node Disconnected.")
    except Exception as e:
        manager.disconnect_sender(websocket)
        print(f"Sender error: {e}")


# -- Endpoint for UI DASHBOARD (listen-only, receives broadcasts) --
@app.websocket("/ws/ui")
async def ui_endpoint(websocket: WebSocket):
    await manager.connect_ui(websocket)
    try:
        # UI clients don't send data -- just keep the connection alive.
        # We listen for close/disconnect frames so we can clean up.
        while True:
            # receive_text() will block until the client sends something
            # or disconnects. The browser WebSocket will send close frames
            # when the tab is closed, which raises WebSocketDisconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_ui(websocket)
    except Exception:
        manager.disconnect_ui(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

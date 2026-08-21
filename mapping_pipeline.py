# ══════════════════════════════════════════════════════════════════
#  🌐 TIDDA MAPPING PIPELINE — Backend orchestrator
#  Connects backend components into an integrated pipeline:
#
#  MOBILE NODE
#      ↓
#  WebSocket (node_register / telemetry / camera_frame)
#      ↓
#  ScanSession → FloorManager → PerceptionEngine → LocalMapManager → FusionEngine → WorldModel
#
#  Does NOT rewrite existing telemetry or physics code.
#  Maintains error isolation across layers and preserves historical data on disconnect.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from mapping_models import (
    FloorState,
    FloorStatus,
    Landmark,
    LocalMapState,
    Pose,
    SemanticObservation,
)
from scan_session import NodeScanState, ScanSession
from floor_manager import FloorManager
from local_map import LocalMapManager
from perception import PerceptionEngine, MockPerceptionEngine
from fusion import FusionEngine, FusionResult
from world_model import WorldModel

log_pipeline = logging.getLogger("MAPPING-PIPELINE")


# ══════════════════════════════════════════════════════════════════
#  MAPPING PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════

class MappingPipeline:
    """Orchestrates cooperative mapping subsystems.

    Wrapped with defensive error handling so failures in perception or fusion
    never crash the host process or break live telemetry streams.
    """

    def __init__(
        self,
        scan_session: Optional[ScanSession] = None,
        floor_manager: Optional[FloorManager] = None,
        perception_engine: Optional[Union[PerceptionEngine, MockPerceptionEngine]] = None,
        local_map_manager: Optional[LocalMapManager] = None,
        fusion_engine: Optional[FusionEngine] = None,
        world_model: Optional[WorldModel] = None,
    ) -> None:
        self.floor_manager = floor_manager or FloorManager()
        self.scan_session = scan_session or ScanSession()
        self.local_map_manager = local_map_manager or LocalMapManager(floor_manager=self.floor_manager)
        self.perception_engine = perception_engine or PerceptionEngine()
        self.fusion_engine = fusion_engine or FusionEngine()
        self.world_model = world_model or WorldModel()

    # ── Node Registration & Lifecycle ─────────────────────────────

    def register_node(self, node_id: str) -> None:
        """Register a mobile node into the scan session (starts IDLE)."""
        try:
            if not node_id:
                return
            self.scan_session.connect_node(node_id)
        except Exception as e:
            log_pipeline.error(f"Error registering node {node_id}: {e}")

    def assign_node_to_floor(self, node_id: str, floor_id: str) -> bool:
        """Assign a mobile node to a specific floor.

        Creates the floor in FloorManager if it doesn't exist and links the local map.
        Returns True if successful, False otherwise.
        """
        try:
            if not node_id or not floor_id:
                return False

            self.floor_manager.create_floor(floor_id)
            ok = self.floor_manager.assign_node(node_id, floor_id)
            if ok:
                # If node has an existing map or is currently active, update local map association
                existing_map = self.local_map_manager.get_map(node_id)
                if existing_map is not None:
                    self.local_map_manager.create_map(node_id, floor_id)
            return ok
        except Exception as e:
            log_pipeline.error(f"Error assigning node {node_id} to floor {floor_id}: {e}")
            return False

    def start_node_scan(self, node_id: str) -> Tuple[bool, str]:
        """Start scanning for a node.

        Fails safely if node has no floor assignment.
        Returns (success: bool, message: str).
        """
        try:
            if not node_id:
                return False, "Invalid node_id"

            floor_id = self.floor_manager.get_node_floor(node_id)
            if not floor_id:
                return False, f"Node {node_id} has no floor assignment"

            ok = self.scan_session.start_node_scan(node_id)
            if not ok:
                return False, f"Failed to transition node {node_id} to SCANNING"

            # Create or update local map for scanning session
            self.local_map_manager.create_map(node_id, floor_id)
            self.floor_manager.activate_floor(floor_id)

            # Ensure WorldModel reflects the active floor immediately
            world_floor = self.world_model.get_or_create_floor(floor_id)
            world_floor.status = FloorStatus.ACTIVE
            if node_id not in world_floor.node_ids:
                world_floor.node_ids.append(node_id)
            self.world_model.participating_nodes.add(node_id)

            return True, f"Node {node_id} scanning started on floor {floor_id}"
        except Exception as e:
            log_pipeline.error(f"Error starting scan for node {node_id}: {e}")
            return False, f"Pipeline error starting scan: {e}"

    def stop_node_scan(self, node_id: str) -> bool:
        """Stop scanning for a single node."""
        try:
            return self.scan_session.stop_node_scan(node_id)
        except Exception as e:
            log_pipeline.error(f"Error stopping scan for node {node_id}: {e}")
            return False

    def handle_node_disconnect(self, node_id: str) -> None:
        """Handle WS disconnect for a node.

        Marks node DISCONNECTED and removes floor assignment.
        Preserves local map data and world model observations.
        """
        try:
            if not node_id:
                return
            self.scan_session.disconnect_node(node_id)
            self.floor_manager.remove_node(node_id)
        except Exception as e:
            log_pipeline.error(f"Error handling disconnect for node {node_id}: {e}")

    # ── Telemetry Ingestion → Pose ─────────────────────────────────

    def process_telemetry(self, node_id: str, payload: dict) -> Optional[Pose]:
        """Convert incoming telemetry dictionary into a Pose model and update local map.

        Preserves GPS and motion metrics without fabricating local x/y coords.
        """
        try:
            if not node_id or not isinstance(payload, dict):
                return None

            lat = payload.get("lat")
            lon = payload.get("lon", payload.get("lng"))
            alt = payload.get("altitude_m", payload.get("altitude"))
            heading = payload.get("heading_deg", payload.get("heading", 0.0))
            speed = payload.get("speed_mps", payload.get("speed", 0.0))
            ts = payload.get("timestamp", time.time())

            pose = Pose(
                x=0.0,
                y=0.0,
                z=float(alt) if alt is not None else 0.0,
                yaw=float(heading),
                pitch=0.0,
                roll=0.0,
                timestamp=float(ts),
                latitude=float(lat) if lat is not None else None,
                longitude=float(lon) if lon is not None else None,
                altitude=float(alt) if alt is not None else None,
            )

            # Update local map if node is currently SCANNING
            if self.scan_session.get_node_state(node_id) == NodeScanState.SCANNING:
                self.local_map_manager.update_pose(node_id, pose)

            return pose
        except Exception as e:
            log_pipeline.error(f"Error processing telemetry for node {node_id}: {e}")
            return None

    # ── Camera / Perception Ingestion ──────────────────────────────

    def process_camera_frame(
        self,
        node_id: str,
        frame: Any,
        frame_id: Optional[str] = None,
    ) -> List[SemanticObservation]:
        """Process a camera frame through PerceptionEngine if node is SCANNING and has a floor assignment."""
        try:
            if not node_id or frame is None:
                return []

            # Verify node is SCANNING
            if self.scan_session.get_node_state(node_id) != NodeScanState.SCANNING:
                return []

            # Verify floor assignment exists
            floor_id = self.floor_manager.get_node_floor(node_id)
            if not floor_id:
                return []

            # Run perception engine
            observations = self.perception_engine.process(
                frame=frame,
                node_id=node_id,
                floor_id=floor_id,
                frame_id=frame_id,
            )

            # Route observations to LocalMapManager and WorldModel
            for obs in observations:
                self.local_map_manager.add_observation(node_id, obs)
                self.world_model.add_observation(obs)

            return observations
        except Exception as e:
            log_pipeline.error(f"Error processing camera frame for node {node_id}: {e}")
            return []

    # ── Fusion & World Model Ingestion ────────────────────────────

    def run_fusion(self) -> FusionResult:
        """Run FusionEngine across all current local maps and apply results to WorldModel."""
        try:
            maps = self.local_map_manager.list_maps()
            result = self.fusion_engine.fuse_maps(maps)
            self.world_model.apply_fusion_result(result)
            return result
        except Exception as e:
            log_pipeline.error(f"Error running fusion pipeline: {e}")
            return FusionResult(
                participating_nodes=[],
                associations=[],
                merge_decisions={},
                confidence=0.0,
                aligned_observations=[],
                warnings=[f"Pipeline fusion error: {e}"],
            )

    # ── Queries & State Snapshots ─────────────────────────────────

    def get_world_state(self) -> dict:
        """Return serialized WorldModel dictionary."""
        try:
            return self.world_model.to_dict()
        except Exception as e:
            log_pipeline.error(f"Error serializing world state: {e}")
            return {"building_id": "ERROR", "floors": {}}

    def get_pipeline_state(self) -> dict:
        """Return a complete diagnostic snapshot of the mapping pipeline."""
        try:
            return {
                "scan_session": self.scan_session.to_state(),
                "floor_manager": self.floor_manager.to_state(),
                "local_maps": [
                    m.node_id for m in self.local_map_manager.list_maps()
                ],
                "world_model": self.world_model.to_dict(),
            }
        except Exception as e:
            log_pipeline.error(f"Error getting pipeline state: {e}")
            return {"error": str(e)}

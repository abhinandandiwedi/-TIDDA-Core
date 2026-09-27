import json
import logging
import os
import time
import uuid
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("reconstruction.world_store")

class WorldStore:
    def __init__(self, workspace_dir: str = "workspaces/default"):
        self.workspace_dir = Path(workspace_dir)
        self.world_dir = self.workspace_dir / "world"
        self.world_state_path = self.world_dir / "world_state.json"
        self.global_map_path = self.world_dir / "global_map.ply"
        
        self.world_dir.mkdir(parents=True, exist_ok=True)

    def save_world(self, global_points: np.ndarray, global_colors: np.ndarray, nodes_metadata: dict, map_version: int) -> dict:
        """Persist the world state safely."""
        try:
            # 1. Save point cloud safely
            tmp_ply = self.global_map_path.parent / (self.global_map_path.name + ".tmp.ply")
            if len(global_points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(global_points)
                if len(global_colors) > 0:
                    pcd.colors = o3d.utility.Vector3dVector(global_colors.astype(np.float64) / 255.0)
                o3d.io.write_point_cloud(str(tmp_ply), pcd)
                os.replace(str(tmp_ply), str(self.global_map_path))
            
            # 2. Save metadata safely
            metadata = self.load_metadata()
            state = {
                "world_id": metadata.get("world_id", str(uuid.uuid4())),
                "map_version": map_version,
                "created_at": metadata.get("created_at", time.time()),
                "updated_at": time.time(),
                "total_points": len(global_points),
                "active_nodes": len(nodes_metadata),
                "persistence_version": 1,
                "nodes": nodes_metadata
            }
            
            tmp_json = self.world_state_path.with_suffix(".json.tmp")
            with open(tmp_json, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(str(tmp_json), str(self.world_state_path))
                
            logger.info(f"[WORLD] Saved world v{map_version} with {len(global_points)} points.")
            return {"status": "OK", "version": map_version, "points": len(global_points)}
        except Exception as e:
            logger.error(f"[WORLD] Failed to save world: {e}")
            return {"status": "ERROR", "message": str(e)}

    def load_metadata(self) -> dict:
        if not self.world_state_path.exists():
            return {}
        try:
            with open(self.world_state_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[WORLD] Metadata corrupted: {e}")
            return {}
            
    def load_world(self) -> dict:
        """Load the persisted world state and points."""
        metadata = self.load_metadata()
        if not metadata:
            return {"status": "EMPTY"}
            
        if metadata.get("persistence_version") != 1:
            logger.warning("[WORLD] Unsupported persistence version, ignoring.")
            return {"status": "EMPTY"}
            
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
        
        if self.global_map_path.exists():
            try:
                pcd = o3d.io.read_point_cloud(str(self.global_map_path))
                points = np.asarray(pcd.points).astype(np.float32)
                colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
            except Exception as e:
                logger.error(f"[WORLD] Failed to load PLY: {e}")
                return {"status": "RECOVERY", "metadata": metadata}
        else:
            if metadata.get("total_points", 0) > 0:
                logger.error("[WORLD] PLY file is missing but metadata claims points exist. State corrupted.")
                return {"status": "RECOVERY", "metadata": metadata}
                
        logger.info(f"[WORLD] Loaded world v{metadata.get('map_version')} with {len(points)} points.")
        return {
            "status": "OK",
            "metadata": metadata,
            "points": points,
            "colors": colors
        }

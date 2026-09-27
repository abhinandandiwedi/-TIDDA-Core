import logging
import threading
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import open3d as o3d

from reconstruction.incremental_mapper import IncrementalMapper, MapperConfig
from reconstruction.world_store import WorldStore

logger = logging.getLogger("reconstruction.cooperative_mapper")

class CooperativeMapper:
    """Manager for multi-node cooperative 3D mapping."""

    def __init__(self, workspace_dir: str = "workspaces/default"):
        self._nodes: Dict[str, IncrementalMapper] = {}
        self._lock = threading.Lock()

        # Global map state
        self._global_points = np.zeros((0, 3), dtype=np.float32)
        self._global_colors = np.zeros((0, 3), dtype=np.uint8)
        self._last_alignment = {}
        
        self.map_version = 0
        self.world_store = WorldStore(workspace_dir)
        self.reload_world()

    def reload_world(self):
        with self._lock:
            self._nodes.clear()
            self._last_alignment.clear()
            state = self.world_store.load_world()
            if state["status"] in ["OK", "RECOVERY"]:
                self._global_points = state.get("points", np.zeros((0, 3), dtype=np.float32))
                self._global_colors = state.get("colors", np.zeros((0, 3), dtype=np.uint8))
                meta = state.get("metadata", {})
                self.map_version = meta.get("map_version", 0)
                logger.info(f"[CO-OP] Reloaded persistent world version {self.map_version}")
            else:
                self._global_points = np.zeros((0, 3), dtype=np.float32)
                self._global_colors = np.zeros((0, 3), dtype=np.uint8)
                self.map_version = 0
                logger.info("[CO-OP] Started fresh empty world.")
            return state["status"]
                
    def save_world_state(self):
        # Assumes lock is held by caller
        nodes_meta = {
            node_id: {
                "points": len(mapper.get_full_pointcloud()[0]),
                "keyframes": len(mapper._keyframes)
            } for node_id, mapper in self._nodes.items()
        }
        return self.world_store.save_world(
            self._global_points,
            self._global_colors,
            nodes_meta,
            self.map_version
        )

    def register_node(self, node_id: str, config: Optional[MapperConfig] = None) -> IncrementalMapper:
        """Register a new mobile mapping node."""
        with self._lock:
            if node_id not in self._nodes:
                self._nodes[node_id] = IncrementalMapper(config)
                logger.info(f"[CO-OP] Registered node {node_id}")
            return self._nodes[node_id]

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from the cooperative mapper."""
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info(f"[CO-OP] Unregistered node {node_id}")

    def get_node_map(self, node_id: str) -> Optional[IncrementalMapper]:
        """Retrieve the local mapper for a specific node."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[str]:
        with self._lock:
            return list(self._nodes.keys())

    def reset_global_map(self) -> None:
        """Clear the global fused map."""
        with self._lock:
            self._global_points = np.zeros((0, 3), dtype=np.float32)
            self._global_colors = np.zeros((0, 3), dtype=np.uint8)
            self._last_alignment = {}
            self.map_version = 0
            for mapper in self._nodes.values():
                mapper.reset()
            self.save_world_state()
            logger.info("[CO-OP] Global map and all local maps reset.")

    def find_correspondences(
        self, source_id: str, target_id: str
    ) -> Tuple[str, int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Find 2D-2D visual correspondences between the latest keyframes of two nodes.

        target_id: The base map (e.g., Node A)
        source_id: The incoming map to align (e.g., Node B)
        """
        source_mapper = self.get_node_map(source_id)
        target_mapper = self.get_node_map(target_id)

        if not source_mapper or not target_mapper:
            return "ALIGNMENT_FAILED", 0, None, None, None, None

        # Get latest keyframes
        src_kfs = source_mapper._keyframes[-5:] # look at last 5
        tgt_kfs = target_mapper._keyframes[-5:]

        if not src_kfs or not tgt_kfs:
            return "INSUFFICIENT_CORRESPONDENCES", 0, None, None, None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        best_good = []
        best_src_kf = None
        best_tgt_kf = None

        for s_kf in reversed(src_kfs):
            if s_kf.descriptors is None or len(s_kf.descriptors) < 2: continue
            for t_kf in reversed(tgt_kfs):
                if t_kf.descriptors is None or len(t_kf.descriptors) < 2: continue

                try:
                    # Lowe's ratio test for ORB
                    knn_matches = bf.knnMatch(s_kf.descriptors, t_kf.descriptors, k=2)
                    good = []
                    for m_n in knn_matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            if m.distance < 0.75 * n.distance:
                                good.append(m)
                        elif len(m_n) == 1:
                            good.append(m_n[0])

                    if len(good) > len(best_good):
                        best_good = good
                        best_src_kf = s_kf
                        best_tgt_kf = t_kf
                except Exception:
                    continue

        if len(best_good) < 15:
            return "INSUFFICIENT_CORRESPONDENCES", len(best_good), None, None, None, None

        pts_src = np.float32([best_src_kf.keypoints[m.queryIdx].pt for m in best_good])
        pts_tgt = np.float32([best_tgt_kf.keypoints[m.trainIdx].pt for m in best_good])

        return "ALIGNMENT_OK", len(best_good), pts_src, pts_tgt, best_src_kf.pose, best_tgt_kf.pose

    def estimate_alignment(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Estimate the rigid transformation from source map to target map."""
        status, num_matches, pts_src, pts_tgt, pose_src, pose_tgt = self.find_correspondences(source_id, target_id)

        if status != "ALIGNMENT_OK":
            return {"status": status, "inliers": 0, "matches": num_matches}

        # 1. Essential Matrix to find relative camera pose (src camera to tgt camera)
        focal = 500.0 # Approximate
        pp = (320.0, 240.0)

        E, mask = cv2.findEssentialMat(
            pts_src, pts_tgt, focal=focal, pp=pp,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or mask is None:
            return {"status": "ALIGNMENT_FAILED", "inliers": 0, "matches": num_matches, "inlier_ratio": 0.0}

        inliers = int(mask.sum())
        inlier_ratio = inliers / num_matches if num_matches > 0 else 0

        if inliers < 15:
            return {"status": "INSUFFICIENT_CORRESPONDENCES", "inliers": inliers, "matches": num_matches, "inlier_ratio": inlier_ratio}

        if inlier_ratio < 0.20:
            return {"status": "LOW_INLIER_RATIO", "inliers": inliers, "matches": num_matches, "inlier_ratio": inlier_ratio}

        _, R, t, pose_mask = cv2.recoverPose(E, pts_src, pts_tgt, focal=focal, pp=pp, mask=mask)
        if R is None or t is None:
            return {"status": "INVALID_TRANSFORM", "inliers": inliers, "matches": num_matches, "inlier_ratio": inlier_ratio}

        # 2. Build Camera-to-Camera transform (T_src_to_tgt)
        T_cam = np.eye(4)
        T_cam[:3, :3] = R
        T_cam[:3, 3] = t.ravel()

        # 3. Compute Map-to-Map transform (M_src_to_tgt)
        inv_pose_src = np.linalg.inv(pose_src)
        M = pose_tgt @ T_cam @ inv_pose_src

        return {
            "status": "ALIGNMENT_OK",
            "inliers": inliers,
            "matches": num_matches,
            "inlier_ratio": inlier_ratio,
            "transform": M,
            "scale_ambiguity": True
        }

    def fuse_maps(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Align and fuse source map into target map's global coordinate space."""
        with self._lock:
            src_mapper = self._nodes.get(source_id)
            tgt_mapper = self._nodes.get(target_id)

            if not src_mapper or not tgt_mapper:
                return {"status": "ERROR", "message": "Nodes not found"}

            pts_src, col_src = src_mapper.get_full_pointcloud()
            pts_tgt, col_tgt = tgt_mapper.get_full_pointcloud()

            if len(pts_src) < 10 or len(pts_tgt) < 10:
                return {"status": "ERROR", "message": "Maps too small for fusion"}

        # Get initial alignment via visual correspondences
        alignment = self.estimate_alignment(source_id, target_id)
        if alignment["status"] != "ALIGNMENT_OK":
            return alignment

        M = alignment["transform"]

        # Apply initial transformation to source points
        pts_src_hom = np.hstack([pts_src, np.ones((len(pts_src), 1))])
        pts_src_aligned = (M @ pts_src_hom.T).T[:, :3]

        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(pts_src_aligned)
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(pts_tgt)

        # Multi-scale ICP Refinement
        # Coarse pass
        reg_coarse = o3d.pipelines.registration.registration_icp(
            pcd_src.voxel_down_sample(0.1), pcd_tgt.voxel_down_sample(0.1),
            0.5, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        )

        # Fine pass
        reg_fine = o3d.pipelines.registration.registration_icp(
            pcd_src, pcd_tgt,
            0.2, reg_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        if float(reg_fine.fitness) < 0.10 or float(reg_fine.inlier_rmse) > 0.4:
            return {
                "status": "ICP_REJECTED",
                "inliers": alignment["inliers"],
                "matches": alignment["matches"],
                "inlier_ratio": alignment["inlier_ratio"],
                "final_rmse": float(reg_fine.inlier_rmse),
                "icp_fitness": float(reg_fine.fitness)
            }

        # Apply refined transformation
        pts_src_final = np.asarray(pcd_src.transform(reg_fine.transformation).points)

        # Fuse into Global Map
        with self._lock:
            global_before = len(self._global_points)

            all_pts = np.vstack([self._global_points, pts_tgt, pts_src_final]) if global_before > 0 else np.vstack([pts_tgt, pts_src_final])
            all_col = np.vstack([self._global_colors, col_tgt, col_src]) if global_before > 0 else np.vstack([col_tgt, col_src])

            pcd_all = o3d.geometry.PointCloud()
            pcd_all.points = o3d.utility.Vector3dVector(all_pts)
            pcd_all.colors = o3d.utility.Vector3dVector(all_col.astype(np.float64) / 255.0)

            pcd_down = pcd_all.voxel_down_sample(voxel_size=0.04) # tuned voxel size

            self._global_points = np.asarray(pcd_down.points).astype(np.float32)
            self._global_colors = (np.asarray(pcd_down.colors) * 255.0).astype(np.uint8)

            global_after = len(self._global_points)
            
            self.map_version += 1
            res = self.save_world_state()
            print(f"DEBUG fuse_maps save_world_state result: {res}")

            self._last_alignment = {
                "source": source_id,
                "target": target_id,
                "status": "ALIGNMENT_OK",
                "inliers": alignment["inliers"],
                "rmse": float(reg_fine.inlier_rmse),
                "icp_fitness": float(reg_fine.fitness),
            }

            return {
                "status": "ALIGNMENT_OK",
                "matches": alignment["matches"],
                "inliers": alignment["inliers"],
                "inlier_ratio": alignment["inlier_ratio"],
                "initial_rmse": None,
                "final_rmse": float(reg_fine.inlier_rmse),
                "icp_fitness": float(reg_fine.fitness),
                "points_added": len(pts_src_final),
                "points_fused": (len(pts_tgt) + len(pts_src_final)) - (global_after - global_before),
                "global_points_before": global_before,
                "global_points_after": global_after,
                "confidence": min(1.0, (alignment["inlier_ratio"] * float(reg_fine.fitness)) * 2.0)
            }

    def get_global_map_snapshot(self) -> Dict[str, Any]:
        """Return the state of the cooperative mapping system."""
        with self._lock:
            return {
                "status": "READY" if len(self._global_points) > 0 else "IDLE",
                "map_version": self.map_version,
                "nodes": len(self._nodes),
                "node_ids": list(self._nodes.keys()),
                "global_points": len(self._global_points),
                "last_alignment": self._last_alignment,
                "persistence": self.world_store.load_metadata().get("persistence_version", 1) if self.map_version > 0 else "NONE"
            }

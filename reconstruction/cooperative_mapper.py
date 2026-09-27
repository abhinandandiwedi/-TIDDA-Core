import logging
import threading
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import open3d as o3d

from reconstruction.incremental_mapper import IncrementalMapper, MapperConfig

logger = logging.getLogger("reconstruction.cooperative_mapper")

class CooperativeMapper:
    """Manager for multi-node cooperative 3D mapping."""
    
    def __init__(self):
        self._nodes: Dict[str, IncrementalMapper] = {}
        self._lock = threading.Lock()
        
        # Global map state
        self._global_points = np.zeros((0, 3), dtype=np.float32)
        self._global_colors = np.zeros((0, 3), dtype=np.uint8)
        self._last_alignment = {}
        
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
            for mapper in self._nodes.values():
                mapper.reset()
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
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        best_matches = []
        best_src_kf = None
        best_tgt_kf = None
        
        for s_kf in reversed(src_kfs):
            if s_kf.descriptors is None or len(s_kf.descriptors) == 0: continue
            for t_kf in reversed(tgt_kfs):
                if t_kf.descriptors is None or len(t_kf.descriptors) == 0: continue
                
                try:
                    matches = bf.match(s_kf.descriptors, t_kf.descriptors)
                    if len(matches) > len(best_matches):
                        best_matches = matches
                        best_src_kf = s_kf
                        best_tgt_kf = t_kf
                except Exception:
                    continue
                    
        if len(best_matches) < 15:
            return "INSUFFICIENT_CORRESPONDENCES", len(best_matches), None, None, None, None
            
        # Extract matched points
        best_matches = sorted(best_matches, key=lambda x: x.distance)
        best_matches = best_matches[: max(30, int(len(best_matches) * 0.2))]
        
        pts_src = np.float32([best_src_kf.keypoints[m.queryIdx].pt for m in best_matches])
        pts_tgt = np.float32([best_tgt_kf.keypoints[m.trainIdx].pt for m in best_matches])
        
        return "ALIGNMENT_OK", len(best_matches), pts_src, pts_tgt, best_src_kf.pose, best_tgt_kf.pose

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
            method=cv2.RANSAC, prob=0.999, threshold=2.0
        )
        
        if E is None or mask is None:
            return {"status": "ALIGNMENT_FAILED", "inliers": 0, "matches": num_matches}
            
        inliers = int(mask.sum())
        if inliers < 10:
            return {"status": "INSUFFICIENT_CORRESPONDENCES", "inliers": inliers, "matches": num_matches}
            
        _, R, t, pose_mask = cv2.recoverPose(E, pts_src, pts_tgt, focal=focal, pp=pp, mask=mask)
        if R is None or t is None:
            return {"status": "ALIGNMENT_FAILED", "inliers": inliers, "matches": num_matches}
            
        # 2. Build Camera-to-Camera transform (T_src_to_tgt)
        T_cam = np.eye(4)
        T_cam[:3, :3] = R
        T_cam[:3, 3] = t.ravel()
        
        # 3. Compute Map-to-Map transform (M_src_to_tgt)
        # M = P_tgt * T_cam * inv(P_src)
        inv_pose_src = np.linalg.inv(pose_src)
        M = pose_tgt @ T_cam @ inv_pose_src
        
        return {
            "status": "ALIGNMENT_OK",
            "inliers": inliers,
            "matches": num_matches,
            "transform": M,
            "scale_ambiguity": True # Explicitly flag that scale is unconstrained
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
        
        # Apply transformation to source points
        pts_src_hom = np.hstack([pts_src, np.ones((len(pts_src), 1))])
        pts_src_aligned = (M @ pts_src_hom.T).T[:, :3]
        
        # ICP Refinement (Open3D)
        # We perform Point-to-Point ICP to refine the rough visual alignment
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(pts_src_aligned)
        
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(pts_tgt)
        
        # ICP max correspondence distance (generous due to scale ambiguity)
        threshold = 0.5 
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_src, pcd_tgt, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Apply refined transformation
        pts_src_final = np.asarray(pcd_src.transform(reg_p2p.transformation).points)
        
        # Fuse into Global Map
        with self._lock:
            global_before = len(self._global_points)
            
            # Simple spatial deduplication
            all_pts = np.vstack([self._global_points, pts_tgt, pts_src_final])
            all_col = np.vstack([self._global_colors, col_tgt, col_src])
            
            # Voxel downsample to deduplicate
            pcd_all = o3d.geometry.PointCloud()
            pcd_all.points = o3d.utility.Vector3dVector(all_pts)
            pcd_all.colors = o3d.utility.Vector3dVector(all_col.astype(np.float64) / 255.0)
            
            pcd_down = pcd_all.voxel_down_sample(voxel_size=0.03)
            
            self._global_points = np.asarray(pcd_down.points).astype(np.float32)
            self._global_colors = (np.asarray(pcd_down.colors) * 255.0).astype(np.uint8)
            
            global_after = len(self._global_points)
            
            self._last_alignment = {
                "source": source_id,
                "target": target_id,
                "status": "ALIGNMENT_OK",
                "inliers": alignment["inliers"],
                "rmse": float(reg_p2p.inlier_rmse),
                "icp_fitness": float(reg_p2p.fitness),
            }
            
            return {
                "status": "ALIGNMENT_OK",
                "inliers": alignment["inliers"],
                "initial_rmse": None, # Visual feature alignment doesn't give RMSE directly
                "final_rmse": float(reg_p2p.inlier_rmse),
                "points_added": len(pts_src_final),
                "points_fused": (len(pts_tgt) + len(pts_src_final)) - (global_after - global_before),
                "global_points_before": global_before,
                "global_points_after": global_after
            }

    def get_global_map_snapshot(self) -> Dict[str, Any]:
        """Return the state of the cooperative mapping system."""
        with self._lock:
            return {
                "status": "READY" if len(self._global_points) > 0 else "IDLE",
                "nodes": len(self._nodes),
                "node_ids": list(self._nodes.keys()),
                "global_points": len(self._global_points),
                "last_alignment": self._last_alignment
            }

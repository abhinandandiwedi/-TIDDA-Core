# ══════════════════════════════════════════════════════════════════
#  🧪 UNIT TESTS — TIDDA Mapping Pipeline
#  Validates mapping_pipeline.py integration across ScanSession, FloorManager,
#  PerceptionEngine, LocalMapManager, FusionEngine, and WorldModel.
#  Run:  python3 -m unittest test_mapping_pipeline -v
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import unittest

from mapping_pipeline import MappingPipeline
from perception import MockPerceptionEngine
from scan_session import NodeScanState
from mapping_models import SemanticObservation, Pose, Landmark


class TestMappingPipelineLifecycle(unittest.TestCase):
    """Tests 1–5: Pipeline initialization, registration, scan start, floor assignment."""

    def test_pipeline_initializes(self):
        """Test 1: Pipeline initializes."""
        pipeline = MappingPipeline()
        self.assertIsNotNone(pipeline.scan_session)
        self.assertIsNotNone(pipeline.floor_manager)
        self.assertIsNotNone(pipeline.local_map_manager)
        self.assertIsNotNone(pipeline.perception_engine)
        self.assertIsNotNone(pipeline.fusion_engine)
        self.assertIsNotNone(pipeline.world_model)

    def test_node_registration_does_not_start_scanning(self):
        """Test 2: Node registration does not start scanning (starts IDLE)."""
        pipeline = MappingPipeline()
        pipeline.register_node("PHONE-01")

        state = pipeline.scan_session.get_node_state("PHONE-01")
        self.assertEqual(state, NodeScanState.IDLE)
        self.assertNotIn("PHONE-01", pipeline.scan_session.scanning_node_ids)

    def test_floor_assignment_works(self):
        """Test 4: Floor assignment works."""
        pipeline = MappingPipeline()
        pipeline.register_node("PHONE-01")

        success = pipeline.assign_node_to_floor("PHONE-01", "FLOOR-1")
        self.assertTrue(success)
        self.assertEqual(pipeline.floor_manager.get_node_floor("PHONE-01"), "FLOOR-1")

    def test_scan_start_without_floor_assignment_fails_safely(self):
        """Test 5: Scan start without floor assignment fails safely."""
        pipeline = MappingPipeline()
        pipeline.register_node("PHONE-01")
        # No floor assigned!

        success, msg = pipeline.start_node_scan("PHONE-01")
        self.assertFalse(success)
        self.assertIn("no floor assignment", msg)
        self.assertEqual(pipeline.scan_session.get_node_state("PHONE-01"), NodeScanState.IDLE)

    def test_scan_start_changes_node_state(self):
        """Test 3: Scan start changes node state (after floor assignment)."""
        pipeline = MappingPipeline()
        pipeline.register_node("PHONE-01")
        pipeline.assign_node_to_floor("PHONE-01", "FLOOR-1")

        success, msg = pipeline.start_node_scan("PHONE-01")
        self.assertTrue(success)
        self.assertEqual(pipeline.scan_session.get_node_state("PHONE-01"), NodeScanState.SCANNING)
        self.assertIn("PHONE-01", pipeline.scan_session.scanning_node_ids)


class TestTelemetryAndPose(unittest.TestCase):
    """Tests 6–7: Pose telemetry handling and invalid telemetry safety."""

    def test_pose_telemetry_updates_local_map_when_valid(self):
        """Test 6: Pose telemetry can update local map when valid."""
        pipeline = MappingPipeline()
        pipeline.register_node("PHONE-01")
        pipeline.assign_node_to_floor("PHONE-01", "FLOOR-1")
        pipeline.start_node_scan("PHONE-01")

        telem = {
            "lat": 28.6139,
            "lon": 77.2090,
            "altitude_m": 5.0,
            "heading_deg": 90.0,
            "speed_mps": 1.2,
            "timestamp": 1000.0,
        }
        pose = pipeline.process_telemetry("PHONE-01", telem)
        self.assertIsNotNone(pose)
        self.assertEqual(pose.latitude, 28.6139)
        self.assertEqual(pose.longitude, 77.2090)

        # Verify updated local map trajectory
        traj = pipeline.local_map_manager.get_trajectory("PHONE-01")
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj[0].latitude, 28.6139)

    def test_invalid_telemetry_does_not_create_fake_pose(self):
        """Test 7: Invalid telemetry does not create fake pose."""
        pipeline = MappingPipeline()
        pipeline.register_node("PHONE-01")

        # Invalid payload (not a dict)
        pose1 = pipeline.process_telemetry("PHONE-01", None)
        self.assertIsNone(pose1)

        # Empty node_id
        pose2 = pipeline.process_telemetry("", {"lat": 10.0})
        self.assertIsNone(pose2)

        # Check trajectory is untouched
        self.assertEqual(len(pipeline.local_map_manager.get_trajectory("PHONE-01")), 0)


class TestPerceptionAndMapIngestion(unittest.TestCase):
    """Tests 8–10: Perception frame processing, local map ingestion, multi-node map isolation."""

    def test_valid_camera_frame_reaches_perception(self):
        """Test 8: Valid camera frame can reach perception."""
        mock_perception = MockPerceptionEngine()
        pipeline = MappingPipeline(perception_engine=mock_perception)

        pipeline.register_node("PHONE-01")
        pipeline.assign_node_to_floor("PHONE-01", "FLOOR-1")
        pipeline.start_node_scan("PHONE-01")

        obs_list = pipeline.process_camera_frame("PHONE-01", "valid_base64_frame_data")
        self.assertGreater(len(obs_list), 0)
        self.assertEqual(obs_list[0].node_id, "PHONE-01")
        self.assertEqual(obs_list[0].floor_id, "FLOOR-1")

    def test_perception_output_reaches_local_map_and_world(self):
        """Test 9 & 12: Perception output reaches local map and WorldModel."""
        mock_perception = MockPerceptionEngine()
        pipeline = MappingPipeline(perception_engine=mock_perception)

        pipeline.register_node("PHONE-01")
        pipeline.assign_node_to_floor("PHONE-01", "FLOOR-1")
        pipeline.start_node_scan("PHONE-01")

        pipeline.process_camera_frame("PHONE-01", "frame_bytes")

        # Check LocalMapManager
        local_obs = pipeline.local_map_manager.get_observations("PHONE-01")
        self.assertGreater(len(local_obs), 0)

        # Check WorldModel (Test 12)
        world_obs = pipeline.world_model.get_observations("FLOOR-1")
        self.assertGreater(len(world_obs), 0)

    def test_multiple_nodes_maintain_independent_local_maps(self):
        """Test 10: Multiple nodes maintain independent local maps."""
        mock_perception = MockPerceptionEngine()
        pipeline = MappingPipeline(perception_engine=mock_perception)

        pipeline.register_node("PHONE-01")
        pipeline.register_node("PHONE-02")
        pipeline.assign_node_to_floor("PHONE-01", "FLOOR-1")
        pipeline.assign_node_to_floor("PHONE-02", "FLOOR-2")

        pipeline.start_node_scan("PHONE-01")
        pipeline.start_node_scan("PHONE-02")

        pipeline.process_camera_frame("PHONE-01", "frame_1")

        map1_obs = pipeline.local_map_manager.get_observations("PHONE-01")
        map2_obs = pipeline.local_map_manager.get_observations("PHONE-02")

        self.assertGreater(len(map1_obs), 0)
        self.assertEqual(len(map2_obs), 0)


class TestFusionWorldModelAndDisconnect(unittest.TestCase):
    """Tests 11, 13, 14: Fusion execution, disconnect data survival, error isolation."""

    def test_fusion_can_receive_multiple_local_maps(self):
        """Test 11: Fusion can receive multiple local maps."""
        mock_perception = MockPerceptionEngine()
        pipeline = MappingPipeline(perception_engine=mock_perception)

        pipeline.register_node("PHONE-01")
        pipeline.register_node("PHONE-02")
        pipeline.assign_node_to_floor("PHONE-01", "FLOOR-1")
        pipeline.assign_node_to_floor("PHONE-02", "FLOOR-1")
        pipeline.start_node_scan("PHONE-01")
        pipeline.start_node_scan("PHONE-02")

        pipeline.process_camera_frame("PHONE-01", "frame_1")
        pipeline.process_camera_frame("PHONE-02", "frame_2")

        fusion_res = pipeline.run_fusion()
        self.assertEqual(len(fusion_res.participating_nodes), 2)
        self.assertIn("PHONE-01", fusion_res.participating_nodes)
        self.assertIn("PHONE-02", fusion_res.participating_nodes)

    def test_node_disconnect_does_not_destroy_world_model_data(self):
        """Test 13: Node disconnect does not destroy historical world-model data."""
        mock_perception = MockPerceptionEngine()
        pipeline = MappingPipeline(perception_engine=mock_perception)

        pipeline.register_node("PHONE-01")
        pipeline.assign_node_to_floor("PHONE-01", "FLOOR-1")
        pipeline.start_node_scan("PHONE-01")
        pipeline.process_camera_frame("PHONE-01", "frame_1")

        self.assertGreater(len(pipeline.world_model.get_observations("FLOOR-1")), 0)

        # Disconnect node
        pipeline.handle_node_disconnect("PHONE-01")

        # Node is no longer scanning
        self.assertEqual(pipeline.scan_session.get_node_state("PHONE-01"), NodeScanState.DISCONNECTED)

        # Historical observations & local map STILL SURVIVE
        self.assertGreater(len(pipeline.world_model.get_observations("FLOOR-1")), 0)
        self.assertIsNotNone(pipeline.local_map_manager.get_map("PHONE-01"))

    def test_one_node_failure_does_not_stop_other_nodes(self):
        """Test 14: One node failure does not stop other nodes."""
        pipeline = MappingPipeline()

        pipeline.register_node("PHONE-01")
        pipeline.register_node("PHONE-02")
        pipeline.assign_node_to_floor("PHONE-02", "FLOOR-1")
        pipeline.start_node_scan("PHONE-02")

        # Invalid operation on PHONE-01 (e.g. processing frame without floor/scan)
        obs1 = pipeline.process_camera_frame("PHONE-01", None)
        self.assertEqual(obs1, [])

        # PHONE-02 remains active and scanning
        self.assertEqual(pipeline.scan_session.get_node_state("PHONE-02"), NodeScanState.SCANNING)


class TestExistingServerHandlers(unittest.TestCase):
    """Tests 15–18: Existing WebSocket registration, telemetry, heartbeat, and scan controls."""

    def test_existing_ws_handlers_and_scan_controls(self):
        """Tests 15–18: Existing registration, telemetry, heartbeat, scan controls work."""
        from mobile_node import MobileNodeRegistry
        from scan_session import ScanSession

        reg = MobileNodeRegistry()
        session = ScanSession()
        pipeline = MappingPipeline(scan_session=session)

        # Test 15: Existing WS node registration
        node = reg.register("PHONE-TEST")
        pipeline.register_node("PHONE-TEST")
        self.assertEqual(node.node_id, "PHONE-TEST")
        self.assertEqual(session.get_node_state("PHONE-TEST"), NodeScanState.IDLE)

        # Test 16: Existing telemetry update
        ok_telem = reg.update_telemetry("PHONE-TEST", {"battery_pct": 95, "lat": 10.0, "lon": 20.0})
        self.assertTrue(ok_telem)

        # Test 17: Existing heartbeat
        ok_hb = reg.heartbeat("PHONE-TEST")
        self.assertTrue(ok_hb)

        # Test 18: Existing Step 2 scan controls
        pipeline.assign_node_to_floor("PHONE-TEST", "FLOOR-1")
        ok_scan, _ = pipeline.start_node_scan("PHONE-TEST")
        self.assertTrue(ok_scan)
        self.assertEqual(session.get_node_state("PHONE-TEST"), NodeScanState.SCANNING)

        ok_stop = pipeline.stop_node_scan("PHONE-TEST")
        self.assertTrue(ok_stop)
        self.assertEqual(session.get_node_state("PHONE-TEST"), NodeScanState.IDLE)


if __name__ == "__main__":
    unittest.main()

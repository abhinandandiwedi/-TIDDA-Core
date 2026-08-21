# ══════════════════════════════════════════════════════════════════
#  🧪 UNIT TESTS — TIDDA Scan Session Manager
#  Validates scan_session.py state machine and serialization.
#  Run:  python3 -m unittest test_scan_session -v
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import unittest

from scan_session import NodeScanState, ScanSession


class TestSessionLifecycle(unittest.TestCase):
    """Tests for session start/stop lifecycle."""

    def test_new_session_inactive(self):
        """Test 1: New session starts inactive."""
        session = ScanSession()
        self.assertFalse(session.active)
        self.assertIsNotNone(session.session_id)
        self.assertTrue(session.session_id.startswith("SCAN-"))

    def test_session_can_start(self):
        """Test 3: Session can be started."""
        session = ScanSession()
        session.start_session()
        self.assertTrue(session.active)

    def test_session_stop_deactivates(self):
        """Session stop deactivates the session."""
        session = ScanSession()
        session.start_session()
        session.stop_session()
        self.assertFalse(session.active)

    def test_custom_session_id(self):
        """Session can be created with a custom ID."""
        session = ScanSession(session_id="SCAN-001")
        self.assertEqual(session.session_id, "SCAN-001")


class TestNodeScanStates(unittest.TestCase):
    """Tests for individual node scan state transitions."""

    def test_connected_node_starts_idle(self):
        """Test 2: Connected node starts IDLE (not SCANNING)."""
        session = ScanSession()
        session.connect_node("PHONE-01")
        self.assertEqual(
            session.get_node_state("PHONE-01"),
            NodeScanState.IDLE,
        )

    def test_one_node_can_start_scanning(self):
        """Test 4: One node can start scanning."""
        session = ScanSession()
        session.start_session()
        session.connect_node("PHONE-01")
        result = session.start_node_scan("PHONE-01")
        self.assertTrue(result)
        self.assertEqual(
            session.get_node_state("PHONE-01"),
            NodeScanState.SCANNING,
        )
        self.assertIn("PHONE-01", session.scanning_node_ids)

    def test_multiple_nodes_scan_simultaneously(self):
        """Test 5: Multiple nodes can scan simultaneously."""
        session = ScanSession()
        session.start_session()
        session.connect_node("PHONE-01")
        session.connect_node("PHONE-02")
        session.connect_node("PHONE-03")
        session.start_node_scan("PHONE-01")
        session.start_node_scan("PHONE-02")
        session.start_node_scan("PHONE-03")
        self.assertEqual(len(session.scanning_node_ids), 3)
        self.assertEqual(
            session.get_node_state("PHONE-01"), NodeScanState.SCANNING
        )
        self.assertEqual(
            session.get_node_state("PHONE-02"), NodeScanState.SCANNING
        )
        self.assertEqual(
            session.get_node_state("PHONE-03"), NodeScanState.SCANNING
        )

    def test_one_node_stops_others_continue(self):
        """Test 6: One node can stop while others continue scanning."""
        session = ScanSession()
        session.start_session()
        session.connect_node("PHONE-01")
        session.connect_node("PHONE-02")
        session.start_node_scan("PHONE-01")
        session.start_node_scan("PHONE-02")

        # Stop only PHONE-01
        session.stop_node_scan("PHONE-01")

        self.assertEqual(
            session.get_node_state("PHONE-01"), NodeScanState.IDLE
        )
        self.assertEqual(
            session.get_node_state("PHONE-02"), NodeScanState.SCANNING
        )
        self.assertNotIn("PHONE-01", session.scanning_node_ids)
        self.assertIn("PHONE-02", session.scanning_node_ids)

    def test_pause_resume(self):
        """Test 7: Pause and resume works correctly."""
        session = ScanSession()
        session.start_session()
        session.connect_node("PHONE-01")
        session.start_node_scan("PHONE-01")

        # Pause
        result = session.pause_node_scan("PHONE-01")
        self.assertTrue(result)
        self.assertEqual(
            session.get_node_state("PHONE-01"), NodeScanState.PAUSED
        )
        self.assertNotIn("PHONE-01", session.scanning_node_ids)

        # Resume
        result = session.resume_node_scan("PHONE-01")
        self.assertTrue(result)
        self.assertEqual(
            session.get_node_state("PHONE-01"), NodeScanState.SCANNING
        )
        self.assertIn("PHONE-01", session.scanning_node_ids)

    def test_invalid_transitions_return_false(self):
        """Invalid state transitions return False."""
        session = ScanSession()
        session.connect_node("PHONE-01")

        # Can't pause an IDLE node
        self.assertFalse(session.pause_node_scan("PHONE-01"))
        # Can't resume an IDLE node
        self.assertFalse(session.resume_node_scan("PHONE-01"))
        # Can't stop an IDLE node
        self.assertFalse(session.stop_node_scan("PHONE-01"))
        # Can't start an unknown node
        self.assertFalse(session.start_node_scan("PHONE-99"))


class TestSessionStopAll(unittest.TestCase):
    """Tests for session-wide stop behavior."""

    def test_session_stop_stops_all_scanning(self):
        """Test 8: Session stop stops all currently scanning nodes."""
        session = ScanSession()
        session.start_session()
        session.connect_node("PHONE-01")
        session.connect_node("PHONE-02")
        session.connect_node("PHONE-03")
        session.start_node_scan("PHONE-01")
        session.start_node_scan("PHONE-02")
        # PHONE-03 remains IDLE

        session.stop_session()

        self.assertEqual(
            session.get_node_state("PHONE-01"), NodeScanState.IDLE
        )
        self.assertEqual(
            session.get_node_state("PHONE-02"), NodeScanState.IDLE
        )
        self.assertEqual(
            session.get_node_state("PHONE-03"), NodeScanState.IDLE
        )
        self.assertEqual(len(session.scanning_node_ids), 0)
        self.assertFalse(session.active)

    def test_session_stop_includes_paused(self):
        """Session stop also stops PAUSED nodes."""
        session = ScanSession()
        session.start_session()
        session.connect_node("PHONE-01")
        session.start_node_scan("PHONE-01")
        session.pause_node_scan("PHONE-01")

        session.stop_session()

        self.assertEqual(
            session.get_node_state("PHONE-01"), NodeScanState.IDLE
        )


class TestDisconnection(unittest.TestCase):
    """Tests for node disconnect behavior."""

    def test_disconnected_removed_from_scanning(self):
        """Test 9: Disconnected node is removed from scanning_node_ids."""
        session = ScanSession()
        session.start_session()
        session.connect_node("PHONE-01")
        session.start_node_scan("PHONE-01")
        self.assertIn("PHONE-01", session.scanning_node_ids)

        session.disconnect_node("PHONE-01")

        self.assertNotIn("PHONE-01", session.scanning_node_ids)
        self.assertNotIn("PHONE-01", session.connected_node_ids)
        self.assertEqual(
            session.get_node_state("PHONE-01"), NodeScanState.DISCONNECTED
        )

    def test_reconnected_node_starts_idle(self):
        """Test 10: Reconnected node starts IDLE."""
        session = ScanSession()
        session.start_session()
        session.connect_node("PHONE-01")
        session.start_node_scan("PHONE-01")

        # Disconnect
        session.disconnect_node("PHONE-01")

        # Reconnect — must be IDLE, not SCANNING
        session.connect_node("PHONE-01")
        self.assertEqual(
            session.get_node_state("PHONE-01"), NodeScanState.IDLE
        )
        self.assertIn("PHONE-01", session.connected_node_ids)
        self.assertNotIn("PHONE-01", session.scanning_node_ids)


class TestStateSerialization(unittest.TestCase):
    """Tests for to_state() JSON serialization."""

    def test_state_serialization(self):
        """Test 11: State serialization works correctly."""
        session = ScanSession(session_id="SCAN-001")
        session.start_session()
        session.connect_node("PHONE-01")
        session.connect_node("PHONE-02")
        session.connect_node("PHONE-03")
        session.start_node_scan("PHONE-01")
        session.start_node_scan("PHONE-02")

        state = session.to_state()

        self.assertEqual(state["session_id"], "SCAN-001")
        self.assertTrue(state["active"])
        self.assertIsNotNone(state["started_at"])
        self.assertIsNone(state["stopped_at"])
        self.assertEqual(len(state["connected_node_ids"]), 3)
        self.assertEqual(len(state["scanning_node_ids"]), 2)
        self.assertIn("PHONE-01", state["scanning_node_ids"])
        self.assertIn("PHONE-02", state["scanning_node_ids"])
        self.assertEqual(state["node_states"]["PHONE-01"], "SCANNING")
        self.assertEqual(state["node_states"]["PHONE-02"], "SCANNING")
        self.assertEqual(state["node_states"]["PHONE-03"], "IDLE")

    def test_state_excludes_disconnected(self):
        """Disconnected nodes are excluded from state output."""
        session = ScanSession()
        session.connect_node("PHONE-01")
        session.connect_node("PHONE-02")
        session.disconnect_node("PHONE-02")

        state = session.to_state()
        self.assertNotIn("PHONE-02", state["node_states"])
        self.assertIn("PHONE-01", state["node_states"])

    def test_stopped_state_has_stopped_at(self):
        """Stopped session has stopped_at timestamp."""
        session = ScanSession()
        session.start_session()
        session.stop_session()

        state = session.to_state()
        self.assertFalse(state["active"])
        self.assertIsNotNone(state["stopped_at"])


class TestExistingModelsUnaffected(unittest.TestCase):
    """Test 12: Verify existing mobile-node models still import and work."""

    def test_mobile_node_import(self):
        """Existing MobileNode can still be imported and created."""
        from mobile_node import MobileNode, MobileNodeRegistry

        node = MobileNode(node_id="TEST-01")
        self.assertEqual(node.node_id, "TEST-01")
        self.assertEqual(node.status, "SCANNING")
        self.assertEqual(node.mode, "mapping")

        registry = MobileNodeRegistry()
        n = registry.register("TEST-02")
        self.assertEqual(n.node_id, "TEST-02")
        self.assertEqual(registry.count(), 1)

    def test_mapping_models_import(self):
        """Step 1 mapping models can still be imported and created."""
        from mapping_models import (
            Pose, SemanticObservation, Landmark,
            LocalMapState, FloorState, WorldModelState,
        )

        p = Pose(x=1.0, y=2.0, z=3.0)
        self.assertEqual(p.x, 1.0)

        obs = SemanticObservation(node_id="N1", floor_id="F1", class_name="wall")
        self.assertEqual(obs.node_id, "N1")


if __name__ == "__main__":
    unittest.main()

# ══════════════════════════════════════════════════════════════════
#  🧪 UNIT TESTS — TIDDA Floor Manager
#  Validates floor_manager.py multi-floor management.
#  Run:  python3 -m unittest test_floor_manager -v
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import unittest

from floor_manager import FloorManager
from mapping_models import FloorState, FloorStatus


class TestFloorCreation(unittest.TestCase):
    """Tests for floor creation and listing."""

    def test_create_floor_1(self):
        """Test 1: Create Floor 1."""
        mgr = FloorManager()
        floor = mgr.create_floor("FLOOR-1")
        self.assertEqual(floor.floor_id, "FLOOR-1")
        self.assertEqual(floor.status, FloorStatus.NOT_STARTED)

    def test_create_floor_2(self):
        """Test 2: Create Floor 2."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        floor = mgr.create_floor("FLOOR-2")
        self.assertEqual(floor.floor_id, "FLOOR-2")
        self.assertEqual(len(mgr.list_floors()), 2)

    def test_create_floor_3(self):
        """Test 3: Create Floor 3."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.create_floor("FLOOR-2")
        floor = mgr.create_floor("FLOOR-3")
        self.assertEqual(floor.floor_id, "FLOOR-3")
        self.assertEqual(len(mgr.list_floors()), 3)

    def test_create_duplicate_returns_existing(self):
        """Creating a floor that already exists returns the existing one."""
        mgr = FloorManager()
        f1 = mgr.create_floor("FLOOR-1")
        f1_dup = mgr.create_floor("FLOOR-1")
        self.assertIs(f1, f1_dup)
        self.assertEqual(len(mgr.list_floors()), 1)


class TestMultiFloorActivation(unittest.TestCase):
    """Tests for simultaneous multi-floor activation."""

    def test_activate_all_three(self):
        """Test 4: Activate all three floors simultaneously."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.create_floor("FLOOR-2")
        mgr.create_floor("FLOOR-3")

        self.assertTrue(mgr.activate_floor("FLOOR-1"))
        self.assertTrue(mgr.activate_floor("FLOOR-2"))
        self.assertTrue(mgr.activate_floor("FLOOR-3"))

        f1 = mgr.get_floor("FLOOR-1")
        f2 = mgr.get_floor("FLOOR-2")
        f3 = mgr.get_floor("FLOOR-3")
        self.assertEqual(f1.status, FloorStatus.ACTIVE)
        self.assertEqual(f2.status, FloorStatus.ACTIVE)
        self.assertEqual(f3.status, FloorStatus.ACTIVE)

    def test_get_active_floors_returns_all_three(self):
        """Test 5: get_active_floors() returns all three active floors."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.create_floor("FLOOR-2")
        mgr.create_floor("FLOOR-3")
        mgr.activate_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-2")
        mgr.activate_floor("FLOOR-3")

        active = mgr.get_active_floors()
        self.assertEqual(len(active), 3)
        active_ids = {f.floor_id for f in active}
        self.assertEqual(active_ids, {"FLOOR-1", "FLOOR-2", "FLOOR-3"})

    def test_activate_nonexistent_floor_returns_false(self):
        """Activating a nonexistent floor returns False."""
        mgr = FloorManager()
        self.assertFalse(mgr.activate_floor("DOES-NOT-EXIST"))


class TestNodeAssignment(unittest.TestCase):
    """Tests for node-to-floor assignment."""

    def test_assign_multiple_nodes_to_floor_1(self):
        """Test 6: Assign multiple nodes to Floor 1."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-1")

        mgr.assign_node("PHONE-01", "FLOOR-1")
        mgr.assign_node("PHONE-02", "FLOOR-1")

        nodes = mgr.get_nodes("FLOOR-1")
        self.assertEqual(len(nodes), 2)
        self.assertIn("PHONE-01", nodes)
        self.assertIn("PHONE-02", nodes)

    def test_assign_nodes_to_different_floors(self):
        """Test 7: Assign nodes to different floors."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.create_floor("FLOOR-2")
        mgr.activate_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-2")

        mgr.assign_node("PHONE-01", "FLOOR-1")
        mgr.assign_node("PHONE-02", "FLOOR-1")
        mgr.assign_node("PHONE-03", "FLOOR-2")

        self.assertEqual(mgr.get_node_floor("PHONE-01"), "FLOOR-1")
        self.assertEqual(mgr.get_node_floor("PHONE-02"), "FLOOR-1")
        self.assertEqual(mgr.get_node_floor("PHONE-03"), "FLOOR-2")

    def test_reassign_node_removes_from_old_floor(self):
        """Test 8 & 9: Reassign node from Floor 1 to Floor 2; verify removed from old."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.create_floor("FLOOR-2")
        mgr.activate_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-2")

        mgr.assign_node("PHONE-03", "FLOOR-1")
        self.assertEqual(mgr.get_node_floor("PHONE-03"), "FLOOR-1")
        self.assertIn("PHONE-03", mgr.get_nodes("FLOOR-1"))

        # Reassign PHONE-03 to FLOOR-2
        mgr.assign_node("PHONE-03", "FLOOR-2")

        # Verify removed from old floor (Test 9)
        self.assertNotIn("PHONE-03", mgr.get_nodes("FLOOR-1"))
        self.assertIn("PHONE-03", mgr.get_nodes("FLOOR-2"))
        self.assertEqual(mgr.get_node_floor("PHONE-03"), "FLOOR-2")

    def test_no_duplicate_node_assignment(self):
        """Test 16: Verify no duplicate node assignment."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-1")

        # Assign same node twice to same floor
        mgr.assign_node("PHONE-01", "FLOOR-1")
        mgr.assign_node("PHONE-01", "FLOOR-1")

        nodes = mgr.get_nodes("FLOOR-1")
        self.assertEqual(nodes.count("PHONE-01"), 1)

    def test_assign_to_nonexistent_floor_returns_false(self):
        """Assigning to a nonexistent floor returns False."""
        mgr = FloorManager()
        self.assertFalse(mgr.assign_node("PHONE-01", "NO-FLOOR"))


class TestNodeDisconnect(unittest.TestCase):
    """Tests for node disconnect behavior."""

    def test_disconnect_node(self):
        """Test 10: Disconnect a node."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-1")
        mgr.assign_node("PHONE-01", "FLOOR-1")
        mgr.assign_node("PHONE-02", "FLOOR-1")

        result = mgr.remove_node("PHONE-02")
        self.assertTrue(result)

    def test_disconnected_node_removed(self):
        """Test 11: Verify disconnected node is removed from floor."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-1")
        mgr.assign_node("PHONE-01", "FLOOR-1")
        mgr.assign_node("PHONE-02", "FLOOR-1")

        mgr.remove_node("PHONE-02")

        self.assertNotIn("PHONE-02", mgr.get_nodes("FLOOR-1"))
        self.assertIsNone(mgr.get_node_floor("PHONE-02"))

    def test_floor_remains_after_disconnect(self):
        """Test 12: Verify floor remains after node disconnect."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-1")
        mgr.assign_node("PHONE-01", "FLOOR-1")
        mgr.assign_node("PHONE-02", "FLOOR-1")

        mgr.remove_node("PHONE-02")

        # Floor-1 must still exist and remain ACTIVE
        floor = mgr.get_floor("FLOOR-1")
        self.assertIsNotNone(floor)
        self.assertEqual(floor.status, FloorStatus.ACTIVE)
        self.assertIn("PHONE-01", floor.node_ids)

    def test_remove_node_with_explicit_floor(self):
        """Remove node specifying the floor explicitly."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.assign_node("PHONE-01", "FLOOR-1")

        result = mgr.remove_node("PHONE-01", floor_id="FLOOR-1")
        self.assertTrue(result)
        self.assertNotIn("PHONE-01", mgr.get_nodes("FLOOR-1"))

    def test_remove_unknown_node_returns_false(self):
        """Removing an unknown node returns False."""
        mgr = FloorManager()
        self.assertFalse(mgr.remove_node("PHONE-99"))


class TestFloorStatusTransitions(unittest.TestCase):
    """Tests for pause/complete while others remain ACTIVE."""

    def test_pause_one_others_active(self):
        """Test 13: Pause one floor while others remain ACTIVE."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.create_floor("FLOOR-2")
        mgr.create_floor("FLOOR-3")
        mgr.activate_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-2")
        mgr.activate_floor("FLOOR-3")

        mgr.pause_floor("FLOOR-2")

        self.assertEqual(mgr.get_floor("FLOOR-1").status, FloorStatus.ACTIVE)
        self.assertEqual(mgr.get_floor("FLOOR-2").status, FloorStatus.PAUSED)
        self.assertEqual(mgr.get_floor("FLOOR-3").status, FloorStatus.ACTIVE)

        # Only 2 active now
        self.assertEqual(len(mgr.get_active_floors()), 2)

    def test_complete_one_others_active(self):
        """Test 14: Complete one floor while others remain ACTIVE."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.create_floor("FLOOR-2")
        mgr.create_floor("FLOOR-3")
        mgr.activate_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-2")
        mgr.activate_floor("FLOOR-3")

        mgr.complete_floor("FLOOR-1")

        self.assertEqual(mgr.get_floor("FLOOR-1").status, FloorStatus.COMPLETE)
        self.assertEqual(mgr.get_floor("FLOOR-2").status, FloorStatus.ACTIVE)
        self.assertEqual(mgr.get_floor("FLOOR-3").status, FloorStatus.ACTIVE)

        # Only 2 active now
        self.assertEqual(len(mgr.get_active_floors()), 2)


class TestFloorSerialization(unittest.TestCase):
    """Tests for state serialization."""

    def test_serialize_floor_state(self):
        """Test 15: Serialize floor state."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.create_floor("FLOOR-2")
        mgr.activate_floor("FLOOR-1")
        mgr.activate_floor("FLOOR-2")
        mgr.assign_node("PHONE-01", "FLOOR-1")
        mgr.assign_node("PHONE-02", "FLOOR-1")
        mgr.assign_node("PHONE-03", "FLOOR-2")

        state = mgr.to_state()

        # Check structure
        self.assertIn("floors", state)
        self.assertIn("node_assignments", state)

        # Check floor data
        f1 = state["floors"]["FLOOR-1"]
        self.assertEqual(f1["status"], "ACTIVE")
        self.assertEqual(len(f1["node_ids"]), 2)
        self.assertIn("PHONE-01", f1["node_ids"])
        self.assertIn("PHONE-02", f1["node_ids"])

        f2 = state["floors"]["FLOOR-2"]
        self.assertEqual(f2["status"], "ACTIVE")
        self.assertEqual(f2["node_ids"], ["PHONE-03"])

        # Check node assignments
        self.assertEqual(state["node_assignments"]["PHONE-01"], "FLOOR-1")
        self.assertEqual(state["node_assignments"]["PHONE-03"], "FLOOR-2")

    def test_serialization_has_timestamps(self):
        """Serialized state includes created_at and updated_at."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        state = mgr.to_state()
        f1 = state["floors"]["FLOOR-1"]
        self.assertIn("created_at", f1)
        self.assertIn("updated_at", f1)
        self.assertIsInstance(f1["created_at"], float)


class TestFloorRemoval(unittest.TestCase):
    """Tests for floor removal."""

    def test_remove_floor_unassigns_nodes(self):
        """Removing a floor unassigns all its nodes."""
        mgr = FloorManager()
        mgr.create_floor("FLOOR-1")
        mgr.assign_node("PHONE-01", "FLOOR-1")
        mgr.assign_node("PHONE-02", "FLOOR-1")

        result = mgr.remove_floor("FLOOR-1")
        self.assertTrue(result)
        self.assertIsNone(mgr.get_floor("FLOOR-1"))
        self.assertIsNone(mgr.get_node_floor("PHONE-01"))
        self.assertIsNone(mgr.get_node_floor("PHONE-02"))

    def test_remove_nonexistent_floor_returns_false(self):
        """Removing a nonexistent floor returns False."""
        mgr = FloorManager()
        self.assertFalse(mgr.remove_floor("NO-FLOOR"))


class TestExistingModels(unittest.TestCase):
    """Verify Step 1 and Step 2 models are unaffected."""

    def test_step1_floorstate_still_works(self):
        """Step 1 FloorState can still be created with original fields."""
        floor = FloorState(
            floor_id="F1",
            status=FloorStatus.ACTIVE,
            node_ids=["PHONE-01", "PHONE-02"],
        )
        self.assertEqual(floor.floor_id, "F1")
        self.assertEqual(floor.status, FloorStatus.ACTIVE)
        self.assertEqual(len(floor.node_ids), 2)
        # New fields have defaults
        self.assertEqual(floor.observation_count, 0)
        self.assertEqual(floor.landmark_count, 0)
        self.assertIsInstance(floor.created_at, float)

    def test_step2_scan_session_still_works(self):
        """Step 2 ScanSession can still be created and used."""
        from scan_session import ScanSession, NodeScanState
        session = ScanSession()
        session.connect_node("PHONE-01")
        self.assertEqual(
            session.get_node_state("PHONE-01"),
            NodeScanState.IDLE,
        )


if __name__ == "__main__":
    unittest.main()

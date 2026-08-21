# ══════════════════════════════════════════════════════════════════
#  🧪 UNIT TESTS — TIDDA Local Map Engine
#  Validates local_map.py per-node map operations and isolation.
#  Run:  python3 -m unittest test_local_map -v
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import unittest

from local_map import LocalMapManager, local_map_to_dict
from floor_manager import FloorManager
from mapping_models import (
    Landmark,
    LocalMapState,
    Pose,
    SemanticObservation,
)


class TestLocalMapLifecycle(unittest.TestCase):
    """Tests for local map creation, retrieval, listing, and removal."""

    def test_create_local_map(self):
        """Test 1: Create local map."""
        mgr = LocalMapManager()
        m = mgr.create_map("PHONE-01", "FLOOR-1")
        self.assertIsNotNone(m)
        self.assertEqual(m.node_id, "PHONE-01")
        self.assertEqual(m.floor_id, "FLOOR-1")

    def test_retrieve_local_map(self):
        """Test 2: Retrieve local map."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")
        m = mgr.get_map("PHONE-01")
        self.assertIsNotNone(m)
        self.assertEqual(m.node_id, "PHONE-01")
        self.assertEqual(m.floor_id, "FLOOR-1")

        # Non-existent node returns None
        self.assertIsNone(mgr.get_map("PHONE-UNKNOWN"))

    def test_remove_local_map(self):
        """Test 3: Remove local map."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")
        self.assertTrue(mgr.remove_map("PHONE-01"))
        self.assertIsNone(mgr.get_map("PHONE-01"))

        # Removing non-existent map returns False
        self.assertFalse(mgr.remove_map("PHONE-01"))

    def test_create_maps_for_multiple_phones(self):
        """Test 4: Create maps for multiple phones."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")
        mgr.create_map("PHONE-02", "FLOOR-1")
        mgr.create_map("PHONE-03", "FLOOR-2")

        maps = mgr.list_maps()
        self.assertEqual(len(maps), 3)

    def test_multiple_phones_same_floor(self):
        """Test 5: Multiple phones can have maps on the same floor."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")
        mgr.create_map("PHONE-02", "FLOOR-1")

        m1 = mgr.get_map("PHONE-01")
        m2 = mgr.get_map("PHONE-02")
        self.assertEqual(m1.floor_id, "FLOOR-1")
        self.assertEqual(m2.floor_id, "FLOOR-1")
        self.assertNotEqual(m1.node_id, m2.node_id)

    def test_phones_different_floors(self):
        """Test 6: Phones can have maps on different floors."""
        mgr = LocalMapManager()
        mgr.create_floor = mgr.create_map  # alias for readability if needed
        mgr.create_map("PHONE-01", "FLOOR-1")
        mgr.create_map("PHONE-02", "FLOOR-2")

        m1 = mgr.get_map("PHONE-01")
        m2 = mgr.get_map("PHONE-02")
        self.assertEqual(m1.floor_id, "FLOOR-1")
        self.assertEqual(m2.floor_id, "FLOOR-2")


class TestPoseUpdates(unittest.TestCase):
    """Tests for updating trajectory poses."""

    def test_pose_update_adds_trajectory_point(self):
        """Test 7: Pose update adds trajectory point."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")
        p = Pose(x=1.0, y=2.0, z=0.0, yaw=0.1)

        success = mgr.update_pose("PHONE-01", p)
        self.assertTrue(success)

        traj = mgr.get_trajectory("PHONE-01")
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj[0].x, 1.0)
        self.assertEqual(traj[0].y, 2.0)

    def test_multiple_pose_updates_preserve_order(self):
        """Test 8: Multiple pose updates preserve order."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        p1 = Pose(x=0.0, y=0.0, z=0.0)
        p2 = Pose(x=1.0, y=0.5, z=0.0)
        p3 = Pose(x=2.0, y=1.0, z=0.0)

        mgr.update_pose("PHONE-01", p1)
        mgr.update_pose("PHONE-01", p2)
        mgr.update_pose("PHONE-01", p3)

        traj = mgr.get_trajectory("PHONE-01")
        self.assertEqual(len(traj), 3)
        self.assertEqual(traj[0].x, 0.0)
        self.assertEqual(traj[1].x, 1.0)
        self.assertEqual(traj[2].x, 2.0)


class TestObservationsAndLandmarks(unittest.TestCase):
    """Tests for adding observations and landmarks with validation."""

    def test_observation_stored(self):
        """Test 9: Observation is stored."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        obs = SemanticObservation(
            node_id="PHONE-01",
            floor_id="FLOOR-1",
            class_name="doorway",
            confidence=0.9,
        )
        success = mgr.add_observation("PHONE-01", obs)
        self.assertTrue(success)

        obs_list = mgr.get_observations("PHONE-01")
        self.assertEqual(len(obs_list), 1)
        self.assertEqual(obs_list[0].class_name, "doorway")

    def test_observation_count_updates(self):
        """Test 10: Observation count updates."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        obs1 = SemanticObservation(class_name="doorway", confidence=0.9)
        obs2 = SemanticObservation(class_name="wall", confidence=0.8)

        mgr.add_observation("PHONE-01", obs1)
        mgr.add_observation("PHONE-01", obs2)

        self.assertEqual(len(mgr.get_observations("PHONE-01")), 2)

    def test_landmark_stored(self):
        """Test 11: Landmark is stored."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        lm = Landmark(
            node_id="PHONE-01",
            position=(2.5, 3.0, 0.0),
            confidence=0.85,
        )
        success = mgr.add_landmark("PHONE-01", lm)
        self.assertTrue(success)

        landmarks = mgr.get_landmarks("PHONE-01")
        self.assertEqual(len(landmarks), 1)
        self.assertEqual(landmarks[0].position, (2.5, 3.0, 0.0))

    def test_landmark_count_updates(self):
        """Test 12: Landmark count updates."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        lm1 = Landmark(position=(1.0, 1.0, 0.0))
        lm2 = Landmark(position=(2.0, 2.0, 0.0))

        mgr.add_landmark("PHONE-01", lm1)
        mgr.add_landmark("PHONE-01", lm2)

        self.assertEqual(len(mgr.get_landmarks("PHONE-01")), 2)

    def test_invalid_node_id_observation_rejected(self):
        """Test 13: Invalid node_id observation is rejected."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        obs = SemanticObservation(
            node_id="PHONE-02",  # Belongs to PHONE-02!
            floor_id="FLOOR-1",
            class_name="doorway",
        )
        success = mgr.add_observation("PHONE-01", obs)
        self.assertFalse(success)
        self.assertEqual(len(mgr.get_observations("PHONE-01")), 0)

    def test_incompatible_floor_observation_rejected(self):
        """Test 14: Incompatible floor observation is rejected."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        obs = SemanticObservation(
            node_id="PHONE-01",
            floor_id="FLOOR-2",  # Incompatible with map's FLOOR-1!
            class_name="staircase",
        )
        success = mgr.add_observation("PHONE-01", obs)
        self.assertFalse(success)
        self.assertEqual(len(mgr.get_observations("PHONE-01")), 0)


class TestSerializationAndMetrics(unittest.TestCase):
    """Tests for serialization, coverage, confidence, and map isolation."""

    def test_serialization_works(self):
        """Test 15: Serialization works."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")
        mgr.update_pose("PHONE-01", Pose(x=1.0, y=2.0, z=0.0))
        mgr.add_observation("PHONE-01", SemanticObservation(class_name="corridor"))
        mgr.add_landmark("PHONE-01", Landmark(position=(1.0, 2.0, 0.0)))
        mgr.set_coverage("PHONE-01", 0.45)
        mgr.set_confidence("PHONE-01", 0.88)

        state_dict = mgr.get_state("PHONE-01")
        self.assertIsNotNone(state_dict)
        self.assertEqual(state_dict["node_id"], "PHONE-01")
        self.assertEqual(state_dict["floor_id"], "FLOOR-1")
        self.assertEqual(len(state_dict["trajectory"]), 1)
        self.assertEqual(len(state_dict["observations"]), 1)
        self.assertEqual(len(state_dict["landmarks"]), 1)
        self.assertEqual(state_dict["coverage"], 0.45)
        self.assertEqual(state_dict["mapping_confidence"], 0.88)
        self.assertIn("last_update", state_dict)

    def test_coverage_valid_range(self):
        """Test 16: Coverage stays within valid range [0.0, 1.0]."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        mgr.set_coverage("PHONE-01", 1.5)  # Over 1.0 -> should clamp to 1.0
        self.assertEqual(mgr.get_map("PHONE-01").coverage, 1.0)

        mgr.set_coverage("PHONE-01", -0.5)  # Under 0.0 -> should clamp to 0.0
        self.assertEqual(mgr.get_map("PHONE-01").coverage, 0.0)

        mgr.set_coverage("PHONE-01", 0.75)
        self.assertEqual(mgr.get_map("PHONE-01").coverage, 0.75)

    def test_confidence_valid_range(self):
        """Test 17: Mapping confidence stays within valid range [0.0, 1.0]."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")

        mgr.set_confidence("PHONE-01", 2.0)  # Should clamp to 1.0
        self.assertEqual(mgr.get_map("PHONE-01").mapping_confidence, 1.0)

        mgr.set_confidence("PHONE-01", -1.0)  # Should clamp to 0.0
        self.assertEqual(mgr.get_map("PHONE-01").mapping_confidence, 0.0)

        mgr.set_confidence("PHONE-01", 0.92)
        self.assertEqual(mgr.get_map("PHONE-01").mapping_confidence, 0.92)

    def test_independent_maps_do_not_contaminate(self):
        """Test 18: Multiple independent local maps do not contaminate each other."""
        mgr = LocalMapManager()
        mgr.create_map("PHONE-01", "FLOOR-1")
        mgr.create_map("PHONE-02", "FLOOR-1")

        mgr.update_pose("PHONE-01", Pose(x=10.0, y=10.0, z=0.0))
        mgr.add_observation("PHONE-01", SemanticObservation(class_name="room"))

        # Check PHONE-02 map remains completely untouched
        self.assertEqual(len(mgr.get_trajectory("PHONE-02")), 0)
        self.assertEqual(len(mgr.get_observations("PHONE-02")), 0)
        self.assertEqual(len(mgr.get_landmarks("PHONE-02")), 0)


class TestFloorManagerIntegration(unittest.TestCase):
    """Test optional integration with FloorManager."""

    def test_floor_manager_integration(self):
        fm = FloorManager()
        mgr = LocalMapManager(floor_manager=fm)

        mgr.create_map("PHONE-01", "FLOOR-1")

        # Verify FloorManager recorded the floor and node assignment
        self.assertIsNotNone(fm.get_floor("FLOOR-1"))
        self.assertEqual(fm.get_node_floor("PHONE-01"), "FLOOR-1")

        # Removing local map unassigns node in FloorManager
        mgr.remove_map("PHONE-01")
        self.assertIsNone(fm.get_node_floor("PHONE-01"))


if __name__ == "__main__":
    unittest.main()

# ══════════════════════════════════════════════════════════════════
#  🧪 UNIT TESTS — TIDDA Mapping Models
#  Validates the data structures in mapping_models.py.
#  Run:  python -m unittest test_mapping_models -v
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
import unittest

from mapping_models import (
    FloorState,
    FloorStatus,
    Landmark,
    LocalMapState,
    Pose,
    SemanticObservation,
    WorldModelState,
)


class TestPose(unittest.TestCase):
    """Validates Pose creation and optional GPS fields."""

    def test_pose_creation(self):
        """Test 1: Pose can be created with default values."""
        p = Pose(x=1.0, y=2.0, z=3.0, yaw=0.5, pitch=0.1, roll=0.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.z, 3.0)
        self.assertEqual(p.yaw, 0.5)
        self.assertEqual(p.pitch, 0.1)
        self.assertEqual(p.roll, 0.0)
        self.assertIsInstance(p.timestamp, float)

    def test_gps_fields_absent(self):
        """Test 2: GPS fields default to None (indoor operation)."""
        p = Pose(x=0.0, y=0.0, z=0.0)
        self.assertIsNone(p.latitude)
        self.assertIsNone(p.longitude)
        self.assertIsNone(p.altitude)

    def test_gps_fields_present(self):
        """GPS fields can be explicitly set when available."""
        p = Pose(
            x=1.0, y=2.0, z=0.0,
            latitude=28.6139, longitude=77.2090, altitude=220.0,
        )
        self.assertAlmostEqual(p.latitude, 28.6139)
        self.assertAlmostEqual(p.longitude, 77.2090)
        self.assertAlmostEqual(p.altitude, 220.0)


class TestSemanticObservation(unittest.TestCase):
    """Validates SemanticObservation preserves key fields."""

    def test_preserves_node_id(self):
        """Test 3: SemanticObservation preserves node_id."""
        obs = SemanticObservation(
            node_id="PHONE-42",
            floor_id="F1",
            class_name="doorway",
            confidence=0.92,
        )
        self.assertEqual(obs.node_id, "PHONE-42")

    def test_preserves_floor_id(self):
        """Test 4: SemanticObservation preserves floor_id."""
        obs = SemanticObservation(
            node_id="PHONE-01",
            floor_id="FLOOR-3",
            class_name="corridor",
            confidence=0.85,
        )
        self.assertEqual(obs.floor_id, "FLOOR-3")

    def test_observation_id_generated(self):
        """observation_id is auto-generated (UUID)."""
        obs = SemanticObservation(node_id="N1", floor_id="F1", class_name="wall")
        self.assertTrue(len(obs.observation_id) > 0)

    def test_optional_fields_default_none(self):
        """Optional fields default to None."""
        obs = SemanticObservation(node_id="N1", floor_id="F1", class_name="wall")
        self.assertIsNone(obs.bounding_box)
        self.assertIsNone(obs.pose)
        self.assertIsNone(obs.position)
        self.assertIsNone(obs.spatial_extent)
        self.assertIsNone(obs.frame_id)


class TestLandmark(unittest.TestCase):
    """Validates Landmark creation."""

    def test_landmark_creation(self):
        """Test 5: Landmark can be created."""
        lm = Landmark(
            node_id="PHONE-01",
            position=(5.0, 10.0, 0.0),
            confidence=0.75,
            observation_count=3,
        )
        self.assertEqual(lm.node_id, "PHONE-01")
        self.assertEqual(lm.position, (5.0, 10.0, 0.0))
        self.assertEqual(lm.confidence, 0.75)
        self.assertEqual(lm.observation_count, 3)
        self.assertTrue(len(lm.landmark_id) > 0)


class TestLocalMapState(unittest.TestCase):
    """Validates LocalMapState with observations and trajectory."""

    def test_multiple_observations(self):
        """Test 6: LocalMapState can contain multiple observations."""
        obs1 = SemanticObservation(
            node_id="N1", floor_id="F1", class_name="wall", confidence=0.9,
        )
        obs2 = SemanticObservation(
            node_id="N1", floor_id="F1", class_name="doorway", confidence=0.8,
        )
        obs3 = SemanticObservation(
            node_id="N1", floor_id="F1", class_name="corridor", confidence=0.7,
        )
        local_map = LocalMapState(
            node_id="N1",
            floor_id="F1",
            observations=[obs1, obs2, obs3],
        )
        self.assertEqual(len(local_map.observations), 3)
        self.assertEqual(local_map.observations[0].class_name, "wall")
        self.assertEqual(local_map.observations[1].class_name, "doorway")
        self.assertEqual(local_map.observations[2].class_name, "corridor")

    def test_trajectory_points(self):
        """Test 7: LocalMapState can contain trajectory points."""
        poses = [
            Pose(x=0.0, y=0.0, z=0.0),
            Pose(x=1.0, y=0.0, z=0.0),
            Pose(x=2.0, y=1.0, z=0.0),
            Pose(x=3.0, y=2.0, z=0.0),
        ]
        local_map = LocalMapState(
            node_id="PHONE-01",
            floor_id="F1",
            trajectory=poses,
        )
        self.assertEqual(len(local_map.trajectory), 4)
        self.assertEqual(local_map.trajectory[0].x, 0.0)
        self.assertEqual(local_map.trajectory[3].x, 3.0)

    def test_preserves_node_ownership(self):
        """LocalMapState preserves source node ownership."""
        lm = LocalMapState(node_id="PHONE-07", floor_id="F2")
        self.assertEqual(lm.node_id, "PHONE-07")


class TestFloorState(unittest.TestCase):
    """Validates FloorState with multiple nodes and concurrent ACTIVE status."""

    def test_multiple_nodes(self):
        """Test 8: FloorState can contain multiple nodes."""
        floor = FloorState(
            floor_id="F1",
            status=FloorStatus.ACTIVE,
            node_ids=["PHONE-01", "PHONE-02", "PHONE-03"],
        )
        self.assertEqual(len(floor.node_ids), 3)
        self.assertIn("PHONE-02", floor.node_ids)

    def test_concurrent_active_floors(self):
        """Test 9: Floor 1 and Floor 2 can BOTH be ACTIVE simultaneously."""
        floor1 = FloorState(
            floor_id="F1",
            status=FloorStatus.ACTIVE,
            node_ids=["PHONE-01"],
        )
        floor2 = FloorState(
            floor_id="F2",
            status=FloorStatus.ACTIVE,
            node_ids=["PHONE-02"],
        )
        # Both floors are independently ACTIVE — no singleton constraint
        self.assertEqual(floor1.status, FloorStatus.ACTIVE)
        self.assertEqual(floor2.status, FloorStatus.ACTIVE)
        self.assertNotEqual(floor1.floor_id, floor2.floor_id)

    def test_floor_statuses(self):
        """All four FloorStatus values can be assigned."""
        for status in FloorStatus:
            floor = FloorState(floor_id="test", status=status)
            self.assertEqual(floor.status, status)


class TestWorldModelState(unittest.TestCase):
    """Validates WorldModelState with multi-floor support."""

    def test_three_floors(self):
        """Test 10: WorldModelState can contain at least three floors."""
        f1 = FloorState(floor_id="F1", status=FloorStatus.ACTIVE)
        f2 = FloorState(floor_id="F2", status=FloorStatus.ACTIVE)
        f3 = FloorState(floor_id="F3", status=FloorStatus.NOT_STARTED)

        world = WorldModelState(
            building_id="BLDG-001",
            floors={"F1": f1, "F2": f2, "F3": f3},
            nodes=["PHONE-01", "PHONE-02", "PHONE-03"],
        )
        self.assertEqual(len(world.floors), 3)
        self.assertIn("F1", world.floors)
        self.assertIn("F2", world.floors)
        self.assertIn("F3", world.floors)
        self.assertEqual(world.building_id, "BLDG-001")

    def test_source_attribution(self):
        """WorldModelState preserves source attribution via node lists."""
        local_map = LocalMapState(node_id="PHONE-01", floor_id="F1")
        floor = FloorState(
            floor_id="F1",
            status=FloorStatus.ACTIVE,
            node_ids=["PHONE-01"],
            local_maps=[local_map],
        )
        world = WorldModelState(
            building_id="BLDG-X",
            floors={"F1": floor},
            nodes=["PHONE-01"],
        )
        # Can trace from world → floor → local_map → node_id
        retrieved_map = world.floors["F1"].local_maps[0]
        self.assertEqual(retrieved_map.node_id, "PHONE-01")

    def test_timestamps_set(self):
        """created_at and updated_at are set automatically."""
        world = WorldModelState(building_id="BLDG-T")
        self.assertIsInstance(world.created_at, float)
        self.assertIsInstance(world.updated_at, float)
        self.assertGreater(world.created_at, 0)


if __name__ == "__main__":
    unittest.main()

# ══════════════════════════════════════════════════════════════════
#  🧪 UNIT TESTS — TIDDA Shared World Model
#  Validates world_model.py multi-floor building model, ingestion,
#  provenance preservation, fusion result processing, and serialization.
#  Run:  python3 -m unittest test_world_model -v
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import unittest

from world_model import WorldModel
from mapping_models import (
    FloorState,
    FloorStatus,
    Landmark,
    LocalMapState,
    Pose,
    SemanticObservation,
)
from fusion import (
    AssociationDecision,
    FusionEngine,
    FusionResult,
    MapAssociation,
)


class TestWorldModelInitializationAndFloors(unittest.TestCase):
    """Tests 1–3: Initialization, multi-floor existence, and multi-floor activation."""

    def test_world_model_initialization(self):
        """Test 1: WorldModel initializes."""
        wm = WorldModel(building_id="BLDG-TIDDA")
        self.assertEqual(wm.building_id, "BLDG-TIDDA")
        self.assertEqual(len(wm.floors), 0)
        self.assertGreater(wm.created_at, 0)
        self.assertGreater(wm.updated_at, 0)

    def test_multiple_floors_exist(self):
        """Test 2: Multiple floors can exist."""
        wm = WorldModel()
        f1 = wm.get_or_create_floor("FLOOR-1")
        f2 = wm.get_or_create_floor("FLOOR-2")
        f3 = wm.get_or_create_floor("FLOOR-3")

        self.assertEqual(len(wm.list_floors()), 3)
        self.assertEqual(f1.floor_id, "FLOOR-1")
        self.assertEqual(f2.floor_id, "FLOOR-2")
        self.assertEqual(f3.floor_id, "FLOOR-3")

    def test_multiple_floors_active(self):
        """Test 3: Multiple floors can be active simultaneously."""
        wm = WorldModel()
        wm.activate_floor("FLOOR-1")
        wm.activate_floor("FLOOR-2")

        active = wm.get_active_floors()
        self.assertEqual(len(active), 2)
        active_ids = {f.floor_id for f in active}
        self.assertEqual(active_ids, {"FLOOR-1", "FLOOR-2"})


class TestObservationAndLandmarkIngestion(unittest.TestCase):
    """Tests 4–6, 11, 14, 15: Observation/landmark routing, provenance, multi-node/floor ingestion."""

    def test_observation_added_to_correct_floor(self):
        """Test 4: Observation is added to correct floor."""
        wm = WorldModel()
        obs = SemanticObservation(
            node_id="PHONE-01",
            floor_id="FLOOR-1",
            class_name="doorway",
            confidence=0.92,
        )
        success = wm.add_observation(obs)
        self.assertTrue(success)

        floor_1_obs = wm.get_observations("FLOOR-1")
        self.assertEqual(len(floor_1_obs), 1)
        self.assertEqual(floor_1_obs[0].class_name, "doorway")
        self.assertEqual(len(wm.get_observations("FLOOR-2")), 0)

    def test_landmark_added_to_correct_floor(self):
        """Test 5: Landmark is added to correct floor."""
        wm = WorldModel()
        lm = Landmark(
            node_id="PHONE-01",
            position=(2.0, 3.0, 0.0),
            confidence=0.88,
        )
        success = wm.add_landmark(lm, floor_id="FLOOR-2")
        self.assertTrue(success)

        floor_2_lms = wm.get_landmarks("FLOOR-2")
        self.assertEqual(len(floor_2_lms), 1)
        self.assertEqual(floor_2_lms[0].position, (2.0, 3.0, 0.0))
        self.assertEqual(len(wm.get_landmarks("FLOOR-1")), 0)

    def test_node_source_attribution_preserved(self):
        """Test 6 & 11: Node source attribution and landmark provenance preserved."""
        wm = WorldModel()
        obs = SemanticObservation(node_id="PHONE-42", floor_id="FLOOR-1", class_name="wall")
        lm = Landmark(node_id="PHONE-42", position=(1.0, 1.0, 0.0))

        wm.add_observation(obs)
        wm.add_landmark(lm, floor_id="FLOOR-1")

        retrieved_obs = wm.get_observations("FLOOR-1")[0]
        retrieved_lm = wm.get_landmarks("FLOOR-1")[0]

        self.assertEqual(retrieved_obs.node_id, "PHONE-42")
        self.assertEqual(retrieved_lm.node_id, "PHONE-42")
        self.assertIn("PHONE-42", wm.participating_nodes)

    def test_multiple_nodes_contribute_to_one_floor(self):
        """Test 14: Multiple nodes can contribute to one floor."""
        wm = WorldModel()
        obs1 = SemanticObservation(node_id="PHONE-01", floor_id="FLOOR-1", class_name="chair")
        obs2 = SemanticObservation(node_id="PHONE-02", floor_id="FLOOR-1", class_name="table")

        wm.add_observation(obs1)
        wm.add_observation(obs2)

        floor_nodes = wm.get_floor_nodes("FLOOR-1")
        self.assertIn("PHONE-01", floor_nodes)
        self.assertIn("PHONE-02", floor_nodes)
        self.assertEqual(len(wm.get_observations("FLOOR-1")), 2)

    def test_multiple_floors_receive_observations_simultaneously(self):
        """Test 15: Multiple floors can receive observations simultaneously."""
        wm = WorldModel()
        obs1 = SemanticObservation(node_id="PHONE-01", floor_id="FLOOR-1", class_name="doorway")
        obs2 = SemanticObservation(node_id="PHONE-02", floor_id="FLOOR-2", class_name="elevator")

        wm.add_observation(obs1)
        wm.add_observation(obs2)

        self.assertEqual(len(wm.get_observations("FLOOR-1")), 1)
        self.assertEqual(len(wm.get_observations("FLOOR-2")), 1)
        self.assertEqual(wm.get_observations("FLOOR-1")[0].class_name, "doorway")
        self.assertEqual(wm.get_observations("FLOOR-2")[0].class_name, "elevator")


class TestFusionIngestionAndFloorRules(unittest.TestCase):
    """Tests 7–10, 16: Applying fusion results, MERGE vs KEEP_SEPARATE, cross-floor safety, non-mutation."""

    def test_fusion_merge_result_applied(self):
        """Test 7: Fusion MERGE result is applied."""
        wm = WorldModel()
        assoc = MapAssociation(
            node_a="PHONE-01", node_b="PHONE-02",
            floor_a="FLOOR-1", floor_b="FLOOR-1",
            confidence=0.92, decision=AssociationDecision.MERGE.value,
            evidence={"semantic_similarity": 0.95},
        )
        obs1 = SemanticObservation(node_id="PHONE-01", floor_id="FLOOR-1", class_name="wall")
        obs2 = SemanticObservation(node_id="PHONE-02", floor_id="FLOOR-1", class_name="wall")

        res = FusionResult(
            participating_nodes=["PHONE-01", "PHONE-02"],
            associations=[assoc],
            merge_decisions={"PHONE-01:PHONE-02": "MERGE"},
            confidence=0.92,
            aligned_observations=[obs1, obs2],
        )

        wm.apply_fusion_result(res)

        # Verify floor activated and nodes merged into floor node list
        f1 = wm.get_floor("FLOOR-1")
        self.assertIsNotNone(f1)
        self.assertEqual(f1.status, FloorStatus.ACTIVE)
        self.assertIn("PHONE-01", f1.node_ids)
        self.assertIn("PHONE-02", f1.node_ids)
        self.assertEqual(len(wm.get_observations("FLOOR-1")), 2)

    def test_keep_separate_result_not_incorrectly_merged(self):
        """Test 8: KEEP_SEPARATE result does not incorrectly merge observations."""
        wm = WorldModel()
        assoc = MapAssociation(
            node_a="PHONE-01", node_b="PHONE-02",
            floor_a="FLOOR-1", floor_b="FLOOR-1",
            confidence=0.40, decision=AssociationDecision.KEEP_SEPARATE.value,
            evidence={"semantic_similarity": 0.30},
        )
        res = FusionResult(
            participating_nodes=["PHONE-01", "PHONE-02"],
            associations=[assoc],
            merge_decisions={"PHONE-01:PHONE-02": "KEEP_SEPARATE"},
            confidence=0.40,
            aligned_observations=[],
        )

        wm.apply_fusion_result(res)

        # Observations should remain empty (no merge applied)
        self.assertEqual(len(wm.get_observations("FLOOR-1")), 0)
        # History recorded
        self.assertEqual(len(wm.get_fusion_history()), 1)

    def test_different_floors_never_merge(self):
        """Test 9: Different floors never merge."""
        wm = WorldModel()
        assoc = MapAssociation(
            node_a="PHONE-01", node_b="PHONE-02",
            floor_a="FLOOR-1", floor_b="FLOOR-2",  # Mismatched floors!
            confidence=0.95, decision=AssociationDecision.MERGE.value,  # Invalid decision for cross-floor
            evidence={"semantic_similarity": 1.0},
        )
        res = FusionResult(
            participating_nodes=["PHONE-01", "PHONE-02"],
            associations=[assoc],
            merge_decisions={"PHONE-01:PHONE-02": "MERGE"},
            confidence=0.95,
            aligned_observations=[],
        )

        wm.apply_fusion_result(res)

        # Cross-floor merge ignored; floors remain unmerged
        f1 = wm.get_floor("FLOOR-1")
        f2 = wm.get_floor("FLOOR-2")
        self.assertIsNone(f1)  # No invalid floor-1 activation
        self.assertIsNone(f2)

    def test_fusion_history_preserved(self):
        """Test 10: Fusion history is preserved."""
        wm = WorldModel()
        res1 = FusionResult(
            participating_nodes=["PHONE-01", "PHONE-02"],
            associations=[], merge_decisions={}, confidence=0.85,
            aligned_observations=[],
        )
        res2 = FusionResult(
            participating_nodes=["PHONE-02", "PHONE-03"],
            associations=[], merge_decisions={}, confidence=0.90,
            aligned_observations=[],
        )

        wm.apply_fusion_result(res1)
        wm.apply_fusion_result(res2)

        history = wm.get_fusion_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["confidence"], 0.85)
        self.assertEqual(history[1]["confidence"], 0.90)

    def test_source_local_maps_not_mutated(self):
        """Test 16: Source LocalMapState objects are not mutated."""
        wm = WorldModel()
        engine = FusionEngine()

        obs_a = [SemanticObservation(class_name="doorway")]
        obs_b = [SemanticObservation(class_name="wall")]

        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1", observations=obs_a)
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1", observations=obs_b)

        # Execute fusion and apply to world model
        res = engine.fuse_maps([map_a, map_b])
        wm.apply_fusion_result(res)

        # Verify source LocalMapState objects were NOT mutated
        self.assertEqual(len(map_a.observations), 1)
        self.assertEqual(len(map_b.observations), 1)
        self.assertEqual(map_a.observations[0].class_name, "doorway")
        self.assertEqual(map_b.observations[0].class_name, "wall")


class TestMetricsAndSerialization(unittest.TestCase):
    """Tests 12 & 13: Coverage metric validation and serialization snapshot."""

    def test_coverage_remains_valid(self):
        """Test 12: Coverage remains valid [0.0, 1.0]."""
        wm = WorldModel()
        wm.set_floor_coverage("FLOOR-1", 1.5)  # Clamps to 1.0
        self.assertEqual(wm.get_floor("FLOOR-1").coverage, 1.0)

        wm.set_floor_coverage("FLOOR-1", -0.2)  # Clamps to 0.0
        self.assertEqual(wm.get_floor("FLOOR-1").coverage, 0.0)

        wm.set_floor_coverage("FLOOR-1", 0.65)
        self.assertEqual(wm.get_floor("FLOOR-1").coverage, 0.65)

    def test_serialization_works(self):
        """Test 13: Serialization works."""
        wm = WorldModel(building_id="BLDG-TEST")
        wm.add_observation(
            SemanticObservation(node_id="PHONE-01", floor_id="FLOOR-1", class_name="doorway")
        )
        wm.add_landmark(
            Landmark(node_id="PHONE-01", position=(1.0, 2.0, 0.0)),
            floor_id="FLOOR-1",
        )
        wm.set_floor_coverage("FLOOR-1", 0.5)
        wm.set_floor_confidence("FLOOR-1", 0.85)

        state_dict = wm.to_dict()

        self.assertEqual(state_dict["building_id"], "BLDG-TEST")
        self.assertIn("FLOOR-1", state_dict["floors"])
        self.assertIn("PHONE-01", state_dict["participating_nodes"])

        f1_data = state_dict["floors"]["FLOOR-1"]
        self.assertEqual(f1_data["coverage"], 0.5)
        self.assertEqual(f1_data["confidence"], 0.85)
        self.assertEqual(len(state_dict["shared_observations"]), 1)
        self.assertEqual(len(state_dict["shared_landmarks"]), 1)


if __name__ == "__main__":
    unittest.main()

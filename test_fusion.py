# ══════════════════════════════════════════════════════════════════
#  🧪 UNIT TESTS — TIDDA Multi-Node Fusion Engine
#  Validates fusion.py signal calculations, weight renormalization,
#  decision thresholding, floor separation, and non-mutation invariants.
#  Run:  python3 -m unittest test_fusion -v
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import unittest

from fusion import (
    AssociationDecision,
    FusionEngine,
    FusionResult,
    MapAssociation,
    compute_angular_diff_deg,
)
from mapping_models import (
    Landmark,
    LocalMapState,
    Pose,
    SemanticObservation,
)


class TestSemanticAndLandmarkSignals(unittest.TestCase):
    """Tests 1–4: Semantic similarity and landmark matching/missing behavior."""

    def test_high_semantic_similarity_high_confidence(self):
        """Test 1: High semantic similarity -> high confidence."""
        engine = FusionEngine()
        obs_a = [
            SemanticObservation(class_name="doorway"),
            SemanticObservation(class_name="wall"),
            SemanticObservation(class_name="chair"),
        ]
        obs_b = [
            SemanticObservation(class_name="doorway"),
            SemanticObservation(class_name="wall"),
            SemanticObservation(class_name="chair"),
        ]
        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1", observations=obs_a)
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1", observations=obs_b)

        sim = engine.calc_semantic_similarity(map_a, map_b)
        self.assertEqual(sim, 1.0)

    def test_low_semantic_similarity_lower_confidence(self):
        """Test 2: Low semantic similarity -> lower confidence."""
        engine = FusionEngine()
        obs_a = [
            SemanticObservation(class_name="doorway"),
            SemanticObservation(class_name="wall"),
        ]
        obs_b = [
            SemanticObservation(class_name="staircase"),
            SemanticObservation(class_name="elevator"),
        ]
        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1", observations=obs_a)
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1", observations=obs_b)

        sim = engine.calc_semantic_similarity(map_a, map_b)
        self.assertEqual(sim, 0.0)

    def test_matching_landmarks_higher_confidence(self):
        """Test 3: Matching landmarks -> higher confidence."""
        engine = FusionEngine()
        lm_a = [Landmark(position=(1.0, 2.0, 0.0), descriptor="feature_x")]
        lm_b = [Landmark(position=(1.0, 2.0, 0.0), descriptor="feature_x")]

        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1", landmarks=lm_a)
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1", landmarks=lm_b)

        sim = engine.calc_landmark_similarity(map_a, map_b)
        self.assertEqual(sim, 1.0)

    def test_missing_landmarks_handled_correctly(self):
        """Test 4: Missing landmarks handled correctly (returns None)."""
        engine = FusionEngine()
        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1", landmarks=[])
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1", landmarks=[])

        sim = engine.calc_landmark_similarity(map_a, map_b)
        self.assertIsNone(sim)


class TestPoseOrientationAndGPS(unittest.TestCase):
    """Tests 5–10: Pose, orientation, GPS, and missing signal handling."""

    def test_matching_poses_higher_consistency(self):
        """Test 5: Matching poses -> higher pose consistency."""
        engine = FusionEngine()
        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.0, y=2.0, z=0.0)],
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.1, y=2.1, z=0.0)],
        )

        score = engine.calc_pose_consistency(map_a, map_b)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.9)

    def test_missing_pose_handled_correctly(self):
        """Test 6: Missing pose handled correctly (returns None)."""
        engine = FusionEngine()
        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1", trajectory=[])
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1", trajectory=[])

        score = engine.calc_pose_consistency(map_a, map_b)
        self.assertIsNone(score)

    def test_orientation_wrap_around(self):
        """Test 7: Orientation wrap-around: 359° vs 1° must be treated as close (2° apart)."""
        diff = compute_angular_diff_deg(359.0, 1.0)
        self.assertEqual(diff, 2.0)

        # Test engine orientation calculation with 359° vs 1°
        engine = FusionEngine(max_angle_deg=45.0)
        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            trajectory=[Pose(x=0.0, y=0.0, yaw=359.0)],
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-1",
            trajectory=[Pose(x=0.0, y=0.0, yaw=1.0)],
        )
        score = engine.calc_orientation_consistency(map_a, map_b)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.9)

    def test_gps_available_contributes(self):
        """Test 8: GPS available -> GPS contributes."""
        engine = FusionEngine()
        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            trajectory=[Pose(latitude=28.6139, longitude=77.2090)],
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-1",
            trajectory=[Pose(latitude=28.6139, longitude=77.2090)],
        )
        score = engine.calc_gps_consistency(map_a, map_b)
        self.assertEqual(score, 1.0)

    def test_gps_unavailable_fusion_works(self):
        """Test 9: GPS unavailable -> fusion still works."""
        engine = FusionEngine()
        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.0, y=2.0, latitude=None, longitude=None)],
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.0, y=2.0, latitude=None, longitude=None)],
        )
        assoc = engine.evaluate_association(map_a, map_b)
        self.assertIsNone(assoc.evidence["gps_consistency"])
        self.assertGreater(assoc.confidence, 0.0)

    def test_visual_similarity_unavailable_fusion_works(self):
        """Test 10: Visual similarity unavailable -> fusion still works."""
        engine = FusionEngine()
        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1")
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1")
        assoc = engine.evaluate_association(map_a, map_b)
        self.assertIsNone(assoc.evidence["visual_similarity"])


class TestWeightRenormalizationAndDecisions(unittest.TestCase):
    """Tests 11–15: Weight renormalization, thresholding, and floor constraints."""

    def test_missing_multiple_signals_renormalize(self):
        """Test 11: Missing multiple signals -> remaining weights renormalize."""
        engine = FusionEngine()
        # Only semantic_similarity is available
        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            observations=[SemanticObservation(class_name="doorway")],
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-1",
            observations=[SemanticObservation(class_name="doorway")],
        )
        assoc = engine.evaluate_association(map_a, map_b)
        # Since semantic similarity is 1.0 and it is the only available signal,
        # normalized confidence should be 1.0!
        self.assertEqual(assoc.confidence, 1.0)

    def test_confidence_remains_in_range(self):
        """Test 12: Confidence always remains 0.0–1.0."""
        engine = FusionEngine()
        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1")
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1")
        assoc = engine.evaluate_association(map_a, map_b)

        self.assertGreaterEqual(assoc.confidence, 0.0)
        self.assertLessEqual(assoc.confidence, 1.0)

    def test_confidence_threshold_merge(self):
        """Test 13: Confidence >= threshold -> MERGE."""
        engine = FusionEngine(merge_threshold=0.80)
        obs_a = [SemanticObservation(class_name="doorway"), SemanticObservation(class_name="wall")]
        obs_b = [SemanticObservation(class_name="doorway"), SemanticObservation(class_name="wall")]

        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.0, y=2.0)], observations=obs_a,
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.0, y=2.0)], observations=obs_b,
        )
        assoc = engine.evaluate_association(map_a, map_b)
        self.assertGreaterEqual(assoc.confidence, 0.80)
        self.assertEqual(assoc.decision, AssociationDecision.MERGE.value)

    def test_confidence_subthreshold_keep_separate(self):
        """Test 14: Confidence < threshold -> KEEP_SEPARATE."""
        engine = FusionEngine(merge_threshold=0.80)
        obs_a = [SemanticObservation(class_name="doorway")]
        obs_b = [SemanticObservation(class_name="staircase")]

        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            trajectory=[Pose(x=0.0, y=0.0)], observations=obs_a,
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-1",
            trajectory=[Pose(x=50.0, y=50.0)], observations=obs_b,
        )
        assoc = engine.evaluate_association(map_a, map_b)
        self.assertLess(assoc.confidence, 0.80)
        self.assertEqual(assoc.decision, AssociationDecision.KEEP_SEPARATE.value)

    def test_different_floors_keep_separate(self):
        """Test 15: Different floors -> KEEP_SEPARATE."""
        engine = FusionEngine(merge_threshold=0.10)
        # Even with identical poses and observations, different floor_ids force KEEP_SEPARATE!
        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.0, y=1.0)],
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-2",  # Different floor!
            trajectory=[Pose(x=1.0, y=1.0)],
        )
        assoc = engine.evaluate_association(map_a, map_b)
        self.assertEqual(assoc.decision, AssociationDecision.KEEP_SEPARATE.value)


class TestInvariantsAndSerialization(unittest.TestCase):
    """Tests 16–20: Source attribution, non-mutation, multi-pair evaluation, serialization, insufficient evidence."""

    def test_source_node_ids_preserved(self):
        """Test 16: Source node IDs preserved."""
        engine = FusionEngine()
        obs_a = SemanticObservation(node_id="PHONE-01", class_name="doorway")
        obs_b = SemanticObservation(node_id="PHONE-02", class_name="wall")

        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1", observations=[obs_a])
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1", observations=[obs_b])

        result = engine.fuse_maps([map_a, map_b])
        sources = {obs.node_id for obs in result.aligned_observations}
        self.assertIn("PHONE-01", sources)
        self.assertIn("PHONE-02", sources)

    def test_fusion_does_not_mutate_source_local_maps(self):
        """Test 17: Fusion does not mutate source local maps."""
        engine = FusionEngine()
        obs_a = [SemanticObservation(class_name="doorway")]
        obs_b = [SemanticObservation(class_name="wall")]

        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1", observations=obs_a)
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1", observations=obs_b)

        # Take copies of original state to verify non-mutation
        orig_a_len = len(map_a.observations)
        orig_b_len = len(map_b.observations)

        _ = engine.fuse_maps([map_a, map_b])

        self.assertEqual(len(map_a.observations), orig_a_len)
        self.assertEqual(len(map_b.observations), orig_b_len)

    def test_multiple_node_pairs_evaluated(self):
        """Test 18: Multiple node pairs can be evaluated."""
        engine = FusionEngine()
        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1")
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1")
        map_c = LocalMapState(node_id="PHONE-03", floor_id="FLOOR-2")

        result = engine.fuse_maps([map_a, map_b, map_c])
        self.assertEqual(len(result.participating_nodes), 3)
        # 3 nodes -> 3 pairwise associations (A-B, A-C, B-C)
        self.assertEqual(len(result.associations), 3)
        self.assertIn("PHONE-01:PHONE-02", result.merge_decisions)
        self.assertIn("PHONE-01:PHONE-03", result.merge_decisions)

    def test_serialization_works(self):
        """Test 19: Serialization works."""
        engine = FusionEngine()
        map_a = LocalMapState(
            node_id="PHONE-01", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.0, y=2.0)],
            observations=[SemanticObservation(class_name="doorway")],
        )
        map_b = LocalMapState(
            node_id="PHONE-02", floor_id="FLOOR-1",
            trajectory=[Pose(x=1.0, y=2.0)],
            observations=[SemanticObservation(class_name="doorway")],
        )

        assoc = engine.evaluate_association(map_a, map_b)
        assoc_dict = assoc.to_dict()
        self.assertEqual(assoc_dict["node_a"], "PHONE-01")
        self.assertEqual(assoc_dict["node_b"], "PHONE-02")
        self.assertIn("decision", assoc_dict)

        res = engine.fuse_maps([map_a, map_b])
        res_dict = res.to_dict()
        self.assertEqual(len(res_dict["participating_nodes"]), 2)
        self.assertIn("associations", res_dict)
        self.assertIn("confidence", res_dict)

    def test_insufficient_evidence_does_not_force_merge(self):
        """Test 20: Insufficient evidence does not cause a forced merge."""
        engine = FusionEngine(merge_threshold=0.80)
        # Empty maps with no evidence signals available
        map_a = LocalMapState(node_id="PHONE-01", floor_id="FLOOR-1")
        map_b = LocalMapState(node_id="PHONE-02", floor_id="FLOOR-1")

        assoc = engine.evaluate_association(map_a, map_b)
        self.assertEqual(assoc.confidence, 0.0)
        self.assertEqual(assoc.decision, AssociationDecision.KEEP_SEPARATE.value)


if __name__ == "__main__":
    unittest.main()

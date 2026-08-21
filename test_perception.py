# ══════════════════════════════════════════════════════════════════
#  🧪 UNIT TESTS — TIDDA AI Perception Layer
#  Validates perception.py real & mock perception engines.
#  Run:  python3 -m unittest test_perception -v
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import unittest

from perception import (
    MockPerceptionEngine,
    PerceptionEngine,
    SUPPORTED_CLASSES,
    UNSUPPORTED_CLASSES,
)
from mapping_models import SemanticObservation


class TestPerceptionInstantiationAndInit(unittest.TestCase):
    """Tests 1, 13, 14: Engine instantiation, initialization, and error handling."""

    def test_perception_engine_instantiation(self):
        """Test 1: PerceptionEngine can be instantiated."""
        engine = PerceptionEngine(model_name="yolov8n.pt")
        self.assertIsNotNone(engine)
        self.assertEqual(engine.model_name, "yolov8n.pt")
        self.assertEqual(engine.confidence_threshold, 0.45)

    def test_real_detector_init_without_breaking_imports(self):
        """Test 13: Real detector integration, if available, can be initialized without breaking imports."""
        engine = PerceptionEngine(model_name="yolov8n.pt")
        # Should initialize or report unavailable without raising unhandled exception
        available = engine.initialize()
        self.assertIsInstance(available, bool)

    def test_detector_init_failure_handled_cleanly(self):
        """Test 14: Failure of detector initialization is handled cleanly."""
        # Invalid model path/file
        engine = PerceptionEngine(model_name="nonexistent_model_weights_123.pt")
        available = engine.initialize()
        self.assertFalse(available)
        self.assertIsNotNone(engine.init_error)

        # Processing with failed detector returns [] cleanly
        obs = engine.process("mock_frame", node_id="PHONE-01", floor_id="FLOOR-1")
        self.assertEqual(obs, [])


class TestInvalidFrameHandling(unittest.TestCase):
    """Test 2: Invalid frames are handled cleanly."""

    def test_invalid_frames(self):
        """Test 2: Invalid frame is handled."""
        mock_engine = MockPerceptionEngine()

        # None frame
        self.assertEqual(mock_engine.process(None, "PHONE-01", "FLOOR-1"), [])
        # Empty string frame
        self.assertEqual(mock_engine.process("", "PHONE-01", "FLOOR-1"), [])
        # Empty bytes frame
        self.assertEqual(mock_engine.process(b"", "PHONE-01", "FLOOR-1"), [])

        # Invalid node_id
        self.assertEqual(mock_engine.process("frame_data", "", "FLOOR-1"), [])
        # Invalid floor_id
        self.assertEqual(mock_engine.process("frame_data", "PHONE-01", ""), [])

        # Real engine with invalid input
        real_engine = PerceptionEngine()
        self.assertEqual(real_engine.process(None, "PHONE-01", "FLOOR-1"), [])
        self.assertEqual(real_engine.process("corrupted_base64_???", "PHONE-01", "FLOOR-1"), [])


class TestMockPerceptionEngine(unittest.TestCase):
    """Tests 3–11: Mock perception deterministic observations and field assertions."""

    def test_mock_perception_deterministic(self):
        """Test 3: Mock perception produces deterministic observations."""
        mock_engine = MockPerceptionEngine()
        obs = mock_engine.process("test_frame", node_id="PHONE-01", floor_id="FLOOR-1")
        self.assertGreater(len(obs), 0)
        self.assertEqual(obs[0].class_name, "person")
        self.assertEqual(obs[1].class_name, "chair")

    def test_observation_fields(self):
        """Tests 4–8: Observation contains node_id, floor_id, class_name, confidence in valid range."""
        mock_engine = MockPerceptionEngine()
        obs_list = mock_engine.process("test_frame", node_id="PHONE-01", floor_id="FLOOR-1")
        obs = obs_list[0]

        # Test 4: node_id
        self.assertEqual(obs.node_id, "PHONE-01")
        # Test 5: floor_id
        self.assertEqual(obs.floor_id, "FLOOR-1")
        # Test 6: class_name
        self.assertEqual(obs.class_name, "person")
        # Test 7: confidence
        self.assertIsInstance(obs.confidence, float)
        # Test 8: confidence in valid range [0.0, 1.0]
        self.assertGreaterEqual(obs.confidence, 0.0)
        self.assertLessEqual(obs.confidence, 1.0)

    def test_multiple_observations_returned(self):
        """Test 9: Multiple observations can be returned."""
        presets = [
            {"class_name": "person", "confidence": 0.95},
            {"class_name": "chair", "confidence": 0.82},
            {"class_name": "laptop", "confidence": 0.78},
        ]
        mock_engine = MockPerceptionEngine(preset_detections=presets)
        obs_list = mock_engine.process("frame_1", node_id="PHONE-01", floor_id="FLOOR-1")

        self.assertEqual(len(obs_list), 3)
        self.assertEqual(obs_list[0].class_name, "person")
        self.assertEqual(obs_list[1].class_name, "chair")
        self.assertEqual(obs_list[2].class_name, "laptop")

    def test_different_nodes_retain_separate_source_ids(self):
        """Test 10: Different nodes retain separate source IDs."""
        mock_engine = MockPerceptionEngine()

        obs1 = mock_engine.process("frame_a", node_id="PHONE-01", floor_id="FLOOR-1")
        obs2 = mock_engine.process("frame_b", node_id="PHONE-02", floor_id="FLOOR-1")

        self.assertEqual(obs1[0].node_id, "PHONE-01")
        self.assertEqual(obs2[0].node_id, "PHONE-02")
        self.assertNotEqual(obs1[0].node_id, obs2[0].node_id)

    def test_different_floors_retain_separate_floor_ids(self):
        """Test 11: Different floors retain separate floor IDs."""
        mock_engine = MockPerceptionEngine()

        obs1 = mock_engine.process("frame_a", node_id="PHONE-01", floor_id="FLOOR-1")
        obs2 = mock_engine.process("frame_b", node_id="PHONE-01", floor_id="FLOOR-2")

        self.assertEqual(obs1[0].floor_id, "FLOOR-1")
        self.assertEqual(obs2[0].floor_id, "FLOOR-2")
        self.assertNotEqual(obs1[0].floor_id, obs2[0].floor_id)


class TestClassSupportDocumentation(unittest.TestCase):
    """Test 12: Unsupported architectural classes are not fabricated."""

    def test_unsupported_classes_not_fabricated(self):
        """Test 12: Unsupported classes are not fabricated by detector."""
        real_engine = PerceptionEngine()
        unsupported = real_engine.get_unsupported_classes()

        # Check that wall, doorway, corridor, room, staircase, elevator are documented unsupported
        for req_cls in ["wall", "doorway", "corridor", "room", "staircase", "elevator"]:
            self.assertIn(req_cls, unsupported)

        # Verify supported classes list matches COCO / object mapping
        supported = real_engine.get_supported_classes()
        self.assertIn("person", supported)
        self.assertIn("chair", supported)
        self.assertIn("obstacle", supported)
        self.assertIn("object", supported)


if __name__ == "__main__":
    unittest.main()

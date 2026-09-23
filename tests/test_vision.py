import unittest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.vision import HumanDetector


class TestVisionSubsystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = HumanDetector()

    def test_yolo_model_loading(self):
        """Verify YOLO model loads on configured device."""
        self.assertTrue(self.detector.is_loaded)
        self.assertIsNotNone(self.detector.model)

    def test_blank_frame_no_person(self):
        """Verify blank black frame detects 0 persons."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        found, annotated, detections, max_conf = self.detector.detect_humans(blank_frame)

        self.assertFalse(found)
        self.assertEqual(len(detections), 0)
        self.assertEqual(max_conf, 0.0)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, blank_frame.shape)

    def test_none_frame_graceful_handling(self):
        """Verify detector handles None frame without crashing."""
        found, annotated, detections, max_conf = self.detector.detect_humans(None)
        self.assertFalse(found)
        self.assertIsNone(annotated)
        self.assertEqual(len(detections), 0)
        self.assertEqual(max_conf, 0.0)

    def test_random_noise_frame(self):
        """Verify detector handles random noise frame safely."""
        noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        found, annotated, detections, max_conf = self.detector.detect_humans(noise_frame)
        self.assertIsInstance(found, bool)
        self.assertIsInstance(detections, list)


if __name__ == "__main__":
    unittest.main()


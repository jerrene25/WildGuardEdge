import unittest
import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.fusion import FusionEngine
from modules.database import WildGuardDatabase


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_e2e.db"
        self.engine = FusionEngine()
        self.engine.db = WildGuardDatabase(db_path=self.db_path)
        self.engine.start()

    def tearDown(self):
        self.engine.stop()
        self.temp_dir.cleanup()

    def test_full_cycle_execution(self):
        """Execute a full live multimodal cycle and verify all output keys and types."""
        cycle_result = self.engine.run_one_cycle()

        expected_keys = [
            "timestamp", "fps", "mode", "state", "threat_score", "threshold",
            "reason", "rms", "peak", "audio_class", "audio_confidence",
            "audio_threat", "is_silence", "waveform", "camera_online",
            "human_detected", "person_count", "person_confidence",
            "detections", "clip_threat", "clip_score", "caption",
            "alert_fired", "total_alerts",
        ]

        for k in expected_keys:
            self.assertIn(k, cycle_result, f"Missing key in cycle result: {k}")

        self.assertIsInstance(cycle_result["threat_score"], float)
        self.assertGreaterEqual(cycle_result["threat_score"], 0.0)
        self.assertLessEqual(cycle_result["threat_score"], 1.0)
        self.assertIsInstance(cycle_result["alert_fired"], bool)
        self.assertIsInstance(cycle_result["caption"], str)


if __name__ == "__main__":
    unittest.main()


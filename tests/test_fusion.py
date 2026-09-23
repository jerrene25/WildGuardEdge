import unittest
import time
import tempfile
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.fusion import FusionEngine, ThreatState
from modules.database import WildGuardDatabase


class TestFusionEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_fusion.db"

        # Initialize FusionEngine with test database
        self.engine = FusionEngine()
        self.engine.db = WildGuardDatabase(db_path=self.db_path)

    def tearDown(self):
        self.engine.stop()
        self.temp_dir.cleanup()

    def test_silence_produces_zero_audio_evidence(self):
        """Verify that silence produces 0.0 threat score in night mode."""
        score, reason = self.engine._calculate_evidence(
            audio_threat=False,
            audio_conf=0.0,
            rms=0.0001,  # below floor
            human_found=False,
            person_conf=0.0,
            clip_threat=False,
            clip_score=0.20,
            night_mode=True,
        )
        self.assertLess(score, config.NIGHT_FUSION_THRESHOLD)
        self.assertIn("Normal", reason)

    def test_day_mode_requires_person(self):
        """Verify that in day mode without a human, threat score is heavily discounted."""
        score, reason = self.engine._calculate_evidence(
            audio_threat=True,
            audio_conf=0.95,
            rms=0.05,
            human_found=False,
            person_conf=0.0,
            clip_threat=True,
            clip_score=0.35,
            night_mode=False,
        )
        # Without a person in day mode, score is discounted
        self.assertLess(score, config.DAY_FUSION_THRESHOLD)

    def test_temporal_confirmation_and_cooldown(self):
        """
        Verify state machine:
        Requires TEMPORAL_CONFIRMATION_COUNT positive observations before firing an alert.
        Once fired, enters COOLDOWN and does not duplicate alerts.
        """
        evidence_dict = {
            "night_mode": False,
            "audio_class": "threat",
            "audio_conf": 0.90,
            "rms": 0.05,
            "human_detected": True,
            "person_conf": 0.95,
            "clip_score": 0.35,
        }

        # Observation 1:
        state, alert_fired, alert_rec = self.engine._update_state_machine(
            instant_threat_detected=True,
            threat_score=0.85,
            reason="Test Threat",
            frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            evidence_dict=evidence_dict,
        )
        self.assertEqual(state, ThreatState.SUSPICIOUS)
        self.assertFalse(alert_fired, "First observation should not immediately fire alert")

        # Observation 2:
        state, alert_fired, alert_rec = self.engine._update_state_machine(
            instant_threat_detected=True,
            threat_score=0.85,
            reason="Test Threat",
            frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            evidence_dict=evidence_dict,
        )
        self.assertEqual(state, ThreatState.SUSPICIOUS)
        self.assertFalse(alert_fired, "Second observation should not fire alert")

        # Observation 3 (Meets confirmation threshold):
        state, alert_fired, alert_rec = self.engine._update_state_machine(
            instant_threat_detected=True,
            threat_score=0.85,
            reason="Test Threat",
            frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            evidence_dict=evidence_dict,
        )
        self.assertEqual(state, ThreatState.COOLDOWN)
        self.assertTrue(alert_fired, "Third observation MUST confirm threat and fire alert")
        self.assertIsNotNone(alert_rec)
        self.assertEqual(self.engine.db.get_total_count(), 1)

        # Observation 4 immediately following (During Cooldown):
        state, alert_fired, alert_rec = self.engine._update_state_machine(
            instant_threat_detected=True,
            threat_score=0.85,
            reason="Test Threat Continued",
            frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            evidence_dict=evidence_dict,
        )
        self.assertEqual(state, ThreatState.COOLDOWN)
        self.assertFalse(alert_fired, "Must NOT fire duplicate alert during cooldown")
        self.assertEqual(self.engine.db.get_total_count(), 1, "Database must still have exactly 1 event")


if __name__ == "__main__":
    unittest.main()


import unittest
import tempfile
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.database import WildGuardDatabase
from modules.fusion import FusionEngine


class TestSecurityFortification(unittest.TestCase):
    """
    Adversarial security test suite targeting OWASP Top 10 vectors:
    - SQL Injection
    - Path Traversal / Arbitrary File Deletion
    - Denial of Service / Extreme Input Bounds
    - Malicious Model Loading Protection
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_sec.db"
        self.snapshot_dir = Path(self.temp_dir.name) / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Re-point config.SNAPSHOT_DIR for testing
        self.orig_snapshot_dir = config.SNAPSHOT_DIR
        config.SNAPSHOT_DIR = self.snapshot_dir

        self.db = WildGuardDatabase(db_path=self.db_path)

    def tearDown(self):
        config.SNAPSHOT_DIR = self.orig_snapshot_dir
        self.temp_dir.cleanup()

    def test_sql_injection_in_event_fields(self):
        """Verify that malicious SQL payloads in text/numeric fields do not inject."""
        malicious_payload = {
            "timestamp": "2026-09-23T12:00:00'; DROP TABLE events; --",
            "mode": "DAY'; DELETE FROM events; --",
            "threat_score": 0.95,
            "reason": "Threat detected'); DROP TABLE events; --",
            "caption": "' OR '1'='1",
            "audio_class": "threat'; DROP TABLE events; --",
        }

        event_id = self.db.insert_event(malicious_payload)
        self.assertGreater(event_id, 0)

        # Verify table still exists and record is safely stored
        self.assertEqual(self.db.get_total_count(), 1)
        event = self.db.get_event_by_id(event_id)
        self.assertIsNotNone(event)
        self.assertEqual(event["reason"], "Threat detected'); DROP TABLE events; --")

    def test_sql_injection_and_overflow_in_query_limit(self):
        """Verify limit parameter is strictly bounded against injection and memory DoS."""
        for i in range(5):
            self.db.insert_event({"mode": "DAY", "threat_score": 0.5, "reason": f"Event {i}"})

        # Test negative limit
        neg_results = self.db.get_recent_events(limit=-5)
        self.assertEqual(len(neg_results), 1)  # clamped to min 1

        # Test gigantic limit (DoS prevention)
        huge_results = self.db.get_recent_events(limit=999999999)
        self.assertLessEqual(len(huge_results), 1000)

        # Test string conversion
        str_results = self.db.get_recent_events(limit="3")
        self.assertEqual(len(str_results), 3)

    def test_path_traversal_prevention_on_snapshot_deletion(self):
        """
        Verify that delete_event() CANNOT delete files outside config.SNAPSHOT_DIR
        even if an attacker injects an arbitrary path into the database record.
        """
        # Create an innocent canary file outside SNAPSHOT_DIR (e.g. in root temp_dir)
        sensitive_canary = Path(self.temp_dir.name) / "sensitive_system_file.txt"
        sensitive_canary.write_text("CRITICAL SYSTEM DATA - DO NOT DELETE")
        self.assertTrue(sensitive_canary.exists())

        # Insert malicious event referencing file outside SNAPSHOT_DIR
        event_id = self.db.insert_event({
            "mode": "DAY",
            "threat_score": 0.9,
            "snapshot_path": str(sensitive_canary),
        })

        # Attempt to delete event with delete_snapshot=True
        deleted = self.db.delete_event(event_id, delete_snapshot=True)
        self.assertTrue(deleted)

        # The canary file MUST NOT be deleted!
        self.assertTrue(
            sensitive_canary.exists(),
            "SECURITY VULNERABILITY: Sensitive file outside SNAPSHOT_DIR was deleted!",
        )

    def test_legitimate_snapshot_deletion(self):
        """Verify that legitimate snapshots inside SNAPSHOT_DIR are correctly deleted."""
        valid_snapshot = self.snapshot_dir / "valid_alert.jpg"
        valid_snapshot.write_text("fake image bytes")
        self.assertTrue(valid_snapshot.exists())

        event_id = self.db.insert_event({
            "mode": "DAY",
            "threat_score": 0.9,
            "snapshot_path": str(valid_snapshot),
        })

        self.db.delete_event(event_id, delete_snapshot=True)
        self.assertFalse(valid_snapshot.exists())

    def test_path_traversal_prevention_on_snapshot_save(self):
        """Verify _save_snapshot strips path traversal components from tag."""
        # Use __new__ to test method logic directly without opening live camera hardware
        engine = FusionEngine.__new__(FusionEngine)
        dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # Malicious tag attempting to write to parent directory
        malicious_tag = "../../escaped_tag"
        saved_path = engine._save_snapshot(dummy_frame, malicious_tag)
        self.assertIsNotNone(saved_path)

        resolved_path = Path(saved_path).resolve()
        resolved_root = config.SNAPSHOT_DIR.resolve()

        # Ensure saved path is strictly within SNAPSHOT_DIR
        self.assertTrue(
            resolved_path.is_relative_to(resolved_root),
            f"Snapshot was saved outside root: {resolved_path}",
        )
        self.assertNotIn("..", Path(saved_path).name)


if __name__ == "__main__":
    unittest.main()

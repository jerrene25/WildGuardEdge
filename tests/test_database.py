import unittest
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.database import WildGuardDatabase


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_wildguard.db"
        self.snapshot_dir = Path(self.temp_dir.name) / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.orig_snapshot_dir = config.SNAPSHOT_DIR
        config.SNAPSHOT_DIR = self.snapshot_dir
        self.db = WildGuardDatabase(db_path=self.db_path)

    def tearDown(self):
        config.SNAPSHOT_DIR = self.orig_snapshot_dir
        self.temp_dir.cleanup()

    def test_database_initialization(self):
        """Verify events table exists and count starts at 0."""
        self.assertEqual(self.db.get_total_count(), 0)

    def test_insert_and_retrieve_event(self):
        """Verify insertion of a confirmed event and retrieval of all fields."""
        event_payload = {
            "timestamp": "2026-09-12T10:00:00",
            "mode": "DAY",
            "threat_score": 0.85,
            "audio_class": "threat",
            "audio_confidence": 0.92,
            "audio_rms": 0.045,
            "person_detected": 1,
            "person_confidence": 0.88,
            "clip_score": 0.35,
            "caption": "A person holding equipment in a forest",
            "snapshot_path": "/path/to/snapshot.jpg",
            "reason": "Person detected + Suspicious context",
        }

        event_id = self.db.insert_event(event_payload)
        self.assertGreater(event_id, 0)
        self.assertEqual(self.db.get_total_count(), 1)

        retrieved = self.db.get_event_by_id(event_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["id"], event_id)
        self.assertEqual(retrieved["mode"], "DAY")
        self.assertAlmostEqual(retrieved["threat_score"], 0.85, places=2)
        self.assertEqual(retrieved["person_detected"], 1)
        self.assertEqual(retrieved["caption"], "A person holding equipment in a forest")

    def test_recent_events_ordering(self):
        """Verify recent events are ordered by ID descending."""
        for i in range(5):
            self.db.insert_event({
                "timestamp": f"2026-09-12T10:0{i}:00",
                "mode": "DAY",
                "threat_score": 0.5 + (i * 0.1),
                "person_detected": 1,
                "reason": f"Event {i}",
            })

        recent = self.db.get_recent_events(limit=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0]["reason"], "Event 4")
        self.assertEqual(recent[1]["reason"], "Event 3")
        self.assertEqual(recent[2]["reason"], "Event 2")

    def test_clear_events(self):
        """Verify clearing events removes all records."""
        self.db.insert_event({"mode": "DAY", "threat_score": 0.9, "person_detected": 1})
        self.assertEqual(self.db.get_total_count(), 1)
        self.db.clear_events()
        self.assertEqual(self.db.get_total_count(), 0)

    def test_delete_event_and_snapshots(self):
        """Verify deleting an event also removes its snapshot file from disk."""
        snap_file = self.snapshot_dir / "test_snapshot.jpg"
        snap_file.write_text("fake image data")
        self.assertTrue(snap_file.exists())

        event_id = self.db.insert_event({
            "mode": "DAY",
            "threat_score": 0.9,
            "person_detected": 1,
            "snapshot_path": str(snap_file),
        })
        self.assertEqual(self.db.get_total_count(), 1)

        deleted = self.db.delete_event(event_id, delete_snapshot=True)
        self.assertTrue(deleted)
        self.assertEqual(self.db.get_total_count(), 0)
        self.assertFalse(snap_file.exists())


if __name__ == "__main__":
    unittest.main()


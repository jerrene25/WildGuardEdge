import unittest
import tempfile
import threading
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.database import WildGuardDatabase
from modules.audio_capture import AudioCapture
from modules.audio_cnn import AudioThreatDetector
from modules.vision import HumanDetector
from modules.vlm import VLMVerifier


class TestBulletproofingAndHardening(unittest.TestCase):
    """
    Hostile QA test suite testing zero-crash tolerance under corrupted inputs,
    numerical extremes (NaN/Inf), invalid memory buffers, and high-frequency concurrency.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_hardening.db"
        self.db = WildGuardDatabase(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_audio_nan_and_inf_resilience(self):
        """Verify AudioCapture and AudioThreatDetector handle NaNs and Infs gracefully."""
        cap = AudioCapture()
        detector = AudioThreatDetector()

        # Array of NaNs and Infs
        corrupted_audio = np.array([np.nan, np.inf, -np.inf, 0.0, 1.0, -1.0], dtype=np.float32)

        # RMS calculation must not crash and must not return NaN
        rms = cap.get_volume_rms(corrupted_audio)
        self.assertFalse(np.isnan(rms), "RMS returned NaN")
        self.assertFalse(np.isinf(rms), "RMS returned Inf")
        self.assertGreaterEqual(rms, 0.0)

        # Peak calculation must not crash
        peak = cap.get_peak_level(corrupted_audio)
        self.assertFalse(np.isnan(peak))
        self.assertFalse(np.isinf(peak))

        # Spectrogram of NaNs must produce valid finite float output
        mel = cap.get_mel_spectrogram(corrupted_audio, rms=rms)
        self.assertFalse(np.isnan(mel).any(), "Mel spectrogram contains NaNs")

        # Detector predict on NaN mel must not crash
        label, conf, cls, is_threat = detector.predict(mel, rms=rms)
        self.assertIn(label, (0, 1))
        self.assertFalse(np.isnan(conf))

    def test_audio_empty_buffer_resilience(self):
        """Verify AudioCapture handles empty buffers without divide-by-zero or crash."""
        cap = AudioCapture()
        empty = np.array([], dtype=np.float32)

        self.assertEqual(cap.get_volume_rms(empty), 0.0)
        self.assertEqual(cap.get_peak_level(empty), 0.0)

        mel = cap.get_mel_spectrogram(empty)
        self.assertEqual(mel.shape, (config.N_MELS, 87))

    def test_vision_corrupted_frame_resilience(self):
        """Verify HumanDetector does not crash on corrupt, malformed, or invalid frames."""
        detector = HumanDetector()

        test_cases = [
            None,
            np.zeros((0, 0, 3), dtype=np.uint8),       # Zero size
            np.zeros((100, 100), dtype=np.uint8),          # 2D Grayscale instead of 3D BGR
            np.zeros((100, 100, 4), dtype=np.uint8),       # 4-channel RGBA
            np.zeros((100, 100, 3), dtype=np.float64),     # Float64 instead of uint8
            "not_an_image",                                # Wrong type entirely
        ]

        for case in test_cases:
            human_found, ann_frame, dets, conf = detector.detect_humans(case)
            self.assertFalse(human_found)
            self.assertEqual(dets, [])
            self.assertEqual(conf, 0.0)

    def test_vlm_corrupted_frame_resilience(self):
        """Verify VLMVerifier handles malformed and invalid frames gracefully."""
        vlm = VLMVerifier(load_captioner=False)

        test_cases = [
            None,
            np.zeros((0, 0, 3), dtype=np.uint8),
            np.zeros((50, 50), dtype=np.uint8),
            "invalid_frame",
        ]

        for case in test_cases:
            is_threat, threat_s, safe_s, prompt, scores = vlm.clip_threat_score(case)
            self.assertFalse(is_threat)
            self.assertEqual(threat_s, 0.0)

    def test_concurrent_database_writes(self):
        """Verify SQLite WAL mode handles rapid concurrent thread insertions without locking errors."""
        num_threads = 10
        events_per_thread = 20
        errors = []

        def worker(thread_idx: int):
            for i in range(events_per_thread):
                try:
                    eid = self.db.insert_event({
                        "mode": "DAY" if thread_idx % 2 == 0 else "NIGHT",
                        "threat_score": 0.5 + (i * 0.01),
                        "reason": f"Thread {thread_idx} Event {i}",
                    })
                    if eid <= 0:
                        errors.append(f"Thread {thread_idx} got invalid id {eid}")
                except Exception as e:
                    errors.append(f"Thread {thread_idx} exception: {e}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent database write errors: {errors}")
        total = self.db.get_total_count()
        self.assertEqual(total, num_threads * events_per_thread)

    def test_database_input_clamping(self):
        """Verify database clamps out-of-bounds scores and sanitizes malformed entries."""
        eid = self.db.insert_event({
            "mode": "INVALID_MODE",
            "threat_score": 999.0,          # Out of bounds (> 1.0)
            "audio_confidence": -5.0,       # Out of bounds (< 0.0)
            "clip_score": float("nan"),      # NaN value
            "person_confidence": 1.5,       # Out of bounds (> 1.0)
        })

        event = self.db.get_event_by_id(eid)
        self.assertIsNotNone(event)
        self.assertEqual(event["mode"], "DAY")            # Fallback to DAY
        self.assertEqual(event["threat_score"], 1.0)      # Clamped to 1.0
        self.assertEqual(event["audio_confidence"], 0.0)  # Clamped to 0.0
        self.assertEqual(event["clip_score"], 0.0)        # Sanitized from NaN to 0.0
        self.assertEqual(event["person_confidence"], 1.0) # Clamped to 1.0


if __name__ == "__main__":
    unittest.main()


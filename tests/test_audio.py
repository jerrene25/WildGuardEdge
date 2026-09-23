import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.audio_capture import AudioCapture
from modules.audio_cnn import AudioCNN, AudioThreatDetector


class TestAudioSubsystem(unittest.TestCase):
    def setUp(self):
        self.cap = AudioCapture()
        self.detector = AudioThreatDetector()
        self.sr = config.SAMPLE_RATE
        self.duration = config.AUDIO_WINDOW_SECONDS

    def tearDown(self):
        self.cap.close()

    def test_audio_cnn_model_loading(self):
        """Verify AudioCNN loads checkpoint weights successfully."""
        self.assertTrue(self.detector.weights_loaded, "AudioCNN weights should be loaded from checkpoint.")
        self.assertIsNotNone(self.detector.model)

    def test_silence_gating_zeros(self):
        """Verify that pure zeros are identified as silence and produce zero threat confidence."""
        silence_window = np.zeros(int(self.sr * self.duration), dtype=np.float32)
        rms = self.cap.get_volume_rms(silence_window)
        self.assertAlmostEqual(rms, 0.0, places=6)
        self.assertTrue(self.cap.is_silence(rms))

        mel = self.cap.get_mel_spectrogram(silence_window, rms=rms)
        label, conf, class_name, is_threat = self.detector.predict(mel, rms=rms)

        self.assertEqual(label, 0, "Silence must be classified as ambient/silence (class 0)")
        self.assertEqual(conf, 0.0, "Silence confidence must be 0.0 to prevent false threat evidence")
        self.assertFalse(is_threat, "Silence must NEVER trigger threat flag")
        self.assertIn("Silence", class_name)

    def test_silence_gating_micronoise(self):
        """
        Verify Bug A fix: Low-level numerical noise (< AUDIO_RMS_FLOOR)
        must NOT trigger threat even after spectrogram normalization.
        """
        noise = np.random.normal(0, 0.0001, int(self.sr * self.duration)).astype(np.float32)
        rms = self.cap.get_volume_rms(noise)
        self.assertLess(rms, config.AUDIO_RMS_FLOOR, f"RMS {rms} should be below floor {config.AUDIO_RMS_FLOOR}")
        self.assertTrue(self.cap.is_silence(rms))

        mel = self.cap.get_mel_spectrogram(noise, rms=rms)
        label, conf, class_name, is_threat = self.detector.predict(mel, rms=rms)

        self.assertEqual(label, 0)
        self.assertEqual(conf, 0.0)
        self.assertFalse(is_threat, "Micro-noise must NEVER trigger threat")

    def test_mel_spectrogram_dimensions(self):
        """Verify log-mel spectrogram has correct shape (n_mels, time_steps)."""
        tone = (0.2 * np.sin(np.linspace(0, 1000, int(self.sr * self.duration)))).astype(np.float32)
        rms = self.cap.get_volume_rms(tone)
        mel = self.cap.get_mel_spectrogram(tone, rms=rms)
        self.assertEqual(mel.shape[0], config.N_MELS)
        self.assertGreater(mel.shape[1], 40)

    def test_audible_signal_inference(self):
        """Verify that audible acoustic signal (RMS >= floor) runs inference without crashing."""
        signal = np.random.normal(0, 0.03, int(self.sr * self.duration)).astype(np.float32)
        rms = self.cap.get_volume_rms(signal)
        self.assertGreaterEqual(rms, config.AUDIO_RMS_FLOOR)

        mel = self.cap.get_mel_spectrogram(signal, rms=rms)
        label, conf, class_name, is_threat = self.detector.predict(mel, rms=rms)

        self.assertIn(label, [0, 1])
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


if __name__ == "__main__":
    unittest.main()


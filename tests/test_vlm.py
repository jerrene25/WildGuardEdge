import unittest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.vlm import VLMVerifier


class TestVLMSubsystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = VLMVerifier(load_captioner=True)

    def test_clip_loaded(self):
        """Verify CLIP ViT-B/32 initialized properly."""
        self.assertTrue(self.verifier.clip_loaded)
        self.assertIsNotNone(self.verifier.clip_model)

    def test_captioner_loaded(self):
        """Verify BLIP captioner initialized properly."""
        self.assertTrue(self.verifier.captioner_loaded)
        self.assertIsNotNone(self.verifier.blip_model)

    def test_clip_scoring_on_image(self):
        """Verify CLIP cosine similarities are bounded in [-1.0, 1.0]."""
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        is_threat, threat_sc, safe_sc, best_p, scores = self.verifier.clip_threat_score(test_img)

        self.assertIsInstance(is_threat, bool)
        self.assertGreater(threat_sc, -1.0)
        self.assertLess(threat_sc, 1.0)
        self.assertGreater(safe_sc, -1.0)
        self.assertLess(safe_sc, 1.0)
        self.assertIn(best_p, self.verifier.all_prompts)
        self.assertEqual(len(scores), len(self.verifier.all_prompts))

    def test_caption_generation(self):
        """Verify BLIP produces a non-empty string caption."""
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        caption = self.verifier.describe_scene(test_img)
        self.assertIsInstance(caption, str)
        self.assertGreater(len(caption), 0)


if __name__ == "__main__":
    unittest.main()


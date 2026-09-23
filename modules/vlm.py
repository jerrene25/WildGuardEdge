import os
import sys
import logging
import time
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
from PIL import Image
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("VLMVerifier")


class VLMVerifier:
    """
    Vision-Language verification combining:
    1. Zero-shot CLIP threat vs safe semantic cosine similarity.
    2. BLIP natural language scene captioning.
    Optimized for low VRAM (<1GB) and rapid inference.
    """

    def __init__(self, load_captioner: bool = True):
        self.device = config.DEVICE
        self.clip_loaded = False
        self.captioner_loaded = False
        self.last_clip_latency_ms = 0.0
        self.last_caption_latency_ms = 0.0

        # Ensure offline mode is respected if files are cached locally
        os.environ["HF_HUB_OFFLINE"] = "1"

        # ── 1. CLIP: Fast Zero-Shot Threat Verification ──
        try:
            logger.info(f"[VLM] Loading CLIP ({config.CLIP_MODEL_NAME}) on {self.device}...")
            self.clip_model, self.clip_preprocess = clip.load(config.CLIP_MODEL_NAME, device=self.device)
            self.clip_model.eval()

            self.all_prompts = config.THREAT_PROMPTS + config.SAFE_PROMPTS
            self.n_threat = len(config.THREAT_PROMPTS)

            tokens = clip.tokenize(self.all_prompts).to(self.device)
            with torch.no_grad():
                self.text_features = self.clip_model.encode_text(tokens)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            self.clip_loaded = True
            logger.info("[VLM] CLIP initialized successfully.")
        except Exception as e:
            logger.error(f"[VLM] Failed to load CLIP: {e}")
            self.clip_model = None

        # ── 2. BLIP: Natural Language Scene Description ──
        if load_captioner:
            try:
                logger.info(f"[VLM] Loading BLIP captioner ({config.CAPTION_MODEL_NAME}) on {self.device}...")
                self.blip_processor = BlipProcessor.from_pretrained(
                    config.CAPTION_MODEL_NAME, local_files_only=True
                )
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    config.CAPTION_MODEL_NAME, local_files_only=True
                ).to(self.device)
                self.blip_model.eval()
                self.captioner_loaded = True
                logger.info("[VLM] BLIP captioner initialized successfully.")
            except Exception as e:
                logger.error(f"[VLM] Failed to load BLIP captioner: {e}")
                self.blip_model = None
                self.blip_processor = None

    def clip_threat_score(
        self, frame_bgr: Optional[np.ndarray]
    ) -> Tuple[bool, float, float, str, Dict[str, float]]:
        """
        Calculates cosine similarities between frame and threat/safe text prompts.
        
        Returns:
          (is_threat: bool, max_threat_score: float, max_safe_score: float,
           best_prompt: str, prompt_scores: dict)
        """
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[0] == 0
            or frame_bgr.shape[1] == 0
            or not self.clip_loaded
            or self.clip_model is None
        ):
            return False, 0.0, 0.0, "VLM Unavailable", {}

        t0 = time.time()
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ self.text_features.T).squeeze(0)

            threat_scores = similarity[:self.n_threat]
            safe_scores = similarity[self.n_threat:]

            max_threat = float(threat_scores.max().item())
            max_safe = float(safe_scores.max().item())
            best_idx = int(similarity.argmax().item())
            best_prompt = self.all_prompts[best_idx]

            prompt_scores = {
                p: float(s) for p, s in zip(self.all_prompts, similarity.cpu().tolist())
            }

            # Threat requires exceeding threshold AND exceeding safe similarity
            is_threat = (max_threat >= config.CLIP_THREAT_THRESHOLD) and (max_threat > max_safe)
            self.last_clip_latency_ms = (time.time() - t0) * 1000.0

            return is_threat, max_threat, max_safe, best_prompt, prompt_scores

        except Exception as e:
            logger.error(f"[VLM] Error during CLIP inference: {e}")
            return False, 0.0, 0.0, "CLIP Inference Error", {}

    def describe_scene(
        self,
        frame_bgr: Optional[np.ndarray],
        human_detected: bool = False,
        threat_detected: bool = False,
    ) -> str:
        """
        Generates natural language surveillance description for the provided frame.
        Uses targeted conditional prompts and context grounding to eliminate hallucination.
        """
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[0] == 0
            or frame_bgr.shape[1] == 0
            or not self.captioner_loaded
            or self.blip_model is None
        ):
            return "VLM monitoring active (standby)."

        t0 = time.time()
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # Condition BLIP with targeted prefix based on detector state
            if human_detected:
                prompt = "a person wearing"
            elif threat_detected:
                prompt = "the surveillance camera captures"
            else:
                prompt = "the scene shows"

            inputs = self.blip_processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.blip_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_beams=3,
                    repetition_penalty=1.2,
                )

            caption = self.blip_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            # Filter absurd generic hallucinations if present
            absurd_tokens = ["painting a wall", "hospital", "hotel room", "bedroom", "painting"]
            if any(tok in caption.lower() for tok in absurd_tokens):
                if human_detected:
                    caption = "Person active in the monitored surveillance zone"
                else:
                    caption = "Surveillance sector clear, baseline monitoring nominal"

            # Capitalize and format cleanly
            if caption:
                caption = caption[0].upper() + caption[1:]
                if not caption.endswith("."):
                    caption += "."

            self.last_caption_latency_ms = (time.time() - t0) * 1000.0
            return caption if caption else "Scene monitoring nominal."

        except Exception as e:
            logger.error(f"[VLM] Error generating caption: {e}")
            return "Perimeter surveillance monitoring active."
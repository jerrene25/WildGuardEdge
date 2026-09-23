import time
import os
import sys
import logging
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.audio_capture import AudioCapture
from modules.audio_cnn import AudioThreatDetector
from modules.vision import HumanDetector
from modules.vlm import VLMVerifier
from modules.database import WildGuardDatabase

logger = logging.getLogger("FusionEngine")


class ThreatState(Enum):
    IDLE = "IDLE"
    SUSPICIOUS = "SUSPICIOUS"
    CONFIRMED = "CONFIRMED"
    ALERTED = "ALERTED"
    COOLDOWN = "COOLDOWN"


class FusionEngine:
    """
    Multimodal fusion engine and threat state machine for WildGuard Edge.
    Synchronizes real microphone and camera feeds, evaluates multimodal evidence
    (Audio CNN + YOLOv8 + CLIP + BLIP), enforces temporal confirmation and cooldowns,
    and writes confirmed events to the authoritative SQLite database.
    """

    def __init__(self, camera_index: Optional[int] = None, mic_index: Optional[int] = None):
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.db = WildGuardDatabase()

        # Audio subsystem
        self.audio_cap = AudioCapture(device_index=mic_index)
        self.audio_detector = AudioThreatDetector()

        # Vision subsystem
        self.vision_detector = HumanDetector()

        # VLM subsystem (CLIP + BLIP)
        self.vlm = VLMVerifier(load_captioner=True)

        # Video capture
        # Zero-latency background camera capture thread
        self._camera_thread = None
        self._camera_lock = threading.Lock()
        self._latest_frame = None
        self._injected_frame: Optional[np.ndarray] = None
        self._init_camera()

        # State Machine tracking
        self.state = ThreatState.IDLE
        self.observation_timestamps: List[float] = []
        self.last_alert_time: float = 0.0
        self.cooldown_until: float = 0.0
        self.alert_count = self.db.get_total_count()

        # Latest asynchronous caption
        self.latest_caption = "Initializing scene description..."
        self.caption_running = False
        self.last_caption_time = 0.0

        # Cycle statistics
        self.last_cycle_time = time.time()
        self.current_fps = 0.0
        self.frame_count = 0
        self.last_clip_result = (False, 0.0, 0.0, "N/A", {})

    def _init_camera(self):
        """Initializes OpenCV video capture and launches background grabber thread."""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                self.camera_online = True
                self._stop_camera = False

                # Launch high-priority background camera reader
                self._camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
                self._camera_thread.start()
                logger.info(f"[Fusion] Camera background worker started at index {self.camera_index} (30 FPS, Buffer=1)")
            else:
                self.camera_online = False
                logger.warning(f"[Fusion] Camera index {self.camera_index} failed to open.")
        except Exception as e:
            self.camera_online = False
            logger.error(f"[Fusion] Camera initialization exception: {e}")

    def _camera_worker(self):
        """Continuously pulls latest frame from sensor at native 30 FPS, eliminating lag accumulation."""
        while not self._stop_camera and self.camera is not None and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    with self._camera_lock:
                        self._latest_frame = frame
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def set_injected_frame(self, frame_bgr: Optional[np.ndarray]):
        """Sets a synthetic or uploaded frame to simulate trail camera video."""
        with self._camera_lock:
            self._injected_frame = frame_bgr.copy() if frame_bgr is not None else None

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Instantly returns the freshest frame from memory in 0.001 ms."""
        with self._camera_lock:
            if self._injected_frame is not None:
                return self._injected_frame.copy()
            if not self.camera_online:
                return None
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _async_caption_worker(self, frame_bgr: np.ndarray, human_detected: bool = False, threat_detected: bool = False):
        """Runs BLIP captioning in a background thread so video inference does not freeze."""
        try:
            caption = self.vlm.describe_scene(frame_bgr, human_detected=human_detected, threat_detected=threat_detected)
            if caption:
                self.latest_caption = caption
        except Exception as e:
            logger.error(f"[Fusion] Background caption error: {e}")
        finally:
            self.caption_running = False

    def _trigger_caption_update(self, frame_bgr: np.ndarray, human_detected: bool = False, threat_detected: bool = False):
        """Dispatches caption generation to a background worker if interval has elapsed."""
        now = time.time()
        if (now - self.last_caption_time >= config.CAPTION_INTERVAL_SECONDS) and not self.caption_running:
            self.caption_running = True
            self.last_caption_time = now
            t = threading.Thread(target=self._async_caption_worker, args=(frame_bgr.copy(), human_detected, threat_detected), daemon=True)
            t.start()

    def _calculate_evidence(
        self,
        audio_threat: bool,
        audio_conf: float,
        rms: float,
        human_found: bool,
        person_conf: float,
        clip_threat: bool,
        clip_score: float,
        night_mode: bool,
    ) -> Tuple[float, str]:
        """
        Computes weighted multimodal threat score in [0.0, 1.0] and rational rationale.
        Near-silence contributes 0.0 audio evidence.
        """
        # 1. Audio evidence
        if rms < config.AUDIO_RMS_FLOOR:
            audio_ev = 0.0
        else:
            audio_ev = audio_conf if audio_threat else 0.0

        # 2. Person detection evidence
        person_ev = person_conf if human_found else 0.0

        # 3. Contextual CLIP evidence
        clip_ev = clip_score if (clip_threat and clip_score >= config.CLIP_THREAT_THRESHOLD) else (clip_score * 0.4)

        # 4. Fusion weighting based on Day/Night Mode
        if night_mode:
            score = (
                config.NIGHT_WEIGHT_AUDIO * audio_ev
                + config.NIGHT_WEIGHT_PERSON * person_ev
                + config.NIGHT_WEIGHT_CLIP * clip_ev
            )
            reasons = []
            if audio_ev > 0.5:
                reasons.append(f"Night threat sound (conf {audio_conf:.2f}, RMS {rms:.4f})")
            if person_ev > 0.4:
                reasons.append(f"Night person spotted (conf {person_conf:.2f})")
            if clip_threat:
                reasons.append(f"Night visual threat context ({clip_score:.2f})")
            reason_str = " + ".join(reasons) if reasons else "Normal night baseline"
        else:
            reasons = []
            score = (
                config.DAY_WEIGHT_PERSON * person_ev
                + config.DAY_WEIGHT_CLIP * clip_ev
                + config.DAY_WEIGHT_AUDIO * audio_ev
            )
            # In Day Mode, an intruder threat requires human presence (loud audio alone -> no automatic threat)
            if not human_found:
                score *= 0.35  # Discount without human in frame so loud audio alone does not trigger false alerts
                if audio_ev > 0.5:
                    reasons.append(f"Acoustic sound noticed without visual human (conf {audio_conf:.2f})")
            else:
                if person_ev > 0.4:
                    reasons.append(f"Person detected (conf {person_conf:.2f})")
                if clip_threat:
                    reasons.append(f"Suspicious context ({clip_score:.2f})")
                if audio_ev > 0.5:
                    reasons.append(f"Corroborating sound (conf {audio_conf:.2f})")

            reason_str = " + ".join(reasons) if reasons else "Normal daytime baseline"

        return float(min(1.0, max(0.0, score))), reason_str

    def _save_snapshot(self, frame_bgr: Optional[np.ndarray], tag: str) -> Optional[str]:
        """Saves current webcam frame to snapshots directory with unique timestamp and strict path confinement."""
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.size == 0
        ):
            return None
        safe_tag = re.sub(r"[^a-zA-Z0-9_-]", "", str(tag)) or "ALERT"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"{safe_tag}_{timestamp}.jpg"
        target_path = (config.SNAPSHOT_DIR / filename).resolve()
        snapshot_root = Path(config.SNAPSHOT_DIR).resolve()

        if not target_path.is_relative_to(snapshot_root):
            logger.warning(f"[Security] Denied snapshot write outside root: {target_path}")
            return None

        cv2.imwrite(str(target_path), frame_bgr)
        return str(target_path)

    def _update_state_machine(
        self,
        instant_threat_detected: bool,
        threat_score: float,
        reason: str,
        frame_bgr: Optional[np.ndarray],
        evidence_dict: Dict[str, Any],
    ) -> Tuple[ThreatState, bool, Optional[Dict[str, Any]]]:
        """
        Evaluates state transitions:
        IDLE -> SUSPICIOUS -> CONFIRMED -> ALERTED -> COOLDOWN -> IDLE
        """
        now = time.time()
        alert_fired = False
        alert_record = None

        # Clean up observations older than confirmation window
        cutoff = now - config.CONFIRMATION_WINDOW_SECONDS
        self.observation_timestamps = [t for t in self.observation_timestamps if t >= cutoff]

        # ── Check COOLDOWN state ──
        if self.state == ThreatState.COOLDOWN:
            if now < self.cooldown_until:
                return ThreatState.COOLDOWN, False, None
            else:
                self.state = ThreatState.IDLE
                self.observation_timestamps.clear()

        # ── State transitions from IDLE / SUSPICIOUS ──
        if instant_threat_detected:
            self.observation_timestamps.append(now)

            if len(self.observation_timestamps) >= config.TEMPORAL_CONFIRMATION_COUNT:
                # Confirmed threat over time!
                self.state = ThreatState.CONFIRMED

                # Generate alert record and snapshot
                snapshot_tag = "NIGHT_ALERT" if evidence_dict.get("night_mode") else "DAY_ALERT"
                snapshot_path = self._save_snapshot(frame_bgr, snapshot_tag)

                # Request immediate caption for the confirmed event
                caption_text = self.latest_caption
                if self.vlm.captioner_loaded and frame_bgr is not None:
                    try:
                        fresh_caption = self.vlm.describe_scene(
                            frame_bgr,
                            human_detected=bool(evidence_dict.get("human_detected")),
                            threat_detected=True,
                        )
                        if fresh_caption and not fresh_caption.startswith("Caption"):
                            caption_text = fresh_caption
                            self.latest_caption = fresh_caption
                    except Exception:
                        pass

                event_data = {
                    "timestamp": datetime.now().isoformat(),
                    "mode": "NIGHT" if evidence_dict.get("night_mode") else "DAY",
                    "threat_score": threat_score,
                    "audio_class": evidence_dict.get("audio_class", "unknown"),
                    "audio_confidence": evidence_dict.get("audio_conf", 0.0),
                    "audio_rms": evidence_dict.get("rms", 0.0),
                    "person_detected": 1 if evidence_dict.get("human_detected") else 0,
                    "person_confidence": evidence_dict.get("person_conf", 0.0),
                    "clip_score": evidence_dict.get("clip_score", 0.0),
                    "caption": caption_text,
                    "snapshot_path": snapshot_path,
                    "reason": reason,
                }

                event_id = self.db.insert_event(event_data)
                event_data["id"] = event_id
                self.alert_count = self.db.get_total_count()

                alert_fired = True
                alert_record = event_data
                self.last_alert_time = now
                self.cooldown_until = now + config.ALERT_COOLDOWN_SECONDS
                self.state = ThreatState.COOLDOWN
                self.observation_timestamps.clear()
            else:
                self.state = ThreatState.SUSPICIOUS
        else:
            # If no instant threat and no recent observations, return to IDLE
            if len(self.observation_timestamps) == 0:
                self.state = ThreatState.IDLE
            else:
                self.state = ThreatState.SUSPICIOUS

        return self.state, alert_fired, alert_record

    def run_one_cycle(self, night_mode: Optional[bool] = None) -> Dict[str, Any]:
        """
        Executes one complete multimodal cycle:
        1. Capture & resample audio window.
        2. Calculate RMS & peak energy.
        3. AudioCNN inference with silence gating.
        4. Grab webcam frame.
        5. YOLO human detection & bounding boxes.
        6. CLIP zero-shot threat verification.
        7. Background BLIP caption trigger.
        8. Multimodal evidence fusion & threat scoring.
        9. Threat state machine transition & authoritative SQLite event logging.
        """
        now = time.time()
        dt = now - self.last_cycle_time
        self.last_cycle_time = now
        if dt > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)

        if night_mode is None:
            night_mode = config.is_night_mode()

        t_cycle_start = time.time()

        # ── 1. Real Audio Pipeline ──
        t_audio_start = time.time()
        window = self.audio_cap.read_window()
        rms = self.audio_cap.get_volume_rms(window)
        peak = self.audio_cap.get_peak_level(window)
        mel = self.audio_cap.get_mel_spectrogram(window, rms=rms)
        t_audio_end = time.time()

        t_audio_cnn_start = time.time()
        audio_label, audio_conf, audio_class, audio_is_threat = self.audio_detector.predict(mel, rms=rms)
        t_audio_cnn_end = time.time()

        # ── 2. Real Camera Pipeline ──
        t_vision_start = time.time()
        frame = self._grab_frame()
        human_found, annotated_frame, detections, person_max_conf = False, frame, [], 0.0

        if frame is not None:
            human_found, annotated_frame, detections, person_max_conf = self.vision_detector.detect_humans(frame)
            self._trigger_caption_update(frame, human_detected=human_found)
        t_vision_end = time.time()

        self.frame_count += 1

        # ── 3. Optimized CLIP Pipeline ──
        t_clip_start = time.time()
        should_run_clip = human_found or audio_is_threat or (self.frame_count % 5 == 0)
        if frame is not None and self.vlm.clip_loaded:
            if should_run_clip:
                clip_threat, clip_threat_score, clip_safe_score, clip_best_prompt, clip_scores = (
                    self.vlm.clip_threat_score(frame)
                )
                self.last_clip_result = (clip_threat, clip_threat_score, clip_safe_score, clip_best_prompt, clip_scores)
            else:
                clip_threat, clip_threat_score, clip_safe_score, clip_best_prompt, clip_scores = self.last_clip_result
        else:
            clip_threat, clip_threat_score, clip_safe_score, clip_best_prompt, clip_scores = (
                False, 0.0, 0.0, "N/A", {}
            )
        t_clip_end = time.time()

        # ── 4. Multimodal Fusion Calculation ──
        t_fusion_start = time.time()
        threat_score, reason = self._calculate_evidence(
            audio_threat=audio_is_threat,
            audio_conf=audio_conf,
            rms=rms,
            human_found=human_found,
            person_conf=person_max_conf,
            clip_threat=clip_threat,
            clip_score=clip_threat_score,
            night_mode=night_mode,
        )

        threshold = config.NIGHT_FUSION_THRESHOLD if night_mode else config.DAY_FUSION_THRESHOLD
        instant_threat = threat_score >= threshold

        # ── 5. State Machine & Event Handling ──
        evidence_dict = {
            "night_mode": night_mode,
            "audio_class": audio_class,
            "audio_conf": audio_conf,
            "rms": rms,
            "human_detected": human_found,
            "person_conf": person_max_conf,
            "clip_score": clip_threat_score,
        }

        current_state, alert_fired, alert_record = self._update_state_machine(
            instant_threat_detected=instant_threat,
            threat_score=threat_score,
            reason=reason,
            frame_bgr=annotated_frame if annotated_frame is not None else frame,
            evidence_dict=evidence_dict,
        )
        t_fusion_end = time.time()

        # Build comprehensive cycle report
        cycle_result = {
            "timestamp": datetime.now().isoformat(),
            "fps": round(self.current_fps, 1),
            "mode": "NIGHT" if night_mode else "DAY",
            "state": current_state.value,
            "threat_score": threat_score,
            "threshold": threshold,
            "reason": reason,
            # Audio metrics
            "rms": rms,
            "peak": peak,
            "audio_class": audio_class,
            "audio_confidence": audio_conf,
            "audio_threat": audio_is_threat,
            "is_silence": self.audio_cap.is_silence(rms),
            "waveform": self.audio_cap.get_waveform_display(300),
            # Vision metrics
            "camera_online": self.camera_online,
            "human_detected": human_found,
            "person_count": len(detections),
            "person_confidence": person_max_conf,
            "detections": detections,
            "raw_frame": frame,
            "annotated_frame": annotated_frame,
            # VLM metrics
            "clip_threat": clip_threat,
            "clip_score": clip_threat_score,
            "clip_safe_score": clip_safe_score,
            "clip_prompt": clip_best_prompt,
            "clip_all_scores": clip_scores,
            "caption": self.latest_caption,
            # Granular Stage Latencies for Conference Telemetry
            "latencies": {
                "audio_dsp_ms": round((t_audio_end - t_audio_start) * 1000.0, 2),
                "audio_cnn_ms": round((t_audio_cnn_end - t_audio_cnn_start) * 1000.0, 2),
                "vision_yolo_ms": round((t_vision_end - t_vision_start) * 1000.0, 2),
                "vlm_clip_ms": round((t_clip_end - t_clip_start) * 1000.0, 2),
                "fusion_fsm_ms": round((t_fusion_end - t_fusion_start) * 1000.0, 2),
                "total_e2e_ms": round((time.time() - t_cycle_start) * 1000.0, 2),
            },
            # Alert metrics
            "alert_fired": alert_fired,
            "alert_record": alert_record,
            "total_alerts": self.alert_count,
            "observation_count": len(self.observation_timestamps),
            "cooldown_remaining": max(0.0, self.cooldown_until - now),
        }

        return cycle_result

    def start(self):
        """Starts audio stream and engine monitoring."""
        self.audio_cap.open_stream()
        logger.info("[Fusion] FusionEngine started.")

    def stop(self):
        """Gracefully releases camera and audio hardware resources."""
        self._stop_camera = True
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=0.5)
        self.audio_cap.close()
        if self.camera:
            self.camera.release()
            self.camera = None
        self.camera_online = False
        logger.info("[Fusion] FusionEngine stopped.")
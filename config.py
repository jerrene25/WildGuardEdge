import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger("WildGuardConfig")

class ConfigurationError(ValueError):
    """Raised when configuration values are malformed or out of safe operating bounds."""
    pass

# ── Base Paths ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

def _load_env_file(env_path: Path):
    """Zero-dependency .env loader that populates os.environ without overwriting existing vars."""
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        logger.warning(f"[Config] Could not parse .env file: {e}")

# Load .env if present
_load_env_file(BASE_DIR / ".env")

def _get_env_float(key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
        if min_val is not None and val < min_val:
            raise ConfigurationError(f"Config '{key}' value {val} is below minimum allowed {min_val}")
        if max_val is not None and val > max_val:
            raise ConfigurationError(f"Config '{key}' value {val} exceeds maximum allowed {max_val}")
        return val
    except ValueError as e:
        raise ConfigurationError(f"Config '{key}' must be a valid float. Received: '{raw}'") from e

def _get_env_int(key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
        if min_val is not None and val < min_val:
            raise ConfigurationError(f"Config '{key}' value {val} is below minimum allowed {min_val}")
        if max_val is not None and val > max_val:
            raise ConfigurationError(f"Config '{key}' value {val} exceeds maximum allowed {max_val}")
        return val
    except ValueError as e:
        raise ConfigurationError(f"Config '{key}' must be a valid integer. Received: '{raw}'") from e

# ── Execution Environment ─────────────────────────
ENV = os.getenv("WILDGUARD_ENV", "production").lower()
if ENV not in ("development", "production", "testing"):
    raise ConfigurationError(f"WILDGUARD_ENV must be 'development', 'production', or 'testing'. Got '{ENV}'")

# ── Directory Structure ───────────────────────────
ALERTS_DIR = BASE_DIR / "alerts"
SNAPSHOT_DIR = ALERTS_DIR / "snapshots"
AUDIO_CLIP_DIR = ALERTS_DIR / "audio_clips"
ALERT_LOG_PATH = ALERTS_DIR / "alert_log.json"

db_env = os.getenv("WILDGUARD_DB_PATH")
DB_PATH = Path(db_env) if db_env else ALERTS_DIR / "wildguard.db"

RETRAINING_DIR = BASE_DIR / "retraining_data"
CONFIRMED_DIR = RETRAINING_DIR / "confirmed"
FALSE_ALARM_DIR = RETRAINING_DIR / "false_alarm"

MODULES_DIR = BASE_DIR / "modules"
AUDIO_MODEL_PATH = MODULES_DIR / "audio_cnn_weights.pth"

# Ensure runtime directories exist securely
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_CLIP_DIR.mkdir(parents=True, exist_ok=True)
CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
FALSE_ALARM_DIR.mkdir(parents=True, exist_ok=True)

# ── Hardware & Compute Device ─────────────────────
device_override = os.getenv("WILDGUARD_DEVICE", "").lower()
if device_override in ("cuda", "cpu"):
    DEVICE = device_override if (device_override != "cuda" or torch.cuda.is_available()) else "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE_NAME = torch.cuda.get_device_name(0) if (DEVICE == "cuda" and torch.cuda.is_available()) else "CPU"

# ── Audio Settings & Validated Thresholds ─────────
SAMPLE_RATE = 22050              # Target rate for AudioCNN
CAPTURE_SAMPLE_RATE = 44100      # Hardware microphone native capture rate
AUDIO_WINDOW_SECONDS = 2.0       # Duration of rolling audio buffer
AUDIO_HOP_SECONDS = 0.5          # Inference hop interval
N_MELS = 128                     # Mel frequency bins

AUDIO_THREAT_THRESHOLD = _get_env_float("WILDGUARD_AUDIO_THREAT_THRESHOLD", 0.60, min_val=0.1, max_val=0.99)
AUDIO_RMS_FLOOR = _get_env_float("WILDGUARD_AUDIO_RMS_FLOOR", 0.0002, min_val=0.00001, max_val=0.1)

# ── Vision Settings ───────────────────────────────
YOLO_MODEL_NAME = str(BASE_DIR / "yolov8n.pt")
YOLO_CONF_THRESHOLD = _get_env_float("WILDGUARD_YOLO_CONF_THRESHOLD", 0.45, min_val=0.05, max_val=0.95)
CAMERA_INDEX = _get_env_int("WILDGUARD_CAMERA_INDEX", 0, min_val=0, max_val=10)

# ── VLM (CLIP + Captioning) Settings ──────────────
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_THREAT_THRESHOLD = _get_env_float("WILDGUARD_CLIP_THREAT_THRESHOLD", 0.28, min_val=0.10, max_val=0.90)
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
CAPTION_INTERVAL_SECONDS = 2.0   # Background caption refresh interval

THREAT_PROMPTS = [
    "a person carrying a weapon in a forest",
    "a poacher setting a trap",
    "an intruder trespassing at night",
    "a person handling an animal trap or snare",
]
SAFE_PROMPTS = [
    "an empty forest scene",
    "a park ranger on patrol",
    "wildlife walking peacefully",
    "an ordinary person walking outdoors",
]

# ── State Machine & Fusion Settings ───────────────
TEMPORAL_CONFIRMATION_COUNT = 3  # Positive observations required before alert triggers
CONFIRMATION_WINDOW_SECONDS = 5.0  # Time window for temporal confirmation
ALERT_COOLDOWN_SECONDS = 10.0      # Minimum seconds between alerts to prevent duplicate spamming

# Fusion evidence weights
DAY_WEIGHT_PERSON = 0.40
DAY_WEIGHT_CLIP = 0.30
DAY_WEIGHT_AUDIO = 0.30
DAY_FUSION_THRESHOLD = _get_env_float("WILDGUARD_DAY_FUSION_THRESHOLD", 0.55, min_val=0.1, max_val=0.95)

NIGHT_WEIGHT_AUDIO = 0.65
NIGHT_WEIGHT_PERSON = 0.20
NIGHT_WEIGHT_CLIP = 0.15
NIGHT_FUSION_THRESHOLD = _get_env_float("WILDGUARD_NIGHT_FUSION_THRESHOLD", 0.45, min_val=0.1, max_val=0.95)

# Aliases for robust code access
DAY_VISION_WEIGHT = DAY_WEIGHT_PERSON
DAY_CLIP_WEIGHT = DAY_WEIGHT_CLIP
DAY_AUDIO_WEIGHT = DAY_WEIGHT_AUDIO
NIGHT_VISION_WEIGHT = NIGHT_WEIGHT_PERSON
NIGHT_CLIP_WEIGHT = NIGHT_WEIGHT_CLIP
NIGHT_AUDIO_WEIGHT = NIGHT_WEIGHT_AUDIO

# ── Day/Night Adaptive Mode ───────────────────────
NIGHT_MODE_START_HOUR = _get_env_int("WILDGUARD_NIGHT_START_HOUR", 20, min_val=0, max_val=23)
NIGHT_MODE_END_HOUR = _get_env_int("WILDGUARD_NIGHT_END_HOUR", 6, min_val=0, max_val=23)

def is_night_mode() -> bool:
    now_hour = datetime.now().hour
    return now_hour >= NIGHT_MODE_START_HOUR or now_hour < NIGHT_MODE_END_HOUR

# ── System ────────────────────────────────────────
DASHBOARD_REFRESH_MS = 100
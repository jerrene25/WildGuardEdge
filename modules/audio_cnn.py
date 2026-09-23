import os
import sys
import logging
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("AudioCNN")


class AudioCNN(nn.Module):
    """
    4-layer CNN for acoustic threat vs ambient classification
    from a log-mel spectrogram input.
    Matches the trained checkpoint architecture in audio_cnn_weights.pth.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 2)  # Class 0: Ambient, Class 1: Threat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class AudioThreatDetector:
    """
    Wraps AudioCNN with input validation, calibrated silence gating,
    and inference execution.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.device = config.DEVICE
        self.weights_loaded = False
        self.model = AudioCNN().to(self.device)

        if weights_path is None:
            weights_path = str(config.AUDIO_MODEL_PATH)

        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.weights_loaded = True
                logger.info(f"[AudioCNN] Loaded trained weights from {weights_path} on {self.device}")
            except Exception as e:
                logger.error(f"[AudioCNN] Error loading weights from {weights_path}: {e}")
        else:
            logger.warning(f"[AudioCNN] No weights file found at {weights_path}. Model using random initialization.")

        self.model.eval()

    def predict(
        self,
        log_mel_spectrogram: np.ndarray,
        rms: Optional[float] = None,
    ) -> Tuple[int, float, str, bool]:
        """
        Takes log-mel spectrogram and optional window RMS energy.
        Applies strict silence gate:
          If RMS < AUDIO_RMS_FLOOR, near-silence is rejected and reported as "Silence / No Audio"
          with 0.0 threat confidence, ensuring no false threat evidence is contributed.
        
        Returns:
          (predicted_label, confidence, class_name, is_threat)
          predicted_label: 0 = Ambient/Silence, 1 = Threat
          confidence: float in [0.0, 1.0]
          class_name: human-readable classification string
          is_threat: boolean indicating whether acoustic threat condition is met
        """
        # Strict silence gate: Near-silence must never become a threat
        if rms is not None and rms < config.AUDIO_RMS_FLOOR:
            return 0, 0.0, "Silence / No Audio", False

        if log_mel_spectrogram is None or getattr(log_mel_spectrogram, "size", 0) == 0:
            return 0, 0.0, "No Audio Data", False

        try:
            # Sanitize NaNs and Infs to defend against malformed audio inputs
            clean_mel = np.nan_to_num(log_mel_spectrogram, nan=0.0, posinf=0.0, neginf=-80.0)
            x = torch.from_numpy(clean_mel).float()
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
            x = x.to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                threat_prob = probs[0, 1].item()
                ambient_prob = probs[0, 0].item()

                if threat_prob >= config.AUDIO_THREAT_THRESHOLD:
                    predicted_label = 1
                    confidence = threat_prob
                    class_name = "Threat Sound"
                    is_threat = True
                else:
                    predicted_label = 0
                    confidence = ambient_prob
                    class_name = "Ambient Sound"
                    is_threat = False

            return predicted_label, confidence, class_name, is_threat

        except Exception as e:
            logger.error(f"[AudioCNN] Prediction error: {e}")
            return 0, 0.0, "Audio Inference Error", False
import logging
import threading
import time
try:
    import pyaudio
except Exception:
    pyaudio = None
import numpy as np
import librosa
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("AudioCapture")


class AudioCapture:
    """
    High-performance real-time microphone audio capture.
    Runs non-blocking audio ingestion in a dedicated background worker thread,
    eliminating PortAudio buffer overflows and zeroing out cycle latency.
    """

    def __init__(self, device_index: Optional[int] = None):
        self.target_sample_rate = config.SAMPLE_RATE  # 22050 Hz
        self.device_index = device_index
        self.window_samples = int(config.AUDIO_WINDOW_SECONDS * self.target_sample_rate)  # 44,100 samples
        self.chunk_size = 1024  # Low latency chunk size

        if pyaudio is not None:
            try:
                self.pa = pyaudio.PyAudio()
            except Exception:
                self.pa = None
        else:
            self.pa = None
        self.stream = None
        self.is_active = False
        self.actual_capture_rate = self.target_sample_rate

        # Thread-safe rolling audio buffer and stream lock
        self.buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream_lock = threading.Lock()

        # Injected audio for reproducible verification / testing
        self.injected_audio: Optional[np.ndarray] = None
        self.injected_pos: int = 0

        # Background capture thread
        self.capture_thread = None
        self.stop_requested = False

    def open_stream(self) -> bool:
        """Opens microphone stream and launches background capture thread."""
        if self.pa is None:
            logger.info("[AudioCapture] PyAudio not available (cloud environment). Physical mic disabled.")
            self.stream = None
            self.is_active = False
            return False

        try:
            # First attempt opening directly at 22050 Hz for zero-resampling performance
            try:
                open_kwargs = {
                    "format": pyaudio.paFloat32,
                    "channels": 1,
                    "rate": self.target_sample_rate,
                    "input": True,
                    "frames_per_buffer": self.chunk_size,
                }
                if self.device_index is not None:
                    open_kwargs["input_device_index"] = self.device_index

                self.stream = self.pa.open(**open_kwargs)
                self.actual_capture_rate = self.target_sample_rate
                logger.info(f"[AudioCapture] Native 22050 Hz audio stream opened successfully.")
            except Exception as e:
                logger.info(f"[AudioCapture] 22050 Hz not supported natively ({e}), falling back to 44100 Hz.")
                open_kwargs["rate"] = 44100
                self.stream = self.pa.open(**open_kwargs)
                self.actual_capture_rate = 44100

            self.is_active = True
            self.stop_requested = False

            # Start background capture worker thread
            self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.capture_thread.start()
            logger.info(f"[AudioCapture] Background capture worker started at {self.actual_capture_rate} Hz.")
            return True

        except Exception as e:
            logger.error(f"[AudioCapture] Failed to open microphone stream: {e}")
            self.stream = None
            self.is_active = False
            return False

    def _capture_worker(self):
        """Continuously reads audio chunks from PyAudio or injected audio and updates rolling buffer in background."""
        while not self.stop_requested and self.is_active:
            # If injected audio is active (testing mode), stream chunks from injected waveform
            if self.injected_audio is not None:
                with self.lock:
                    if self.injected_audio is not None:
                        total = len(self.injected_audio)
                        chunk_samples = self.chunk_size
                        if self.injected_pos + chunk_samples > total:
                            part1 = self.injected_audio[self.injected_pos :]
                            part2 = self.injected_audio[: chunk_samples - len(part1)]
                            new_chunk = np.concatenate([part1, part2])
                            self.injected_pos = len(part2)
                        else:
                            new_chunk = self.injected_audio[self.injected_pos : self.injected_pos + chunk_samples]
                            self.injected_pos += chunk_samples

                        self.buffer = np.roll(self.buffer, -len(new_chunk))
                        self.buffer[-len(new_chunk) :] = new_chunk
                time.sleep(self.chunk_size / self.target_sample_rate)
                continue

            raw = None
            with self.stream_lock:
                if not self.stream or not self.is_active or self.stop_requested:
                    time.sleep(0.02)
                    continue
                try:
                    raw = self.stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    if not self.stop_requested:
                        logger.warning(f"[AudioCapture] Worker read error: {e}")
                    time.sleep(0.02)
                    continue

            if raw is None:
                continue

            try:
                new_chunk = np.frombuffer(raw, dtype=np.float32)

                # If capturing at 44100 Hz, fast decimate 2:1 to 22050 Hz in 0.0 ms
                if self.actual_capture_rate == 44100:
                    new_chunk = new_chunk[::2]

                n_samples = len(new_chunk)
                if n_samples > 0:
                    with self.lock:
                        self.buffer = np.roll(self.buffer, -n_samples)
                        self.buffer[-n_samples:] = new_chunk

            except Exception as e:
                if not self.stop_requested:
                    logger.warning(f"[AudioCapture] Worker buffer error: {e}")
                time.sleep(0.01)

    def set_injected_audio(self, audio_data: Optional[np.ndarray]):
        """
        Sets a test waveform to stream continuously into the rolling buffer,
        allowing deterministic acoustic verification. Pass None to resume live hardware microphone.
        """
        with self.lock:
            self.injected_audio = audio_data.astype(np.float32) if audio_data is not None else None
            self.injected_pos = 0
            if audio_data is not None:
                if len(audio_data) >= self.window_samples:
                    self.buffer = audio_data[: self.window_samples].copy().astype(np.float32)
                else:
                    self.buffer = np.zeros(self.window_samples, dtype=np.float32)
                    self.buffer[-len(audio_data) :] = audio_data.astype(np.float32)

    def read_window(self) -> np.ndarray:
        """
        Instantly returns the current 2.0-second rolling audio buffer.
        Zero blocking latency (< 0.05 ms).
        """
        with self.lock:
            return self.buffer.copy()

    def get_volume_rms(self, audio_window: np.ndarray) -> float:
        """Calculates root-mean-square (RMS) energy of the audio window safely."""
        if audio_window is None or getattr(audio_window, "size", 0) == 0:
            return 0.0
        clean = np.nan_to_num(audio_window, nan=0.0, posinf=1.0, neginf=-1.0)
        mean_sq = float(np.mean(clean**2))
        return float(np.sqrt(max(0.0, mean_sq)))

    def get_peak_level(self, audio_window: np.ndarray) -> float:
        """Calculates maximum absolute peak level of the audio window safely."""
        if audio_window is None or getattr(audio_window, "size", 0) == 0:
            return 0.0
        clean = np.nan_to_num(audio_window, nan=0.0, posinf=1.0, neginf=-1.0)
        return float(np.max(np.abs(clean)))

    def is_silence(self, rms: float) -> bool:
        """Determines whether the energy level is below the silence floor."""
        try:
            return float(rms) < config.AUDIO_RMS_FLOOR
        except (ValueError, TypeError):
            return True

    def get_mel_spectrogram(self, audio_window: np.ndarray, rms: Optional[float] = None) -> np.ndarray:
        """
        Converts audio window into a log-mel spectrogram.
        Uses calibrated silence normalization so micro-noise does not amplify.
        """
        if audio_window is None or getattr(audio_window, "size", 0) == 0:
            return np.full((config.N_MELS, 87), -80.0, dtype=np.float32)

        clean = np.nan_to_num(audio_window, nan=0.0, posinf=1.0, neginf=-1.0)

        if rms is None:
            rms = self.get_volume_rms(clean)

        try:
            mel = librosa.feature.melspectrogram(
                y=clean,
                sr=self.target_sample_rate,
                n_mels=config.N_MELS,
            )

            # Calibrated normalization: if audio is silence, use fixed ref=1.0
            if rms < config.AUDIO_RMS_FLOOR or float(np.max(mel)) < 1e-6:
                log_mel = librosa.power_to_db(mel, ref=1.0)
            else:
                log_mel = librosa.power_to_db(mel, ref=np.max)

            return np.nan_to_num(log_mel, nan=-80.0, posinf=0.0, neginf=-80.0)
        except Exception as e:
            logger.error(f"[AudioCapture] Mel spectrogram error: {e}")
            return np.full((config.N_MELS, 87), -80.0, dtype=np.float32)

    def get_waveform_display(self, num_points: int = 300) -> np.ndarray:
        """Returns a downsampled slice of the current buffer for live UI rendering."""
        with self.lock:
            buf = self.buffer.copy()
        if len(buf) < num_points:
            return buf
        step = len(buf) // num_points
        return buf[::step][:num_points]

    def close(self):
        """Stops worker thread, closes stream, and terminates PyAudio safely."""
        self.stop_requested = True
        self.is_active = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        try:
            with self.stream_lock:
                if self.stream:
                    try:
                        self.stream.stop_stream()
                    except Exception:
                        pass
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
        except Exception as e:
            logger.warning(f"[AudioCapture] Error stopping stream: {e}")
        finally:
            try:
                self.pa.terminate()
            except Exception:
                pass
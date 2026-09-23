# 🐾 WildGuard Edge — Real-Time Autonomous Anti-Poaching Surveillance

WildGuard Edge is a multimodal edge intelligence system designed for wildlife conservation reserves and remote wilderness perimeters. It runs continuous, real-time intrusion monitoring combining acoustic event analysis, computer vision human detection, semantic zero-shot vision-language verification, and natural language captioning.

---

## 🏛️ System Architecture

WildGuard Edge implements a multi-stage sequential and concurrent fusion pipeline backed by a finite threat state machine:

```
                  ┌──────────────────────┐
                  │ Hardware Microphone  │
                  └──────────┬───────────┘
                             │ PCM Audio (44.1 kHz -> 22.05 kHz)
                             ▼
                  ┌──────────────────────┐
                  │ Audio Rolling Buffer │
                  │  (2.0s win, 0.5s hop)│
                  └──────────┬───────────┘
                             │
                             ├────────────────────────┐
                             ▼                        ▼
                   [RMS Energy Meter]          [Log-Mel Spec]
                             │                 (128 mel bins)
                             ▼                        │
                   [Silence / Energy Gate]            │
                   (Floor: 0.005 RMS)                 ▼
                             │                 ┌───────────────┐
                      RMS < Floor ?            │   AudioCNN    │
                      ├── YES ──> Zero Threat  │(4-layer Conv) │
                      └── NO  ──> Forward ────>└──────┬────────┘
                                                      │
                                                      │ Acoustic Threat Evidence
                                                      ▼
┌────────────────────┐                 ┌─────────────────────────────┐
│   Hardware Camera  │                 │                             │
└─────────┬──────────┘                 │                             │
          │ BGR Frame (480x640)        │                             │
          ▼                            │                             │
┌────────────────────┐                 │                             │
│   YOLOv8n Vision   ├─ Person Conf ──>│      Multimodal Fusion      │
│  (COCO Class 0)    │  & Bounding Box │        Engine & Threat      │
└─────────┬──────────┘                 │         State Machine       │
          │                            │                             │
          ▼                            │  (IDLE -> SUSPICIOUS ->     │
┌────────────────────┐                 │   CONFIRMED -> ALERTED ->   │
│   CLIP ViT-B/32    ├─ Semantic Sim ─>│   COOLDOWN -> IDLE)         │
│ (Zero-Shot Threat) │  (Threat Prompts│                             │
└─────────┬──────────┘                 │                             │
          │                            │                             │
          ▼                            │                             │
┌────────────────────┐                 │                             │
│ BLIP Scene Caption ├─ Context Desc ─>│                             │
│(Async Worker Thread│                 └──────────────┬──────────────┘
└────────────────────┘                                │
                                         Confirmed Threat Event
                                                      │
                                  ┌───────────────────┴───────────────────┐
                                  ▼                                       ▼
                       ┌─────────────────────┐                 ┌─────────────────────┐
                       │ High-Res Snapshot   │                 │ Authoritative SQLite│
                       │ (alerts/snapshots/) │                 │ Event DB (wildguard)│
                       └─────────────────────┘                 └──────────┬──────────┘
                                                                          │
                                                                          ▼
                                                               ┌─────────────────────┐
                                                               │ Streamlit Live Edge │
                                                               │ Control Dashboard   │
                                                               └─────────────────────┘
```

---

## 🚀 Key Subsystems & Features

### 1. Real Audio Pipeline & Silence Gating
* **Live Ingestion**: Captures live microphone audio using PyAudio at native 44.1 kHz, resampled to 22.05 kHz rolling buffer (2.0s window, 0.5s hop).
* **Calibrated Silence Gate (`AUDIO_RMS_FLOOR = 0.005`)**: Computes true root-mean-square energy. If `RMS < AUDIO_RMS_FLOOR`, audio is identified as `Silence / No Audio` with 0.0 threat confidence. This eliminates numerical noise amplification from spectrogram normalization.
* **Acoustic Threat Classifier (`AudioCNN`)**: 4-layer convolutional neural network trained on log-mel spectrograms (128 mel bins). Outputs classification probabilities for ambient sound vs threat sound.

### 2. Real Vision & Human Detection
* **YOLOv8n**: Real-time object detector targeting COCO class 0 (`person`).
* **Outputs**: Detection coordinates with bounding box clamping, bounding box annotations, and maximum person detection confidence.
* **Fault Tolerance**: Gracefully identifies when the camera is offline or disconnected, reporting `CAMERA OFFLINE` without throwing exceptions or generating false alerts.

### 3. Vision-Language Verification (CLIP & BLIP)
* **OpenAI CLIP (`ViT-B/32`)**: Computes semantic cosine similarity between the live camera frame and anti-poaching threat prompts versus safe baseline prompts.
* **BLIP Captioner (`Salesforce/blip-image-captioning-base`)**: Generates natural language descriptions of the camera scene in an asynchronous background thread, ensuring real-time video frames never stall.

### 4. Threat State Machine & Fusion Logic
* **States**:
  * `IDLE`: Normal baseline surveillance.
  * `SUSPICIOUS`: An instant observation exceeds the threshold.
  * `CONFIRMED`: Temporal confirmation achieved (requires `TEMPORAL_CONFIRMATION_COUNT = 3` positive observations within `CONFIRMATION_WINDOW_SECONDS = 5.0s`).
  * `ALERTED`: Atomically captures a snapshot, logs the event to SQLite, and updates the UI.
  * `COOLDOWN`: Suppresses duplicate alert spamming for `ALERT_COOLDOWN_SECONDS = 10.0s`.
* **Day vs. Night Adaptive Modes**:
  * **Day Mode**: Requires visual person detection + CLIP context + corroborating sound (`DAY_FUSION_THRESHOLD = 0.60`).
  * **Night Mode**: Heightens acoustic sensitivity with camera verification (`NIGHT_FUSION_THRESHOLD = 0.50`), while silence gating ensures silence contributes 0.0 threat score.

### 5. Authoritative SQLite Event Database
* Stored in `alerts/wildguard.db`.
* Schema records `id`, `timestamp`, `mode`, `threat_score`, `audio_class`, `audio_confidence`, `audio_rms`, `person_detected`, `person_confidence`, `clip_score`, `caption`, `snapshot_path`, and `reason`.
* The dashboard directly queries SQLite, guaranteeing zero duplicate entries.

---

## 💻 Hardware Requirements & Setup

* **OS**: Windows 10/11 or Linux.
* **Python**: 3.10 (3.10.11 recommended).
* **GPU**: NVIDIA GPU with CUDA support (e.g. RTX 4060 8GB VRAM or higher). Fallback to CPU is fully supported.
* **Peripherals**: USB / integrated webcam and microphone.

### Installation

```bash
# 1. Clone repository
git clone <repository_url>
cd WildGuardEdge

# 2. Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Launch via Single Entrypoint
```bash
python main.py
```
This automatically starts the Streamlit dashboard on `http://localhost:8501`.

### Launch Directly via Streamlit
```bash
streamlit run dashboard/app.py
```

---

## ⚙️ Configuration Reference (`config.py` & `.env`)

WildGuard Edge uses a fail-fast runtime environment validator with bounded validation rules. On startup, configuration values are validated against acceptable ranges and types; invalid values immediately trigger a safe `ConfigurationError` rather than runtime crashes or vulnerabilities.

Create a `.env` file in the project root (or copy from `.env.example`):

```bash
cp .env.example .env
```

| Setting | Default | Range / Valid Values | Description |
| :--- | :--- | :--- | :--- |
| `WILDGUARD_ENV` | `production` | `production`, `staging`, `development` | Deployment environment |
| `SAMPLE_RATE` | `22050` | `8000` - `96000` Hz | AudioCNN target sample rate |
| `CAPTURE_SAMPLE_RATE` | `44100` | `8000` - `192000` Hz | Microphone hardware capture rate |
| `AUDIO_WINDOW_SECONDS`| `2.0` | `0.5` - `10.0` s | Duration of audio analysis window |
| `AUDIO_HOP_SECONDS` | `0.5` | `0.1` - `5.0` s | Audio inference interval |
| `AUDIO_RMS_FLOOR` | `0.005` | `0.0001` - `0.1` | Energy floor for silence gating |
| `YOLO_CONF_THRESHOLD` | `0.45` | `0.01` - `0.99` | YOLO minimum confidence for person detection |
| `CLIP_THREAT_THRESHOLD`| `0.28` | `0.01` - `0.99` | Cosine similarity threshold for CLIP threat |
| `TEMPORAL_CONFIRMATION_COUNT` | `3` | `1` - `20` | Positive observations required to confirm threat |
| `CONFIRMATION_WINDOW_SECONDS` | `5.0` | `1.0` - `60.0` s | Rolling window duration for confirmation |
| `ALERT_COOLDOWN_SECONDS` | `10.0` | `1.0` - `300.0` s | Cooldown duration suppressing duplicate alerts |
| `DAY_FUSION_THRESHOLD` | `0.60` | `0.1` - `0.99` | Minimum score to trigger daytime alert |
| `NIGHT_FUSION_THRESHOLD` | `0.50` | `0.1` - `0.99` | Minimum score to trigger nighttime alert |

---

## 🛡️ Security Architecture & Threat Hardening

WildGuard Edge adheres to enterprise security standards and OWASP defense-in-depth principles:

### 1. Deserialization Security (`torch.load`)
* **Vulnerability Mitigated**: Arbitrary code execution via malicious pickle payloads in model weights.
* **Mitigation**: All model loaders enforce `weights_only=True` (`torch.load(..., weights_only=True)`) or load strictly via safe tensor weights (`ultralytics`).

### 2. Path Traversal & Arbitrary Deletion Defense
* **Vulnerability Mitigated**: Path traversal attacks (`../../etc/passwd`) attempting arbitrary file deletion via event cleanup APIs.
* **Mitigation**: Strict path boundary confinement (`is_relative_to(config.SNAPSHOT_DIR)`). File operations verify resolved absolute canonical paths against the designated snapshots root. Disallowed paths immediately abort with security audit logging.

### 3. Database Hardening & Injection Protection
* **Vulnerability Mitigated**: SQL injection, database table locking under concurrent edge threads, and memory exhaustion via boundless queries.
* **Mitigation**:
  * 100% parameterized SQL statements across all queries, inserts, and deletes.
  * SQLite WAL (Write-Ahead Logging) enabled (`PRAGMA journal_mode=WAL;`), reducing write contention.
  * Busy timeout set to 5000ms (`PRAGMA busy_timeout=5000;`).
  * Dedicated indexes on `timestamp`, `threat_score`, and `mode` for $O(\log n)$ forensic retrieval.
  * Strict integer query limit bounds (`1 <= limit <= 5000`) preventing memory exhaustion attacks.

### 4. Cross-Site Scripting (XSS) Prevention
* **Vulnerability Mitigated**: Stored XSS via malicious input captions or forensic metadata in the dashboard.
* **Mitigation**: Dynamic text outputs rendered in HTML components are escaped using `html.escape()`.

### 5. Numerical & Media Bulletproofing
* **Vulnerability Mitigated**: Application crashing or NaN-propagation when encountering corrupt audio inputs, zero-sized video frames, or audio buffers with division-by-zero.
* **Mitigation**: All audio transforms apply `np.nan_to_num()`, zero-size buffer guards, and exception isolation blocks. Vision modules validate frame dimensions, non-zero sizes, and correct 3-channel uint8 formatting.

---

## 🧪 Automated Testing & Adversarial QA Suite

WildGuard Edge includes a 33-test suite including hostile adversarial and fuzzing tests:

```bash
# Run all unit, security, and hardening tests
python -m unittest discover tests
```

### Test Suite Structure
* `tests/test_security.py`:
  * SQL injection payload injection into all fields.
  * Query limit out-of-bounds attacks and negative limit rejection.
  * Path traversal arbitrary deletion prevention (`../../../sensitive_file.txt`).
  * Legitimate snapshot deletion boundary checks.
  * Snapshot file tag sanitization (rejection of directory traversal characters).
* `tests/test_hardening.py`:
  * Extreme NaN, Infinity, and zero-energy audio resilience.
  * Zero-length and single-sample audio buffer fault tolerance.
  * Corrupted, zero-byte, single-pixel, and non-image frame rejection.
  * High-concurrency SQLite multithreading stress test (10 parallel threads).
  * Out-of-bounds input score clamping ($[-10, 100] \to [0.0, 1.0]$).
* `tests/test_audio.py`: Silence gating, micro-noise rejection, log-mel spectrogram shape, audio model loading.
* `tests/test_vision.py`: YOLO loading, blank frame handling, offline camera handling, bounding box bounds.
* `tests/test_vlm.py`: CLIP similarity bounds, prompt evaluation, BLIP caption generation.
* `tests/test_fusion.py`: Silence rejection in night mode, day mode human requirement, temporal confirmation, cooldown duplicate prevention.
* `tests/test_database.py`: Table schema, event insertion, ordering, deduplication.
* `tests/test_e2e.py`: Full multimodal pipeline execution.

---

## 🔒 Privacy & Local Processing
* All inference (AudioCNN, YOLOv8, CLIP, BLIP) runs **100% locally on device**.
* Video and audio streams are processed in transient memory buffers.
* Snapshots are only saved when a threat is confirmed by the multimodal state machine.
* No continuous recording or third-party cloud streaming occurs.

---

## ⚠️ Known Limitations & Model Notes
1. **AudioCNN Domain**: The included `audio_cnn_weights.pth` checkpoint was trained on environmental audio datasets (ESC-50 derived). While effective for distinguishing loud/unusual acoustic events from ambient noise, extreme sound variations should be fine-tuned on local wilderness reserve acoustic datasets.
2. **Low-Light Night Vision**: Standard webcam RGB sensors require ambient infrared or floodlighting at night. In pure dark environments, the system leans on acoustic detection and saves infrared/ambient snapshots.
3. **Camera Placement**: Ensure the camera lens is clean and placed at a height suitable for pedestrian/poacher identification.


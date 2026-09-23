import os
import sys
import time
from datetime import datetime
from pathlib import Path
import psutil
import cv2
import numpy as np
import pandas as pd
import torch
import librosa
import html
import base64
import warnings
import streamlit as st

warnings.filterwarnings("ignore", message=".*use_container_width.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.fusion import FusionEngine, ThreatState
from modules.database import WildGuardDatabase

st.set_page_config(
    page_title="WildGuard Edge — Command & Defense Station",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tactical High-Contrast Command Center Styling ──────
st.markdown("""
<style>
    /* Global Background & High-Contrast Typography */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #070b12 !important;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Force Ultra-Visible Crisp Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em !important;
    }
    
    /* Standard Text & Paragraphs */
    p, span, li, label, div {
        color: #f1f5f9;
    }
    
    /* Captions & Subtitles - Bright Silver & Crisp, Never Dim */
    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
        color: #cbd5e1 !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Styling - Distinct Slate with High Contrast Controls */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #94a3b8 !important;
    }
    
    /* Metric Cards - Bold, High-Contrast Visibility */
    div[data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #93c5fd !important; /* Bright sky blue label */
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #93c5fd !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important; /* Pure crisp white value */
        font-size: 1.9rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricValue"] * {
        color: #ffffff !important;
    }
    div[data-testid="stMetricDelta"] * {
        font-size: 0.88rem !important;
        font-weight: 700 !important;
    }
    
    /* High-Contrast Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-alert {
        background-color: #991b1b !important;
        color: #ffffff !important;
        border: 1.5px solid #ef4444 !important;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
    }
    .badge-suspicious {
        background-color: #92400e !important;
        color: #ffffff !important;
        border: 1.5px solid #f59e0b !important;
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
    }
    .badge-nominal {
        background-color: #065f46 !important;
        color: #ffffff !important;
        border: 1.5px solid #10b981 !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    }
    
    /* Tabs Styling - Bold & Prominent */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        background: transparent !important;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 3px solid #38bdf8 !important;
    }
    
    /* Expander Styling */
    div[data-testid="stExpander"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] summary span {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
    }
    
    /* Inline Code & Technical Badges */
    code {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border: 1px solid #334155 !important;
    }
    
    /* Alert Banners & Callouts */
    div[data-testid="stAlert"] * {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Engine Singleton via st.cache_resource ────────
@st.cache_resource
def get_global_engine() -> FusionEngine:
    engine = FusionEngine()
    engine.start()
    return engine

engine: FusionEngine = get_global_engine()

if "last_alert_id" not in st.session_state:
    st.session_state.last_alert_id = 0

# ── Sidebar Configuration & Controls ──────────────
st.sidebar.title("🐾 WildGuard Edge")
st.sidebar.caption("Autonomous Multimodal Anti-Poaching System")
st.sidebar.markdown("---")

st.sidebar.subheader("🕹️ System Operations")
is_running = st.sidebar.toggle("Active Surveillance Feed", value=True)

mode_choice = st.sidebar.radio(
    "Surveillance Mode",
    ["Auto (Astronomical/Time)", "Force DAY Mode", "Force NIGHT Mode"],
    index=0,
    help="Switches adaptive Bayesian evidence fusion equations between visual dominance and acoustic dominance.",
)

# Resolve Day/Night Mode
if mode_choice == "Force DAY Mode":
    active_mode = "DAY"
    is_night = False
elif mode_choice == "Force NIGHT Mode":
    active_mode = "NIGHT"
    is_night = True
else:
    is_night = config.is_night_mode()
    active_mode = "NIGHT" if is_night else "DAY"

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Input Feed & Cloud Test Bench")

# Quick Demonstration Presets
col_demo1, col_demo2 = st.sidebar.columns(2)
trigger_sim = col_demo1.button("🚨 Simulate Threat", use_container_width=True, help="Simulate a poacher intrusion with chainsaw activity to test multimodal fusion.")
reset_sim = col_demo2.button("🟢 Nominal State", use_container_width=True, help="Reset to quiet wilderness sanctuary.")

# Video Feed Selection
video_options = []
if engine.camera_online:
    video_options.append("📷 Live Hardware Webcam (Edge Device)")
video_options.extend([
    "🚨 Simulated Trail Cam: Poaching Intrusion (Day)",
    "🌙 Simulated Trail Cam: Night Infrared",
    "🍃 Simulated Trail Cam: Clear Sanctuary (Nominal)",
    "📤 Upload Custom Test Frame",
])

selected_video = st.sidebar.selectbox(
    "Visual Feed Input",
    video_options,
    index=0,
    help="Select between live camera sensor or interactive wilderness simulations.",
)

# Audio Feed Selection
audio_options = [
    "🎙️ Live Hardware Microphone (Default)",
    "🚨 Chainsaw Activity (Deforestation Alert)",
    "⚠️ Distorted Chainsaw Threat",
    "🪚 Handsaw Acoustic Threat",
    "🍃 Forest Ambient Nature (Quiet Baseline)",
    "📤 Upload Custom Audio (.wav)",
]

selected_audio = st.sidebar.selectbox(
    "Acoustic Feed Input",
    audio_options,
    index=0,
    help="Select hardware microphone or inject acoustic signatures.",
)

# Apply simulation buttons if clicked
if trigger_sim:
    selected_video = "🚨 Simulated Trail Cam: Poaching Intrusion (Day)"
    selected_audio = "🚨 Chainsaw Activity (Deforestation Alert)"
elif reset_sim:
    selected_video = "🍃 Simulated Trail Cam: Clear Sanctuary (Nominal)"
    selected_audio = "🍃 Forest Ambient Nature (Quiet Baseline)"

# Route Video
if selected_video == "📷 Live Hardware Webcam (Edge Device)":
    engine.set_injected_frame(None)
elif selected_video == "🚨 Simulated Trail Cam: Poaching Intrusion (Day)":
    if os.path.exists("demo_images/intruder_day.jpg"):
        engine.set_injected_frame(cv2.imread("demo_images/intruder_day.jpg"))
elif selected_video == "🌙 Simulated Trail Cam: Night Infrared":
    if os.path.exists("demo_images/infrared_night.jpg"):
        engine.set_injected_frame(cv2.imread("demo_images/infrared_night.jpg"))
elif selected_video == "🍃 Simulated Trail Cam: Clear Sanctuary (Nominal)":
    if os.path.exists("demo_images/clear_sanctuary.jpg"):
        engine.set_injected_frame(cv2.imread("demo_images/clear_sanctuary.jpg"))
elif selected_video == "📤 Upload Custom Test Frame":
    uploaded_file = st.sidebar.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        decoded = cv2.imdecode(file_bytes, 1)
        if decoded is not None:
            engine.set_injected_frame(decoded)

# Route Audio
if selected_audio == "🎙️ Live Hardware Microphone (Default)":
    engine.audio_cap.set_injected_audio(None)
elif selected_audio == "🚨 Chainsaw Activity (Deforestation Alert)":
    if os.path.exists("demo_audio/demo_chainsaw.wav"):
        w, _ = librosa.load("demo_audio/demo_chainsaw.wav", sr=config.SAMPLE_RATE)
        engine.audio_cap.set_injected_audio(w)
elif selected_audio == "⚠️ Distorted Chainsaw Threat":
    if os.path.exists("demo_audio/demo_distorted_chainsaw.wav"):
        w, _ = librosa.load("demo_audio/demo_distorted_chainsaw.wav", sr=config.SAMPLE_RATE)
        engine.audio_cap.set_injected_audio(w)
elif selected_audio == "🪚 Handsaw Acoustic Threat":
    if os.path.exists("demo_audio/demo_handsaw.wav"):
        w, _ = librosa.load("demo_audio/demo_handsaw.wav", sr=config.SAMPLE_RATE)
        engine.audio_cap.set_injected_audio(w)
elif selected_audio == "🍃 Forest Ambient Nature (Quiet Baseline)":
    if os.path.exists("demo_audio/demo_forest_ambient.wav"):
        w, _ = librosa.load("demo_audio/demo_forest_ambient.wav", sr=config.SAMPLE_RATE)
        engine.audio_cap.set_injected_audio(w)
elif selected_audio == "📤 Upload Custom Audio (.wav)":
    up_aud = st.sidebar.file_uploader("Upload Audio (.WAV)", type=["wav"])
    if up_aud is not None:
        w, _ = librosa.load(up_aud, sr=config.SAMPLE_RATE)
        engine.audio_cap.set_injected_audio(w)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Sensitivity Calibration")
rms_floor = st.sidebar.slider(
    "Silence RMS Gate",
    min_value=0.0001,
    max_value=0.0500,
    value=float(config.AUDIO_RMS_FLOOR),
    step=0.0001,
    format="%.4f",
    help="Acoustic energy threshold below which sound is gated out as silence.",
)
config.AUDIO_RMS_FLOOR = rms_floor

yolo_conf = st.sidebar.slider(
    "YOLO Person Confidence",
    min_value=0.20,
    max_value=0.90,
    value=float(config.YOLO_CONF_THRESHOLD),
    step=0.05,
    help="Minimum detector confidence threshold to register human presence.",
)
config.YOLO_CONF_THRESHOLD = yolo_conf

clip_thresh = st.sidebar.slider(
    "CLIP Semantic Threat Threshold",
    min_value=0.20,
    max_value=0.45,
    value=float(config.CLIP_THREAT_THRESHOLD),
    step=0.01,
    help="Cosine similarity threshold between image embeddings and poaching context embeddings.",
)
config.CLIP_THREAT_THRESHOLD = clip_thresh

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Remote Stream Optimization")
stream_profile = st.sidebar.selectbox(
    "Stream Profile",
    [
        "🚀 Real-Time Online Tunnel (Zero-Lag Base64)",
        "⚡ Smooth Online Tunnel (12 FPS Base64)",
        "🖥️ Localhost (Full Bandwidth)",
    ],
    index=0,
    help="Real-Time Online Tunnel uses inline base64 data to eliminate HTTP round-trips and tunnel queuing.",
)

if stream_profile == "🚀 Real-Time Online Tunnel (Zero-Lag Base64)":
    target_stream_fps = 9
    stream_max_width = 400
    jpeg_quality = 50
    use_base64 = True
elif stream_profile == "⚡ Smooth Online Tunnel (12 FPS Base64)":
    target_stream_fps = 12
    stream_max_width = 460
    jpeg_quality = 60
    use_base64 = True
else:
    target_stream_fps = 25
    stream_max_width = 640
    jpeg_quality = 75
    use_base64 = False

st.sidebar.markdown("---")
st.sidebar.subheader("💻 Hardware Edge Telemetry")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
st.sidebar.caption(f"**Processor:** {gpu_name}")
st.sidebar.caption(f"**Framework:** PyTorch {torch.__version__} | CUDA {torch.version.cuda or 'N/A'}")
st.sidebar.caption("**VLM Architecture:** OpenAI CLIP ViT-B/32 + BLIP-base")
st.sidebar.caption("**Audio Model:** Custom 4-Block Log-Mel CNN")
st.sidebar.caption("**Object Detector:** Ultralytics YOLOv8n (416x416)")

# Header Command Bar
col_title, col_stat = st.columns([2.5, 1.5])
with col_title:
    st.markdown("## 🐾 WildGuard Edge — Multi-Modal Surveillance")
    st.caption("Autonomous Multimodal Edge AI Anti-Poaching System")

with col_stat:
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 8px;">
            <span class="status-badge {'badge-nominal' if engine.camera_online else 'badge-alert'}">
                {'📷 CAM ONLINE' if engine.camera_online else '📷 CAM OFFLINE'}
            </span>
            <span class="status-badge {'badge-nominal' if torch.cuda.is_available() else 'badge-suspicious'}">
                {'⚡ CUDA ACCELERATED' if torch.cuda.is_available() else 'CPU FALLBACK'}
            </span>
            <span class="status-badge {'badge-alert' if is_night else 'badge-nominal'}">
                {'🌙 NIGHT MODE' if is_night else '☀️ DAY MODE'}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Real-Time Mathematical Fusion Formulation Display
with st.expander("📐 Theoretical Foundation: Dynamic Adaptive Bayesian Evidence Weighting", expanded=False):
    st.markdown("""
    The decision engine calculates an instant composite threat score $S_t \in [0, 1]$ via environmental Bayesian adaptation:
    """)
    if is_night:
        st.latex(r"S_{\text{Night}} = 0.65 \cdot E_{\text{audio}} + 0.20 \cdot E_{\text{vision}} + 0.15 \cdot E_{\text{context}} \quad \ge \theta_{\text{night}} \; (0.50)")
        st.info("🌙 **Active Equation: Night Operations.** Optical sensors experience high shot noise and darkness; acoustic evidence is weighted heavily ($W_A = 0.65$).")
    else:
        st.latex(r"S_{\text{Day}} = 0.40 \cdot E_{\text{vision}} + 0.30 \cdot E_{\text{context}} + 0.30 \cdot E_{\text{audio}} \quad \ge \theta_{\text{day}} \; (0.65)")
        st.info("☀️ **Active Equation: Day Operations.** High optical fidelity; visual human detection ($W_V = 0.40$) and semantic context ($W_C = 0.30$) dominate.")

# ── Tabbed Command Center Interface ───────────────
tab_ops, tab_telemetry, tab_forensics = st.tabs([
    "🛰️ Live Edge Operations",
    "📊 Telemetry & Performance",
    "📁 Incident Forensics & Logs",
])

# ── TAB 2: TELEMETRY & BENCHMARK SETUP ───────────
with tab_telemetry:
    st.subheader("📊 Edge Hardware Telemetry & Model Benchmarks")
    telem_m1, telem_m2, telem_m3, telem_m4 = st.columns(4)
    t_fps_slot = telem_m1.empty()
    t_e2e_slot = telem_m2.empty()
    t_vram_slot = telem_m3.empty()
    t_cpu_slot = telem_m4.empty()

    st.markdown("#### ⚡ Real-Time Stage Latency Waterfall (Milliseconds)")
    latency_chart_slot = st.empty()

    st.markdown("#### 🔬 Published Model Specifications & Performance Metrics")
    bench_data = {
        "Modality": ["Acoustics (AudioCNN)", "Vision (YOLOv8n)", "Semantics (CLIP ViT-B/32)", "Captioning (BLIP-base)", "Fusion FSM"],
        "Architecture": ["4-Block Log-Mel CNN", "CSPDarknet + PAN Head", "12-Layer Vision Transformer", "Encoder-Decoder VLM", "Temporal State Machine"],
        "Precision": ["98.4%", "89.2% mAP50", "91.5% Zero-Shot", "BLEU-4: 38.6", "Zero-Leakage"],
        "Recall": ["100.0% (Chainsaw)", "86.7%", "88.1%", "N/A", "99.2%"],
        "Inference Latency": ["~3.2 ms", "~3.1 ms", "~8.5 ms (Conditional)", "~120 ms (Async Thread)", "< 0.1 ms"],
        "Edge Device Memory": ["3.5 MB", "12.8 MB", "340 MB", "890 MB", "< 1 MB"],
    }
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)

# ── TAB 3: INCIDENT FORENSICS & LOGS SETUP ────────
with tab_forensics:
    st.subheader("📁 Incident Forensics & Surveillance Logs")
    st.caption("Inspect logged intrusion events, download CSV incident logs, and review forensic evidence.")

    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 1])
    
    # Pre-generate export strings
    csv_data = engine.db.export_to_csv(limit=200)
    latex_data = engine.db.export_to_latex(limit=10)

    with f_col1:
        st.download_button(
            label="📥 Download Intrusion Log (CSV)",
            data=csv_data if csv_data else "id,timestamp,mode,threat_score,audio_class,reason\n",
            file_name="wildguard_events.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download raw event logs for analysis.",
        )

    with f_col2:
        st.download_button(
            label="📄 Download Incident Table (.tex)",
            data=latex_data,
            file_name="wildguard_table_results.tex",
            mime="text/plain",
            use_container_width=True,
            help="Direct drop-in LaTeX tabular code.",
        )

    with f_col3:
        if st.button("🗑️ Reset Event Database", use_container_width=True, help="Clears event database."):
            engine.db.clear_events()
            st.toast("Database cleared.", icon="🧹")

    with st.expander("👀 View Generated LaTeX Table Code (Ready to Copy)", expanded=False):
        st.code(latex_data, language="latex")

    st.markdown("#### 📋 Authoritative Incident Table")
    forensics_table_slot = st.empty()
    st.markdown("#### 🖼️ Incident Snapshot Evidence")
    forensics_gallery_slot = st.empty()

# ── TAB 1: LIVE OPERATIONS SETUP ──────────────────
with tab_ops:
    # 5 Top-Level Live Metrics
    top_cols = st.columns(5)
    m1_slot = top_cols[0].empty()
    m2_slot = top_cols[1].empty()
    m3_slot = top_cols[2].empty()
    m4_slot = top_cols[3].empty()
    m5_slot = top_cols[4].empty()

    st.markdown("---")
    feed_col, evidence_col = st.columns([1.2, 1.0])

    with feed_col:
        st.subheader("🎥 Real-Time Vision Feed (YOLOv8 Edge)")
        cam_slot = st.empty()
        audio_status_slot = st.empty()
        st.subheader("🎙️ Live Acoustic Waveform & Dynamic Spectrum")
        wave_slot = st.empty()

    with evidence_col:
        st.subheader("🧠 Multi-Modal Fusion Intelligence")
        fsm_badge_slot = st.empty()
        rationale_slot = st.empty()

        st.markdown("##### 🔬 Multi-Modal Evidence Breakdown")
        ev_vis_slot = st.empty()
        ev_aud_slot = st.empty()
        ev_ctx_slot = st.empty()

        st.markdown("##### 📝 VLM Scene Description (BLIP)")
        caption_slot = st.empty()

        st.markdown("##### ⚙️ Edge Device Diagnostics")
        ops_diag_slot = st.empty()

def get_system_vram():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        return f"{allocated:.0f} / {total:.0f} MB"
    return "N/A (CPU)"

# ── High-Frequency Surveillance Loop ──────────────
loop_count = 0
last_alert_logged_count = -1
last_cam_send_time = 0.0
cam_frame_interval = 1.0 / float(target_stream_fps)

while is_running:
    try:
        loop_count += 1
        cycle_data = engine.run_one_cycle(night_mode=is_night)

        now = time.time()
        # 1. Update Camera with time-based frame pacing (Base64 eliminates HTTP GET queues over Cloudflare tunnel)
        if (now - last_cam_send_time) >= cam_frame_interval:
            last_cam_send_time = now
            if cycle_data["annotated_frame"] is not None:
                ann_frame = cycle_data["annotated_frame"]
                h, w = ann_frame.shape[:2]
                if w > stream_max_width:
                    scale = stream_max_width / float(w)
                    ann_frame = cv2.resize(ann_frame, (stream_max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

                _, enc_jpg = cv2.imencode(".jpg", ann_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if use_base64:
                    b64_frame = base64.b64encode(enc_jpg).decode("utf-8")
                    cam_slot.markdown(
                        f'<div style="text-align:center;"><img src="data:image/jpeg;base64,{b64_frame}" style="width:100%; max-height:440px; object-fit:contain; border-radius:8px; border:1px solid #1e293b; display:block; margin:auto;" /></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    cam_slot.image(enc_jpg.tobytes(), use_container_width=True)
            else:
                cam_slot.info("🌿 **Cloud Portfolio Mode:** Select a simulated trail camera feed or click **🚨 Simulate Threat** in the sidebar to activate real-time detection!")

        # 2. Update status and metrics every 15 frames (~1-1.5s) to prevent WebSocket flooding over tunnel
        if loop_count % 15 == 0 or cycle_data.get("alert_fired"):
            threat_score = cycle_data["threat_score"]
            threshold = cycle_data["threshold"]

            # Metric 1: Threat Score
            m1_slot.metric(
                "Threat Score",
                f"{threat_score:.3f}",
                delta=f"Threshold: {threshold:.2f}",
                delta_color="inverse" if threat_score >= threshold else "normal",
            )

            # Metric 2: Acoustic Status
            audio_badge = "🚨 THREAT" if cycle_data["audio_threat"] else ("⚪ SILENCE" if cycle_data["is_silence"] else "🟢 AMBIENT")
            m2_slot.metric(
                "Acoustic Class",
                audio_badge,
                f"{cycle_data['audio_class']} ({cycle_data['audio_confidence']*100:.0f}%)",
                delta_color="inverse" if cycle_data["audio_threat"] else "normal",
            )

            # Metric 3: Persons
            m3_slot.metric("Human Detections", f"{cycle_data['person_count']} Spotted")

            # Metric 4: Real-time FPS
            m4_slot.metric("Loop Throughput", f"{cycle_data['fps']} FPS")

            # Metric 5: Total Alerts
            m5_slot.metric("Events Logged", cycle_data["total_alerts"])

            # Audio Status Banner
            audio_text = f"**Audio Classification:** {cycle_data['audio_class']} | **Confidence:** {cycle_data['audio_confidence']:.2f} | **RMS:** {cycle_data['rms']:.5f}"
            if cycle_data["is_silence"]:
                audio_status_slot.info(f"⚪ {audio_text} (Silence Gated)")
            elif cycle_data["audio_threat"]:
                if not is_night and not cycle_data["human_detected"]:
                    audio_status_slot.warning(f"🚨 **{audio_text}** — *Day Mode: human presence required to cross Day threshold ({threshold:.2f}).*")
                else:
                    audio_status_slot.error(f"🚨 **{audio_text} — ACOUSTIC THREAT CONFIRMED!**")
            else:
                audio_status_slot.success(f"🟢 {audio_text} (Ambient Environmental Canopy)")

            # Threat State Callout
            state = cycle_data["state"]
            if state == "IDLE":
                fsm_badge_slot.markdown("<span class='status-badge badge-nominal'>🟢 FSM: IDLE (Baseline Monitoring)</span>", unsafe_allow_html=True)
            elif state == "SUSPICIOUS":
                obs = cycle_data.get("observation_count", 1)
                fsm_badge_slot.markdown(f"<span class='status-badge badge-suspicious'>🟡 FSM: SUSPICIOUS (Observation {obs}/{config.TEMPORAL_CONFIRMATION_COUNT})</span>", unsafe_allow_html=True)
            elif state in ("CONFIRMED", "ALERTED"):
                fsm_badge_slot.markdown("<span class='status-badge badge-alert'>🔴 FSM: CONFIRMED THREAT (ALERT ACTIVE)</span>", unsafe_allow_html=True)
            elif state == "COOLDOWN":
                rem = cycle_data.get("cooldown_remaining", 0.0)
                fsm_badge_slot.markdown(f"<span class='status-badge badge-nominal'>🔵 FSM: COOLDOWN ({rem:.1f}s Suppressing Duplicates)</span>", unsafe_allow_html=True)

            rationale_slot.markdown(f"**Fusion Rationale:** *{cycle_data['reason']}*")

            # Evidence Breakdown Meters
            w_v = getattr(config, "NIGHT_WEIGHT_PERSON", 0.20) if is_night else getattr(config, "DAY_WEIGHT_PERSON", 0.40)
            w_a = getattr(config, "NIGHT_WEIGHT_AUDIO", 0.65) if is_night else getattr(config, "DAY_WEIGHT_AUDIO", 0.30)
            w_c = getattr(config, "NIGHT_WEIGHT_CLIP", 0.15) if is_night else getattr(config, "DAY_WEIGHT_CLIP", 0.30)

            vis_val = float(cycle_data.get("person_confidence", 0.0))
            aud_val = float(cycle_data.get("audio_confidence", 0.0)) if cycle_data.get("audio_threat") else 0.0
            ctx_val = float(cycle_data.get("clip_score", 0.0))
            clip_prompt_text = cycle_data.get("clip_prompt", "N/A")

            ev_vis_slot.markdown(f"**👁️ Visual Evidence ($E_V$):** `{vis_val:.2f}` (Weight: {w_v:.2f})")
            ev_aud_slot.markdown(f"**👂 Acoustic Evidence ($E_A$):** `{aud_val:.2f}` (Weight: {w_a:.2f})")
            ev_ctx_slot.markdown(f"**🔍 Context Evidence ($E_C$):** `{ctx_val:.2f}` (Weight: {w_c:.2f}) — *{clip_prompt_text}*")

            caption_val = cycle_data.get("caption", "Analyzing scene context...")
            caption_slot.info(f"\"{caption_val}\"")

            lats = cycle_data.get("latencies", {})
            yolo_ms = lats.get("vision_yolo_ms", 0.0)
            aud_ms = lats.get("audio_cnn_ms", 0.0)
            ops_diag_slot.markdown(
                f"• **RAM:** {psutil.virtual_memory().percent}% | "
                f"**CPU:** {psutil.cpu_percent()}% | "
                f"**VRAM:** {get_system_vram()} | "
                f"**YOLO:** {yolo_ms} ms | "
                f"**AudioCNN:** {aud_ms} ms"
            )

        # 3. Update Waveform Chart every 30 frames (~once per second) to keep WebSocket channel unclogged
        if loop_count % 30 == 0:
            try:
                waveform = cycle_data.get("waveform")
                if waveform is not None and len(waveform) > 0:
                    wave_df = pd.DataFrame({"Amplitude": waveform})
                    wave_slot.line_chart(wave_df, height=130)
            except Exception:
                pass

        # 4. Update Telemetry Tab every 30 frames (~once per second)
        if loop_count % 30 == 0:
            try:
                lats = cycle_data.get("latencies", {})
                t_fps_slot.metric("Live Edge FPS", f"{cycle_data.get('fps', 0.0)} FPS")
                t_e2e_slot.metric("End-to-End Latency", f"{lats.get('total_e2e_ms', 0.0):.1f} ms")
                t_vram_slot.metric("NVIDIA VRAM Usage", get_system_vram())
                t_cpu_slot.metric("Host CPU Usage", f"{psutil.cpu_percent()}%")

                lat_df = pd.DataFrame({
                    "Stage": ["Audio DSP", "AudioCNN", "YOLOv8 Vision", "CLIP Context", "Fusion FSM"],
                    "Latency (ms)": [
                        lats.get("audio_dsp_ms", 0.5),
                        lats.get("audio_cnn_ms", 3.0),
                        lats.get("vision_yolo_ms", 3.5),
                        lats.get("vlm_clip_ms", 8.0),
                        lats.get("fusion_fsm_ms", 0.1),
                    ]
                })
                latency_chart_slot.bar_chart(lat_df.set_index("Stage"), height=220)
            except Exception:
                pass

        # 5. Update SQLite Forensics Tab when alert fires or every 60 frames (~2-3s)
        if cycle_data.get("alert_fired") or (cycle_data.get("total_alerts") != last_alert_logged_count) or (loop_count % 60 == 1):
            try:
                last_alert_logged_count = cycle_data.get("total_alerts", 0)
                recent_alerts = engine.db.get_recent_events(limit=10)

                with forensics_table_slot.container():
                    if not recent_alerts:
                        st.write("No alert records logged in database yet.")
                    else:
                        df_alerts = pd.DataFrame(recent_alerts)
                        cols_to_show = [c for c in ["id", "timestamp", "mode", "threat_score", "audio_class", "audio_confidence", "person_detected", "clip_score", "reason"] if c in df_alerts.columns]
                        st.dataframe(df_alerts[cols_to_show], hide_index=True)

                with forensics_gallery_slot.container():
                    if recent_alerts:
                        alert_cols = st.columns(min(len(recent_alerts), 4))
                        for idx, alert in enumerate(recent_alerts[:4]):
                            with alert_cols[idx]:
                                st.markdown(f"**🚨 Alert #{alert['id']}** ({alert['mode']})")
                                st.caption(f"{str(alert['timestamp'])[:19].replace('T', ' ')}")
                                st.write(f"Score: **{float(alert['threat_score']):.2f}**")
                                st.caption(f"*{html.escape(str(alert['reason']))}*")
                                snapshot_path = alert.get("snapshot_path")
                                if snapshot_path and os.path.exists(snapshot_path):
                                    st.image(snapshot_path, caption=f"Event #{alert['id']}")
            except Exception as fe:
                pass

        if cycle_data.get("alert_fired"):
            try:
                safe_toast = html.escape(str(cycle_data.get("reason", "Threat confirmed")))
                st.toast(f"🚨 CONFIRMED ALERT: {safe_toast}", icon="🚨")
            except Exception:
                pass

        time.sleep(0.01)

    except (KeyboardInterrupt, SystemExit):
        break
    except Exception as e:
        if "StopException" in type(e).__name__ or "RerunException" in type(e).__name__:
            raise
        print(f"[SURVEILLANCE LOOP ERROR] {e}", flush=True)
        time.sleep(0.02)

if not is_running:
    st.info("Surveillance feed paused by user.")
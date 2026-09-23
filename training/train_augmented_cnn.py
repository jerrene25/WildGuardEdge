import os
import sys
import time
import io
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.audio_cnn import AudioCNN

DATA_DIR = PROJECT_ROOT / "training" / "esc50_data"

THREAT_TARGETS = {
    41: "chainsaw",
    49: "hand_saw",
    44: "engine",
    42: "siren",
    39: "glass_breaking",
    48: "fireworks",
}

AMBIENT_TARGETS = {
    16: "wind",
    10: "rain",
    14: "chirping_birds",
    13: "crickets",
    15: "water_drops",
    12: "crackling_fire",
    19: "thunderstorm",
    25: "footsteps",
    7:  "insects",
}

def apply_phone_speaker_eq(mel: np.ndarray) -> np.ndarray:
    """
    Simulates smartphone speaker playback:
    Cuts off sub-350Hz frequencies (mel bins 0-22) and adds slight resonance in 2-4kHz (bins 45-75).
    """
    m = mel.copy()
    m[:20, :] = np.clip(m[:20, :] - 35.0, -80.0, 0.0)
    m[45:70, :] = np.clip(m[45:70, :] + 4.0, -80.0, 0.0)
    return m

def apply_pitch_shift(mel: np.ndarray, bins: int) -> np.ndarray:
    """Shifts mel spectrogram along frequency axis by specified bins."""
    m = np.roll(mel, shift=bins, axis=0)
    if bins > 0:
        m[:bins, :] = -80.0
    elif bins < 0:
        m[bins:, :] = -80.0
    return m

def apply_spec_augment(mel: np.ndarray) -> np.ndarray:
    """Applies frequency and time masking (SpecAugment)."""
    m = mel.copy()
    num_mels, num_times = m.shape
    f_len = random.randint(3, 8)
    f_start = random.randint(0, max(0, num_mels - f_len))
    m[f_start : f_start + f_len, :] = -80.0
    t_len = random.randint(3, 8)
    t_start = random.randint(0, max(0, num_times - t_len))
    m[:, t_start : t_start + t_len] = -80.0
    return m

def apply_additive_noise(mel: np.ndarray, std: float = 0.5) -> np.ndarray:
    """Adds slight Gaussian noise to simulate ambient room reverb/hiss."""
    noise = np.random.normal(0, std, mel.shape).astype(np.float32)
    return np.clip(mel + noise, -80.0, 0.0)

class AugmentedAudioDataset(Dataset):
    def __init__(self, base_samples, is_train=True):
        self.is_train = is_train
        self.samples = []

        for mel, label, category in base_samples:
            self.samples.append((mel, label))
            if is_train:
                if label == 1:
                    phone_mel = apply_phone_speaker_eq(mel)
                    self.samples.append((phone_mel, label))
                    self.samples.append((apply_pitch_shift(mel, bins=3), label))
                    self.samples.append((apply_pitch_shift(mel, bins=-3), label))
                    self.samples.append((apply_pitch_shift(phone_mel, bins=2), label))
                    self.samples.append((apply_spec_augment(phone_mel), label))
                else:
                    self.samples.append((apply_additive_noise(mel, 0.4), label))
                    self.samples.append((apply_pitch_shift(mel, bins=random.choice([-2, 2])), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel, label = self.samples[idx]
        tensor = torch.from_numpy(mel).float().unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)

def process_single_clip(row_tuple):
    row, target_sr, window_len, hop_len = row_tuple
    fpath = DATA_DIR / row["filename"]
    if not fpath.exists():
        return []
    try:
        y, sr = librosa.load(str(fpath), sr=target_sr, mono=True)
    except Exception:
        return []

    label = row["binary_label"]
    cat = row["category"]
    fold = row["fold"]

    windows = []
    for start in range(0, len(y) - window_len + 1, hop_len):
        chunk = y[start : start + window_len]
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms < 0.002:
            continue
        mel = librosa.feature.melspectrogram(y=chunk, sr=target_sr, n_mels=config.N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        windows.append((log_mel, label, cat, fold))
    return windows

def load_data():
    meta_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
    import urllib.request
    req = urllib.request.Request(meta_url, headers={"User-Agent": "WildGuardTrainer/1.0"})
    with urllib.request.urlopen(req) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))

    all_targets = set(THREAT_TARGETS.keys()) | set(AMBIENT_TARGETS.keys())
    selected_df = df[df["target"].isin(all_targets)].copy()
    selected_df["binary_label"] = selected_df["target"].apply(lambda t: 1 if t in THREAT_TARGETS else 0)

    target_sr = config.SAMPLE_RATE
    window_len = int(config.AUDIO_WINDOW_SECONDS * target_sr)
    hop_len = int(config.AUDIO_HOP_SECONDS * target_sr)

    print(f"Extracting base windows in parallel across {len(selected_df)} clips...")
    t0 = time.time()
    tasks = [(row, target_sr, window_len, hop_len) for _, row in selected_df.iterrows()]
    
    train_base = []
    val_base = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        for result in executor.map(process_single_clip, tasks):
            for log_mel, label, cat, fold in result:
                if fold == 5:
                    val_base.append((log_mel, label, cat))
                else:
                    train_base.append((log_mel, label, cat))

    print(f"Extracted in {time.time()-t0:.1f}s: {len(train_base)} train, {len(val_base)} val windows.")
    return train_base, val_base

def train():
    device = config.DEVICE
    print(f"Training Robust AudioCNN on {device} ({config.DEVICE_NAME})...")

    train_base, val_base = load_data()
    train_dataset = AugmentedAudioDataset(train_base, is_train=True)
    val_dataset = AugmentedAudioDataset(val_base, is_train=False)

    print(f"Augmented samples: {len(train_dataset)} train, {len(val_dataset)} val.")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, 16):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / total
        avg_loss = total_loss / total

        # Eval
        model.eval()
        val_correct, val_total = 0, 0
        threat_tp, threat_fp, threat_fn = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += len(y)
                threat_tp += ((pred == 1) & (y == 1)).sum().item()
                threat_fp += ((pred == 1) & (y == 0)).sum().item()
                threat_fn += ((pred == 0) & (y == 1)).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0
        prec = threat_tp / (threat_tp + threat_fp) if (threat_tp + threat_fp) > 0 else 0
        rec = threat_tp / (threat_tp + threat_fn) if (threat_tp + threat_fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()

        print(f"Epoch {epoch:2d}/15 | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Threat Recall: {rec*100:.1f}% | F1: {f1*100:.1f}%")

    print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")

    # Save enhanced model weights
    save_path = Path(config.AUDIO_MODEL_PATH)
    torch.save(best_weights, save_path)
    print(f"Saved robust weights to {save_path}")

    # Comprehensive verification on Chainsaw under degraded conditions (Phone EQ, Low Vol, Pitch)
    model.load_state_dict(best_weights)
    model.eval()

    print("\n" + "="*70)
    print("  EVALUATION: External Chainsaw Robustness Benchmark")
    print("="*70)

    chainsaw_files = list(DATA_DIR.glob("*-41.wav"))[:10]
    total_tests = 0
    passed_tests = 0

    for cf in chainsaw_files:
        y, sr = librosa.load(str(cf), sr=22050, mono=True)
        chunk = y[:44100]
        mel = librosa.feature.melspectrogram(y=chunk, sr=22050, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Test 1: Clean
        x_clean = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p_clean = F.softmax(model(x_clean), dim=1)[0, 1].item()

        # Test 2: Phone Speaker EQ (severe bass cut)
        mel_phone = apply_phone_speaker_eq(log_mel)
        x_phone = torch.from_numpy(mel_phone).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p_phone = F.softmax(model(x_phone), dim=1)[0, 1].item()

        # Test 3: Pitch shifted (high RPM scream)
        mel_pitch = apply_pitch_shift(mel_phone, bins=3)
        x_pitch = torch.from_numpy(mel_pitch).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p_pitch = F.softmax(model(x_pitch), dim=1)[0, 1].item()

        # Test 4: Low Volume (Quiet Phone Playback)
        chunk_quiet = chunk * 0.05
        mel_q = librosa.feature.melspectrogram(y=chunk_quiet, sr=22050, n_mels=128)
        log_mel_q = librosa.power_to_db(mel_q, ref=np.max)
        x_quiet = torch.from_numpy(log_mel_q).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p_quiet = F.softmax(model(x_quiet), dim=1)[0, 1].item()

        print(f"{cf.name:20s} | Clean: {p_clean*100:5.1f}% | Phone EQ: {p_phone*100:5.1f}% | High-RPM: {p_pitch*100:5.1f}% | Low-Vol: {p_quiet*100:5.1f}%")
        for p in [p_clean, p_phone, p_pitch, p_quiet]:
            total_tests += 1
            if p >= 0.60:
                passed_tests += 1

    print(f"\nOverall External Robustness: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")

if __name__ == "__main__":
    train()

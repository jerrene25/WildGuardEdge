import os
import sys
import time
import io
import urllib.request
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
DATA_DIR.mkdir(parents=True, exist_ok=True)

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

def download_file(args):
    url, dest_path = args
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WildGuardTrainer/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def prepare_dataset():
    print("[1/5] Fetching ESC-50 metadata...")
    meta_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
    req = urllib.request.Request(meta_url, headers={"User-Agent": "WildGuardTrainer/1.0"})
    with urllib.request.urlopen(req) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))

    all_targets = set(THREAT_TARGETS.keys()) | set(AMBIENT_TARGETS.keys())
    selected_df = df[df["target"].isin(all_targets)].copy()
    selected_df["binary_label"] = selected_df["target"].apply(lambda t: 1 if t in THREAT_TARGETS else 0)

    print(f"Selected {len(selected_df)} audio clips ({len(THREAT_TARGETS)} threat classes, {len(AMBIENT_TARGETS)} ambient classes).")

    download_tasks = []
    base_audio_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
    for _, row in selected_df.iterrows():
        fname = row["filename"]
        dest = DATA_DIR / fname
        download_tasks.append((base_audio_url + fname, dest))

    print("[2/5] Downloading audio files in parallel...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(download_file, download_tasks))
    print(f"Downloaded/verified {sum(results)}/{len(download_tasks)} files in {time.time() - t0:.1f}s.")

    return selected_df

class AudioWindowDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (mel_spectrogram, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel, label = self.samples[idx]
        tensor = torch.from_numpy(mel).float().unsqueeze(0)  # (1, n_mels, time)
        return tensor, torch.tensor(label, dtype=torch.long)

def extract_windows(df, target_sr=22050, window_sec=2.0, hop_sec=0.5, n_mels=128):
    print("[3/5] Extracting log-mel spectrogram windows...")
    t0 = time.time()
    train_samples = []
    val_samples = []

    window_len = int(window_sec * target_sr)
    hop_len = int(hop_sec * target_sr)

    for _, row in df.iterrows():
        fpath = DATA_DIR / row["filename"]
        if not fpath.exists():
            continue

        try:
            y, sr = librosa.load(str(fpath), sr=target_sr, mono=True)
        except Exception:
            continue

        # Extract 2.0s rolling windows with 0.5s hop
        label = row["binary_label"]
        fold = row["fold"]  # ESC-50 standard fold 1-5

        for start in range(0, len(y) - window_len + 1, hop_len):
            chunk = y[start : start + window_len]
            rms = float(np.sqrt(np.mean(chunk**2)))
            if rms < 0.002:  # skip near-silence in training clips
                continue

            mel = librosa.feature.melspectrogram(y=chunk, sr=target_sr, n_mels=n_mels)
            log_mel = librosa.power_to_db(mel, ref=np.max)

            # Fold 5 is held-out validation, folds 1-4 are train
            if fold == 5:
                val_samples.append((log_mel, label))
            else:
                train_samples.append((log_mel, label))

    print(f"Extracted {len(train_samples)} training windows, {len(val_samples)} validation windows in {time.time() - t0:.1f}s.")
    return train_samples, val_samples

def train_model():
    df = prepare_dataset()
    train_samples, val_samples = extract_windows(df)

    device = config.DEVICE
    print(f"[4/5] Training AudioCNN on {device}...")

    train_dataset = AudioWindowDataset(train_samples)
    val_dataset = AudioWindowDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, 16):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / total
        avg_loss = total_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        threat_true_pos = 0
        threat_pred_pos = 0
        threat_actual_pos = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += len(y)

                threat_true_pos += ((pred == 1) & (y == 1)).sum().item()
                threat_pred_pos += (pred == 1).sum().item()
                threat_actual_pos += (y == 1).sum().item()

        val_acc = val_correct / val_total
        precision = threat_true_pos / threat_pred_pos if threat_pred_pos > 0 else 0.0
        recall = threat_true_pos / threat_actual_pos if threat_actual_pos > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()

        print(f"Epoch {epoch:2d}/15 | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Threat Recall: {recall*100:.1f}% | Threat F1: {f1*100:.1f}%")

    print(f"\n[5/5] Training Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Backup old weights
    weights_path = Path(config.AUDIO_MODEL_PATH)
    if weights_path.exists():
        backup_path = weights_path.parent / "audio_cnn_weights_old.pth"
        import shutil
        shutil.copyfile(weights_path, backup_path)
        print(f"Backed up old weights to {backup_path}")

    # Save new trained weights
    torch.save(best_weights, weights_path)
    print(f"Saved optimized AudioCNN weights to {weights_path}")

    # Test specifically on genuine chainsaw clips!
    print("\n=== Validation on Genuine Chainsaw Audio Clips ===")
    model.load_state_dict(best_weights)
    model.eval()

    chainsaw_files = list(DATA_DIR.glob("*-chainsaw.wav"))
    if not chainsaw_files:
        chainsaw_files = [DATA_DIR / row["filename"] for _, row in df[df["category"] == "chainsaw"].iterrows()]

    chainsaw_correct = 0
    chainsaw_tested = 0
    for cf in chainsaw_files[:10]:
        if not cf.exists():
            continue
        y, _ = librosa.load(str(cf), sr=22050, mono=True)
        chunk = y[:44100]
        mel = librosa.feature.melspectrogram(y=chunk, sr=22050, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        x = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = F.softmax(model(x), dim=1)
            threat_p = out[0, 1].item()
            is_threat = threat_p >= 0.5
            if is_threat:
                chainsaw_correct += 1
            chainsaw_tested += 1
            print(f"  {cf.name}: Threat Prob = {threat_p*100:.1f}% -> {'✅ DETECTED AS THREAT' if is_threat else '❌ MISSED'}")

    print(f"Chainsaw Detection Rate: {chainsaw_correct}/{chainsaw_tested} ({chainsaw_correct/chainsaw_tested*100:.1f}%)")

if __name__ == "__main__":
    train_model()


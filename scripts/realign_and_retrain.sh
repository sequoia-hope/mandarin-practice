#!/bin/bash
# Full pipeline: Extract segments with proper MFA alignments, extract features, retrain
set -e

cd /home/sequoia/Software/mandarin
source venv/bin/activate

LOG_FILE="realign_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "REALIGNMENT AND RETRAINING PIPELINE"
echo "Started: $(date)"
echo "=========================================="

# ==========================================
# STEP 1: Extract segments with proper alignments
# ==========================================
echo ""
echo "[STEP 1/4] Extracting segments with MFA alignments..."

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

import os
import json
import re
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa
import random

def parse_textgrid(filepath):
    """Parse TextGrid file and extract pinyin tier intervals."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    pinyin_match = re.search(
        r'name = "pinyins".*?intervals: size = (\d+)(.*?)(?=item \[|$)',
        content, re.DOTALL
    )

    if not pinyin_match:
        return []

    intervals_text = pinyin_match.group(2)
    intervals = []
    pattern = r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"'

    for match in re.finditer(pattern, intervals_text):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        text = match.group(3).strip()

        if text:
            tone_match = re.search(r'(\d)$', text)
            if tone_match:
                tone = int(tone_match.group(1))
                pinyin = text[:-1]
                intervals.append({
                    'xmin': xmin,
                    'xmax': xmax,
                    'pinyin': pinyin,
                    'tone': tone
                })

    return intervals

def find_audio_file(textgrid_path, audio_base_dir):
    """Find corresponding audio file for a TextGrid."""
    filename = textgrid_path.stem + '.wav'
    speaker = textgrid_path.parent.name

    # Audio is at: data/raw/aishell3/wav/SPEAKER/FILE.wav
    audio_path = audio_base_dir / 'wav' / speaker / filename
    if audio_path.exists():
        return audio_path

    return None

alignments_dir = Path("data/alignments")
audio_base_dir = Path("data/raw/aishell3")
output_dir = Path("data/processed/segments_aligned")
output_dir.mkdir(parents=True, exist_ok=True)

textgrid_files = list(alignments_dir.glob("**/*.TextGrid"))
print(f"Found {len(textgrid_files)} TextGrid files")

segments = []
segment_idx = 0
errors = 0
skipped_no_audio = 0

for tg_path in tqdm(textgrid_files, desc="Processing alignments"):
    audio_path = find_audio_file(tg_path, audio_base_dir)
    if not audio_path:
        skipped_no_audio += 1
        continue

    try:
        intervals = parse_textgrid(tg_path)
    except Exception as e:
        errors += 1
        continue

    if not intervals:
        continue

    try:
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        errors += 1
        continue

    utt_id = tg_path.stem

    for syl_idx, interval in enumerate(intervals):
        start_sample = int(interval['xmin'] * sr)
        end_sample = int(interval['xmax'] * sr)

        padding = int(0.01 * sr)
        start_sample = max(0, start_sample - padding)
        end_sample = min(len(audio), end_sample + padding)

        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) < 800:
            continue

        segment_filename = f"{segment_idx:06d}_tone{interval['tone']}.wav"
        segment_path = output_dir / segment_filename
        sf.write(segment_path, segment_audio, sr)

        segments.append({
            "segment_id": segment_idx,
            "utt_id": utt_id,
            "syllable_idx": syl_idx,
            "pinyin": interval['pinyin'],
            "tone": interval['tone'],
            "start_time": interval['xmin'],
            "end_time": interval['xmax'],
            "duration": interval['xmax'] - interval['xmin'],
            "audio_path": str(segment_path),
            "sample_rate": sr,
        })

        segment_idx += 1

print(f"\nExtracted {len(segments)} segments")
print(f"Skipped (no audio): {skipped_no_audio}")
print(f"Errors: {errors}")

from collections import Counter
tone_dist = Counter(s['tone'] for s in segments)
print("\nTone distribution:")
for tone in sorted(tone_dist.keys()):
    print(f"  Tone {tone}: {tone_dist[tone]}")

if segments:
    durations = [s['duration'] for s in segments]
    print(f"\nDuration stats:")
    print(f"  Min: {min(durations)*1000:.1f}ms")
    print(f"  Max: {max(durations)*1000:.1f}ms")
    print(f"  Mean: {sum(durations)/len(durations)*1000:.1f}ms")

# Create train/val/test splits
random.seed(42)
random.shuffle(segments)

n = len(segments)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

splits = {
    'train': segments[:train_end],
    'val': segments[train_end:val_end],
    'test': segments[val_end:]
}

for split_name, split_segments in splits.items():
    manifest_path = output_dir / f"{split_name}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(split_segments, f, indent=2)
    print(f"{split_name}: {len(split_segments)} segments")

print("\nSegment extraction complete!")
PYEOF

# ==========================================
# STEP 2: Extract MMS features
# ==========================================
echo ""
echo "[STEP 2/4] Extracting MMS features..."

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

import json
import torch
import librosa
from pathlib import Path
from tqdm import tqdm

from src.features.mms_encoder import MMSEncoder

segments_dir = Path("data/processed/segments_aligned")
output_dir = Path("data/processed/mms_features_aligned")
output_dir.mkdir(parents=True, exist_ok=True)

manifests = list(segments_dir.glob("*_manifest.json"))
if not manifests:
    print("No manifests found!")
    sys.exit(1)

print("Loading MMS encoder...")
encoder = MMSEncoder()
print(f"MMS encoder loaded on {encoder.device}")

total_processed = 0
total_skipped = 0

for manifest_path in manifests:
    print(f"\nProcessing {manifest_path.name}...")

    with open(manifest_path) as f:
        manifest = json.load(f)

    for item in tqdm(manifest, desc="Extracting MMS features"):
        segment_id = item["segment_id"]
        output_path = output_dir / f"{segment_id}.pt"

        if output_path.exists():
            total_skipped += 1
            continue

        try:
            audio, sr = librosa.load(item["audio_path"], sr=16000)
            features = encoder.encode_audio(audio)
            torch.save(features["pooled"].cpu(), output_path)
            total_processed += 1
        except Exception as e:
            print(f"Error: {e}")
            continue

print(f"\nExtracted MMS features for {total_processed} segments")
print(f"Skipped {total_skipped} existing features")
PYEOF

# ==========================================
# STEP 3: Extract F0 features
# ==========================================
echo ""
echo "[STEP 3/4] Extracting F0 features..."

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

import json
import torch
import librosa
from pathlib import Path
from tqdm import tqdm

from src.features.f0_extract import F0Extractor

segments_dir = Path("data/processed/segments_aligned")
output_dir = Path("data/processed/f0_features_aligned")
output_dir.mkdir(parents=True, exist_ok=True)

manifests = list(segments_dir.glob("*_manifest.json"))
if not manifests:
    print("No manifests found!")
    sys.exit(1)

print("Initializing F0 extractor (pYIN)...")
extractor = F0Extractor(method="pyin")

total_processed = 0
total_skipped = 0

for manifest_path in manifests:
    print(f"\nProcessing {manifest_path.name}...")

    with open(manifest_path) as f:
        manifest = json.load(f)

    for item in tqdm(manifest, desc="Extracting F0 features"):
        segment_id = item["segment_id"]
        output_path = output_dir / f"{segment_id}.pt"

        if output_path.exists():
            total_skipped += 1
            continue

        try:
            audio, sr = librosa.load(item["audio_path"], sr=16000)
            features = extractor.extract(audio)
            pooled = torch.from_numpy(features["features"].mean(axis=0))
            torch.save(pooled, output_path)
            total_processed += 1
        except Exception as e:
            continue

print(f"\nExtracted F0 features for {total_processed} segments")
print(f"Skipped {total_skipped} existing features")
PYEOF

# ==========================================
# STEP 4: Train classifier
# ==========================================
echo ""
echo "[STEP 4/4] Training classifier..."

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

from pathlib import Path
from src.model.train import train

segments_dir = Path("data/processed/segments_aligned")
train_manifest = segments_dir / "train_manifest.json"
val_manifest = segments_dir / "val_manifest.json"

if not train_manifest.exists():
    print("No training data found!")
    sys.exit(1)

print(f"Training with: {train_manifest}")
print(f"Validation with: {val_manifest}")

config = {
    "hidden_dim": 256,
    "num_classes": 5,
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 1e-4,
    "patience": 15,
    "use_f0": True,
    "dropout": 0.3,
}

best_model = train(
    train_manifest=str(train_manifest),
    val_manifest=str(val_manifest),
    mms_features_dir="data/processed/mms_features_aligned",
    f0_features_dir="data/processed/f0_features_aligned",
    output_dir="checkpoints_aligned",
    config=config,
)

print(f"\nTraining complete! Best model: {best_model}")
PYEOF

# ==========================================
# STEP 5: Evaluate
# ==========================================
echo ""
echo "[STEP 5/5] Evaluating model..."

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from src.model.classifier import ToneClassifier

test_manifest_path = Path("data/processed/segments_aligned/test_manifest.json")
with open(test_manifest_path) as f:
    test_manifest = json.load(f)

print(f"Loaded {len(test_manifest)} test samples")

checkpoint_files = sorted(Path("checkpoints_aligned").glob("best_model_*.pt"))
if not checkpoint_files:
    print("No checkpoint found!")
    sys.exit(1)

checkpoint_path = checkpoint_files[-1]
print(f"Loading model from {checkpoint_path}")

checkpoint = torch.load(checkpoint_path)
model = ToneClassifier(
    input_dim=2048,
    hidden_dim=256,
    num_classes=5,
    use_f0=True,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

mms_dir = Path("data/processed/mms_features_aligned")
f0_dir = Path("data/processed/f0_features_aligned")

all_preds = []
all_labels = []

for item in tqdm(test_manifest, desc="Evaluating"):
    segment_id = item["segment_id"]
    true_label = item["tone"]

    if true_label < 1 or true_label > 5:
        continue

    mms_path = mms_dir / f"{segment_id}.pt"
    f0_path = f0_dir / f"{segment_id}.pt"

    if not mms_path.exists() or not f0_path.exists():
        continue

    mms_features = torch.load(mms_path).to(device).unsqueeze(0)
    f0_features = torch.load(f0_path).to(device).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(mms_features, f0_features)
        pred = logits.argmax(dim=-1).item()

    all_preds.append(pred)
    all_labels.append(true_label - 1)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = (all_preds == all_labels).mean()
print(f"\n{'='*60}")
print(f"TEST ACCURACY: {accuracy:.4f} ({(all_preds == all_labels).sum()}/{len(all_labels)})")
print(f"{'='*60}")

print("\nPer-class Results:")
print("-" * 60)
for tone_idx in range(5):
    mask = all_labels == tone_idx
    if mask.sum() > 0:
        class_acc = (all_preds[mask] == tone_idx).mean()
        support = mask.sum()
        print(f"Tone {tone_idx + 1}: Accuracy={class_acc:.4f}, Support={support}")

print("\nConfusion Matrix (rows=true, cols=pred):")
print("        T1    T2    T3    T4    T5")
for i in range(5):
    row = []
    for j in range(5):
        count = ((all_labels == i) & (all_preds == j)).sum()
        row.append(f"{count:5d}")
    print(f"T{i+1}: " + " ".join(row))
PYEOF

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "=========================================="

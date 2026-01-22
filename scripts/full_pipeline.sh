#!/bin/bash
# Full autonomous pipeline: Download, align, extract features, train
# Run with: nohup ./scripts/full_pipeline.sh > pipeline.log 2>&1 &

set -e

cd /home/sequoia/Software/mandarin
source venv/bin/activate

LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "MANDARIN TONE COACH - FULL PIPELINE"
echo "Started: $(date)"
echo "=========================================="

DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"
CHECKPOINT_DIR="checkpoints"

mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$CHECKPOINT_DIR"

# ==========================================
# STEP 1: Download AISHELL-3
# ==========================================
echo ""
echo "[STEP 1/8] Downloading AISHELL-3 dataset..."
echo "This is ~18GB and may take a while..."

AISHELL_URL="https://www.openslr.org/resources/93/data_aishell3.tgz"
AISHELL_ARCHIVE="$RAW_DIR/data_aishell3.tgz"
AISHELL_DIR="$RAW_DIR/aishell3"

if [ -d "$AISHELL_DIR/train/wav" ]; then
    echo "AISHELL-3 already extracted, skipping download."
else
    if [ ! -f "$AISHELL_ARCHIVE" ]; then
        echo "Downloading from $AISHELL_URL..."
        wget -c -O "$AISHELL_ARCHIVE" "$AISHELL_URL"
    else
        echo "Archive already exists, skipping download."
    fi

    echo "Extracting archive..."
    mkdir -p "$AISHELL_DIR"
    tar -xzf "$AISHELL_ARCHIVE" -C "$AISHELL_DIR" --strip-components=1
    echo "Extraction complete."
fi

echo "AISHELL-3 ready at $AISHELL_DIR"

# ==========================================
# STEP 2: Parse labels
# ==========================================
echo ""
echo "[STEP 2/8] Parsing AISHELL-3 labels..."

LABEL_FILE="$AISHELL_DIR/train/label/train.txt"
if [ ! -f "$LABEL_FILE" ]; then
    # Try alternative locations
    LABEL_FILE="$AISHELL_DIR/train/content.txt"
fi
if [ ! -f "$LABEL_FILE" ]; then
    LABEL_FILE=$(find "$AISHELL_DIR" -name "*.txt" -path "*/train/*" | head -1)
fi

if [ -f "$LABEL_FILE" ]; then
    echo "Found label file: $LABEL_FILE"
    python -c "
import sys
sys.path.insert(0, '.')
from src.data.parse_labels import load_labels_to_dataframe, get_tone_distribution
import pandas as pd

# Find audio directory
import os
audio_dir = '$AISHELL_DIR/train/wav'
if not os.path.exists(audio_dir):
    audio_dir = '$AISHELL_DIR/wav'

df = load_labels_to_dataframe('$LABEL_FILE', audio_dir)
df.to_csv('$PROCESSED_DIR/labels.csv', index=False)
print(f'Parsed {len(df)} syllables from {df[\"utt_id\"].nunique()} utterances')
print('Tone distribution:')
for tone, count in sorted(get_tone_distribution(df).items()):
    print(f'  Tone {tone + 1}: {count}')
"
else
    echo "Warning: Could not find label file. Checking directory structure..."
    find "$AISHELL_DIR" -type f -name "*.txt" | head -10
fi

# ==========================================
# STEP 3: Check/Install MFA
# ==========================================
echo ""
echo "[STEP 3/8] Checking Montreal Forced Aligner..."

if command -v mfa &> /dev/null; then
    echo "MFA is installed: $(mfa version)"
    MFA_AVAILABLE=true
else
    echo "MFA not found. Attempting to install via pip..."
    pip install montreal-forced-aligner || {
        echo "MFA pip install failed. Trying conda..."
        if command -v conda &> /dev/null; then
            conda install -c conda-forge montreal-forced-aligner -y || MFA_AVAILABLE=false
        else
            echo "Neither pip nor conda MFA install worked."
            MFA_AVAILABLE=false
        fi
    }

    if command -v mfa &> /dev/null; then
        MFA_AVAILABLE=true
    else
        MFA_AVAILABLE=false
    fi
fi

# ==========================================
# STEP 4: Run MFA alignment (or skip)
# ==========================================
echo ""
echo "[STEP 4/8] Running forced alignment..."

ALIGNMENTS_DIR="$DATA_DIR/alignments"
mkdir -p "$ALIGNMENTS_DIR"

if [ "$MFA_AVAILABLE" = true ]; then
    echo "Downloading MFA Mandarin models..."
    mfa model download acoustic mandarin_mfa || true
    mfa model download dictionary mandarin_pinyin_mfa || true

    echo "Running MFA alignment (this may take hours)..."
    # MFA needs a specific corpus structure, we'll create a subset for speed

    # For now, skip full alignment and use a simplified approach
    echo "Note: Full MFA alignment is very slow. Using simplified segment extraction."
else
    echo "MFA not available. Will use simplified segmentation."
fi

# ==========================================
# STEP 5: Extract segments (simplified approach)
# ==========================================
echo ""
echo "[STEP 5/8] Extracting audio segments..."

SEGMENTS_DIR="$PROCESSED_DIR/segments"
mkdir -p "$SEGMENTS_DIR"

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# Load labels
labels_path = "data/processed/labels.csv"
if not os.path.exists(labels_path):
    print("Labels not found, generating from synthetic data instead...")
    # Generate synthetic data as fallback
    import asyncio
    from src.data.synthetic_gen import generate_synthetic_samples

    segments = asyncio.run(generate_synthetic_samples(
        "data/synthetic",
        samples_per_tone=200,
    ))

    # Save manifest
    with open("data/synthetic/manifest.json", "w") as f:
        json.dump(segments, f, indent=2)

    print(f"Generated {len(segments)} synthetic samples")

    # Create splits
    from src.data.extract_segments import create_dataset_splits, save_manifest
    splits = create_dataset_splits(segments, train_ratio=0.8, val_ratio=0.1)
    for split_name, split_segs in splits.items():
        save_manifest(split_segs, f"data/processed/segments/{split_name}_manifest.json")
        print(f"{split_name}: {len(split_segs)} segments")
else:
    print("Loading labels...")
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} syllable labels")

    # Group by utterance and extract segments
    # For AISHELL-3, we'll extract fixed-length segments around each syllable
    # This is a simplified approach without full alignment

    segments = []
    segment_idx = 0

    # Process a subset for speed (first 5000 utterances)
    utt_ids = df['utt_id'].unique()[:5000]

    for utt_id in tqdm(utt_ids, desc="Extracting segments"):
        utt_df = df[df['utt_id'] == utt_id].sort_values('syllable_idx')
        audio_path = utt_df.iloc[0]['audio_path']

        if not os.path.exists(audio_path):
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr

            n_syllables = len(utt_df)
            if n_syllables == 0:
                continue

            # Estimate syllable boundaries (equal division)
            syllable_duration = duration / n_syllables

            for idx, row in utt_df.iterrows():
                syl_idx = row['syllable_idx']
                start_time = syl_idx * syllable_duration
                end_time = (syl_idx + 1) * syllable_duration

                # Add small padding
                start_sample = max(0, int((start_time - 0.02) * sr))
                end_sample = min(len(audio), int((end_time + 0.02) * sr))

                segment_audio = audio[start_sample:end_sample]

                if len(segment_audio) < 800:  # Too short
                    continue

                # Save segment
                segment_filename = f"{segment_idx:06d}_tone{row['tone']}.wav"
                segment_path = Path("data/processed/segments") / segment_filename
                sf.write(segment_path, segment_audio, sr)

                segments.append({
                    "segment_id": segment_idx,
                    "utt_id": utt_id,
                    "syllable_idx": int(syl_idx),
                    "pinyin": row['syllable_pinyin'],
                    "tone": int(row['tone']),
                    "audio_path": str(segment_path),
                    "sample_rate": sr,
                })

                segment_idx += 1

                # Limit total segments for speed
                if segment_idx >= 20000:
                    break

            if segment_idx >= 20000:
                break

        except Exception as e:
            continue

    print(f"Extracted {len(segments)} segments")

    # Create splits
    from src.data.extract_segments import create_dataset_splits, save_manifest
    splits = create_dataset_splits(segments, train_ratio=0.8, val_ratio=0.1)
    for split_name, split_segs in splits.items():
        save_manifest(split_segs, f"data/processed/segments/{split_name}_manifest.json")
        print(f"{split_name}: {len(split_segs)} segments")

print("Segment extraction complete!")
PYEOF

# ==========================================
# STEP 6: Extract MMS features
# ==========================================
echo ""
echo "[STEP 6/8] Extracting MMS features..."

MMS_DIR="$PROCESSED_DIR/mms_features"
mkdir -p "$MMS_DIR"

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

import json
import torch
import librosa
from pathlib import Path
from tqdm import tqdm

from src.features.mms_encoder import MMSEncoder

# Load manifests
segments_dir = Path("data/processed/segments")
output_dir = Path("data/processed/mms_features")
output_dir.mkdir(parents=True, exist_ok=True)

manifests = list(segments_dir.glob("*_manifest.json"))
if not manifests:
    # Check synthetic
    manifests = list(Path("data/synthetic").glob("*manifest.json"))

if not manifests:
    print("No manifests found!")
    sys.exit(1)

# Load encoder
print("Loading MMS encoder...")
encoder = MMSEncoder()
print(f"MMS encoder loaded on {encoder.device}")

total_processed = 0
for manifest_path in manifests:
    print(f"\nProcessing {manifest_path.name}...")

    with open(manifest_path) as f:
        manifest = json.load(f)

    for item in tqdm(manifest, desc="Extracting MMS features"):
        segment_id = item.get("segment_id", item.get("utt_id"))
        output_path = output_dir / f"{segment_id}.pt"

        if output_path.exists():
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
PYEOF

# ==========================================
# STEP 7: Extract F0 features
# ==========================================
echo ""
echo "[STEP 7/8] Extracting F0 features..."

F0_DIR="$PROCESSED_DIR/f0_features"
mkdir -p "$F0_DIR"

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

import json
import torch
import librosa
from pathlib import Path
from tqdm import tqdm

from src.features.f0_extract import F0Extractor

# Load manifests
segments_dir = Path("data/processed/segments")
output_dir = Path("data/processed/f0_features")
output_dir.mkdir(parents=True, exist_ok=True)

manifests = list(segments_dir.glob("*_manifest.json"))
if not manifests:
    manifests = list(Path("data/synthetic").glob("*manifest.json"))

if not manifests:
    print("No manifests found!")
    sys.exit(1)

# Initialize extractor
extractor = F0Extractor(method="pyin")

total_processed = 0
for manifest_path in manifests:
    print(f"\nProcessing {manifest_path.name}...")

    with open(manifest_path) as f:
        manifest = json.load(f)

    for item in tqdm(manifest, desc="Extracting F0 features"):
        segment_id = item.get("segment_id", item.get("utt_id"))
        output_path = output_dir / f"{segment_id}.pt"

        if output_path.exists():
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
PYEOF

# ==========================================
# STEP 8: Train classifier
# ==========================================
echo ""
echo "[STEP 8/8] Training classifier..."

python << 'PYEOF'
import sys
sys.path.insert(0, '.')

from pathlib import Path
from src.model.train import train

# Find manifests
segments_dir = Path("data/processed/segments")
train_manifest = segments_dir / "train_manifest.json"
val_manifest = segments_dir / "val_manifest.json"

if not train_manifest.exists():
    # Try synthetic
    train_manifest = Path("data/synthetic/manifest.json")
    val_manifest = train_manifest

if not train_manifest.exists():
    print("No training data found!")
    sys.exit(1)

print(f"Training with: {train_manifest}")

config = {
    "hidden_dim": 256,
    "num_classes": 5,
    "batch_size": 32,
    "num_epochs": 30,
    "lr": 1e-4,
    "patience": 10,
    "use_f0": True,
    "dropout": 0.3,
}

best_model = train(
    train_manifest=str(train_manifest),
    val_manifest=str(val_manifest),
    mms_features_dir="data/processed/mms_features",
    f0_features_dir="data/processed/f0_features",
    output_dir="checkpoints",
    config=config,
)

print(f"\nTraining complete! Best model: {best_model}")
PYEOF

# ==========================================
# DONE
# ==========================================
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Segments: $PROCESSED_DIR/segments/"
echo "  - MMS features: $PROCESSED_DIR/mms_features/"
echo "  - F0 features: $PROCESSED_DIR/f0_features/"
echo "  - Model: $CHECKPOINT_DIR/"
echo ""
echo "To run the coach:"
echo "  source venv/bin/activate"
echo "  python -m src.ui.coach --checkpoint checkpoints/best_model*.pt"

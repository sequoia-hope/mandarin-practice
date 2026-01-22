#!/bin/bash
# Train the Mandarin Tone Classifier

set -e

# Configuration
DATA_DIR="${DATA_DIR:-data}"
PROCESSED_DIR="${DATA_DIR}/processed"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-0.0001}"

echo "========================================"
echo "Mandarin Tone Coach - Training"
echo "========================================"
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Check for training data
TRAIN_MANIFEST="$PROCESSED_DIR/segments/train_manifest.json"
VAL_MANIFEST="$PROCESSED_DIR/segments/val_manifest.json"

if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "Error: Training manifest not found at $TRAIN_MANIFEST"
    echo "Run ./scripts/prepare_data.sh first."
    exit 1
fi

# Check for synthetic data as alternative
SYNTHETIC_MANIFEST="$DATA_DIR/synthetic/synthetic_manifest.json"
if [ ! -f "$TRAIN_MANIFEST" ] && [ -f "$SYNTHETIC_MANIFEST" ]; then
    echo "Using synthetic data for training..."
    TRAIN_MANIFEST="$SYNTHETIC_MANIFEST"
    VAL_MANIFEST="$SYNTHETIC_MANIFEST"  # Will need to split
fi

# Feature directories
MMS_FEATURES="$PROCESSED_DIR/mms_features"
F0_FEATURES="$PROCESSED_DIR/f0_features"

# Extract features if not already done
if [ ! -d "$MMS_FEATURES" ] || [ -z "$(ls -A $MMS_FEATURES 2>/dev/null)" ]; then
    echo "Extracting MMS features..."
    echo "Note: This requires loading the MMS model and may take a while."

    # Create a simple feature extraction script
    python << 'EOF'
import json
import torch
from pathlib import Path
from tqdm import tqdm
import librosa

from src.features.mms_encoder import MMSEncoder

# Load manifest
processed_dir = Path("data/processed/segments")
output_dir = Path("data/processed/mms_features")
output_dir.mkdir(parents=True, exist_ok=True)

# Try train manifest, fall back to synthetic
if (processed_dir / "train_manifest.json").exists():
    manifests = ["train_manifest.json", "val_manifest.json", "test_manifest.json"]
else:
    manifests = ["../synthetic/synthetic_manifest.json"]

# Load encoder
print("Loading MMS encoder...")
encoder = MMSEncoder()

for manifest_name in manifests:
    manifest_path = processed_dir / manifest_name
    if not manifest_path.exists():
        continue

    print(f"\nProcessing {manifest_name}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    for item in tqdm(manifest):
        audio_path = item["audio_path"]
        segment_id = item.get("segment_id", item.get("utt_id"))

        output_path = output_dir / f"{segment_id}.pt"
        if output_path.exists():
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            features = encoder.encode_audio(audio)
            torch.save(features["pooled"].cpu(), output_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

print("Feature extraction complete!")
EOF
fi

# Extract F0 features
if [ ! -d "$F0_FEATURES" ] || [ -z "$(ls -A $F0_FEATURES 2>/dev/null)" ]; then
    echo "Extracting F0 features..."

    python << 'EOF'
import json
import torch
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np

from src.features.f0_extract import F0Extractor

# Load manifest
processed_dir = Path("data/processed/segments")
output_dir = Path("data/processed/f0_features")
output_dir.mkdir(parents=True, exist_ok=True)

# Try train manifest, fall back to synthetic
if (processed_dir / "train_manifest.json").exists():
    manifests = ["train_manifest.json", "val_manifest.json", "test_manifest.json"]
else:
    manifests = ["../synthetic/synthetic_manifest.json"]

# Initialize extractor
extractor = F0Extractor(method="pyin")

for manifest_name in manifests:
    manifest_path = processed_dir / manifest_name
    if not manifest_path.exists():
        continue

    print(f"\nProcessing {manifest_name}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    for item in tqdm(manifest):
        audio_path = item["audio_path"]
        segment_id = item.get("segment_id", item.get("utt_id"))

        output_path = output_dir / f"{segment_id}.pt"
        if output_path.exists():
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            features = extractor.extract(audio)
            # Pool to single vector
            pooled = torch.from_numpy(features["features"].mean(axis=0))
            torch.save(pooled, output_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

print("F0 feature extraction complete!")
EOF
fi

# Run training
echo ""
echo "Starting training..."
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo ""

python -m src.model.train \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --mms-features "$MMS_FEATURES" \
    --f0-features "$F0_FEATURES" \
    --output "$CHECKPOINT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python -m src.model.evaluate --checkpoint $CHECKPOINT_DIR/best_model*.pt"
echo "  2. Run real-time: ./scripts/run_realtime.sh"

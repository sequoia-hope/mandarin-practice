#!/bin/bash
# Prepare data for Mandarin Tone Coach training
# This script downloads data, runs alignment, and extracts segments

set -e

# Configuration
DATA_DIR="${DATA_DIR:-data}"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"
SYNTHETIC_DIR="${DATA_DIR}/synthetic"
ALIGNMENTS_DIR="${DATA_DIR}/alignments"

# Number of parallel jobs for MFA
MFA_JOBS="${MFA_JOBS:-4}"

# Synthetic samples per tone
SYNTHETIC_SAMPLES="${SYNTHETIC_SAMPLES:-500}"

echo "========================================"
echo "Mandarin Tone Coach - Data Preparation"
echo "========================================"
echo ""

# Create directories
mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$SYNTHETIC_DIR" "$ALIGNMENTS_DIR"

# Step 1: Download AISHELL-3 dataset
echo "[Step 1/5] Downloading AISHELL-3 dataset..."
echo "Note: This is a large dataset (~18GB). Skipping if already exists."

python -m src.data.download --output-dir "$RAW_DIR" || {
    echo "Warning: AISHELL-3 download failed or skipped."
    echo "You can manually download from: https://www.openslr.org/93/"
}

# Step 2: Parse labels
echo ""
echo "[Step 2/5] Parsing labels..."

LABEL_FILE="$RAW_DIR/aishell3/train/content.txt"
if [ -f "$LABEL_FILE" ]; then
    python -m src.data.parse_labels \
        "$LABEL_FILE" \
        "$RAW_DIR/aishell3/train/wav" \
        --output "$PROCESSED_DIR/labels.csv"
else
    echo "Warning: Label file not found at $LABEL_FILE"
    echo "Skipping label parsing."
fi

# Step 3: Generate synthetic data (always available as fallback)
echo ""
echo "[Step 3/5] Generating synthetic data with edge-tts..."

python -m src.data.synthetic_gen \
    --output "$SYNTHETIC_DIR" \
    --samples-per-tone "$SYNTHETIC_SAMPLES"

# Step 4: Run MFA alignment (if AISHELL-3 available)
echo ""
echo "[Step 4/5] Running forced alignment with MFA..."

if [ -d "$RAW_DIR/aishell3" ]; then
    # Check if MFA is installed
    if command -v mfa &> /dev/null; then
        python -m src.data.align \
            --corpus "$RAW_DIR/aishell3/train" \
            --output "$ALIGNMENTS_DIR" \
            --jobs "$MFA_JOBS" \
            --download-models || {
            echo "Warning: MFA alignment failed."
            echo "You may need to install MFA: conda install -c conda-forge montreal-forced-aligner"
        }
    else
        echo "Warning: MFA not installed. Skipping alignment."
        echo "Install with: conda install -c conda-forge montreal-forced-aligner"
    fi
else
    echo "Warning: AISHELL-3 not found. Skipping alignment."
fi

# Step 5: Extract segments
echo ""
echo "[Step 5/5] Extracting syllable segments..."

if [ -f "$PROCESSED_DIR/labels.csv" ] && [ -d "$ALIGNMENTS_DIR" ]; then
    python -m src.data.extract_segments \
        --alignments "$ALIGNMENTS_DIR" \
        --audio "$RAW_DIR/aishell3/train/wav" \
        --labels "$PROCESSED_DIR/labels.csv" \
        --output "$PROCESSED_DIR/segments"
else
    echo "Warning: Skipping segment extraction (missing labels or alignments)."
fi

# Summary
echo ""
echo "========================================"
echo "Data preparation complete!"
echo "========================================"
echo ""
echo "Directories created:"
echo "  Raw data: $RAW_DIR"
echo "  Processed: $PROCESSED_DIR"
echo "  Synthetic: $SYNTHETIC_DIR"
echo "  Alignments: $ALIGNMENTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Extract MMS features: python -m src.features.mms_encoder"
echo "  2. Extract F0 features: python -m src.features.f0_extract"
echo "  3. Train model: ./scripts/train.sh"

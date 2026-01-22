#!/bin/bash
# Run the real-time Mandarin Tone Coach

set -e

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

echo "========================================"
echo "Mandarin Tone Coach - Real-time Mode"
echo "========================================"
echo ""

# Find the best checkpoint
CHECKPOINT=""
if [ -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/best_model*.pt 2>/dev/null | head -n 1)
fi

# Parse arguments
MODE="coach"
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            shift
            ;;
        --benchmark)
            MODE="benchmark"
            shift
            ;;
        --interactive)
            MODE="interactive"
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --file)
            MODE="file"
            FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test         Run latency test"
            echo "  --benchmark    Run full benchmark"
            echo "  --interactive  Run interactive mode (without coach UI)"
            echo "  --checkpoint   Path to model checkpoint"
            echo "  --file PATH    Classify a specific audio file"
            echo ""
            exit 1
            ;;
    esac
done

# Check for checkpoint
if [ -n "$CHECKPOINT" ]; then
    echo "Using checkpoint: $CHECKPOINT"
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
else
    echo "Warning: No checkpoint found. Using untrained model."
    echo "Run ./scripts/train.sh first for accurate predictions."
    CHECKPOINT_ARG=""
fi

echo ""

case $MODE in
    test)
        echo "Running latency test..."
        python -m src.realtime.pipeline --test $CHECKPOINT_ARG
        ;;
    benchmark)
        echo "Running benchmark..."
        python -m src.realtime.pipeline --benchmark $CHECKPOINT_ARG
        ;;
    interactive)
        echo "Starting interactive mode..."
        echo "Speak Mandarin syllables into the microphone."
        echo "Press Ctrl+C to stop."
        echo ""
        python -m src.realtime.pipeline --interactive $CHECKPOINT_ARG
        ;;
    file)
        if [ -z "$FILE" ]; then
            echo "Error: No file specified."
            exit 1
        fi
        echo "Classifying file: $FILE"
        python -m src.realtime.pipeline --file "$FILE" $CHECKPOINT_ARG
        ;;
    coach)
        echo "Starting Tone Coach..."
        echo ""
        python -m src.ui.coach $CHECKPOINT_ARG
        ;;
esac

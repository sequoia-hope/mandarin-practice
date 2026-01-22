"""Extract syllable audio segments using alignment information."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm


@dataclass
class SyllableSegment:
    """A single syllable audio segment with metadata."""

    utt_id: str
    syllable_idx: int
    pinyin: str
    tone: int  # 0-4
    start_time: float
    end_time: float
    audio: np.ndarray
    sample_rate: int


def extract_segment(
    audio_path: str | Path,
    start_time: float,
    end_time: float,
    padding: float = 0.02,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Extract a segment from an audio file.

    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        padding: Padding in seconds to add on both sides
        target_sr: Target sample rate

    Returns:
        (audio_array, sample_rate)
    """
    import librosa

    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)

    # Calculate sample indices with padding
    start_sample = max(0, int((start_time - padding) * sr))
    end_sample = min(len(audio), int((end_time + padding) * sr))

    # Extract segment
    segment = audio[start_sample:end_sample]

    return segment, sr


def extract_syllable_segments(
    alignments_dir: str | Path,
    audio_dir: str | Path,
    labels_df,  # pandas DataFrame from parse_labels
    output_dir: str | Path,
    target_sr: int = 16000,
    min_duration: float = 0.05,
    max_duration: float = 1.0,
    padding: float = 0.02,
) -> list[dict]:
    """
    Extract syllable segments using alignment info.

    Args:
        alignments_dir: Directory with TextGrid files
        audio_dir: Directory with original audio files
        labels_df: DataFrame with pinyin/tone labels
        output_dir: Output directory for segments
        target_sr: Target sample rate
        min_duration: Minimum segment duration
        max_duration: Maximum segment duration
        padding: Padding around segments

    Returns:
        List of segment metadata dicts
    """
    from .parse_textgrid import parse_textgrid_directory

    alignments_dir = Path(alignments_dir)
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group labels by utterance
    utt_labels = labels_df.groupby("utt_id")

    segments_meta = []
    segment_idx = 0

    for alignment in tqdm(
        parse_textgrid_directory(alignments_dir),
        desc="Extracting segments",
    ):
        utt_id = alignment.utt_id

        # Find audio file
        audio_path = None
        for pattern in [
            audio_dir / f"{utt_id}.wav",
            audio_dir / utt_id[:7] / f"{utt_id}.wav",
            audio_dir / "**" / f"{utt_id}.wav",
        ]:
            matches = list(audio_dir.glob(str(pattern).replace(str(audio_dir) + "/", "")))
            if matches:
                audio_path = matches[0]
                break

        if audio_path is None or not audio_path.exists():
            continue

        # Get labels for this utterance
        if utt_id not in utt_labels.groups:
            continue

        utt_df = utt_labels.get_group(utt_id).sort_values("syllable_idx")

        # Match words to syllables
        for word_idx, word in enumerate(alignment.words):
            if word_idx >= len(utt_df):
                break

            row = utt_df.iloc[word_idx]

            # Check duration constraints
            duration = word.end_time - word.start_time
            if duration < min_duration or duration > max_duration:
                continue

            try:
                # Extract audio segment
                segment_audio, sr = extract_segment(
                    audio_path,
                    word.start_time,
                    word.end_time,
                    padding=padding,
                    target_sr=target_sr,
                )

                # Save segment
                segment_filename = f"{utt_id}_{word_idx:03d}_tone{row['tone']}.wav"
                segment_path = output_dir / segment_filename
                sf.write(segment_path, segment_audio, sr)

                # Store metadata
                segments_meta.append({
                    "segment_id": segment_idx,
                    "utt_id": utt_id,
                    "syllable_idx": word_idx,
                    "pinyin": row["syllable_pinyin"],
                    "tone": int(row["tone"]),
                    "start_time": word.start_time,
                    "end_time": word.end_time,
                    "duration": duration,
                    "audio_path": str(segment_path),
                    "sample_rate": sr,
                })

                segment_idx += 1

            except Exception as e:
                print(f"Error extracting {utt_id} word {word_idx}: {e}")
                continue

    return segments_meta


def create_dataset_splits(
    segments_meta: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Split segments into train/val/test sets.

    Stratified by tone class.
    """
    import random

    random.seed(seed)

    # Group by tone
    by_tone = {}
    for seg in segments_meta:
        tone = seg["tone"]
        if tone not in by_tone:
            by_tone[tone] = []
        by_tone[tone].append(seg)

    splits = {"train": [], "val": [], "test": []}

    for tone, segs in by_tone.items():
        random.shuffle(segs)

        n = len(segs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits["train"].extend(segs[:n_train])
        splits["val"].extend(segs[n_train : n_train + n_val])
        splits["test"].extend(segs[n_train + n_val :])

    # Shuffle each split
    for split in splits.values():
        random.shuffle(split)

    return splits


def save_manifest(segments: list[dict], output_path: str | Path) -> None:
    """Save segment manifest as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)


def load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load segment manifest from JSON."""
    with open(manifest_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Extract syllable segments")
    parser.add_argument("--alignments", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True, help="Labels CSV")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels)

    segments = extract_syllable_segments(
        args.alignments,
        args.audio,
        labels_df,
        args.output,
        target_sr=args.sample_rate,
    )

    print(f"Extracted {len(segments)} segments")

    # Create splits
    splits = create_dataset_splits(segments)
    for split_name, split_segs in splits.items():
        manifest_path = Path(args.output) / f"{split_name}_manifest.json"
        save_manifest(split_segs, manifest_path)
        print(f"{split_name}: {len(split_segs)} segments")

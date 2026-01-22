"""Parse pinyin and tone labels from AISHELL-3 dataset."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pandas as pd


# Tone number mapping
TONE_MAP = {
    "1": 0,  # High level (ˉ)
    "2": 1,  # Rising (ˊ)
    "3": 2,  # Dipping (ˇ)
    "4": 3,  # Falling (ˋ)
    "5": 4,  # Neutral tone (轻声)
}

# Valid pinyin initials
INITIALS = [
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "zh", "ch", "sh", "r",
    "z", "c", "s",
    "y", "w",
]

# Pinyin syllable pattern: optional initial + final + tone number
PINYIN_PATTERN = re.compile(
    r"([a-z]+)([1-5])",
    re.IGNORECASE,
)


@dataclass
class Syllable:
    """Represents a single Mandarin syllable with tone."""

    pinyin: str  # Full pinyin without tone number (e.g., "ma")
    tone: int  # Tone class 0-4
    tone_number: str  # Original tone number 1-5
    raw: str  # Original string (e.g., "ma1")


@dataclass
class Utterance:
    """Represents a complete utterance with all syllables."""

    utt_id: str
    audio_path: Path
    syllables: list[Syllable]
    text: str  # Original transcription


def parse_pinyin_syllable(pinyin_str: str) -> Syllable | None:
    """
    Parse a single pinyin syllable with tone number.

    Args:
        pinyin_str: Pinyin string like "ma1", "shi4", etc.

    Returns:
        Syllable object or None if parsing fails
    """
    match = PINYIN_PATTERN.match(pinyin_str.strip())
    if not match:
        return None

    base_pinyin = match.group(1).lower()
    tone_number = match.group(2)

    if tone_number not in TONE_MAP:
        return None

    return Syllable(
        pinyin=base_pinyin,
        tone=TONE_MAP[tone_number],
        tone_number=tone_number,
        raw=pinyin_str,
    )


def parse_pinyin_sequence(text: str) -> list[Syllable]:
    """
    Parse a sequence of pinyin syllables.

    Args:
        text: Space-separated pinyin sequence like "ma1 ma2 ma3 ma4"

    Returns:
        List of Syllable objects
    """
    syllables = []
    for token in text.split():
        syllable = parse_pinyin_syllable(token)
        if syllable:
            syllables.append(syllable)
    return syllables


def parse_aishell3_labels(
    label_file: str | Path,
    audio_dir: str | Path,
) -> Generator[Utterance, None, None]:
    """
    Parse AISHELL-3 label file and yield utterances.

    AISHELL-3 label format:
    utt_id|speaker_id|pinyin sequence|character sequence

    Args:
        label_file: Path to label.txt or content.txt
        audio_dir: Directory containing audio files

    Yields:
        Utterance objects with parsed syllables
    """
    label_file = Path(label_file)
    audio_dir = Path(audio_dir)

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 3:
                continue

            utt_id = parts[0].strip()
            pinyin_seq = parts[2].strip() if len(parts) > 2 else parts[1].strip()

            # Find audio file
            audio_path = audio_dir / f"{utt_id}.wav"
            if not audio_path.exists():
                # Try nested structure (speaker/utt_id.wav)
                speaker_id = utt_id[:7] if len(utt_id) > 7 else utt_id[:4]
                audio_path = audio_dir / speaker_id / f"{utt_id}.wav"

            syllables = parse_pinyin_sequence(pinyin_seq)

            if syllables:
                yield Utterance(
                    utt_id=utt_id,
                    audio_path=audio_path,
                    syllables=syllables,
                    text=pinyin_seq,
                )


def load_labels_to_dataframe(
    label_file: str | Path,
    audio_dir: str | Path,
) -> pd.DataFrame:
    """
    Load AISHELL-3 labels into a pandas DataFrame.

    Returns DataFrame with columns:
    - utt_id: Utterance ID
    - audio_path: Path to audio file
    - pinyin: Full pinyin sequence
    - syllable_idx: Index of syllable in utterance
    - syllable_pinyin: Pinyin of individual syllable
    - tone: Tone class (0-4)
    """
    records = []
    for utt in parse_aishell3_labels(label_file, audio_dir):
        for idx, syl in enumerate(utt.syllables):
            records.append({
                "utt_id": utt.utt_id,
                "audio_path": str(utt.audio_path),
                "pinyin": utt.text,
                "syllable_idx": idx,
                "syllable_pinyin": syl.pinyin,
                "tone": syl.tone,
                "tone_number": syl.tone_number,
            })

    return pd.DataFrame(records)


def get_tone_distribution(df: pd.DataFrame) -> dict[int, int]:
    """Get distribution of tones in dataset."""
    return df["tone"].value_counts().to_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse AISHELL-3 labels")
    parser.add_argument("label_file", type=str, help="Path to label file")
    parser.add_argument("audio_dir", type=str, help="Audio directory")
    parser.add_argument("--output", type=str, help="Output CSV file")
    args = parser.parse_args()

    df = load_labels_to_dataframe(args.label_file, args.audio_dir)
    print(f"Loaded {len(df)} syllables from {df['utt_id'].nunique()} utterances")
    print("\nTone distribution:")
    for tone, count in sorted(get_tone_distribution(df).items()):
        print(f"  Tone {tone + 1}: {count}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

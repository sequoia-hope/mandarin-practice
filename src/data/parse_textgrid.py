"""Parse MFA TextGrid output for syllable timing information."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import tgt


@dataclass
class PhoneInterval:
    """A single phone with timing information."""

    phone: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class WordInterval:
    """A word (syllable) with timing and phone breakdown."""

    text: str
    start_time: float
    end_time: float
    duration: float
    phones: list[PhoneInterval]


@dataclass
class AlignmentResult:
    """Complete alignment result for an utterance."""

    utt_id: str
    words: list[WordInterval]
    phones: list[PhoneInterval]
    total_duration: float


def parse_textgrid(textgrid_path: str | Path) -> AlignmentResult:
    """
    Parse a TextGrid file from MFA alignment.

    Args:
        textgrid_path: Path to .TextGrid file

    Returns:
        AlignmentResult with word and phone timing
    """
    textgrid_path = Path(textgrid_path)
    utt_id = textgrid_path.stem

    tg = tgt.io.read_textgrid(str(textgrid_path))

    # Get tiers
    words_tier = None
    phones_tier = None

    for tier in tg.tiers:
        tier_name = tier.name.lower()
        if "word" in tier_name:
            words_tier = tier
        elif "phone" in tier_name:
            phones_tier = tier

    # Parse phones
    phones = []
    if phones_tier:
        for interval in phones_tier.intervals:
            if interval.text and interval.text.strip():
                phones.append(
                    PhoneInterval(
                        phone=interval.text.strip(),
                        start_time=interval.start_time,
                        end_time=interval.end_time,
                        duration=interval.end_time - interval.start_time,
                    )
                )

    # Parse words and associate phones
    words = []
    if words_tier:
        for interval in words_tier.intervals:
            if interval.text and interval.text.strip():
                # Find phones within this word's time range
                word_phones = [
                    p
                    for p in phones
                    if p.start_time >= interval.start_time - 0.001
                    and p.end_time <= interval.end_time + 0.001
                ]

                words.append(
                    WordInterval(
                        text=interval.text.strip(),
                        start_time=interval.start_time,
                        end_time=interval.end_time,
                        duration=interval.end_time - interval.start_time,
                        phones=word_phones,
                    )
                )

    total_duration = tg.end_time if hasattr(tg, "end_time") else 0.0
    if words:
        total_duration = max(total_duration, words[-1].end_time)

    return AlignmentResult(
        utt_id=utt_id,
        words=words,
        phones=phones,
        total_duration=total_duration,
    )


def parse_textgrid_directory(
    textgrid_dir: str | Path,
    pattern: str = "**/*.TextGrid",
) -> Generator[AlignmentResult, None, None]:
    """
    Parse all TextGrid files in a directory.

    Args:
        textgrid_dir: Directory containing TextGrid files
        pattern: Glob pattern for finding files

    Yields:
        AlignmentResult for each file
    """
    textgrid_dir = Path(textgrid_dir)

    for tg_path in sorted(textgrid_dir.glob(pattern)):
        try:
            yield parse_textgrid(tg_path)
        except Exception as e:
            print(f"Error parsing {tg_path}: {e}")
            continue


def get_syllable_timings(
    alignment: AlignmentResult,
    pinyin_sequence: list[str],
) -> list[tuple[str, float, float, int]]:
    """
    Match aligned words to pinyin sequence with tones.

    Args:
        alignment: Parsed alignment result
        pinyin_sequence: List of pinyin syllables with tone numbers

    Returns:
        List of (pinyin_with_tone, start_time, end_time, tone_class)
    """
    import re

    results = []

    # Extract tones from pinyin
    pinyin_tones = []
    for pinyin in pinyin_sequence:
        match = re.match(r"([a-z]+)(\d)", pinyin, re.IGNORECASE)
        if match:
            base = match.group(1).lower()
            tone = int(match.group(2)) - 1  # Convert to 0-indexed
            pinyin_tones.append((base, tone, pinyin))

    # Match words to pinyin (simple sequential matching)
    word_idx = 0
    for base, tone, full_pinyin in pinyin_tones:
        if word_idx >= len(alignment.words):
            break

        word = alignment.words[word_idx]
        # Check if word matches (MFA may have slightly different pinyin)
        if word.text.lower() == base or word.text.lower().startswith(base[:2]):
            results.append((full_pinyin, word.start_time, word.end_time, tone))
            word_idx += 1
        else:
            # Try to find matching word
            for i, w in enumerate(alignment.words[word_idx:], word_idx):
                if w.text.lower() == base or w.text.lower().startswith(base[:2]):
                    results.append((full_pinyin, w.start_time, w.end_time, tone))
                    word_idx = i + 1
                    break

    return results


def alignment_to_dict(alignment: AlignmentResult) -> dict:
    """Convert AlignmentResult to serializable dict."""
    return {
        "utt_id": alignment.utt_id,
        "total_duration": alignment.total_duration,
        "words": [
            {
                "text": w.text,
                "start": w.start_time,
                "end": w.end_time,
                "duration": w.duration,
                "phones": [
                    {"phone": p.phone, "start": p.start_time, "end": p.end_time}
                    for p in w.phones
                ],
            }
            for w in alignment.words
        ],
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse MFA TextGrid files")
    parser.add_argument("input", type=str, help="TextGrid file or directory")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        result = parse_textgrid(input_path)
        results = [alignment_to_dict(result)]
    else:
        results = [alignment_to_dict(r) for r in parse_textgrid_directory(input_path)]

    print(f"Parsed {len(results)} alignments")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        for r in results[:3]:
            print(json.dumps(r, indent=2))

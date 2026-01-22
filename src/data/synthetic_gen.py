"""Generate synthetic Mandarin syllable samples using edge-tts."""

import asyncio
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm


# Common syllables for each tone (representative examples)
TONE_SYLLABLES = {
    0: [  # Tone 1 (high level)
        "mā", "bā", "fēi", "gāo", "shū", "tīng", "xīn", "zhōng", "chuāng", "guāng",
        "tiān", "jiā", "kāi", "sān", "bān", "dān", "fān", "gān", "hān", "kān",
    ],
    1: [  # Tone 2 (rising)
        "má", "bá", "lái", "nán", "shí", "rén", "xuē", "yóu", "chéng", "míng",
        "hái", "nián", "qián", "cái", "hé", "xué", "guó", "shén", "méi", "néng",
    ],
    2: [  # Tone 3 (dipping)
        "mǎ", "bǎ", "hǎo", "xiǎo", "wǒ", "nǐ", "yǔ", "jiǔ", "shǒu", "zǒu",
        "kě", "lǐ", "měi", "nǎ", "yě", "zhǐ", "bǐ", "fǔ", "gǔ", "hǔ",
    ],
    3: [  # Tone 4 (falling)
        "mà", "bà", "dà", "shì", "yào", "zuò", "lù", "qù", "huà", "jiàn",
        "dào", "hòu", "kuài", "lèi", "mèi", "nèi", "pài", "sài", "tài", "wài",
    ],
    4: [  # Tone 5 (neutral)
        "ma", "ba", "de", "le", "ne", "me", "zi", "tou", "men", "ge",
        "a", "ya", "wa", "na", "la",
    ],
}

# Mapping from tone-marked pinyin to numbered pinyin
TONE_MARKS_TO_NUMBERS = {
    "ā": "a1", "á": "a2", "ǎ": "a3", "à": "a4",
    "ē": "e1", "é": "e2", "ě": "e3", "è": "e4",
    "ī": "i1", "í": "i2", "ǐ": "i3", "ì": "i4",
    "ō": "o1", "ó": "o2", "ǒ": "o3", "ò": "o4",
    "ū": "u1", "ú": "u2", "ǔ": "u3", "ù": "u4",
    "ǖ": "v1", "ǘ": "v2", "ǚ": "v3", "ǜ": "v4",
}

# Chinese voices available in edge-tts
CHINESE_VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunyangNeural",
    "zh-CN-XiaochenNeural",
    "zh-CN-XiaohanNeural",
    "zh-CN-XiaomengNeural",
    "zh-CN-XiaomoNeural",
    "zh-CN-XiaoqiuNeural",
    "zh-CN-XiaoruiNeural",
    "zh-CN-XiaoshuangNeural",
    "zh-CN-XiaoxuanNeural",
    "zh-CN-XiaoyanNeural",
    "zh-CN-XiaoyouNeural",
    "zh-CN-YunfengNeural",
    "zh-CN-YunhaoNeural",
    "zh-CN-YunxiaNeural",
    "zh-CN-YunyeNeural",
]


def pinyin_marked_to_numbered(pinyin: str) -> tuple[str, int]:
    """
    Convert tone-marked pinyin to numbered format.

    Returns (base_pinyin, tone_number)
    """
    base = pinyin
    tone = 5  # Default neutral tone

    for mark, replacement in TONE_MARKS_TO_NUMBERS.items():
        if mark in pinyin:
            base = pinyin.replace(mark, replacement[0])
            tone = int(replacement[1])
            break

    return base, tone


async def synthesize_syllable(
    text: str,
    output_path: str | Path,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "-10%",
) -> bool:
    """
    Synthesize a single syllable using edge-tts.

    Args:
        text: Text to synthesize (single syllable with tone mark)
        output_path: Output audio file path
        voice: Edge-tts voice name
        rate: Speech rate adjustment

    Returns:
        True if successful
    """
    import edge_tts

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    communicate = edge_tts.Communicate(text, voice, rate=rate)

    try:
        await communicate.save(str(output_path))
        return True
    except Exception as e:
        print(f"Error synthesizing '{text}': {e}")
        return False


async def generate_synthetic_samples(
    output_dir: str | Path,
    samples_per_tone: int = 100,
    voices: Optional[list[str]] = None,
    target_sr: int = 16000,
) -> list[dict]:
    """
    Generate synthetic syllable samples for all tones.

    Args:
        output_dir: Output directory
        samples_per_tone: Number of samples per tone class
        voices: List of voices to use (cycles through)
        target_sr: Target sample rate for output

    Returns:
        List of segment metadata
    """
    import librosa

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if voices is None:
        voices = CHINESE_VOICES[:5]  # Use first 5 voices

    segments_meta = []
    segment_idx = 0

    for tone_class, syllables in TONE_SYLLABLES.items():
        tone_dir = output_dir / f"tone{tone_class}"
        tone_dir.mkdir(exist_ok=True)

        samples_generated = 0
        syllable_idx = 0

        pbar = tqdm(
            total=samples_per_tone,
            desc=f"Tone {tone_class + 1}",
        )

        while samples_generated < samples_per_tone:
            syllable = syllables[syllable_idx % len(syllables)]
            voice = voices[samples_generated % len(voices)]

            # Generate unique filename
            base_pinyin, _ = pinyin_marked_to_numbered(syllable)
            filename = f"{base_pinyin}_tone{tone_class}_{samples_generated:04d}.wav"
            temp_path = tone_dir / f"temp_{filename}.mp3"
            final_path = tone_dir / filename

            # Synthesize
            success = await synthesize_syllable(
                syllable,
                temp_path,
                voice=voice,
            )

            if success and temp_path.exists():
                try:
                    # Convert to target sample rate and format
                    audio, sr = librosa.load(temp_path, sr=target_sr)
                    sf.write(final_path, audio, target_sr)

                    # Clean up temp file
                    temp_path.unlink()

                    # Store metadata
                    segments_meta.append({
                        "segment_id": segment_idx,
                        "utt_id": f"synthetic_{segment_idx}",
                        "syllable_idx": 0,
                        "pinyin": base_pinyin,
                        "tone": tone_class,
                        "duration": len(audio) / target_sr,
                        "audio_path": str(final_path),
                        "sample_rate": target_sr,
                        "voice": voice,
                        "synthetic": True,
                    })

                    segment_idx += 1
                    samples_generated += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"Error processing {temp_path}: {e}")
                    if temp_path.exists():
                        temp_path.unlink()

            syllable_idx += 1

            # Avoid infinite loop
            if syllable_idx > samples_per_tone * 3:
                break

        pbar.close()

    return segments_meta


def generate_balanced_dataset(
    output_dir: str | Path,
    total_samples: int = 5000,
    voices: Optional[list[str]] = None,
) -> list[dict]:
    """
    Generate a balanced dataset across all tone classes.

    Args:
        output_dir: Output directory
        total_samples: Total number of samples to generate
        voices: Voices to use

    Returns:
        List of segment metadata
    """
    samples_per_tone = total_samples // 5

    return asyncio.run(
        generate_synthetic_samples(
            output_dir,
            samples_per_tone=samples_per_tone,
            voices=voices,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic samples")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--samples-per-tone", type=int, default=100)
    parser.add_argument("--total-samples", type=int, default=None)
    args = parser.parse_args()

    if args.total_samples:
        segments = generate_balanced_dataset(
            args.output,
            total_samples=args.total_samples,
        )
    else:
        segments = asyncio.run(
            generate_synthetic_samples(
                args.output,
                samples_per_tone=args.samples_per_tone,
            )
        )

    # Save manifest
    manifest_path = Path(args.output) / "synthetic_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"Generated {len(segments)} synthetic samples")
    print(f"Manifest saved to {manifest_path}")

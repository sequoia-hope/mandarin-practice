#!/usr/bin/env python3
"""
Pre-generate audio for all built-in lesson phrases.

This creates a cache of TTS audio so lessons load quickly without
waiting for synthesis.

Usage:
    python scripts/generate_lesson_audio.py --backend qwen
    python scripts/generate_lesson_audio.py --backend light
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Pre-generate lesson audio")
    parser.add_argument(
        "--backend",
        choices=["light", "qwen"],
        default="qwen",
        help="TTS backend to use (default: qwen for high quality)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data" / "lessons" / "audio"),
        help="Output directory for audio files"
    )
    parser.add_argument(
        "--speed",
        choices=["teacher", "slow", "normal", "fast"],
        default="teacher",
        help="Speaking speed (default: teacher - slow and deliberate for language learning)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't generate audio, just show what would be generated"
    )
    parser.add_argument(
        "--lesson",
        type=str,
        default=None,
        help="Generate audio for specific lesson only (by ID)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import lesson data
    from src.coach.voice_menu import BUILT_IN_LESSONS

    # Filter to specific lesson if requested
    if args.lesson:
        lessons = [l for l in BUILT_IN_LESSONS if l.id == args.lesson]
        if not lessons:
            print(f"Lesson not found: {args.lesson}")
            print(f"Available lessons: {[l.id for l in BUILT_IN_LESSONS]}")
            return
    else:
        lessons = BUILT_IN_LESSONS

    # Count total phrases
    total_phrases = sum(len(l.phrases) for l in lessons)
    print(f"Generating audio for {total_phrases} phrases across {len(lessons)} lessons")
    print(f"Backend: {args.backend}")
    print(f"Output: {output_dir}")
    print()

    if args.dry_run:
        for lesson in lessons:
            print(f"\n{lesson.name} ({lesson.id}):")
            for phrase in lesson.phrases:
                print(f"  - {phrase['characters']} ({phrase['english']})")
        return

    # Initialize TTS
    from src.coach.tts import create_tts
    tts = create_tts(args.backend)

    # Generate audio for each phrase
    manifest = {}
    generated = 0
    skipped = 0

    for lesson in lessons:
        lesson_dir = output_dir / lesson.id
        lesson_dir.mkdir(exist_ok=True)

        print(f"\n{lesson.name}:")

        lesson_manifest = []

        for i, phrase in enumerate(lesson.phrases):
            chars = phrase["characters"]
            pinyin = phrase["pinyin"]
            english = phrase["english"]

            # Create filename from characters (sanitized)
            safe_name = chars.replace(" ", "_")[:20]
            audio_path = lesson_dir / f"{i:02d}_{safe_name}.wav"

            if audio_path.exists():
                print(f"  [skip] {chars} (already exists)")
                skipped += 1
            else:
                print(f"  [gen]  {chars}...", end=" ", flush=True)
                try:
                    audio = tts.speak(chars, language="zh", speed=args.speed, play=False)

                    # Save audio
                    import soundfile as sf
                    sf.write(str(audio_path), audio, 16000)

                    duration_ms = len(audio) / 16
                    print(f"OK ({duration_ms:.0f}ms)")
                    generated += 1

                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            # Add to manifest
            lesson_manifest.append({
                "index": i,
                "characters": chars,
                "pinyin": pinyin,
                "english": english,
                "audio_path": str(audio_path.relative_to(output_dir)),
            })

        manifest[lesson.id] = lesson_manifest

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Generated: {generated} files")
    print(f"  Skipped:   {skipped} files (already exist)")
    print(f"  Manifest:  {manifest_path}")


if __name__ == "__main__":
    main()

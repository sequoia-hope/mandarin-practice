#!/usr/bin/env python3
"""Run the Mandarin Tone Coach in voice-controlled mode.

Designed for hands-free operation with Bluetooth headphones.
Uses a phone tree style menu system with audio feedback.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Mandarin Tone Coach - Voice Controlled Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This mode is designed for hands-free operation with Bluetooth headphones.
Navigate using voice commands or keyboard input as fallback.

Input Modes:
  --text-mode     Use keyboard input for commands (default)
  --voice-mode    Use voice recognition for commands (requires Whisper)

TTS Options:
  --tts light     Use edge-tts for all audio (fast, free)
  --tts qwen      Use Qwen3-TTS for all audio (high quality, requires GPU)
  --tts dual      Use edge-tts for English, Qwen3-TTS for Chinese (default)
  --no-tts        Disable TTS (text output only)

Menu commands:
  one/two/three...  - Select menu option
  menu / back       - Return to main menu
  skip / next       - Skip current phrase
  repeat / again    - Repeat current phrase
  quit / exit       - Exit the program

Audio feedback:
  Success chime  - You got it right!
  Try again tone - Almost, let's retry
  Menu blip      - Navigating menus

Examples:
  python run_voice.py                    # Text mode, dual TTS
  python run_voice.py --text-mode        # Text mode (explicit)
  python run_voice.py --voice-mode       # Voice mode with ASR
  python run_voice.py --tts qwen         # Use Qwen3-TTS for everything
  python run_voice.py --no-tts           # No TTS, text only
"""
    )

    # Input mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--text-mode",
        action="store_true",
        default=True,
        help="Use keyboard input for commands (default)"
    )
    mode_group.add_argument(
        "--voice-mode",
        action="store_true",
        help="Use voice recognition for commands"
    )

    # TTS options
    parser.add_argument(
        "--tts",
        choices=["light", "qwen", "dual"],
        default="dual",
        help="TTS backend: light (edge-tts), qwen (Qwen3-TTS), dual (both)"
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech (text only)"
    )

    # ASR options
    parser.add_argument(
        "--asr",
        choices=["whisper", "faster"],
        default="whisper",
        help="ASR backend for voice mode (default: whisper)"
    )
    parser.add_argument(
        "--asr-model",
        choices=["tiny", "base", "small"],
        default="tiny",
        help="ASR model size (default: tiny for speed)"
    )

    # Audio options
    parser.add_argument(
        "--no-sounds",
        action="store_true",
        help="Disable audio feedback sounds"
    )

    # Device options
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode (no CUDA/GPU)"
    )

    # Testing
    parser.add_argument(
        "--test-sounds",
        action="store_true",
        help="Test audio feedback sounds and exit"
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Test TTS and exit"
    )
    parser.add_argument(
        "--test-eval",
        action="store_true",
        help="Test evaluator and exit"
    )

    args = parser.parse_args()

    # Set CPU mode if requested (before any model imports)
    if args.cpu:
        os.environ["TONE_COACH_DEVICE"] = "cpu"

    # Test sounds mode
    if args.test_sounds:
        print("Testing audio feedback sounds...")
        from src.coach.audio_feedback import AudioFeedback
        import time

        feedback = AudioFeedback()

        sounds = [
            ("Success (correct answer)", "success"),
            ("Perfect (all correct)", "perfect"),
            ("Try again (incorrect)", "try_again"),
            ("Menu transition", "menu"),
            ("Listening indicator", "listening"),
            ("Error/invalid", "error"),
        ]

        for name, sound in sounds:
            print(f"  {name}...")
            feedback.play(sound, blocking=True)
            time.sleep(0.3)

        print("Done!")
        return

    # Test TTS mode
    if args.test_tts:
        print(f"Testing TTS backend: {args.tts}...")
        from src.coach.tts import create_tts

        tts = create_tts(args.tts)

        print("\nEnglish:")
        tts.speak("Welcome to the Mandarin Tone Coach!", language="en")

        print("\nChinese:")
        tts.speak("你好", language="zh")

        print("\nDone!")
        return

    # Test evaluator mode
    if args.test_eval:
        print("Testing evaluator...")
        from src.coach.evaluator import Evaluator

        evaluator = Evaluator(lazy_load=False)

        print("\nSay '你好' (ni3 hao3)...")
        result = evaluator.evaluate_from_mic(
            expected_pinyin=["ni", "hao"],
            expected_tones=[3, 3],
            characters="你好",
        )

        print(f"\nStatus: {result.status}")
        print(f"Overall accuracy: {result.overall_accuracy:.1%}")

        for r in result.syllable_results:
            status = "OK" if r.correct else "X"
            print(
                f"  {status} {r.character} ({r.pinyin}): "
                f"expected T{r.expected_tone}, got T{r.detected_tone} "
                f"(conf: {r.confidence:.2f})"
            )

        print("\nDone!")
        return

    # Import components (lazy to speed up --help)
    from src.coach.voice_menu import VoiceMenu, InputMode

    # Set up TTS
    tts = None
    if not args.no_tts:
        try:
            from src.coach.tts import create_tts

            use_qwen_for_all = (args.tts == "qwen")
            tts = create_tts(
                backend=args.tts if args.tts != "qwen" else "dual",
                use_qwen_for_all=use_qwen_for_all,
            )
            print(f"[TTS] Using {args.tts} backend")
        except Exception as e:
            print(f"[TTS] Failed to initialize: {e}")
            print("[TTS] Running without TTS")

    # Set up ASR (only in voice mode)
    asr = None
    if args.voice_mode:
        try:
            from src.coach.command_asr import create_asr

            asr = create_asr(
                backend=args.asr,
                model_size=args.asr_model,
            )
            print(f"[ASR] Using {args.asr} ({args.asr_model}) backend")
        except Exception as e:
            print(f"[ASR] Failed to initialize: {e}")
            print("[ASR] Falling back to text mode")
            args.voice_mode = False

    # Determine input mode
    input_mode = InputMode.VOICE if args.voice_mode else InputMode.TEXT

    # Set up conversation callback (links to coach)
    def conversation_callback():
        """Handle conversation mode - link to LLM coach."""
        print("\n  Conversation mode - Type your question:")
        print("  (Type 'menu' to go back)\n")

        try:
            from src.coach.coach import create_coach
            from src.coach.tools import ToolExecutor

            # Create coach with Ollama backend (local-first)
            try:
                coach = create_coach(backend="ollama", model="qwen2.5:7b")
                executor = ToolExecutor(tts=tts, lazy_load=True)
                coach.tools = executor
            except Exception:
                print("  [Ollama not available, trying Claude...]")
                try:
                    coach = create_coach(backend="claude")
                except Exception as e:
                    print(f"  [No LLM backend available: {e}]")
                    return "menu"

            while True:
                try:
                    user_input = input("  You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    return "menu"

                if not user_input:
                    continue

                if user_input.lower() in ["menu", "back", "quit", "exit"]:
                    return "menu"

                response = coach.chat(user_input)
                print(f"\n  Coach: {response}\n")

        except Exception as e:
            print(f"  [Conversation error: {e}]")
            return "menu"

    # Create voice menu
    menu = VoiceMenu(
        tts=tts,
        asr=asr,
        input_mode=input_mode,
        conversation_callback=conversation_callback,
    )

    if args.no_sounds:
        menu.feedback.enabled = False

    # Run
    try:
        menu.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()

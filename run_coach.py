#!/usr/bin/env python3
"""Run the Mandarin Tone Coach in conversational mode.

Uses an LLM to provide interactive, adaptive tone learning.
Default: Uses local Ollama with Qwen2.5 (offline capable).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Mandarin Tone Coach - Interactive Conversation Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This mode uses an LLM to provide interactive, conversational tone learning.
The coach can translate phrases, generate audio, evaluate your pronunciation,
and give personalized feedback.

LLM Backends:
  ollama (default)  - Local LLM via Ollama (offline, free)
  claude            - Claude API (online, requires API key)
  transformers      - HuggingFace models (local, requires GPU)

Recommended Models:
  Ollama:   qwen2.5:7b (default), llama3.1:8b
  Claude:   claude-sonnet-4-20250514 (default)

Examples:
  # Use local Ollama (default, offline)
  python run_coach.py

  # Use specific Ollama model
  python run_coach.py --model llama3.1:8b

  # Use Claude API
  python run_coach.py --backend claude

  # Enable TTS for spoken responses
  python run_coach.py --tts

Setup:
  # For Ollama (recommended):
  1. Install Ollama: https://ollama.ai
  2. Pull model: ollama pull qwen2.5:7b
  3. Run: python run_coach.py

  # For Claude:
  1. Set ANTHROPIC_API_KEY environment variable
  2. Run: python run_coach.py --backend claude
"""
    )

    # Backend options
    parser.add_argument(
        "--backend",
        choices=["ollama", "claude", "transformers"],
        default="ollama",
        help="LLM backend (default: ollama for local inference)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (uses backend default if not specified)"
    )

    # TTS options
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable TTS for spoken responses"
    )
    parser.add_argument(
        "--tts-backend",
        choices=["light", "qwen", "dual"],
        default="dual",
        help="TTS backend if --tts enabled (default: dual)"
    )

    # Other options
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="LLM request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if backend is available and exit"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode (no CUDA/GPU)"
    )

    args = parser.parse_args()

    # Set CPU mode if requested (before any model imports)
    if args.cpu:
        os.environ["TONE_COACH_DEVICE"] = "cpu"

    # Check mode
    if args.check:
        if args.backend == "ollama":
            from src.coach.llm import check_ollama_available
            model = args.model or "qwen2.5:7b"
            if check_ollama_available(model):
                print(f"Ollama is available with model: {model}")
                sys.exit(0)
            else:
                print(f"Ollama not available or model not found: {model}")
                print("Install Ollama: https://ollama.ai")
                print(f"Pull model: ollama pull {model}")
                sys.exit(1)
        elif args.backend == "claude":
            import os
            if os.environ.get("ANTHROPIC_API_KEY"):
                print("Claude API key is configured")
                sys.exit(0)
            else:
                print("ANTHROPIC_API_KEY environment variable not set")
                sys.exit(1)
        else:
            print(f"Backend check not implemented for: {args.backend}")
            sys.exit(1)

    # Import components
    from src.coach.coach import create_coach, run_interactive

    # Set up TTS if enabled
    tts = None
    if args.tts:
        try:
            from src.coach.tts import create_tts
            tts = create_tts(args.tts_backend)
            print(f"[TTS] Using {args.tts_backend} backend")
        except Exception as e:
            print(f"[TTS] Failed to initialize: {e}")

    # Determine backend and model
    backend = args.backend
    model = args.model

    print(f"Starting coach with {backend} backend...")
    if model:
        print(f"Model: {model}")

    # Try to create coach
    try:
        # Build backend kwargs
        backend_kwargs = {}
        if backend == "ollama":
            backend_kwargs["timeout"] = args.timeout

        coach = create_coach(
            backend=backend,
            model=model,
            tts=tts,
            **backend_kwargs,
        )

    except ImportError as e:
        print(f"\nError: {e}")
        if backend == "claude":
            print("Install with: pip install anthropic")
        elif backend == "ollama":
            print("Install with: pip install ollama")
            print("And make sure Ollama is running: ollama serve")
            print("Then pull a model: ollama pull qwen2.5:7b")
        elif backend == "transformers":
            print("Install with: pip install transformers accelerate")
        sys.exit(1)

    except Exception as e:
        print(f"\nError creating coach: {e}")

        # Offer fallback
        if backend == "ollama":
            print("\nOllama not available. Options:")
            print("  1. Install and start Ollama: https://ollama.ai")
            print("  2. Use Claude: python run_coach.py --backend claude")
        elif backend == "claude":
            print("\nClaude API error. Make sure ANTHROPIC_API_KEY is set.")
            print("Or use local Ollama: python run_coach.py --backend ollama")

        sys.exit(1)

    # Run interactive session
    try:
        run_interactive(coach)
    except KeyboardInterrupt:
        print("\n\n再见! (Goodbye!)")
        sys.exit(0)


if __name__ == "__main__":
    main()

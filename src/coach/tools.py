"""Tool definitions and executor for the tone coach."""

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Tool definitions for the LLM
COACH_TOOLS = [
    {
        "name": "translate_to_chinese",
        "description": "Translate English text to Chinese. Returns characters, pinyin with tone marks, and tone numbers for each syllable.",
        "parameters": {
            "english": {
                "type": "string",
                "description": "The English text to translate"
            }
        },
        "required": ["english"]
    },
    {
        "name": "generate_audio",
        "description": "Generate native speaker audio for Chinese text. Use this to create reference audio for the user to listen to.",
        "parameters": {
            "characters": {
                "type": "string",
                "description": "Chinese characters to speak"
            },
            "speed": {
                "type": "string",
                "enum": ["slow", "normal", "fast"],
                "description": "Speaking speed (default: normal)"
            }
        },
        "required": ["characters"]
    },
    {
        "name": "record_and_evaluate",
        "description": "Record user's speech and evaluate their tone pronunciation for each syllable. Returns detailed per-syllable results.",
        "parameters": {
            "characters": {
                "type": "string",
                "description": "The Chinese characters the user should say"
            },
            "pinyin": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of pinyin syllables (without tone marks)"
            },
            "tones": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of expected tone numbers (1-5)"
            }
        },
        "required": ["characters", "pinyin", "tones"]
    },
    {
        "name": "play_audio",
        "description": "Play an audio file. Use to replay reference audio or user's recording.",
        "parameters": {
            "audio_id": {
                "type": "string",
                "description": "ID of audio to play (from generate_audio or record_and_evaluate)"
            }
        },
        "required": ["audio_id"]
    },
    {
        "name": "show_pitch_comparison",
        "description": "Show a visual comparison of user's pitch contour vs the expected tone pattern. Useful for helping users understand what went wrong.",
        "parameters": {
            "syllable_index": {
                "type": "integer",
                "description": "Which syllable to show (0-indexed)"
            },
            "expected_tone": {
                "type": "integer",
                "description": "The expected tone number (1-5)"
            }
        },
        "required": ["syllable_index", "expected_tone"]
    },
]


@dataclass
class EvaluationResultLegacy:
    """Legacy result from evaluating user's speech (for backward compatibility)."""
    syllable_results: list[dict]
    overall_accuracy: float
    audio_id: str
    f0_contours: list[np.ndarray]


class ToolExecutor:
    """Execute tools called by the LLM."""

    def __init__(
        self,
        tts=None,  # TTS instance (LightTTS or QwenTTS)
        evaluator=None,  # Evaluator instance
        use_vad: bool = True,
        translation_llm=None,  # LLM for translation
        lazy_load: bool = True,  # Load evaluator lazily
    ):
        """
        Initialize tool executor.

        Args:
            tts: Text-to-speech instance
            evaluator: Tone evaluator instance. If None and lazy_load=True,
                       will be loaded on first use.
            use_vad: Use VAD for utterance detection
            translation_llm: LLM for translation
            lazy_load: If True, load evaluator lazily on first use
        """
        self.tts = tts
        self._evaluator = evaluator
        self.translation_llm = translation_llm
        self.use_vad = use_vad
        self._lazy_load = lazy_load

        # Audio storage
        self.audio_cache: dict[str, np.ndarray] = {}
        self.last_evaluation = None

        # Temp directory for audio files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tone_coach_"))

    @property
    def evaluator(self):
        """Get evaluator, loading lazily if needed."""
        if self._evaluator is None and self._lazy_load:
            try:
                from .evaluator import Evaluator
                print("[ToolExecutor] Loading evaluator...")
                self._evaluator = Evaluator(lazy_load=True)
            except Exception as e:
                print(f"[ToolExecutor] Failed to load evaluator: {e}")
        return self._evaluator

    def _record_utterance(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        """
        Record a single utterance using VAD.

        Waits for speech to start, then records until silence.
        Returns the complete utterance audio.
        """
        if not self.use_vad:
            # Fallback: fixed duration recording
            import sounddevice as sd
            print("  [Recording for 3 seconds...]")
            audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            return audio.flatten()

        try:
            from ..realtime.audio_capture import UtteranceDetector
        except ImportError:
            print("  [VAD not available, using fixed recording]")
            import sounddevice as sd
            audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            return audio.flatten()

        # Use event to wait for utterance
        utterance_event = threading.Event()
        captured_audio = [None]

        def on_speech_start():
            print("  [Listening...]")

        def on_speech_end():
            print("  [Processing...]")

        def on_utterance(audio):
            captured_audio[0] = audio
            utterance_event.set()

        detector = UtteranceDetector(
            on_utterance=on_utterance,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            min_speech_duration_ms=200,
            max_speech_duration_ms=8000,  # Max 8 seconds for a phrase
        )

        print("  [Speak now...]")
        detector.start()

        # Wait for utterance or timeout
        got_utterance = utterance_event.wait(timeout=timeout)
        detector.stop()

        if got_utterance and captured_audio[0] is not None:
            return captured_audio[0]
        else:
            print("  [No speech detected]")
            return None

    def execute(self, tool_name: str, params: dict) -> dict:
        """Execute a tool and return results."""
        print(f"  [Tool: {tool_name}]")

        if tool_name == "translate_to_chinese":
            return self._translate(params)
        elif tool_name == "generate_audio":
            return self._generate_audio(params)
        elif tool_name == "record_and_evaluate":
            return self._record_and_evaluate(params)
        elif tool_name == "play_audio":
            return self._play_audio(params)
        elif tool_name == "show_pitch_comparison":
            return self._show_pitch_comparison(params)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _translate(self, params: dict) -> dict:
        """Translate English to Chinese with pinyin and tones."""
        english = params["english"]

        if self.translation_llm:
            # Use LLM for translation
            prompt = f"""Translate this English to Chinese. Return JSON with:
- characters: the Chinese characters
- pinyin: list of pinyin syllables with tone marks (e.g., ["nǐ", "hǎo"])
- tones: list of tone numbers (1-5)
- literal: word-by-word breakdown

English: {english}

Return only valid JSON, no other text."""

            return {
                "status": "needs_llm",
                "prompt": prompt,
                "note": "Main LLM should provide translation"
            }
        else:
            return {
                "error": "No translation backend configured",
                "suggestion": "The LLM should provide the translation directly"
            }

    def _generate_audio(self, params: dict) -> dict:
        """Generate TTS audio for Chinese text."""
        characters = params["characters"]
        speed = params.get("speed", "normal")

        if self.tts is None:
            # Mock for testing without TTS
            print(f"  [Would generate audio for: {characters}]")
            audio_id = f"audio_{int(time.time())}"
            # Create silent placeholder
            self.audio_cache[audio_id] = np.zeros(16000, dtype=np.float32)
            return {
                "audio_id": audio_id,
                "duration_ms": 1000,
                "status": "mock",
                "note": "TTS not configured - using placeholder"
            }

        # Generate with TTS
        try:
            audio = self.tts.speak(
                text=characters,
                language="zh",
                speed=speed,
            )
            audio_id = f"audio_{int(time.time())}"
            self.audio_cache[audio_id] = audio

            return {
                "audio_id": audio_id,
                "duration_ms": int(len(audio) / 16),  # Assuming 16kHz
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def _record_and_evaluate(self, params: dict) -> dict:
        """Record user and evaluate tone accuracy."""
        characters = params["characters"]
        pinyin = params["pinyin"]
        expected_tones = params["tones"]

        # Strip tone marks from pinyin if present
        pinyin_plain = [self._strip_tone_marks(p) for p in pinyin]

        # Record audio
        print(f"  [Target: {characters}]")
        audio = self._record_utterance(timeout=10.0)

        if audio is None:
            return {"error": "No speech detected", "status": "timeout"}

        audio_id = f"user_{int(time.time())}"
        self.audio_cache[audio_id] = audio

        # Check if we have a real evaluator
        if self.evaluator is not None:
            try:
                result = self.evaluator.evaluate(
                    audio=audio,
                    expected_pinyin=pinyin_plain,
                    expected_tones=expected_tones,
                    characters=characters,
                )

                # Store for pitch comparison
                self.last_evaluation = result

                # Convert to dict format
                syllable_results = [
                    {
                        "character": r.character,
                        "pinyin": r.pinyin,
                        "expected_tone": r.expected_tone,
                        "detected_tone": r.detected_tone,
                        "confidence": r.confidence,
                        "correct": r.correct,
                    }
                    for r in result.syllable_results
                ]

                return {
                    "syllable_results": syllable_results,
                    "overall_accuracy": result.overall_accuracy,
                    "audio_id": audio_id,
                    "duration_ms": result.duration_ms,
                    "status": result.status,
                }

            except Exception as e:
                print(f"  [Evaluation error: {e}]")
                # Fall through to mock mode

        # Mock mode for testing conversation flow
        import random
        results = []
        for i, tone in enumerate(expected_tones):
            char = characters[i] if i < len(characters) else "?"
            pin = pinyin[i] if i < len(pinyin) else "?"
            # 70% chance of correct for testing
            detected = tone if random.random() > 0.3 else random.randint(1, 5)
            results.append({
                "character": char,
                "pinyin": pin,
                "expected_tone": tone,
                "detected_tone": detected,
                "confidence": 0.75 + random.random() * 0.2,
                "correct": detected == tone
            })

        accuracy = sum(r["correct"] for r in results) / len(results) if results else 0

        return {
            "syllable_results": results,
            "overall_accuracy": accuracy,
            "audio_id": audio_id,
            "duration_ms": int(len(audio) / 16),
            "status": "mock_eval"
        }

    def _play_audio(self, params: dict) -> dict:
        """Play an audio file."""
        audio_id = params["audio_id"]

        if audio_id not in self.audio_cache:
            return {"error": f"Audio not found: {audio_id}"}

        audio = self.audio_cache[audio_id]

        try:
            import sounddevice as sd
            sd.play(audio, samplerate=16000)
            sd.wait()
            return {"status": "played", "duration_ms": int(len(audio) / 16)}
        except Exception as e:
            print(f"  [Would play audio: {audio_id}]")
            return {"status": "mock", "note": str(e)}

    def _show_pitch_comparison(self, params: dict) -> dict:
        """Show F0 comparison visualization."""
        syllable_idx = params["syllable_index"]
        expected_tone = params["expected_tone"]

        if self.last_evaluation is None:
            return {"error": "No recent evaluation to show"}

        # Handle both old and new evaluation result formats
        if hasattr(self.last_evaluation, 'syllable_results'):
            syllable_results = self.last_evaluation.syllable_results
            if syllable_idx >= len(syllable_results):
                return {"error": f"Syllable index {syllable_idx} out of range"}

            result = syllable_results[syllable_idx]
            user_f0 = result.f0_contour if hasattr(result, 'f0_contour') and result.f0_contour is not None else np.array([])
        else:
            return {"error": "Evaluation format not supported"}

        # Generate ASCII visualization
        viz = self._ascii_f0_viz(user_f0, expected_tone)
        print(viz)

        return {
            "status": "displayed",
            "visualization": viz
        }

    def _ascii_f0_viz(self, user_f0: np.ndarray, expected_tone: int) -> str:
        """Generate ASCII visualization of F0 contour."""
        # Reference patterns (normalized 0-1)
        patterns = {
            1: [0.8] * 10,  # High level
            2: [0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],  # Rising
            3: [0.5, 0.4, 0.3, 0.25, 0.2, 0.2, 0.25, 0.35, 0.5, 0.6],  # Dipping
            4: [0.9, 0.8, 0.65, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1],  # Falling
            5: [0.5] * 10,  # Neutral
        }

        ref = patterns.get(expected_tone, patterns[5])
        height = 8
        width = 40

        # Normalize user F0 to 0-1
        if len(user_f0) > 0 and user_f0.max() > 0:
            user_norm = (user_f0 - user_f0.min()) / (user_f0.max() - user_f0.min() + 1e-6)
            user_resampled = np.interp(
                np.linspace(0, 1, width),
                np.linspace(0, 1, len(user_norm)),
                user_norm
            )
        else:
            user_resampled = np.ones(width) * 0.5

        ref_resampled = np.interp(np.linspace(0, 1, width), np.linspace(0, 1, len(ref)), ref)

        # Build ASCII grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        for x in range(width):
            # Reference (dots)
            ref_y = int((1 - ref_resampled[x]) * (height - 1))
            if 0 <= ref_y < height:
                grid[ref_y][x] = '·'

            # User (asterisks)
            user_y = int((1 - user_resampled[x]) * (height - 1))
            if 0 <= user_y < height:
                grid[user_y][x] = '*'

        # Build output
        lines = [f"  Tone {expected_tone} comparison (· = reference, * = you)"]
        lines.append("  ┌" + "─" * width + "┐")
        for row in grid:
            lines.append("  │" + "".join(row) + "│")
        lines.append("  └" + "─" * width + "┘")
        lines.append("   Time →")

        return "\n".join(lines)

    def _strip_tone_marks(self, pinyin: str) -> str:
        """Remove tone marks from pinyin."""
        replacements = {
            'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
            'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
            'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
            'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
            'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
            'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v', 'ü': 'v',
        }
        result = pinyin.lower()
        for mark, plain in replacements.items():
            result = result.replace(mark, plain)
        return result

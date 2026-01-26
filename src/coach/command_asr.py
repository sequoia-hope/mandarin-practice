"""Lightweight ASR for voice command recognition."""

import threading
from typing import Literal, Optional

import numpy as np


class CommandASR:
    """
    Lightweight ASR for recognizing menu commands.

    Uses Whisper (tiny/small) for fast recognition of a limited vocabulary:
    - Numbers: "one", "two", "three", etc.
    - Navigation: "menu", "back", "skip", "repeat", "quit"

    This is optimized for command recognition, not full transcription.
    """

    # Limited vocabulary for command recognition
    COMMAND_VOCABULARY = {
        # Numbers
        "one", "1", "first",
        "two", "2", "second",
        "three", "3", "third",
        "four", "4", "fourth",
        "five", "5", "fifth",
        "six", "6", "sixth",
        # Navigation
        "menu", "main", "back", "go back", "return",
        "skip", "next",
        "repeat", "again", "replay",
        "quit", "exit", "stop",
        "help",
        # Conversation mode
        "conversation", "chat", "free",
    }

    def __init__(
        self,
        model_size: Literal["tiny", "base", "small"] = "tiny",
        device: Optional[str] = None,
        language: str = "en",
    ):
        """
        Initialize command ASR.

        Args:
            model_size: Whisper model size ("tiny", "base", or "small")
            device: Device to run on (auto-detected if None)
            language: Language for recognition
        """
        self.model_size = model_size
        self.language = language

        # Auto-detect device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._model = None
        self._lock = threading.Lock()

        # Check availability
        try:
            import whisper
            self._available = True
        except ImportError:
            self._available = False
            print("[CommandASR] whisper not installed. Install with: pip install openai-whisper")

    def _load_model(self):
        """Load Whisper model lazily."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            print(f"[CommandASR] Loading Whisper {self.model_size}...")
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)
            print("[CommandASR] Ready!")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples (1D numpy array)
            sample_rate: Audio sample rate

        Returns:
            Dict with "text" key containing transcription
        """
        if not self._available:
            return {"text": "", "error": "Whisper not available"}

        self._load_model()

        # Ensure correct dtype and shape
        audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Transcribe with settings optimized for short commands
        result = self._model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            # Optimize for short utterances
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            # Fast decoding
            fp16=(self.device == "cuda"),
            beam_size=1,
            best_of=1,
        )

        text = result.get("text", "").strip().lower()

        return {
            "text": text,
            "language": result.get("language", self.language),
            "confidence": 1.0 - result.get("no_speech_prob", 0.0),
        }

    def recognize_command(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[str]:
        """
        Recognize a command from audio.

        Returns the recognized command text, or None if no valid command.
        """
        result = self.transcribe(audio, sample_rate)
        text = result.get("text", "")

        if not text:
            return None

        # Check if any command words are present
        text_lower = text.lower().strip()
        words = set(text_lower.split())

        # Direct match
        for cmd in self.COMMAND_VOCABULARY:
            if cmd in text_lower:
                return text_lower

        # Fuzzy matching for common misrecognitions
        fuzzy_map = {
            "won": "one",
            "want": "one",
            "too": "two",
            "to": "two",
            "tree": "three",
            "free": "three",
            "for": "four",
            "fore": "four",
            "five": "five",
            "sex": "six",
            "seeks": "six",
            "main menu": "menu",
            "go back": "menu",
        }

        for misheard, correct in fuzzy_map.items():
            if misheard in text_lower:
                return correct

        return text_lower if text_lower else None


class FasterWhisperASR:
    """
    Alternative ASR using faster-whisper for better performance.

    Requires: pip install faster-whisper
    """

    COMMAND_VOCABULARY = CommandASR.COMMAND_VOCABULARY

    def __init__(
        self,
        model_size: Literal["tiny", "base", "small"] = "tiny",
        device: Optional[str] = None,
        compute_type: str = "auto",
    ):
        """
        Initialize faster-whisper ASR.

        Args:
            model_size: Model size
            device: "cuda" or "cpu"
            compute_type: Computation type ("auto", "float16", "int8", etc.)
        """
        self.model_size = model_size

        # Auto-detect device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.compute_type = compute_type

        self._model = None
        self._lock = threading.Lock()

        try:
            from faster_whisper import WhisperModel
            self._available = True
        except ImportError:
            self._available = False
            print("[FasterWhisperASR] faster-whisper not installed.")

    def _load_model(self):
        """Load model lazily."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            print(f"[FasterWhisperASR] Loading {self.model_size}...")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            print("[FasterWhisperASR] Ready!")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        """Transcribe audio to text."""
        if not self._available:
            return {"text": "", "error": "faster-whisper not available"}

        self._load_model()

        # Prepare audio
        audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Transcribe
        segments, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=100,
                max_speech_duration_s=5.0,
                min_silence_duration_ms=100,
            ),
        )

        # Collect text from segments
        text = " ".join(segment.text for segment in segments).strip().lower()

        return {
            "text": text,
            "language": info.language,
            "confidence": 1.0,
        }

    def recognize_command(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[str]:
        """Recognize a command from audio."""
        result = self.transcribe(audio, sample_rate)
        text = result.get("text", "")
        return text.strip().lower() if text else None


def create_asr(
    backend: Literal["whisper", "faster"] = "whisper",
    model_size: Literal["tiny", "base", "small"] = "tiny",
    **kwargs,
):
    """
    Create an ASR backend.

    Args:
        backend: "whisper" for OpenAI Whisper, "faster" for faster-whisper
        model_size: Model size
        **kwargs: Additional arguments for the backend

    Returns:
        ASR instance
    """
    if backend == "whisper":
        return CommandASR(model_size=model_size, **kwargs)
    elif backend == "faster":
        return FasterWhisperASR(model_size=model_size, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test command ASR")
    parser.add_argument("--backend", choices=["whisper", "faster"], default="whisper")
    parser.add_argument("--model", choices=["tiny", "base", "small"], default="tiny")
    parser.add_argument("--test", action="store_true", help="Run interactive test")
    args = parser.parse_args()

    print(f"Creating {args.backend} ASR with {args.model} model...")
    asr = create_asr(args.backend, args.model)

    if args.test:
        print("\nInteractive test - speak commands into your microphone")
        print("Say: one, two, three, menu, skip, repeat, quit")
        print("Press Ctrl+C to exit\n")

        import sounddevice as sd

        while True:
            try:
                print("Listening...", end=" ", flush=True)
                audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="float32")
                sd.wait()
                audio = audio.flatten()

                result = asr.transcribe(audio)
                text = result.get("text", "")
                print(f"Heard: '{text}'")

                cmd = asr.recognize_command(audio)
                if cmd:
                    print(f"  -> Command: {cmd}")
                print()

            except KeyboardInterrupt:
                print("\nDone!")
                break

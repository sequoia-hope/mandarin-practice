"""ASR transcription module using Meta's Omnilingual ASR."""

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


@dataclass
class TranscriptionResult:
    """Result of ASR transcription."""

    text: str
    language: str
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0


class ASRTranscriber:
    """
    ASR transcription using Meta's Omnilingual ASR.

    Provides Mandarin transcription to complement tone classification.
    Uses lazy loading to defer model initialization until first use.
    """

    MANDARIN_LANG_CODE = "cmn_Hans"

    def __init__(
        self,
        model_card: str = "omniASR_CTC_300M_v2",
        device: Optional[str] = None,
        lazy_load: bool = False,
    ):
        """
        Initialize ASR transcriber.

        Args:
            model_card: Omnilingual ASR model identifier
            device: Torch device (cuda/cpu)
            lazy_load: If True, defer model loading until first transcribe call
        """
        self.model_card = model_card
        self.device = device
        self._pipeline = None
        self._loaded = False

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the ASR model."""
        if self._loaded:
            return

        try:
            from omnilingual_asr import ASRInferencePipeline
        except ImportError:
            raise ImportError(
                "omnilingual-asr is required for ASR functionality. "
                "Install with: pip install omnilingual-asr"
            )

        print(f"Loading ASR model: {self.model_card}...")
        self._pipeline = ASRInferencePipeline(
            model_card=self.model_card,
            device=self.device,
        )
        self._loaded = True
        print("ASR model loaded.")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples (numpy array, mono)
            sample_rate: Sample rate of audio
            language: Language code (defaults to Mandarin)

        Returns:
            TranscriptionResult with transcribed text
        """
        # Ensure model is loaded
        if not self._loaded:
            self._load_model()

        start_time = time.perf_counter()

        # Use Mandarin by default
        if language is None:
            language = self.MANDARIN_LANG_CODE

        # Omnilingual ASR API requires a file path, so we use a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            # Ensure float32 and normalize
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Write to temp file
            sf.write(tmp.name, audio, sample_rate)

            # Transcribe
            result = self._pipeline.transcribe(
                audio_path=tmp.name,
                language=language,
            )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Extract text from result
        text = result.get("text", "") if isinstance(result, dict) else str(result)

        return TranscriptionResult(
            text=text.strip(),
            language=language,
            confidence=result.get("confidence") if isinstance(result, dict) else None,
            processing_time_ms=processing_time_ms,
        )

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (defaults to Mandarin)

        Returns:
            TranscriptionResult with transcribed text
        """
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000)
        return self.transcribe(audio, sample_rate=sr, language=language)

    def unload(self):
        """Unload model to free VRAM."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False

            # Force garbage collection
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("ASR model unloaded.")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Transcription")
    parser.add_argument("--file", type=str, required=True, help="Audio file to transcribe")
    parser.add_argument("--model", type=str, default="omniASR_CTC_300M_v2", help="Model card")
    args = parser.parse_args()

    transcriber = ASRTranscriber(model_card=args.model)
    result = transcriber.transcribe_file(args.file)

    print(f"Text: {result.text}")
    print(f"Language: {result.language}")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")

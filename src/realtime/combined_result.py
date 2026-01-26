"""Combined result for tone classification and ASR transcription."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pipeline import ToneResult
from .asr import TranscriptionResult


@dataclass
class CombinedResult:
    """
    Combined result containing both tone classification and ASR transcription.

    Provides unified output for coaching interface and future LLM integration.
    """

    # Tone classification results
    tone: int
    confidence: float
    tone_name: str
    f0_contour: Optional[np.ndarray] = None

    # ASR results
    transcription: Optional[str] = None
    transcription_language: Optional[str] = None
    transcription_confidence: Optional[float] = None

    # Timing information
    tone_processing_time_ms: float = 0.0
    asr_processing_time_ms: float = 0.0
    audio_duration_ms: float = 0.0

    @classmethod
    def from_tone_result(
        cls,
        tone_result: ToneResult,
        transcription_result: Optional[TranscriptionResult] = None,
    ) -> "CombinedResult":
        """
        Create CombinedResult from ToneResult and optional TranscriptionResult.

        Args:
            tone_result: Result from tone classification
            transcription_result: Optional result from ASR transcription

        Returns:
            CombinedResult combining both results
        """
        return cls(
            # Tone results
            tone=tone_result.tone,
            confidence=tone_result.confidence,
            tone_name=tone_result.tone_name,
            f0_contour=tone_result.f0_contour,
            # ASR results
            transcription=transcription_result.text if transcription_result else None,
            transcription_language=transcription_result.language if transcription_result else None,
            transcription_confidence=transcription_result.confidence if transcription_result else None,
            # Timing
            tone_processing_time_ms=tone_result.processing_time_ms,
            asr_processing_time_ms=transcription_result.processing_time_ms if transcription_result else 0.0,
            audio_duration_ms=tone_result.audio_duration_ms,
        )

    @property
    def total_processing_time_ms(self) -> float:
        """Total processing time for both tone and ASR."""
        return self.tone_processing_time_ms + self.asr_processing_time_ms

    @property
    def has_transcription(self) -> bool:
        """Check if transcription is available."""
        return self.transcription is not None and len(self.transcription) > 0

    def to_coaching_context(self) -> dict:
        """
        Convert to dictionary for LLM coaching context.

        Returns:
            Dictionary with all relevant information for LLM prompting
        """
        context = {
            "tone": {
                "number": self.tone + 1,  # 1-indexed for readability
                "name": self.tone_name,
                "confidence": self.confidence,
            },
            "timing": {
                "audio_duration_ms": self.audio_duration_ms,
                "tone_processing_ms": self.tone_processing_time_ms,
                "asr_processing_ms": self.asr_processing_time_ms,
                "total_processing_ms": self.total_processing_time_ms,
            },
        }

        if self.has_transcription:
            context["transcription"] = {
                "text": self.transcription,
                "language": self.transcription_language,
                "confidence": self.transcription_confidence,
            }

        if self.f0_contour is not None:
            # Include F0 summary statistics for LLM context
            valid_f0 = self.f0_contour[self.f0_contour > 0]
            if len(valid_f0) > 0:
                context["f0_summary"] = {
                    "mean_hz": float(valid_f0.mean()),
                    "min_hz": float(valid_f0.min()),
                    "max_hz": float(valid_f0.max()),
                    "range_hz": float(valid_f0.max() - valid_f0.min()),
                }

        return context

    def __str__(self) -> str:
        """String representation."""
        parts = [
            f"Tone: {self.tone_name} ({self.confidence:.0%})",
        ]
        if self.has_transcription:
            parts.append(f"Text: {self.transcription}")
        parts.append(f"Processing: {self.total_processing_time_ms:.0f}ms")
        return " | ".join(parts)

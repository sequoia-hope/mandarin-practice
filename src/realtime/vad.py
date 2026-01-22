"""Silero VAD wrapper for voice activity detection."""

from collections import deque
from typing import Optional

import numpy as np
import torch


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD.

    Processes audio in small chunks (512 samples) for low-latency detection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        device: Optional[str] = None,
    ):
        """
        Initialize Silero VAD.

        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000)
            threshold: Speech probability threshold (0-1)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Minimum silence duration to end speech
            window_size_samples: Processing window size (512 or 1024 for 16kHz)
            device: Torch device
        """
        assert sample_rate in (8000, 16000), "Sample rate must be 8000 or 16000"

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_samples = int(min_speech_duration_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)
        self.window_size = window_size_samples

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load Silero VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.model.to(self.device)

        # State tracking
        self.reset()

    def reset(self):
        """Reset VAD state."""
        self.model.reset_states()
        self.is_speech = False
        self.speech_samples = 0
        self.silence_samples = 0
        self.current_speech_start = 0
        self.sample_count = 0

    @torch.no_grad()
    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process a single audio chunk.

        Args:
            audio_chunk: Audio samples (numpy array, typically 512 samples)

        Returns:
            Dict with:
                - speech_prob: Speech probability (0-1)
                - is_speech: Whether current chunk is speech
                - speech_start: Sample index where speech started (if applicable)
                - speech_end: Sample index where speech ended (if applicable)
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float().to(self.device)

        # Ensure correct shape
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Get speech probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        result = {
            "speech_prob": speech_prob,
            "is_speech": False,
            "speech_start": None,
            "speech_end": None,
        }

        chunk_is_speech = speech_prob >= self.threshold

        if chunk_is_speech:
            self.speech_samples += len(audio_chunk)
            self.silence_samples = 0

            if not self.is_speech and self.speech_samples >= self.min_speech_samples:
                # Speech started
                self.is_speech = True
                self.current_speech_start = self.sample_count - self.speech_samples
                result["speech_start"] = self.current_speech_start

        else:
            self.silence_samples += len(audio_chunk)

            if self.is_speech and self.silence_samples >= self.min_silence_samples:
                # Speech ended
                result["speech_end"] = self.sample_count
                self.is_speech = False
                self.speech_samples = 0

        result["is_speech"] = self.is_speech
        self.sample_count += len(audio_chunk)

        return result

    def process_audio(
        self,
        audio: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Process complete audio and return speech segments.

        Args:
            audio: Complete audio array

        Returns:
            List of (start_sample, end_sample) tuples for speech segments
        """
        self.reset()

        segments = []
        current_start = None

        for i in range(0, len(audio), self.window_size):
            chunk = audio[i:i + self.window_size]

            if len(chunk) < self.window_size:
                # Pad last chunk
                chunk = np.pad(chunk, (0, self.window_size - len(chunk)))

            result = self.process_chunk(chunk)

            if result["speech_start"] is not None:
                current_start = result["speech_start"]

            if result["speech_end"] is not None and current_start is not None:
                segments.append((current_start, result["speech_end"]))
                current_start = None

        # Handle case where speech continues to end
        if self.is_speech and current_start is not None:
            segments.append((current_start, len(audio)))

        return segments


class StreamingVAD:
    """
    Streaming VAD with buffering for real-time use.

    Accumulates audio chunks and detects speech segments in real-time.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_ms: int = 5000,
        speech_pad_ms: int = 50,
        **kwargs,
    ):
        """
        Initialize streaming VAD.

        Args:
            sample_rate: Audio sample rate
            threshold: Speech probability threshold
            min_speech_duration_ms: Minimum speech duration
            max_speech_duration_ms: Maximum speech duration before forced split
            speech_pad_ms: Padding around speech segments
        """
        self.sample_rate = sample_rate
        self.max_speech_samples = int(max_speech_duration_ms * sample_rate / 1000)
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)

        self.vad = SileroVAD(
            sample_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            **kwargs,
        )

        # Buffer for accumulating speech
        self.buffer = deque(maxlen=int(sample_rate * 10))  # 10 second max buffer
        self.speech_buffer = []
        self.is_collecting = False

    def reset(self):
        """Reset streaming state."""
        self.vad.reset()
        self.buffer.clear()
        self.speech_buffer = []
        self.is_collecting = False

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Process audio chunk and return complete utterance if detected.

        Args:
            audio_chunk: Audio chunk to process

        Returns:
            Complete speech segment as numpy array if speech ended, None otherwise
        """
        # Add to rolling buffer
        self.buffer.extend(audio_chunk)

        result = self.vad.process_chunk(audio_chunk)

        if result["speech_start"] is not None:
            # Start collecting speech
            self.is_collecting = True
            # Include some pre-speech context
            start_idx = max(0, len(self.buffer) - len(audio_chunk) - self.speech_pad_samples)
            self.speech_buffer = list(self.buffer)[start_idx:]

        elif self.is_collecting:
            # Continue collecting
            self.speech_buffer.extend(audio_chunk)

            # Check for max duration
            if len(self.speech_buffer) >= self.max_speech_samples:
                # Force return current segment
                speech = np.array(self.speech_buffer, dtype=np.float32)
                self.speech_buffer = []
                self.is_collecting = False
                return speech

        if result["speech_end"] is not None and self.speech_buffer:
            # Add post-speech padding
            self.speech_buffer.extend(audio_chunk[:self.speech_pad_samples])

            speech = np.array(self.speech_buffer, dtype=np.float32)
            self.speech_buffer = []
            self.is_collecting = False
            return speech

        return None


if __name__ == "__main__":
    # Test VAD
    import sounddevice as sd

    print("Testing Silero VAD...")

    vad = SileroVAD(sample_rate=16000, threshold=0.5)

    # Generate test audio with speech-like burst
    sr = 16000
    duration = 3.0

    # Silent -> Tone -> Silent
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)

    # Add a "speech-like" segment in the middle
    start_idx = int(0.5 * sr)
    end_idx = int(2.0 * sr)
    audio[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * 200 * t[start_idx:end_idx])
    audio = audio.astype(np.float32)

    # Process
    segments = vad.process_audio(audio)
    print(f"Detected segments: {segments}")

    for start, end in segments:
        print(f"  {start/sr:.2f}s - {end/sr:.2f}s (duration: {(end-start)/sr:.2f}s)")

"""Audio capture and utterance detection for real-time processing."""

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from .vad import StreamingVAD


class AudioCapture:
    """
    Real-time audio capture using sounddevice.

    Captures audio from microphone and streams chunks for processing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        channels: int = 1,
        device: Optional[int] = None,
        dtype: str = "float32",
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Audio sample rate
            chunk_size: Samples per chunk
            channels: Number of audio channels (1 for mono)
            device: Audio device index (None for default)
            dtype: Audio data type
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device = device
        self.dtype = dtype

        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")

        # Copy data to avoid reference issues
        self.audio_queue.put(indata[:, 0].copy() if self.channels == 1 else indata.copy())

    def start(self):
        """Start audio capture."""
        if self.is_running:
            return

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device,
            callback=self._audio_callback,
        )

        self.stream.start()
        self.is_running = True

    def stop(self):
        """Stop audio capture."""
        if not self.is_running:
            return

        self.is_running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next audio chunk from queue.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Audio chunk or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class UtteranceDetector:
    """
    Combines audio capture with VAD to detect complete utterances.

    Runs in a background thread and calls callback when utterance is detected.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 200,
        max_speech_duration_ms: int = 5000,
        on_utterance: Optional[Callable[[np.ndarray], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        device: Optional[int] = None,
    ):
        """
        Initialize utterance detector.

        Args:
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size
            vad_threshold: VAD speech probability threshold
            min_speech_duration_ms: Minimum speech duration
            max_speech_duration_ms: Maximum speech duration
            on_utterance: Callback when utterance is detected
            on_speech_start: Callback when speech starts
            on_speech_end: Callback when speech ends
            device: Audio device index
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self.audio_capture = AudioCapture(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            device=device,
        )

        self.vad = StreamingVAD(
            sample_rate=sample_rate,
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_ms=max_speech_duration_ms,
        )

        self.on_utterance = on_utterance
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self.is_running = False
        self.thread = None
        self.was_speaking = False

    def _processing_loop(self):
        """Main processing loop running in background thread."""
        while self.is_running:
            chunk = self.audio_capture.get_chunk(timeout=0.1)

            if chunk is None:
                continue

            # Check speech state before processing
            was_collecting = self.vad.is_collecting

            # Process chunk through VAD
            utterance = self.vad.process_chunk(chunk)

            # Detect speech start
            if self.vad.is_collecting and not was_collecting:
                if self.on_speech_start:
                    self.on_speech_start()

            # Detect speech end
            if not self.vad.is_collecting and was_collecting and utterance is not None:
                if self.on_speech_end:
                    self.on_speech_end()

            # Callback for complete utterance
            if utterance is not None and self.on_utterance:
                self.on_utterance(utterance)

    def start(self):
        """Start utterance detection."""
        if self.is_running:
            return

        self.is_running = True
        self.audio_capture.start()

        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop utterance detection."""
        self.is_running = False

        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        self.audio_capture.stop()
        self.vad.reset()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class AudioRecorder:
    """
    Simple audio recorder for capturing fixed-duration clips.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device

    def record(self, duration: float) -> np.ndarray:
        """
        Record audio for specified duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            Recorded audio as numpy array
        """
        samples = int(duration * self.sample_rate)

        print(f"Recording for {duration:.1f} seconds...")
        audio = sd.rec(
            samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        sd.wait()

        return audio.flatten()


def list_audio_devices():
    """List available audio input devices."""
    print("\nAvailable audio devices:")
    print(sd.query_devices())

    default_input = sd.query_devices(kind="input")
    print(f"\nDefault input device: {default_input['name']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test audio capture")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--record", type=float, default=0, help="Record for N seconds")
    parser.add_argument("--detect", action="store_true", help="Run utterance detection")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()

    elif args.record > 0:
        recorder = AudioRecorder()
        audio = recorder.record(args.record)
        print(f"Recorded {len(audio)} samples ({len(audio)/16000:.2f}s)")

        # Save to file
        import soundfile as sf
        sf.write("recorded.wav", audio, 16000)
        print("Saved to recorded.wav")

    elif args.detect:
        print("Starting utterance detection (Ctrl+C to stop)...")
        print("Speak into the microphone...")

        def on_utterance(audio):
            duration = len(audio) / 16000
            print(f"  Detected utterance: {duration:.2f}s, {len(audio)} samples")

        def on_speech_start():
            print("  [Speech started]")

        def on_speech_end():
            print("  [Speech ended]")

        detector = UtteranceDetector(
            on_utterance=on_utterance,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
        )

        try:
            with detector:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")

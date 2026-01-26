"""Controller for audio recording and playback."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable
import numpy as np

from gi.repository import GLib


class AudioController:
    """
    Controller for managing audio operations.

    Handles:
    - TTS playback (reference audio)
    - Recording from microphone
    - Audio caching
    """

    def __init__(self):
        # Thread pool for background operations
        self._executor = ThreadPoolExecutor(max_workers=2)

        # TTS backend (lazy loaded)
        self._tts = None
        self._tts_backend = "light"

        # Audio cache
        self._audio_cache = {}

        # Recording state
        self._is_recording = False
        self._recording_stop_event = threading.Event()

    def set_tts_backend(self, backend: str):
        """Set the TTS backend to use."""
        if backend != self._tts_backend:
            self._tts_backend = backend
            self._tts = None  # Force reload

    def _get_tts(self):
        """Get TTS instance, creating if needed."""
        if self._tts is None:
            try:
                from ...coach.tts import create_tts
                self._tts = create_tts(backend=self._tts_backend)
            except Exception as e:
                print(f"[AudioController] Failed to load TTS: {e}")
        return self._tts

    def play_reference(
        self,
        text: str,
        language: str = "zh",
        speed: str = "normal",
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Play reference audio for a phrase.

        Args:
            text: Text to synthesize and play
            language: Language code ("en" or "zh")
            speed: Speed ("slow", "normal", "fast")
            on_complete: Callback when playback completes
            on_error: Callback if playback fails
        """
        def do_play():
            try:
                tts = self._get_tts()
                if tts:
                    audio = tts.speak(text, language=language, speed=speed, play=True)

                    # Cache the audio
                    cache_key = f"{text}_{language}_{speed}"
                    self._audio_cache[cache_key] = audio

                if on_complete:
                    GLib.idle_add(on_complete)

            except Exception as e:
                if on_error:
                    GLib.idle_add(on_error, str(e))

        self._executor.submit(do_play)

    def get_cached_audio(
        self,
        text: str,
        language: str = "zh",
        speed: str = "normal",
    ) -> Optional[np.ndarray]:
        """Get cached audio if available."""
        cache_key = f"{text}_{language}_{speed}"
        return self._audio_cache.get(cache_key)

    def play_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        on_complete: Optional[Callable] = None,
    ):
        """
        Play audio samples directly.

        Args:
            audio: Audio samples (numpy array)
            sample_rate: Sample rate in Hz
            on_complete: Callback when playback completes
        """
        def do_play():
            try:
                import sounddevice as sd
                sd.play(audio, samplerate=sample_rate)
                sd.wait()

                if on_complete:
                    GLib.idle_add(on_complete)
            except Exception as e:
                print(f"[AudioController] Playback error: {e}")

        self._executor.submit(do_play)

    def record(
        self,
        timeout: float = 10.0,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_complete: Optional[Callable[[np.ndarray], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Record audio from microphone with VAD.

        Args:
            timeout: Maximum recording time in seconds
            on_speech_start: Called when speech is detected
            on_speech_end: Called when speech ends
            on_complete: Called with recorded audio when done
            on_error: Called if recording fails
        """
        if self._is_recording:
            return

        self._is_recording = True
        self._recording_stop_event.clear()

        def do_record():
            try:
                from ...realtime.audio_capture import UtteranceDetector

                utterance_event = threading.Event()
                captured_audio = [None]

                def on_utterance(audio):
                    captured_audio[0] = audio
                    utterance_event.set()

                def speech_start_handler():
                    if on_speech_start:
                        GLib.idle_add(on_speech_start)

                def speech_end_handler():
                    if on_speech_end:
                        GLib.idle_add(on_speech_end)

                detector = UtteranceDetector(
                    on_utterance=on_utterance,
                    on_speech_start=speech_start_handler,
                    on_speech_end=speech_end_handler,
                    min_speech_duration_ms=200,
                    max_speech_duration_ms=8000,
                )

                detector.start()
                got_utterance = utterance_event.wait(timeout=timeout)
                detector.stop()

                self._is_recording = False

                if got_utterance and captured_audio[0] is not None:
                    if on_complete:
                        GLib.idle_add(on_complete, captured_audio[0])
                else:
                    if on_error:
                        GLib.idle_add(on_error, "No speech detected")

            except ImportError:
                # Fallback to fixed-duration recording
                self._record_fixed_duration(timeout, on_complete, on_error)

            except Exception as e:
                self._is_recording = False
                if on_error:
                    GLib.idle_add(on_error, str(e))

        self._executor.submit(do_record)

    def _record_fixed_duration(
        self,
        duration: float,
        on_complete: Optional[Callable[[np.ndarray], None]],
        on_error: Optional[Callable[[str], None]],
    ):
        """Record for a fixed duration (fallback when VAD unavailable)."""
        try:
            import sounddevice as sd

            sample_rate = 16000
            samples = int(duration * sample_rate)

            audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
            sd.wait()

            audio = audio.flatten()
            self._is_recording = False

            if on_complete:
                GLib.idle_add(on_complete, audio)

        except Exception as e:
            self._is_recording = False
            if on_error:
                GLib.idle_add(on_error, str(e))

    def stop_recording(self):
        """Stop current recording."""
        self._recording_stop_event.set()

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def shutdown(self):
        """Shutdown the controller and release resources."""
        self._executor.shutdown(wait=False)

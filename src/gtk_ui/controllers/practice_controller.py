"""Controller for coordinating the practice flow."""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable
import numpy as np

from gi.repository import GLib

from .audio_controller import AudioController


class PracticeController:
    """
    Controller for managing the practice workflow.

    Coordinates:
    - Lesson and phrase selection
    - Audio playback and recording
    - Tone evaluation
    - Progress tracking
    """

    def __init__(self):
        # Sub-controllers
        self.audio = AudioController()

        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Components (lazy loaded)
        self._evaluator = None
        self._progress_tracker = None

        # Current state
        self._current_lesson = None
        self._current_phrase_idx = 0

    def _get_evaluator(self):
        """Get evaluator instance, creating if needed."""
        if self._evaluator is None:
            try:
                from ...coach.evaluator import Evaluator
                self._evaluator = Evaluator(lazy_load=True)
            except Exception as e:
                print(f"[PracticeController] Failed to load Evaluator: {e}")
        return self._evaluator

    def _get_progress_tracker(self):
        """Get progress tracker, creating if needed."""
        if self._progress_tracker is None:
            try:
                from ...coach.progress import ProgressTracker
                self._progress_tracker = ProgressTracker()
            except Exception as e:
                print(f"[PracticeController] Failed to load ProgressTracker: {e}")
        return self._progress_tracker

    def set_lesson(self, lesson):
        """Set the current lesson."""
        self._current_lesson = lesson
        self._current_phrase_idx = 0

    def get_current_phrase(self) -> Optional[dict]:
        """Get the current phrase dict."""
        if not self._current_lesson or not self._current_lesson.phrases:
            return None

        if 0 <= self._current_phrase_idx < len(self._current_lesson.phrases):
            return self._current_lesson.phrases[self._current_phrase_idx]

        return None

    def next_phrase(self) -> Optional[dict]:
        """Move to next phrase and return it."""
        if not self._current_lesson:
            return None

        if self._current_phrase_idx < len(self._current_lesson.phrases) - 1:
            self._current_phrase_idx += 1

        return self.get_current_phrase()

    def prev_phrase(self) -> Optional[dict]:
        """Move to previous phrase and return it."""
        if self._current_phrase_idx > 0:
            self._current_phrase_idx -= 1

        return self.get_current_phrase()

    def play_reference(
        self,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """Play reference audio for current phrase."""
        phrase = self.get_current_phrase()
        if not phrase:
            if on_error:
                on_error("No phrase selected")
            return

        self.audio.play_reference(
            text=phrase["characters"],
            language="zh",
            on_complete=on_complete,
            on_error=on_error,
        )

    def evaluate_recording(
        self,
        on_recording_start: Optional[Callable] = None,
        on_recording_end: Optional[Callable] = None,
        on_complete: Optional[Callable[[object], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Record and evaluate the user's pronunciation.

        Args:
            on_recording_start: Called when recording begins
            on_recording_end: Called when recording ends
            on_complete: Called with EvaluationResult when done
            on_error: Called with error message if failed
        """
        phrase = self.get_current_phrase()
        if not phrase:
            if on_error:
                on_error("No phrase selected")
            return

        def do_evaluate():
            try:
                evaluator = self._get_evaluator()
                if evaluator is None:
                    if on_error:
                        GLib.idle_add(on_error, "Evaluator not available")
                    return

                # Strip tone marks from pinyin
                def strip_tone_marks(pinyin: str) -> str:
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

                # Evaluate from microphone
                result = evaluator.evaluate_from_mic(
                    expected_pinyin=[strip_tone_marks(p) for p in phrase["pinyin"]],
                    expected_tones=phrase["tones"],
                    characters=phrase["characters"],
                    timeout=10.0,
                )

                # Record progress
                self._record_result(result)

                if on_complete:
                    GLib.idle_add(on_complete, result)

            except Exception as e:
                if on_error:
                    GLib.idle_add(on_error, str(e))

        self._executor.submit(do_evaluate)

    def _record_result(self, result):
        """Record evaluation result in progress tracker."""
        tracker = self._get_progress_tracker()
        if tracker is None:
            return

        phrase = self.get_current_phrase()
        if not phrase:
            return

        # Convert to dict format
        syllable_dicts = [
            {
                "expected_tone": r.expected_tone,
                "detected_tone": r.detected_tone,
                "correct": r.correct,
            }
            for r in result.syllable_results
        ]

        tracker.record_evaluation(syllable_dicts, phrase=phrase["characters"])

    def start_session(self):
        """Start a new practice session."""
        tracker = self._get_progress_tracker()
        if tracker:
            tracker.start_session()

    def end_session(self):
        """End the current practice session."""
        tracker = self._get_progress_tracker()
        if tracker:
            tracker.end_session()

    def get_progress_summary(self) -> dict:
        """Get progress summary."""
        tracker = self._get_progress_tracker()
        if tracker:
            return tracker.get_summary()
        return {}

    def shutdown(self):
        """Shutdown the controller."""
        self.audio.shutdown()
        self._executor.shutdown(wait=False)

        # End session if active
        self.end_session()

"""Main practice view for tone training."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, Adw, GLib, GObject
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np

from ..widgets.phrase_display import PhraseDisplay
from ..widgets.record_button import RecordButton, RecordState
from ..widgets.pitch_visualizer import PitchVisualizer
from ..widgets.tone_result_card import ToneResultCard


class PracticeView(Gtk.Box):
    """
    Main practice interface for learning tones.

    Contains:
    - Phrase display (characters, pinyin, translation)
    - Play reference button
    - Pitch visualizer
    - Record button
    - Results display
    """

    __gtype_name__ = "PracticeView"

    __gsignals__ = {
        "evaluation-complete": (GObject.SignalFlags.RUN_FIRST, None, (object,)),
    }

    def __init__(self):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=0,
        )

        self.set_vexpand(True)
        self.set_hexpand(True)

        # Current lesson and phrase
        self._lesson = None
        self._current_phrase_idx = 0

        # Thread pool for background operations
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Lazy-loaded components
        self._tts = None
        self._evaluator = None
        self._tts_backend = "light"

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the practice view UI."""
        # Main scrolled container
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.append(scrolled)

        # Content box
        content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=24,
            halign=Gtk.Align.CENTER,
        )
        content.set_margin_top(24)
        content.set_margin_bottom(24)
        content.set_margin_start(24)
        content.set_margin_end(24)
        content.set_size_request(500, -1)
        scrolled.set_child(content)

        # Phrase display
        self._phrase_display = PhraseDisplay()
        content.append(self._phrase_display)

        # Play reference button
        self._play_button = Gtk.Button()
        self._play_button.set_child(
            Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL,
                spacing=8,
                halign=Gtk.Align.CENTER,
            )
        )
        play_icon = Gtk.Image.new_from_icon_name("media-playback-start-symbolic")
        play_label = Gtk.Label(label="Play Reference")
        self._play_button.get_child().append(play_icon)
        self._play_button.get_child().append(play_label)
        self._play_button.set_halign(Gtk.Align.CENTER)
        self._play_button.add_css_class("pill")
        self._play_button.connect("clicked", self._on_play_clicked)
        content.append(self._play_button)

        # Pitch visualizer
        self._pitch_viz = PitchVisualizer()
        self._pitch_viz.set_margin_top(16)
        self._pitch_viz.set_margin_bottom(16)
        content.append(self._pitch_viz)

        # Record button
        self._record_button = RecordButton()
        self._record_button.set_halign(Gtk.Align.CENTER)
        self._record_button.connect("record-start", self._on_record_start)
        self._record_button.connect("record-stop", self._on_record_stop)
        content.append(self._record_button)

        # Results card
        self._results_card = ToneResultCard()
        content.append(self._results_card)

        # Navigation buttons
        nav_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.CENTER,
        )
        nav_box.set_margin_top(16)

        self._prev_button = Gtk.Button(label="Previous")
        self._prev_button.add_css_class("flat")
        self._prev_button.connect("clicked", self._on_prev_clicked)
        nav_box.append(self._prev_button)

        self._next_button = Gtk.Button(label="Next")
        self._next_button.add_css_class("suggested-action")
        self._next_button.connect("clicked", self._on_next_clicked)
        nav_box.append(self._next_button)

        content.append(nav_box)

        # Phrase counter
        self._phrase_counter = Gtk.Label()
        self._phrase_counter.add_css_class("dim-label")
        self._phrase_counter.set_margin_top(8)
        content.append(self._phrase_counter)

    def set_lesson(self, lesson):
        """Set the current lesson."""
        self._lesson = lesson
        self._current_phrase_idx = 0
        self._update_phrase()

    def set_tts_backend(self, backend: str):
        """Set the TTS backend to use."""
        self._tts_backend = backend
        self._tts = None  # Force reload

    def _get_tts(self):
        """Get TTS instance, creating if needed."""
        if self._tts is None:
            try:
                from ...coach.tts import create_tts
                self._tts = create_tts(backend=self._tts_backend)
            except Exception as e:
                print(f"[PracticeView] Failed to load TTS: {e}")
        return self._tts

    def _get_evaluator(self):
        """Get evaluator instance, creating if needed."""
        if self._evaluator is None:
            try:
                from ...coach.evaluator import Evaluator
                self._evaluator = Evaluator(lazy_load=True)
            except Exception as e:
                print(f"[PracticeView] Failed to load Evaluator: {e}")
        return self._evaluator

    def _update_phrase(self):
        """Update the display for the current phrase."""
        if not self._lesson or not self._lesson.phrases:
            self._phrase_display.clear()
            self._results_card.clear()
            self._pitch_viz.clear()
            return

        # Clamp index
        self._current_phrase_idx = max(
            0,
            min(self._current_phrase_idx, len(self._lesson.phrases) - 1)
        )

        phrase = self._lesson.phrases[self._current_phrase_idx]

        # Apply sandhi to get spoken tones
        from src.coach.sandhi import apply_sandhi
        sandhi_result = apply_sandhi(
            phrase["tones"],
            phrase["pinyin"],
            phrase["characters"],
        )

        # Update phrase display with sandhi info
        self._phrase_display.set_phrase(
            characters=phrase["characters"],
            pinyin=phrase["pinyin"],
            tones=phrase["tones"],
            english=phrase.get("english", ""),
            spoken_tones=sandhi_result.spoken_tones,
        )

        # Update pitch visualizer expected tone
        if phrase["tones"]:
            self._pitch_viz.set_expected_tone(phrase["tones"][0])
        else:
            self._pitch_viz.set_expected_tone(None)

        # Clear previous results
        self._results_card.clear()
        self._pitch_viz.set_user_f0(None)
        self._pitch_viz.set_reference_f0(None)

        # Update navigation
        self._prev_button.set_sensitive(self._current_phrase_idx > 0)
        self._next_button.set_sensitive(
            self._current_phrase_idx < len(self._lesson.phrases) - 1
        )

        # Update counter
        self._phrase_counter.set_text(
            f"{self._current_phrase_idx + 1} / {len(self._lesson.phrases)}"
        )

    def _on_play_clicked(self, button):
        """Handle play reference button click."""
        if not self._lesson:
            return

        phrase = self._lesson.phrases[self._current_phrase_idx]
        button.set_sensitive(False)

        def play_audio():
            tts = self._get_tts()
            if tts:
                try:
                    tts.speak(phrase["characters"], language="zh", play=True)
                except Exception as e:
                    print(f"[PracticeView] TTS error: {e}")

            GLib.idle_add(lambda: button.set_sensitive(True))

        self._executor.submit(play_audio)

    def _on_record_start(self, button):
        """Handle record start."""
        self._record_button.set_state(RecordState.RECORDING)
        self._results_card.clear()

        # Start recording in background
        self._executor.submit(self._do_record_and_evaluate)

    def _on_record_stop(self, button):
        """Handle record stop (manual stop not implemented - auto-detects end)."""
        pass

    def _do_record_and_evaluate(self):
        """Record and evaluate in background thread."""
        if not self._lesson:
            GLib.idle_add(self._record_button.reset)
            return

        phrase = self._lesson.phrases[self._current_phrase_idx]

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

        try:
            evaluator = self._get_evaluator()
            if evaluator is None:
                GLib.idle_add(self._on_evaluation_error, "Evaluator not available")
                return

            GLib.idle_add(
                lambda: self._record_button.set_state(RecordState.RECORDING)
            )

            # Evaluate from microphone
            result = evaluator.evaluate_from_mic(
                expected_pinyin=[strip_tone_marks(p) for p in phrase["pinyin"]],
                expected_tones=phrase["tones"],
                characters=phrase["characters"],
                timeout=10.0,
            )

            GLib.idle_add(self._on_evaluation_complete, result)

        except Exception as e:
            GLib.idle_add(self._on_evaluation_error, str(e))

    def _on_evaluation_complete(self, result):
        """Handle evaluation completion on main thread."""
        self._record_button.set_state(RecordState.IDLE)

        if result.status == "error" or result.status == "timeout":
            self._show_error(result.error or "Recording failed")
            return

        # Convert to dict format for results card
        syllable_dicts = [
            {
                "character": r.character,
                "pinyin": r.pinyin,
                "expected_tone": r.expected_tone,
                "spoken_tone": r.spoken_tone,  # Tone after sandhi
                "sandhi_applied": r.sandhi_applied,
                "detected_tone": r.detected_tone,
                "correct": r.correct,
                "confidence": r.confidence,
                "char_recognized": r.char_recognized,
                "asr_tone": r.asr_tone,
                "correct_source": r.correct_source,
            }
            for r in result.syllable_results
        ]

        # Update results card with tone + word recognition results
        self._results_card.set_results(
            syllable_dicts,
            result.overall_accuracy,
            word_accuracy=result.word_accuracy,
            combined_score=result.combined_score,
            transcription=result.transcription,
            transcription_pinyin=result.transcription_pinyin,
        )

        # Update pitch visualizer with user's F0
        if result.syllable_results and result.syllable_results[0].f0_contour is not None:
            # Combine F0 contours
            all_f0 = []
            for r in result.syllable_results:
                if r.f0_contour is not None:
                    all_f0.extend(r.f0_contour)

            if all_f0:
                self._pitch_viz.set_user_f0(np.array(all_f0))

        self.emit("evaluation-complete", result)

    def _on_evaluation_error(self, error_message: str):
        """Handle evaluation error on main thread."""
        self._record_button.set_state(RecordState.IDLE)
        self._show_error(error_message)

    def _show_error(self, message: str):
        """Show an error toast."""
        # Find the window to show toast
        window = self.get_root()
        if window and hasattr(window, "add_toast"):
            toast = Adw.Toast(title=message, timeout=3)
            window.add_toast(toast)
        else:
            print(f"[PracticeView] Error: {message}")

    def _on_prev_clicked(self, button):
        """Go to previous phrase."""
        if self._current_phrase_idx > 0:
            self._current_phrase_idx -= 1
            self._update_phrase()

    def _on_next_clicked(self, button):
        """Go to next phrase."""
        if self._lesson and self._current_phrase_idx < len(self._lesson.phrases) - 1:
            self._current_phrase_idx += 1
            self._update_phrase()

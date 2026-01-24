"""Widget for displaying per-syllable tone evaluation results."""

import gi
import math

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, Adw, GObject, Gdk
from typing import Optional


# Tone descriptions and visual representations
TONE_INFO = {
    1: {
        "name": "high flat",
        "symbol": "—",
        "description": "Start high, stay high",
        "color": "#4a90d9",  # Blue
    },
    2: {
        "name": "rising",
        "symbol": "／",
        "description": "Start mid, rise up",
        "color": "#73d216",  # Green
    },
    3: {
        "name": "dipping",
        "symbol": "∨",
        "description": "Dip low, then rise",
        "color": "#f57900",  # Orange
    },
    4: {
        "name": "falling",
        "symbol": "＼",
        "description": "Start high, fall down",
        "color": "#cc0000",  # Red
    },
    5: {
        "name": "neutral",
        "symbol": "·",
        "description": "Light and short",
        "color": "#888888",  # Gray
    },
}


class TonePitchWidget(Gtk.DrawingArea):
    """Small widget that draws a tone pitch contour."""

    # Standard colors for tone feedback
    COLOR_CORRECT = "#4caf50"   # Green
    COLOR_INCORRECT = "#e53935"  # Red
    COLOR_EXPECTED = "#888888"   # Grey

    def __init__(self, tone: int, width: int = 40, height: int = 24, color: str = None):
        super().__init__()
        self.tone = tone
        self.custom_color = color
        self.set_size_request(width, height)
        self.set_draw_func(self._draw)

    def _draw(self, area, cr, width, height):
        """Draw the tone contour."""
        # Padding
        pad = 3
        w = width - 2 * pad
        h = height - 2 * pad

        # Set line properties
        cr.set_line_width(2.5)
        cr.set_line_cap(1)  # Round caps

        # Use custom color if provided, otherwise use tone's default color
        if self.custom_color:
            color = Gdk.RGBA()
            color.parse(self.custom_color)
        else:
            info = TONE_INFO.get(self.tone, TONE_INFO[1])
            color = Gdk.RGBA()
            color.parse(info["color"])
        cr.set_source_rgba(color.red, color.green, color.blue, 1.0)

        # Draw contour based on tone
        if self.tone == 1:  # High flat: ——
            y = pad + h * 0.2
            cr.move_to(pad, y)
            cr.line_to(pad + w, y)

        elif self.tone == 2:  # Rising: ／
            cr.move_to(pad, pad + h * 0.8)
            cr.line_to(pad + w, pad + h * 0.2)

        elif self.tone == 3:  # Dipping: ∨
            cr.move_to(pad, pad + h * 0.4)
            cr.curve_to(
                pad + w * 0.3, pad + h * 0.9,
                pad + w * 0.7, pad + h * 0.9,
                pad + w, pad + h * 0.3
            )

        elif self.tone == 4:  # Falling: ＼
            cr.move_to(pad, pad + h * 0.2)
            cr.line_to(pad + w, pad + h * 0.85)

        elif self.tone == 5:  # Neutral: short mid
            y = pad + h * 0.5
            cr.move_to(pad + w * 0.3, y)
            cr.line_to(pad + w * 0.7, y)

        cr.stroke()


class ToneResultCard(Gtk.Box):
    """
    Display evaluation results for each syllable.

    Shows a row of syllable results with:
    - Character
    - Tone comparison (expected vs detected)
    - Correct/incorrect indicator
    - Overall score
    """

    __gtype_name__ = "ToneResultCard"

    def __init__(self):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.CENTER,
        )

        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)

        # Syllable results container (horizontal box instead of flow)
        self._results_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
        )
        self.append(self._results_box)

        # Separator
        self._sep1 = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self._sep1.set_margin_start(4)
        self._sep1.set_margin_end(4)
        self.append(self._sep1)

        # Transcription display (what ASR heard) - characters + pinyin
        self._transcription_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=4,
        )
        self.append(self._transcription_box)

        self._heard_label = Gtk.Label(label="Heard:")
        self._heard_label.add_css_class("dim-label")
        self._heard_label.add_css_class("caption")
        self._transcription_box.append(self._heard_label)

        self._transcription_chars = Gtk.Label(label="")
        self._transcription_box.append(self._transcription_chars)

        self._transcription_pinyin = Gtk.Label(label="")
        self._transcription_pinyin.add_css_class("dim-label")
        self._transcription_box.append(self._transcription_pinyin)

        # Separator
        self._sep2 = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self._sep2.set_margin_start(4)
        self._sep2.set_margin_end(4)
        self.append(self._sep2)

        # Score summary box - horizontal inline
        self._score_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
        )
        self.append(self._score_box)

        # Tone accuracy - inline
        self._tone_score_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._tone_score_label = Gtk.Label(label="Tones:")
        self._tone_score_label.add_css_class("caption")
        self._tone_score_label.add_css_class("dim-label")
        self._tone_score_box.append(self._tone_score_label)
        self._tone_score_value = Gtk.Label(label="-")
        self._tone_score_box.append(self._tone_score_value)
        self._score_box.append(self._tone_score_box)

        # Word accuracy - inline
        self._word_score_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._word_score_label = Gtk.Label(label="Words:")
        self._word_score_label.add_css_class("caption")
        self._word_score_label.add_css_class("dim-label")
        self._word_score_box.append(self._word_score_label)
        self._word_score_value = Gtk.Label(label="-")
        self._word_score_box.append(self._word_score_value)
        self._score_box.append(self._word_score_box)

        # Combined score - inline
        self._combined_score_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._combined_score_label = Gtk.Label(label="Overall:")
        self._combined_score_label.add_css_class("caption")
        self._combined_score_label.add_css_class("dim-label")
        self._combined_score_box.append(self._combined_score_label)
        self._combined_score_value = Gtk.Label(label="-")
        self._combined_score_value.add_css_class("title-4")
        self._combined_score_box.append(self._combined_score_value)
        self._score_box.append(self._combined_score_box)

        # Initially hidden
        self.set_visible(False)

    def set_results(
        self,
        syllable_results: list[dict],
        overall_accuracy: float,
        word_accuracy: float = None,
        combined_score: float = None,
        transcription: str = None,
        transcription_pinyin: str = None,
    ):
        """
        Set the evaluation results to display.

        Args:
            syllable_results: List of syllable result dicts with keys:
                - character: Chinese character
                - expected_tone: Expected tone number (1-5)
                - detected_tone: Detected tone number (1-5)
                - correct: Boolean (tone correct)
                - confidence: Float 0-1
                - char_recognized: Boolean (word recognized) or None
            overall_accuracy: Tone accuracy as float 0-1
            word_accuracy: Word/character recognition accuracy 0-1
            combined_score: Combined tone + word score 0-1
            transcription: What ASR heard (Chinese characters)
            transcription_pinyin: Pinyin of what ASR heard
        """
        # Clear previous results
        self.clear()

        if not syllable_results:
            return

        # Add syllable result widgets
        for result in syllable_results:
            widget = self._create_syllable_result(result)
            self._results_box.append(widget)

        # Update transcription display (characters + pinyin)
        if transcription:
            # Detect junk ASR output (repetitive characters or too long)
            is_junk = self._is_junk_transcription(transcription)

            if is_junk:
                self._transcription_chars.set_text("[ASR error]")
                self._transcription_pinyin.set_text("")
            else:
                # Truncate if too long
                max_chars = 20
                display_text = transcription[:max_chars] + "..." if len(transcription) > max_chars else transcription
                self._transcription_chars.set_text(display_text)

                if transcription_pinyin:
                    max_pinyin = 30
                    display_pinyin = transcription_pinyin[:max_pinyin] + "..." if len(transcription_pinyin) > max_pinyin else transcription_pinyin
                    self._transcription_pinyin.set_text(f"({display_pinyin})")
                else:
                    self._transcription_pinyin.set_text("")

            self._transcription_box.set_visible(True)
        else:
            self._transcription_box.set_visible(False)

        # Update tone score
        tone_percent = int(overall_accuracy * 100)
        self._tone_score_value.set_text(f"{tone_percent}%")
        self._apply_score_color(self._tone_score_value, overall_accuracy)

        # Update word score
        if word_accuracy is not None:
            word_percent = int(word_accuracy * 100)
            self._word_score_value.set_text(f"{word_percent}%")
            self._apply_score_color(self._word_score_value, word_accuracy)
            self._word_score_box.set_visible(True)
        else:
            self._word_score_box.set_visible(False)

        # Update combined score
        if combined_score is not None:
            combined_percent = int(combined_score * 100)
            self._combined_score_value.set_text(f"{combined_percent}%")
            self._apply_score_color(self._combined_score_value, combined_score)
            self._combined_score_box.set_visible(True)
        else:
            # Fall back to showing just tone score as "Overall" if no combined
            self._combined_score_value.set_text(f"{tone_percent}%")
            self._apply_score_color(self._combined_score_value, overall_accuracy)
            self._combined_score_box.set_visible(True)

        # Show separators
        self._sep1.set_visible(True)
        self._sep2.set_visible(True)

        self.set_visible(True)

    def _is_junk_transcription(self, text: str) -> bool:
        """
        Detect junk ASR output (repetitive characters, too long, etc.)

        Returns True if the transcription looks like garbage.
        """
        if not text:
            return False

        # Too long for expected input (Chinese phrases are typically short)
        if len(text) > 30:
            return True

        # Highly repetitive (same character repeated many times)
        if len(text) >= 4:
            # Check if dominated by one character
            from collections import Counter
            counts = Counter(text)
            most_common_char, most_common_count = counts.most_common(1)[0]
            if most_common_count / len(text) > 0.6:  # >60% same character
                return True

        return False

    def _apply_score_color(self, label: Gtk.Label, score: float):
        """Apply color coding to a score label."""
        label.remove_css_class("success")
        label.remove_css_class("warning")
        label.remove_css_class("error")

        if score >= 0.8:
            label.add_css_class("success")
        elif score >= 0.5:
            label.add_css_class("warning")
        else:
            label.add_css_class("error")

    def _create_syllable_result(self, result: dict) -> Gtk.Widget:
        """Create a widget for a single syllable result."""
        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=2,
            halign=Gtk.Align.CENTER,
        )
        box.set_margin_start(6)
        box.set_margin_end(6)
        box.set_margin_top(4)
        box.set_margin_bottom(4)

        # Frame with background color based on correct/incorrect
        frame = Gtk.Frame()
        frame.set_child(box)

        correct = result.get("correct", False)
        if correct:
            frame.add_css_class("card")
            frame.add_css_class("result-correct")
        else:
            frame.add_css_class("card")
            frame.add_css_class("result-incorrect")

        # Character
        char_label = Gtk.Label(label=result.get("character", "?"))
        char_label.add_css_class("title-3")
        box.append(char_label)

        # Use spoken_tone (after sandhi) for what we expect to hear
        underlying_tone = result.get("expected_tone", 1)
        spoken_tone = result.get("spoken_tone") or underlying_tone
        detected = result.get("detected_tone", 1)
        sandhi_applied = result.get("sandhi_applied", False)

        spoken_info = TONE_INFO.get(spoken_tone, TONE_INFO[1])
        detected_info = TONE_INFO.get(detected, TONE_INFO[1])

        if correct:
            # Correct: just GREEN tone contour
            pitch_widget = TonePitchWidget(
                spoken_tone, width=32, height=18,
                color=TonePitchWidget.COLOR_CORRECT  # Green
            )
            box.append(pitch_widget)
        else:
            # Incorrect: RED detected first, then GREY expected
            comparison_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            comparison_box.set_halign(Gtk.Align.CENTER)

            # Detected tone in red (what we got - first)
            det_pitch = TonePitchWidget(
                detected, width=24, height=16,
                color=TonePitchWidget.COLOR_INCORRECT  # Red
            )
            comparison_box.append(det_pitch)

            # Expected tone in grey (what we wanted - second)
            exp_pitch = TonePitchWidget(
                spoken_tone, width=24, height=16,
                color=TonePitchWidget.COLOR_EXPECTED  # Grey
            )
            comparison_box.append(exp_pitch)

            box.append(comparison_box)

        # Compact bottom row: word recognition + confidence
        bottom_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        bottom_row.set_halign(Gtk.Align.CENTER)

        char_recognized = result.get("char_recognized")
        if char_recognized is not None:
            word_icon = "✓" if char_recognized else "✗"
            word_class = "success" if char_recognized else "error"
            word_label = Gtk.Label(label=word_icon)
            word_label.add_css_class(word_class)
            word_label.add_css_class("caption")
            bottom_row.append(word_label)

        confidence = result.get("confidence", 0.0)
        conf_label = Gtk.Label(label=f"{confidence:.0%}")
        conf_label.add_css_class("dim-label")
        conf_label.add_css_class("caption")
        bottom_row.append(conf_label)

        box.append(bottom_row)

        return frame

    def clear(self):
        """Clear all results."""
        # Remove all children from results box
        while True:
            child = self._results_box.get_first_child()
            if child is None:
                break
            self._results_box.remove(child)

        # Reset all score displays
        self._tone_score_value.set_text("-")
        self._word_score_value.set_text("-")
        self._combined_score_value.set_text("-")
        self._transcription_chars.set_text("")
        self._transcription_pinyin.set_text("")
        self._transcription_box.set_visible(False)
        self._sep1.set_visible(False)
        self._sep2.set_visible(False)
        self.set_visible(False)


# Custom CSS for the widget
TONE_RESULT_CARD_CSS = """
.result-correct {
    background-color: alpha(@success_color, 0.1);
    border-color: @success_color;
}

.result-incorrect {
    background-color: alpha(@error_color, 0.1);
    border-color: @error_color;
}

.score-value.success {
    color: @success_color;
}

.score-value.warning {
    color: @warning_color;
}

.score-value.error {
    color: @error_color;
}
"""

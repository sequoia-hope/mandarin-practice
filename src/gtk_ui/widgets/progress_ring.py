"""Circular progress indicator widget."""

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import Gtk, GObject
import math


class ProgressRing(Gtk.DrawingArea):
    """
    Circular progress ring showing accuracy percentage.

    Displays a ring with filled portion based on progress,
    and text showing the percentage in the center.
    """

    __gtype_name__ = "ProgressRing"

    # Colors
    BACKGROUND_COLOR = (0.9, 0.9, 0.9, 1.0)
    SUCCESS_COLOR = (0.2, 0.7, 0.3, 1.0)  # Green
    WARNING_COLOR = (0.9, 0.7, 0.2, 1.0)  # Yellow/Orange
    ERROR_COLOR = (0.8, 0.2, 0.2, 1.0)    # Red
    TEXT_COLOR = (0.2, 0.2, 0.2, 1.0)

    def __init__(self, size: int = 120):
        super().__init__()

        self._size = size
        self._progress = 0.0
        self._line_width = 12

        self.set_size_request(size, size)
        self.set_draw_func(self._on_draw)

    @GObject.Property(type=float, default=0.0, minimum=0.0, maximum=1.0)
    def progress(self) -> float:
        """Get progress value (0.0 to 1.0)."""
        return self._progress

    @progress.setter
    def progress(self, value: float):
        """Set progress value (0.0 to 1.0)."""
        self._progress = max(0.0, min(1.0, value))
        self.queue_draw()

    def set_progress(self, value: float):
        """Set progress value (0.0 to 1.0)."""
        self.progress = value

    def _get_progress_color(self) -> tuple:
        """Get color based on progress level."""
        if self._progress >= 0.8:
            return self.SUCCESS_COLOR
        elif self._progress >= 0.5:
            return self.WARNING_COLOR
        else:
            return self.ERROR_COLOR

    def _on_draw(self, area, cr, width, height):
        """Cairo draw callback."""
        # Center and radius
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) / 2 - self._line_width

        # Background ring
        cr.set_source_rgba(*self.BACKGROUND_COLOR)
        cr.set_line_width(self._line_width)
        cr.arc(center_x, center_y, radius, 0, 2 * math.pi)
        cr.stroke()

        # Progress ring
        if self._progress > 0:
            cr.set_source_rgba(*self._get_progress_color())
            cr.set_line_width(self._line_width)
            cr.set_line_cap(1)  # Round caps

            # Start from top (-π/2) and go clockwise
            start_angle = -math.pi / 2
            end_angle = start_angle + (2 * math.pi * self._progress)

            cr.arc(center_x, center_y, radius, start_angle, end_angle)
            cr.stroke()

        # Center text
        cr.set_source_rgba(*self.TEXT_COLOR)
        cr.select_font_face("Sans", 0, 1)  # Bold

        # Large percentage
        percentage_text = f"{int(self._progress * 100)}"
        cr.set_font_size(28)

        extents = cr.text_extents(percentage_text)
        text_x = center_x - extents.width / 2 - extents.x_bearing
        text_y = center_y - extents.height / 2 - extents.y_bearing - 5

        cr.move_to(text_x, text_y)
        cr.show_text(percentage_text)

        # Percent sign (smaller)
        cr.set_font_size(14)
        cr.select_font_face("Sans", 0, 0)  # Normal weight

        cr.move_to(text_x + extents.width + 2, text_y)
        cr.show_text("%")

        # "Accuracy" label below
        cr.set_font_size(10)
        label = "Accuracy"
        extents = cr.text_extents(label)
        label_x = center_x - extents.width / 2 - extents.x_bearing
        label_y = center_y + 15

        cr.move_to(label_x, label_y)
        cr.show_text(label)


class ToneAccuracyBars(Gtk.Box):
    """
    Horizontal bar chart showing accuracy per tone.
    """

    __gtype_name__ = "ToneAccuracyBars"

    TONE_NAMES = {
        1: "T1 (high)",
        2: "T2 (rising)",
        3: "T3 (dipping)",
        4: "T4 (falling)",
        5: "T5 (neutral)",
    }

    def __init__(self):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=8,
        )

        self._bars = {}

        # Create bars for each tone
        for tone in range(1, 6):
            row = self._create_tone_row(tone)
            self.append(row)

    def _create_tone_row(self, tone: int) -> Gtk.Box:
        """Create a row for a single tone."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        # Label
        label = Gtk.Label(label=self.TONE_NAMES.get(tone, f"T{tone}"))
        label.set_xalign(0)
        label.set_width_chars(12)
        row.append(label)

        # Progress bar
        bar = Gtk.ProgressBar()
        bar.set_hexpand(True)
        bar.set_valign(Gtk.Align.CENTER)
        row.append(bar)

        # Percentage label
        percent_label = Gtk.Label(label="0%")
        percent_label.set_width_chars(5)
        percent_label.set_xalign(1)
        row.append(percent_label)

        self._bars[tone] = (bar, percent_label)

        return row

    def set_accuracies(self, accuracies: dict[int, float]):
        """
        Set accuracy values for each tone.

        Args:
            accuracies: Dict mapping tone number (1-5) to accuracy (0.0-1.0)
        """
        for tone, (bar, label) in self._bars.items():
            accuracy = accuracies.get(tone, 0.0)
            bar.set_fraction(accuracy)
            label.set_text(f"{int(accuracy * 100)}%")

            # Color based on accuracy
            bar.remove_css_class("success")
            bar.remove_css_class("warning")
            bar.remove_css_class("error")

            if accuracy >= 0.8:
                bar.add_css_class("success")
            elif accuracy >= 0.5:
                bar.add_css_class("warning")
            else:
                bar.add_css_class("error")

    def clear(self):
        """Reset all bars to zero."""
        for tone, (bar, label) in self._bars.items():
            bar.set_fraction(0.0)
            label.set_text("0%")

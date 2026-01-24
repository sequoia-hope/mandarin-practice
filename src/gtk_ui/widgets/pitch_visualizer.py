"""F0 pitch contour visualization widget using Cairo."""

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import Gtk, Gdk, GObject
import math
from typing import Optional
import numpy as np


class PitchVisualizer(Gtk.DrawingArea):
    """
    Visualize F0 pitch contours for reference and user recordings.

    Shows:
    - Reference pitch contour (gray)
    - User's pitch contour (blue)
    - Ideal tone pattern overlay
    """

    __gtype_name__ = "PitchVisualizer"

    # Tone pattern colors
    REFERENCE_COLOR = (0.6, 0.6, 0.6, 0.8)  # Gray
    USER_COLOR = (0.2, 0.5, 0.9, 1.0)  # Blue
    GRID_COLOR = (0.8, 0.8, 0.8, 0.5)
    BACKGROUND_COLOR = (0.98, 0.98, 0.98, 1.0)

    def __init__(self):
        super().__init__()

        self.set_size_request(300, 120)
        self.set_vexpand(False)
        self.set_hexpand(True)

        # Data
        self._reference_f0: Optional[np.ndarray] = None
        self._user_f0: Optional[np.ndarray] = None
        self._expected_tones: Optional[list[int]] = None  # All tones in phrase

        # Display range (Hz)
        self._min_hz = 50
        self._max_hz = 400

        # Set up drawing
        self.set_draw_func(self._on_draw)

    def set_reference_f0(self, f0: Optional[np.ndarray]):
        """Set reference F0 contour."""
        self._reference_f0 = f0
        self.queue_draw()

    def set_user_f0(self, f0: Optional[np.ndarray]):
        """Set user's F0 contour."""
        self._user_f0 = f0
        self.queue_draw()

    def set_expected_tones(self, tones: Optional[list[int]]):
        """Set expected tones for ideal pattern display."""
        self._expected_tones = tones
        self.queue_draw()

    def set_expected_tone(self, tone: Optional[int]):
        """Set single expected tone (legacy, wraps set_expected_tones)."""
        self._expected_tones = [tone] if tone else None
        self.queue_draw()

    def set_hz_range(self, min_hz: float, max_hz: float):
        """Set display range in Hz."""
        self._min_hz = min_hz
        self._max_hz = max_hz
        self.queue_draw()

    def clear(self):
        """Clear all F0 data."""
        self._reference_f0 = None
        self._user_f0 = None
        self._expected_tones = None
        self.queue_draw()

    def _on_draw(self, area, cr, width, height):
        """Cairo draw callback."""
        # Background
        cr.set_source_rgba(*self.BACKGROUND_COLOR)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        # Margins
        margin_left = 50
        margin_right = 20
        margin_top = 20
        margin_bottom = 30

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        if plot_width <= 0 or plot_height <= 0:
            return

        # Draw grid
        self._draw_grid(cr, margin_left, margin_top, plot_width, plot_height)

        # Draw Y axis labels (Hz)
        self._draw_y_labels(cr, margin_left, margin_top, plot_height)

        # Draw ideal tone pattern if set
        if self._expected_tones:
            self._draw_ideal_tones(
                cr, margin_left, margin_top, plot_width, plot_height,
                self._expected_tones
            )

        # Draw reference F0
        if self._reference_f0 is not None and len(self._reference_f0) > 0:
            self._draw_f0_contour(
                cr, margin_left, margin_top, plot_width, plot_height,
                self._reference_f0, self.REFERENCE_COLOR, line_width=3
            )

        # Draw user F0
        if self._user_f0 is not None and len(self._user_f0) > 0:
            self._draw_f0_contour(
                cr, margin_left, margin_top, plot_width, plot_height,
                self._user_f0, self.USER_COLOR, line_width=2
            )

        # Draw legend
        self._draw_legend(cr, width, margin_top)

    def _draw_grid(self, cr, x, y, width, height):
        """Draw background grid."""
        cr.set_source_rgba(*self.GRID_COLOR)
        cr.set_line_width(1)

        # Horizontal lines (pitch levels)
        n_lines = 5
        for i in range(n_lines + 1):
            line_y = y + (i / n_lines) * height
            cr.move_to(x, line_y)
            cr.line_to(x + width, line_y)
            cr.stroke()

        # Vertical lines (time divisions)
        for i in range(5):
            line_x = x + (i / 4) * width
            cr.move_to(line_x, y)
            cr.line_to(line_x, y + height)
            cr.stroke()

    def _draw_y_labels(self, cr, x, y, height):
        """Draw Hz labels on Y axis."""
        cr.set_source_rgba(0.3, 0.3, 0.3, 1.0)
        cr.select_font_face("Sans", 0, 0)
        cr.set_font_size(10)

        n_labels = 5
        for i in range(n_labels + 1):
            hz = self._max_hz - (i / n_labels) * (self._max_hz - self._min_hz)
            label_y = y + (i / n_labels) * height

            cr.move_to(5, label_y + 4)
            cr.show_text(f"{int(hz)}Hz")

    def _draw_f0_contour(self, cr, x, y, width, height, f0, color, line_width=2):
        """Draw an F0 contour line."""
        # Filter out zeros/NaN
        valid_f0 = np.where(f0 > 0, f0, np.nan)

        if np.all(np.isnan(valid_f0)):
            return

        cr.set_source_rgba(*color)
        cr.set_line_width(line_width)
        cr.set_line_cap(1)  # Round caps
        cr.set_line_join(1)  # Round joins

        n_points = len(valid_f0)
        started = False

        for i in range(n_points):
            if np.isnan(valid_f0[i]):
                if started:
                    cr.stroke()
                    started = False
                continue

            # Map to plot coordinates
            px = x + (i / (n_points - 1)) * width if n_points > 1 else x + width / 2

            # Map Hz to Y (inverted - higher Hz at top)
            normalized = (valid_f0[i] - self._min_hz) / (self._max_hz - self._min_hz)
            normalized = max(0, min(1, normalized))
            py = y + (1 - normalized) * height

            if not started:
                cr.move_to(px, py)
                started = True
            else:
                cr.line_to(px, py)

        if started:
            cr.stroke()

    def _draw_ideal_tones(self, cr, x, y, width, height, tones):
        """Draw ideal tone patterns for all syllables as a shaded region."""
        # Simplified tone patterns (normalized 0-1 range within each syllable)
        patterns = {
            1: [(0, 0.8), (1, 0.8)],  # High flat
            2: [(0, 0.4), (1, 0.8)],  # Rising
            3: [(0, 0.5), (0.3, 0.3), (0.7, 0.3), (1, 0.5)],  # Dipping
            4: [(0, 0.8), (1, 0.2)],  # Falling
            5: [(0, 0.5), (1, 0.5)],  # Neutral (mid flat)
        }

        if not tones:
            return

        n_tones = len(tones)
        syllable_width = width / n_tones

        # Build combined path for all tones
        all_points = []

        for i, tone in enumerate(tones):
            pattern = patterns.get(tone, patterns[5])  # Default to neutral
            syllable_x = x + i * syllable_width

            for px_norm, py_norm in pattern:
                px = syllable_x + px_norm * syllable_width
                py = y + (1 - py_norm) * height
                all_points.append((px, py))

        if not all_points:
            return

        # Draw as filled region with low opacity
        cr.set_source_rgba(0.3, 0.7, 0.3, 0.15)

        # Create path
        cr.move_to(x, y + height)  # Bottom left

        for px, py in all_points:
            cr.line_to(px, py)

        cr.line_to(x + width, y + height)  # Bottom right
        cr.close_path()
        cr.fill()

        # Draw line
        cr.set_source_rgba(0.3, 0.7, 0.3, 0.4)
        cr.set_line_width(2)
        cr.set_dash([5, 5])

        first = True
        for px, py in all_points:
            if first:
                cr.move_to(px, py)
                first = False
            else:
                cr.line_to(px, py)

        cr.stroke()
        cr.set_dash([])  # Reset dash

    def _draw_legend(self, cr, width, y):
        """Draw legend in top-right corner."""
        cr.select_font_face("Sans", 0, 0)
        cr.set_font_size(10)

        legend_x = width - 130

        # Reference
        cr.set_source_rgba(*self.REFERENCE_COLOR)
        cr.set_line_width(3)
        cr.move_to(legend_x, y + 10)
        cr.line_to(legend_x + 20, y + 10)
        cr.stroke()

        cr.set_source_rgba(0.3, 0.3, 0.3, 1.0)
        cr.move_to(legend_x + 25, y + 14)
        cr.show_text("Reference")

        # User
        cr.set_source_rgba(*self.USER_COLOR)
        cr.set_line_width(2)
        cr.move_to(legend_x, y + 25)
        cr.line_to(legend_x + 20, y + 25)
        cr.stroke()

        cr.set_source_rgba(0.3, 0.3, 0.3, 1.0)
        cr.move_to(legend_x + 25, y + 29)
        cr.show_text("Your attempt")

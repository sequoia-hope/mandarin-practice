"""Animated recording button widget."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, GLib, GObject
from enum import Enum


class RecordState(Enum):
    """Recording button states."""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class RecordButton(Gtk.Button):
    """
    Recording button with state machine and visual feedback.

    States:
        IDLE: Ready to record (shows microphone icon)
        RECORDING: Currently recording (shows stop icon, pulsing)
        PROCESSING: Evaluating recording (shows spinner)
    """

    __gtype_name__ = "RecordButton"

    __gsignals__ = {
        "record-start": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "record-stop": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self):
        super().__init__()

        self.set_size_request(80, 80)
        self.add_css_class("circular")
        self.add_css_class("suggested-action")
        self.add_css_class("record-button")

        # State
        self._state = RecordState.IDLE
        self._pulse_animation_id = None

        # Content box
        self._content_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=4,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )
        self.set_child(self._content_box)

        # Icon
        self._icon = Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic")
        self._icon.set_pixel_size(48)
        self._content_box.append(self._icon)

        # Label
        self._label = Gtk.Label(label="Record")
        self._content_box.append(self._label)

        # Spinner (hidden by default)
        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(48, 48)
        self._spinner.set_visible(False)

        # Connect click handler
        self.connect("clicked", self._on_clicked)

        self._update_appearance()

    @property
    def state(self) -> RecordState:
        """Get current state."""
        return self._state

    def set_state(self, state: RecordState):
        """Set button state."""
        if self._state == state:
            return

        old_state = self._state
        self._state = state

        self._update_appearance()

        # Handle state transitions
        if state == RecordState.RECORDING:
            self._start_pulse_animation()
        else:
            self._stop_pulse_animation()

    def _update_appearance(self):
        """Update button appearance based on state."""
        # Remove state classes
        self.remove_css_class("recording")
        self.remove_css_class("processing")

        if self._state == RecordState.IDLE:
            self._icon.set_from_icon_name("audio-input-microphone-symbolic")
            self._icon.set_visible(True)
            self._label.set_text("Record")
            self._label.set_visible(True)
            self._spinner.set_visible(False)
            self._spinner.stop()
            self.set_sensitive(True)

        elif self._state == RecordState.RECORDING:
            self._icon.set_from_icon_name("media-playback-stop-symbolic")
            self._icon.set_visible(True)
            self._label.set_text("Stop")
            self._label.set_visible(True)
            self._spinner.set_visible(False)
            self.add_css_class("recording")
            self.set_sensitive(True)

        elif self._state == RecordState.PROCESSING:
            self._icon.set_visible(False)
            self._label.set_text("Processing...")
            self._label.set_visible(True)

            # Replace icon with spinner in content box
            if self._spinner.get_parent() is None:
                self._content_box.prepend(self._spinner)
            self._spinner.set_visible(True)
            self._spinner.start()

            self.add_css_class("processing")
            self.set_sensitive(False)

    def _on_clicked(self, button):
        """Handle button click."""
        if self._state == RecordState.IDLE:
            self.emit("record-start")
        elif self._state == RecordState.RECORDING:
            self.emit("record-stop")
        # Processing state is not clickable

    def _start_pulse_animation(self):
        """Start pulsing animation for recording state."""
        self._pulse_count = 0

        def pulse():
            if self._state != RecordState.RECORDING:
                return False

            self._pulse_count += 1
            if self._pulse_count % 2 == 0:
                self.add_css_class("pulse-on")
                self.remove_css_class("pulse-off")
            else:
                self.add_css_class("pulse-off")
                self.remove_css_class("pulse-on")

            return True  # Continue animation

        self._pulse_animation_id = GLib.timeout_add(500, pulse)

    def _stop_pulse_animation(self):
        """Stop pulsing animation."""
        if self._pulse_animation_id:
            GLib.source_remove(self._pulse_animation_id)
            self._pulse_animation_id = None

        self.remove_css_class("pulse-on")
        self.remove_css_class("pulse-off")

    def reset(self):
        """Reset button to idle state."""
        self.set_state(RecordState.IDLE)


# Custom CSS for the button
RECORD_BUTTON_CSS = """
.record-button {
    border-radius: 60px;
    padding: 16px;
    min-width: 120px;
    min-height: 120px;
}

.record-button.recording {
    background-color: @error_bg_color;
}

.record-button.processing {
    background-color: @warning_bg_color;
}

.record-button.pulse-on {
    opacity: 1.0;
}

.record-button.pulse-off {
    opacity: 0.7;
}
"""

"""Settings view for application configuration."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, Adw, GObject


class SettingsView(Gtk.Box):
    """
    Settings view for configuring the application.

    Options:
    - TTS backend selection
    - LLM backend selection (for conversational mode)
    - Audio device selection
    - Theme selection
    """

    __gtype_name__ = "SettingsView"

    __gsignals__ = {
        "settings-changed": (GObject.SignalFlags.RUN_FIRST, None, (str, str)),
    }

    def __init__(self):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=0,
        )

        self.set_vexpand(True)
        self.set_hexpand(True)

        # Current settings
        self._settings = {
            "tts_backend": "light",
            "llm_backend": "ollama",
            "theme": "system",
        }

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the settings view UI."""
        # Scrolled container
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.append(scrolled)

        # Clamp for comfortable reading width
        clamp = Adw.Clamp()
        clamp.set_maximum_size(600)
        scrolled.set_child(clamp)

        # Main content
        content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=24,
        )
        content.set_margin_top(24)
        content.set_margin_bottom(24)
        content.set_margin_start(12)
        content.set_margin_end(12)
        clamp.set_child(content)

        # Audio settings group
        audio_group = Adw.PreferencesGroup()
        audio_group.set_title("Audio")
        audio_group.set_description("Configure text-to-speech and recording")
        content.append(audio_group)

        # TTS backend
        self._tts_row = Adw.ComboRow()
        self._tts_row.set_title("TTS Backend")
        self._tts_row.set_subtitle("Voice synthesis for reference audio")

        tts_model = Gtk.StringList.new(["Light (Fast)", "Qwen (High Quality)", "Dual (Recommended)"])
        self._tts_row.set_model(tts_model)
        self._tts_row.set_selected(2)  # Dual by default
        self._tts_row.connect("notify::selected", self._on_tts_changed)

        audio_group.add(self._tts_row)

        # LLM settings group
        llm_group = Adw.PreferencesGroup()
        llm_group.set_title("AI Model")
        llm_group.set_description("Configure LLM for conversational mode")
        content.append(llm_group)

        # LLM backend
        self._llm_row = Adw.ComboRow()
        self._llm_row.set_title("LLM Backend")
        self._llm_row.set_subtitle("For conversation and explanations")

        llm_model = Gtk.StringList.new(["Ollama (Local)", "Claude (Cloud)"])
        self._llm_row.set_model(llm_model)
        self._llm_row.set_selected(0)  # Ollama by default
        self._llm_row.connect("notify::selected", self._on_llm_changed)

        llm_group.add(self._llm_row)

        # Appearance group
        appearance_group = Adw.PreferencesGroup()
        appearance_group.set_title("Appearance")
        content.append(appearance_group)

        # Theme
        self._theme_row = Adw.ComboRow()
        self._theme_row.set_title("Theme")
        self._theme_row.set_subtitle("Application color scheme")

        theme_model = Gtk.StringList.new(["System Default", "Light", "Dark"])
        self._theme_row.set_model(theme_model)
        self._theme_row.set_selected(0)  # System by default
        self._theme_row.connect("notify::selected", self._on_theme_changed)

        appearance_group.add(self._theme_row)

        # About section
        about_group = Adw.PreferencesGroup()
        about_group.set_title("About")
        content.append(about_group)

        # Version row
        version_row = Adw.ActionRow()
        version_row.set_title("Version")
        version_row.set_subtitle("1.0.0")
        about_group.add(version_row)

        # Reset progress button
        reset_row = Adw.ActionRow()
        reset_row.set_title("Reset Progress")
        reset_row.set_subtitle("Clear all learning history and statistics")

        reset_button = Gtk.Button(label="Reset")
        reset_button.add_css_class("destructive-action")
        reset_button.set_valign(Gtk.Align.CENTER)
        reset_button.connect("clicked", self._on_reset_clicked)
        reset_row.add_suffix(reset_button)
        reset_row.set_activatable_widget(reset_button)

        about_group.add(reset_row)

    def _on_tts_changed(self, row, param):
        """Handle TTS backend change."""
        selected = row.get_selected()
        backend_map = {0: "light", 1: "qwen", 2: "dual"}
        backend = backend_map.get(selected, "dual")

        self._settings["tts_backend"] = backend
        self.emit("settings-changed", "tts_backend", backend)

    def _on_llm_changed(self, row, param):
        """Handle LLM backend change."""
        selected = row.get_selected()
        backend_map = {0: "ollama", 1: "claude"}
        backend = backend_map.get(selected, "ollama")

        self._settings["llm_backend"] = backend
        self.emit("settings-changed", "llm_backend", backend)

    def _on_theme_changed(self, row, param):
        """Handle theme change."""
        selected = row.get_selected()
        theme_map = {0: "system", 1: "light", 2: "dark"}
        theme = theme_map.get(selected, "system")

        self._settings["theme"] = theme
        self.emit("settings-changed", "theme", theme)

    def _on_reset_clicked(self, button):
        """Handle reset progress button click."""
        # Show confirmation dialog
        dialog = Adw.AlertDialog()
        dialog.set_heading("Reset Progress?")
        dialog.set_body(
            "This will permanently delete all your learning history, "
            "statistics, and mastered phrases. This action cannot be undone."
        )

        dialog.add_response("cancel", "Cancel")
        dialog.add_response("reset", "Reset")
        dialog.set_response_appearance("reset", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")

        dialog.connect("response", self._on_reset_response)
        dialog.present(self.get_root())

    def _on_reset_response(self, dialog, response):
        """Handle reset confirmation dialog response."""
        if response == "reset":
            self._do_reset_progress()

    def _do_reset_progress(self):
        """Actually reset the progress."""
        try:
            from ...coach.progress import ProgressTracker
            from pathlib import Path

            tracker = ProgressTracker()
            progress_file = tracker.progress_file

            if progress_file.exists():
                progress_file.unlink()

            # Show success toast
            window = self.get_root()
            if window and hasattr(window, "add_toast"):
                toast = Adw.Toast(title="Progress has been reset", timeout=3)
                window.add_toast(toast)

        except Exception as e:
            print(f"[SettingsView] Reset error: {e}")

    def get_setting(self, key: str) -> str:
        """Get a setting value."""
        return self._settings.get(key, "")

    def set_setting(self, key: str, value: str):
        """Set a setting value programmatically."""
        if key == "tts_backend":
            backend_map = {"light": 0, "qwen": 1, "dual": 2}
            self._tts_row.set_selected(backend_map.get(value, 2))
        elif key == "llm_backend":
            backend_map = {"ollama": 0, "claude": 1}
            self._llm_row.set_selected(backend_map.get(value, 0))
        elif key == "theme":
            theme_map = {"system": 0, "light": 1, "dark": 2}
            self._theme_row.set_selected(theme_map.get(value, 0))

        self._settings[key] = value

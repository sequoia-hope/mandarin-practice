"""Adwaita application for Mandarin Tone Coach."""

import gi
import threading

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gio, GLib

from .window import ToneCoachWindow


class ToneCoachApp(Adw.Application):
    """Main application class for Mandarin Tone Coach."""

    def __init__(self):
        super().__init__(
            application_id="com.example.MandarinToneCoach",
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )

        self.window = None
        self._preload_done = False

        # Set up actions
        self._setup_actions()

    def _setup_actions(self):
        """Set up application actions."""
        # Quit action
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self._on_quit)
        self.add_action(quit_action)
        self.set_accels_for_action("app.quit", ["<Control>q"])

        # About action
        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self._on_about)
        self.add_action(about_action)

        # Preferences action
        preferences_action = Gio.SimpleAction.new("preferences", None)
        preferences_action.connect("activate", self._on_preferences)
        self.add_action(preferences_action)
        self.set_accels_for_action("app.preferences", ["<Control>comma"])

    def do_activate(self):
        """Called when the application is activated."""
        if not self.window:
            self.window = ToneCoachWindow(application=self)

        self.window.present()

        # Schedule model preload 2 seconds after app loads
        if not self._preload_done:
            GLib.timeout_add(2000, self._preload_models)

    def _preload_models(self):
        """Preload models by running first lesson through pipeline."""
        if self._preload_done:
            return False  # Don't repeat

        self._preload_done = True

        def preload_task():
            try:
                print("[Preload] Warming up models...")

                # Get first lesson's first phrase
                from src.coach.voice_menu import BUILT_IN_LESSONS
                if not BUILT_IN_LESSONS or not BUILT_IN_LESSONS[0].phrases:
                    return

                phrase = BUILT_IN_LESSONS[0].phrases[0]
                chars = phrase["characters"]
                pinyin = phrase["pinyin"]
                tones = phrase["tones"]

                # Generate TTS audio (teacher speed for learning)
                from src.coach.tts import LightTTS
                tts = LightTTS()
                audio = tts.speak(chars, language="zh", speed="teacher", play=False)

                if audio is None or len(audio) == 0:
                    print("[Preload] TTS returned no audio")
                    return

                # Run through evaluator (this loads all models)
                import os
                from src.coach.evaluator import Evaluator

                # Strip tone marks from pinyin
                def strip_tones(p):
                    replacements = {
                        'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
                        'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
                        'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
                        'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
                        'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
                        'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v', 'ü': 'v',
                    }
                    result = p.lower()
                    for mark, plain in replacements.items():
                        result = result.replace(mark, plain)
                    return result

                # Get ASR backend setting
                asr_backend = os.environ.get("TONE_COACH_ASR_BACKEND", "both")
                use_asr = asr_backend != "none"

                evaluator = Evaluator.get_instance(
                    use_asr=use_asr,
                    asr_backend=asr_backend,
                )
                evaluator.evaluate(
                    audio=audio,
                    expected_pinyin=[strip_tones(p) for p in pinyin],
                    expected_tones=tones,
                    characters=chars,
                )

                print("[Preload] Models ready!")

            except Exception as e:
                print(f"[Preload] Error: {e}")

        # Run in background thread
        thread = threading.Thread(target=preload_task, daemon=True)
        thread.start()

        return False  # Don't repeat the timeout

    def _on_quit(self, action, param):
        """Handle quit action."""
        self.quit()

    def _on_about(self, action, param):
        """Show about dialog."""
        about = Adw.AboutDialog(
            application_name="Mandarin Tone Coach",
            application_icon="audio-speakers-symbolic",
            developer_name="Mandarin Tone Coach Project",
            version="1.0.0",
            copyright="2024",
            license_type=Gtk.License.MIT_X11,
            comments="Learn Mandarin Chinese tones with real-time feedback",
            website="https://github.com/example/mandarin-tone-coach",
        )
        about.present(self.window)

    def _on_preferences(self, action, param):
        """Show preferences/settings view."""
        if self.window:
            self.window.show_settings()


# Import Gtk for license type
from gi.repository import Gtk

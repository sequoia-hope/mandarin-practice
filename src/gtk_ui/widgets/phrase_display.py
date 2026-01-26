"""Widget for displaying Chinese phrase with pinyin and tone indicators."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, Pango, GObject


class PhraseDisplay(Gtk.Box):
    """
    Display widget showing Chinese characters, pinyin, tones, and translation.

    Layout:
        你好
        nǐ hǎo (T3 T3)
        "hello"
    """

    __gtype_name__ = "PhraseDisplay"

    def __init__(self):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=4,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )

        self.set_margin_top(12)
        self.set_margin_bottom(12)

        # Chinese characters (large)
        self.characters_label = Gtk.Label()
        self.characters_label.add_css_class("title-1")
        self.characters_label.set_attributes(self._create_large_font_attrs())
        self.append(self.characters_label)

        # Pinyin with tones
        self.pinyin_label = Gtk.Label()
        self.append(self.pinyin_label)

        # English translation
        self.english_label = Gtk.Label()
        self.english_label.add_css_class("dim-label")
        self.english_label.add_css_class("caption")
        self.append(self.english_label)

        # Phrase data
        self._characters = ""
        self._pinyin = []
        self._tones = []
        self._english = ""

    def _create_large_font_attrs(self) -> Pango.AttrList:
        """Create Pango attributes for large Chinese text."""
        attrs = Pango.AttrList()
        size_attr = Pango.attr_size_new(36 * Pango.SCALE)
        attrs.insert(size_attr)
        return attrs

    def set_phrase(
        self,
        characters: str,
        pinyin: list[str],
        tones: list[int],
        english: str = "",
        spoken_tones: list[int] = None,
    ):
        """
        Set the phrase to display.

        Args:
            characters: Chinese characters
            pinyin: List of pinyin syllables with tone marks
            tones: List of tone numbers (1-5) - underlying/lexical tones
            english: English translation
            spoken_tones: List of spoken tone numbers after sandhi (optional)
        """
        self._characters = characters
        self._pinyin = pinyin
        self._tones = tones
        self._english = english

        # Update display
        self.characters_label.set_text(characters)

        # Format pinyin with tone numbers
        pinyin_parts = []
        for i, p in enumerate(pinyin):
            pinyin_parts.append(f"{p}")

        # Show tone numbers - if sandhi applies, show both underlying and spoken
        underlying_str = " ".join(f"T{t}" for t in tones)

        if spoken_tones and spoken_tones != tones:
            # Sandhi applies - show transformation
            spoken_str = " ".join(f"T{t}" for t in spoken_tones)
            tone_str = f"{underlying_str} → {spoken_str}"
        else:
            tone_str = underlying_str

        pinyin_text = f"{' '.join(pinyin_parts)}  ({tone_str})"
        self.pinyin_label.set_text(pinyin_text)

        # English translation in quotes
        if english:
            self.english_label.set_text(f'"{english}"')
            self.english_label.set_visible(True)
        else:
            self.english_label.set_visible(False)

    def get_characters(self) -> str:
        """Get the current Chinese characters."""
        return self._characters

    def get_pinyin(self) -> list[str]:
        """Get the current pinyin list."""
        return self._pinyin.copy()

    def get_tones(self) -> list[int]:
        """Get the current tone list."""
        return self._tones.copy()

    def clear(self):
        """Clear the display."""
        self._characters = ""
        self._pinyin = []
        self._tones = []
        self._english = ""

        self.characters_label.set_text("")
        self.pinyin_label.set_text("")
        self.english_label.set_text("")
        self.english_label.set_visible(False)


# Custom CSS for the widget
PHRASE_DISPLAY_CSS = """
.phrase-display .title-1 {
    font-size: 48px;
    font-weight: bold;
}

.phrase-display .title-3 {
    font-size: 18px;
}

.phrase-display .dim-label {
    font-style: italic;
    opacity: 0.7;
}
"""

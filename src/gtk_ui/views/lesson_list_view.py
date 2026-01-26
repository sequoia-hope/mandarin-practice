"""Sidebar view for lesson selection (integrated into main window sidebar)."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, Adw, GObject


class LessonListView(Gtk.Box):
    """
    Lesson list view for the sidebar.

    Note: This is primarily used as a reference - the main lesson list
    is built directly in the window sidebar. This class can be used
    for standalone lesson browsing or as a component.
    """

    __gtype_name__ = "LessonListView"

    __gsignals__ = {
        "lesson-selected": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "phrase-selected": (GObject.SignalFlags.RUN_FIRST, None, (str, int)),
    }

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Scrolled container
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.append(scrolled)

        # Main list
        self._list_box = Gtk.ListBox()
        self._list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._list_box.add_css_class("navigation-sidebar")
        self._list_box.connect("row-selected", self._on_row_selected)
        scrolled.set_child(self._list_box)

        # Track expanded lessons
        self._expanded_lessons = set()
        self._lesson_rows = {}
        self._phrase_rows = {}

        # Load lessons
        self._load_lessons()

    def _load_lessons(self):
        """Load lessons from BUILT_IN_LESSONS."""
        from ...coach.voice_menu import BUILT_IN_LESSONS

        for lesson in BUILT_IN_LESSONS:
            self._add_lesson_row(lesson)

    def _add_lesson_row(self, lesson):
        """Add a lesson row with expander."""
        # Main row
        row = Adw.ExpanderRow()
        row.set_title(lesson.name)
        row.set_subtitle(f"{len(lesson.phrases)} phrases")
        row.set_icon_name("view-list-symbolic")

        # Store lesson id
        row.lesson_id = lesson.id
        self._lesson_rows[lesson.id] = row

        # Add phrase rows as children
        for i, phrase in enumerate(lesson.phrases):
            phrase_row = self._create_phrase_row(lesson.id, i, phrase)
            row.add_row(phrase_row)

        self._list_box.append(row)

    def _create_phrase_row(self, lesson_id: str, index: int, phrase: dict) -> Adw.ActionRow:
        """Create a row for a phrase."""
        row = Adw.ActionRow()
        row.set_title(phrase["characters"])
        row.set_subtitle(phrase.get("english", ""))

        # Store identifiers
        row.lesson_id = lesson_id
        row.phrase_index = index

        # Make activatable
        row.set_activatable(True)
        row.connect("activated", self._on_phrase_activated)

        # Store in tracking dict
        key = f"{lesson_id}:{index}"
        self._phrase_rows[key] = row

        return row

    def _on_row_selected(self, list_box, row):
        """Handle row selection."""
        if row is None:
            return

        if hasattr(row, "lesson_id") and hasattr(row, "phrase_index"):
            # Phrase row selected
            self.emit("phrase-selected", row.lesson_id, row.phrase_index)
        elif hasattr(row, "lesson_id"):
            # Lesson row selected
            self.emit("lesson-selected", row.lesson_id)

    def _on_phrase_activated(self, row):
        """Handle phrase row activation."""
        if hasattr(row, "lesson_id") and hasattr(row, "phrase_index"):
            self.emit("phrase-selected", row.lesson_id, row.phrase_index)

    def select_lesson(self, lesson_id: str):
        """Programmatically select a lesson."""
        if lesson_id in self._lesson_rows:
            row = self._lesson_rows[lesson_id]
            self._list_box.select_row(row)
            row.set_expanded(True)

    def select_phrase(self, lesson_id: str, phrase_index: int):
        """Programmatically select a phrase."""
        key = f"{lesson_id}:{phrase_index}"
        if key in self._phrase_rows:
            row = self._phrase_rows[key]
            self._list_box.select_row(row)

            # Expand parent lesson
            if lesson_id in self._lesson_rows:
                self._lesson_rows[lesson_id].set_expanded(True)

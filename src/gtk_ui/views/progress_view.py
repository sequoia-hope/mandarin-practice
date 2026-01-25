"""Progress view showing learning statistics."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, Adw, GObject

from ..widgets.progress_ring import ProgressRing, ToneAccuracyBars


class ProgressView(Gtk.Box):
    """
    View displaying user's learning progress and statistics.

    Shows:
    - Overall accuracy ring
    - Per-tone accuracy bars
    - Session history
    - Phrases mastered
    - Weakest tones
    """

    __gtype_name__ = "ProgressView"

    def __init__(self):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=0,
        )

        self.set_vexpand(True)
        self.set_hexpand(True)

        # Progress tracker
        self._tracker = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the progress view UI."""
        # Scrolled container
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.append(scrolled)

        # Main content
        content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=24,
        )
        content.set_margin_top(24)
        content.set_margin_bottom(24)
        content.set_margin_start(24)
        content.set_margin_end(24)
        scrolled.set_child(content)

        # Header
        header = Gtk.Label(label="Your Learning Progress")
        header.add_css_class("title-1")
        header.set_halign(Gtk.Align.START)
        content.append(header)

        # Top stats row
        stats_row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=24,
            homogeneous=True,
        )
        content.append(stats_row)

        # Overall accuracy ring
        accuracy_card = self._create_card("Overall Accuracy")
        self._accuracy_ring = ProgressRing(size=140)
        accuracy_card.get_first_child().append(self._accuracy_ring)
        stats_row.append(accuracy_card)

        # Stats card
        stats_card = self._create_card("Statistics")
        stats_box = stats_card.get_first_child()

        self._stats_grid = Gtk.Grid()
        self._stats_grid.set_row_spacing(8)
        self._stats_grid.set_column_spacing(24)
        stats_box.append(self._stats_grid)

        # Add stat items
        self._stat_labels = {}
        stat_items = [
            ("sessions", "Sessions", "0"),
            ("practice_time", "Practice Time", "0h"),
            ("attempts", "Total Attempts", "0"),
            ("lessons", "Lessons Completed", "0"),
            ("mastered", "Phrases Mastered", "0"),
        ]

        for i, (key, label, default) in enumerate(stat_items):
            name_label = Gtk.Label(label=label)
            name_label.set_xalign(0)
            name_label.add_css_class("dim-label")

            value_label = Gtk.Label(label=default)
            value_label.set_xalign(0)
            value_label.add_css_class("title-4")

            self._stats_grid.attach(name_label, 0, i, 1, 1)
            self._stats_grid.attach(value_label, 1, i, 1, 1)

            self._stat_labels[key] = value_label

        stats_row.append(stats_card)

        # Tone accuracy section
        tone_card = self._create_card("Accuracy by Tone")
        tone_box = tone_card.get_first_child()

        self._tone_bars = ToneAccuracyBars()
        tone_box.append(self._tone_bars)

        content.append(tone_card)

        # Focus areas section
        focus_card = self._create_card("Focus Areas")
        focus_box = focus_card.get_first_child()

        self._focus_label = Gtk.Label()
        self._focus_label.set_xalign(0)
        self._focus_label.set_wrap(True)
        focus_box.append(self._focus_label)

        content.append(focus_card)

        # Recent sessions section
        sessions_card = self._create_card("Recent Sessions")
        sessions_box = sessions_card.get_first_child()

        self._sessions_list = Gtk.ListBox()
        self._sessions_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._sessions_list.add_css_class("boxed-list")
        sessions_box.append(self._sessions_list)

        content.append(sessions_card)

    def _create_card(self, title: str) -> Gtk.Frame:
        """Create a card with title."""
        frame = Gtk.Frame()
        frame.add_css_class("card")

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
        )
        box.set_margin_top(16)
        box.set_margin_bottom(16)
        box.set_margin_start(16)
        box.set_margin_end(16)

        header = Gtk.Label(label=title)
        header.add_css_class("heading")
        header.set_xalign(0)
        box.append(header)

        frame.set_child(box)
        return frame

    def _get_tracker(self):
        """Get progress tracker, creating if needed."""
        if self._tracker is None:
            try:
                from ...coach.progress import ProgressTracker
                self._tracker = ProgressTracker()
            except Exception as e:
                print(f"[ProgressView] Failed to load tracker: {e}")
        return self._tracker

    def refresh(self):
        """Refresh progress data from tracker."""
        # Force reload from disk to get latest data
        self._tracker = None
        tracker = self._get_tracker()
        if tracker is None:
            return

        summary = tracker.get_summary()

        # Update accuracy ring
        self._accuracy_ring.set_progress(summary.get("lifetime_accuracy", 0.0))

        # Update stats
        self._stat_labels["sessions"].set_text(str(summary.get("total_sessions", 0)))

        hours = summary.get("practice_time_hours", 0)
        if hours >= 1:
            self._stat_labels["practice_time"].set_text(f"{hours:.1f}h")
        else:
            minutes = int(hours * 60)
            self._stat_labels["practice_time"].set_text(f"{minutes}m")

        self._stat_labels["attempts"].set_text(str(summary.get("lifetime_attempts", 0)))
        self._stat_labels["lessons"].set_text(str(summary.get("lessons_completed", 0)))
        self._stat_labels["mastered"].set_text(str(summary.get("phrases_mastered", 0)))

        # Update tone bars
        tone_accuracies = summary.get("tone_accuracies", {})
        # Convert string keys to int
        tone_acc_int = {int(k): v for k, v in tone_accuracies.items()}
        self._tone_bars.set_accuracies(tone_acc_int)

        # Update focus areas
        weak_tones = summary.get("weakest_tones", [])
        if weak_tones:
            tone_names = {
                1: "Tone 1 (high level)",
                2: "Tone 2 (rising)",
                3: "Tone 3 (dipping)",
                4: "Tone 4 (falling)",
                5: "Tone 5 (neutral)",
            }
            focus_text = "Focus on practicing: " + ", ".join(
                tone_names.get(t, f"Tone {t}") for t in weak_tones
            )
            self._focus_label.set_text(focus_text)
        else:
            self._focus_label.set_text(
                "Complete more practice sessions to identify areas for improvement."
            )

        # Update recent sessions
        self._update_sessions_list(tracker.progress.recent_sessions)

    def _update_sessions_list(self, sessions: list):
        """Update the recent sessions list."""
        # Clear existing rows
        while True:
            row = self._sessions_list.get_first_child()
            if row is None:
                break
            self._sessions_list.remove(row)

        if not sessions:
            # Empty state
            empty_label = Gtk.Label(label="No sessions yet. Start practicing!")
            empty_label.add_css_class("dim-label")
            empty_label.set_margin_top(12)
            empty_label.set_margin_bottom(12)

            row = Gtk.ListBoxRow()
            row.set_child(empty_label)
            row.set_selectable(False)
            self._sessions_list.append(row)
            return

        # Add session rows
        for session in sessions[:10]:  # Show last 10
            row = self._create_session_row(session)
            self._sessions_list.append(row)

    def _create_session_row(self, session: dict) -> Gtk.ListBoxRow:
        """Create a row for a session."""
        row = Adw.ActionRow()

        # Parse date
        date_str = session.get("date", "")
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(date_str)
            formatted_date = dt.strftime("%b %d, %Y %H:%M")
        except Exception:
            formatted_date = date_str

        row.set_title(formatted_date)

        # Subtitle with stats
        attempts = session.get("attempts", 0)
        correct = session.get("correct", 0)
        accuracy = session.get("accuracy", 0)

        row.set_subtitle(f"{correct}/{attempts} correct ({accuracy:.0%})")

        # Accuracy indicator
        accuracy_label = Gtk.Label(label=f"{int(accuracy * 100)}%")
        if accuracy >= 0.8:
            accuracy_label.add_css_class("success")
        elif accuracy >= 0.5:
            accuracy_label.add_css_class("warning")
        else:
            accuracy_label.add_css_class("error")

        row.add_suffix(accuracy_label)

        return row

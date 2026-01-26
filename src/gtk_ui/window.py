"""Main application window with sidebar navigation."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk, GLib, Gio


class ToneCoachWindow(Adw.ApplicationWindow):
    """Main application window with AdwNavigationSplitView."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("Mandarin Tone Coach")
        self.set_default_size(1000, 700)

        # Controllers (created lazily)
        self._practice_controller = None
        self._audio_controller = None

        # Build UI
        self._build_ui()

        # Load views after UI is built
        GLib.idle_add(self._initialize_views)

    def _build_ui(self):
        """Build the main UI structure."""
        # Main layout box
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Header bar with menu
        header = Adw.HeaderBar()
        main_box.append(header)

        # Add menu button
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu_button.set_menu_model(self._create_menu())
        header.pack_end(menu_button)

        # Navigation split view (sidebar + content)
        self.split_view = Adw.NavigationSplitView()
        self.split_view.set_vexpand(True)
        main_box.append(self.split_view)

        # Sidebar
        self._build_sidebar()

        # Content area (starts with welcome/practice view)
        self._build_content_area()

    def _create_menu(self) -> Gio.Menu:
        """Create the application menu."""
        menu = Gio.Menu()

        menu.append("Preferences", "app.preferences")
        menu.append("About", "app.about")
        menu.append("Quit", "app.quit")

        return menu

    def _build_sidebar(self):
        """Build the sidebar with navigation options."""
        sidebar_page = Adw.NavigationPage(title="Navigation")

        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        sidebar_page.set_child(sidebar_box)

        # Sidebar header
        sidebar_header = Adw.HeaderBar()
        sidebar_header.set_show_end_title_buttons(False)
        sidebar_box.append(sidebar_header)

        # Scrolled window for list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sidebar_box.append(scrolled)

        # List box for navigation items
        self.nav_list = Gtk.ListBox()
        self.nav_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.nav_list.add_css_class("navigation-sidebar")
        self.nav_list.connect("row-selected", self._on_nav_selected)
        scrolled.set_child(self.nav_list)

        # Add navigation sections
        self._add_nav_header("Lessons")

        # Lesson items (will be populated from BUILT_IN_LESSONS)
        self.lesson_rows = {}

        # Add navigation items for lessons
        from ..coach.voice_menu import BUILT_IN_LESSONS

        for lesson in BUILT_IN_LESSONS:
            row = self._create_nav_row(
                lesson.name,
                "view-list-symbolic",
                f"lesson:{lesson.id}",
            )
            self.nav_list.append(row)
            self.lesson_rows[lesson.id] = row

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(12)
        separator.set_margin_bottom(12)
        self.nav_list.append(separator)

        # Other navigation items
        self._add_nav_header("Tools")

        self.progress_row = self._create_nav_row(
            "Progress",
            "document-open-recent-symbolic",
            "view:progress",
        )
        self.nav_list.append(self.progress_row)

        self.settings_row = self._create_nav_row(
            "Settings",
            "preferences-system-symbolic",
            "view:settings",
        )
        self.nav_list.append(self.settings_row)

        self.split_view.set_sidebar(sidebar_page)

    def _add_nav_header(self, title: str):
        """Add a header row to navigation list."""
        label = Gtk.Label(label=title)
        label.set_xalign(0)
        label.add_css_class("heading")
        label.set_margin_start(12)
        label.set_margin_top(12)
        label.set_margin_bottom(6)

        row = Gtk.ListBoxRow()
        row.set_selectable(False)
        row.set_activatable(False)
        row.set_child(label)

        self.nav_list.append(row)

    def _create_nav_row(self, label: str, icon_name: str, action: str) -> Gtk.ListBoxRow:
        """Create a navigation row with icon and label."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        icon = Gtk.Image.new_from_icon_name(icon_name)
        box.append(icon)

        text = Gtk.Label(label=label)
        text.set_xalign(0)
        text.set_hexpand(True)
        box.append(text)

        row = Gtk.ListBoxRow()
        row.set_child(box)
        row.action_name = action

        return row

    def _build_content_area(self):
        """Build the main content area."""
        self.content_page = Adw.NavigationPage(title="Practice")

        # Stack for different views
        self.view_stack = Gtk.Stack()
        self.view_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.view_stack.set_transition_duration(200)

        # Wrap stack in a box with header
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # Content header
        self.content_header = Adw.HeaderBar()
        self.content_header.set_show_end_title_buttons(True)
        content_box.append(self.content_header)

        content_box.append(self.view_stack)
        self.content_page.set_child(content_box)

        self.split_view.set_content(self.content_page)

    def _initialize_views(self):
        """Initialize and add views to the stack."""
        # Import views
        from .views import LessonListView, PracticeView, ProgressView, SettingsView

        # Create practice view (main view)
        self.practice_view = PracticeView()
        self.view_stack.add_named(self.practice_view, "practice")

        # Create progress view
        self.progress_view = ProgressView()
        self.view_stack.add_named(self.progress_view, "progress")

        # Create settings view
        self.settings_view = SettingsView()
        self.view_stack.add_named(self.settings_view, "settings")

        # Connect settings changes
        self.settings_view.connect("settings-changed", self._on_settings_changed)

        # Set initial view to practice
        self.view_stack.set_visible_child_name("practice")

        # Select first lesson by default
        if self.lesson_rows:
            first_lesson_id = list(self.lesson_rows.keys())[0]
            first_row = self.lesson_rows[first_lesson_id]
            self.nav_list.select_row(first_row)

        return False  # Don't repeat idle callback

    def _on_nav_selected(self, list_box, row):
        """Handle navigation selection."""
        if row is None or not hasattr(row, "action_name"):
            return

        action = row.action_name

        if action.startswith("lesson:"):
            lesson_id = action.split(":")[1]
            self._show_lesson(lesson_id)
        elif action == "view:progress":
            self._show_progress()
        elif action == "view:settings":
            self.show_settings()

    def _show_lesson(self, lesson_id: str):
        """Show the practice view with a specific lesson."""
        from ..coach.voice_menu import BUILT_IN_LESSONS

        # Find the lesson
        lesson = None
        for l in BUILT_IN_LESSONS:
            if l.id == lesson_id:
                lesson = l
                break

        if lesson:
            self.content_page.set_title(lesson.name)
            self.practice_view.set_lesson(lesson)
            self.view_stack.set_visible_child_name("practice")

            # Collapse sidebar on mobile
            self.split_view.set_show_content(True)

    def _show_progress(self):
        """Show the progress view."""
        self.content_page.set_title("Progress")
        self.progress_view.refresh()
        self.view_stack.set_visible_child_name("progress")
        self.split_view.set_show_content(True)

    def show_settings(self):
        """Show the settings view."""
        self.content_page.set_title("Settings")
        self.view_stack.set_visible_child_name("settings")
        self.split_view.set_show_content(True)

    def _on_settings_changed(self, settings_view, key, value):
        """Handle settings changes."""
        if key == "tts_backend":
            # Update TTS backend in practice view
            self.practice_view.set_tts_backend(value)
        elif key == "theme":
            # Apply theme change
            style_manager = Adw.StyleManager.get_default()
            if value == "light":
                style_manager.set_color_scheme(Adw.ColorScheme.FORCE_LIGHT)
            elif value == "dark":
                style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)
            else:
                style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)

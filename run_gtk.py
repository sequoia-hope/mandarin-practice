#!/usr/bin/env python3
"""
GTK4 GUI entry point for Mandarin Tone Coach.

This application provides a graphical interface for practicing Mandarin Chinese
tone pronunciation with real-time feedback.

Requirements:
    - GTK4 and libadwaita (system packages)
    - Python GObject introspection bindings

On Ubuntu/Debian:
    sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1

On Fedora:
    sudo dnf install python3-gobject gtk4 libadwaita

Usage:
    python run_gtk.py [--asr mms|whisper|both]
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def check_dependencies():
    """Check that required GTK dependencies are available."""
    try:
        import gi
        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")
        from gi.repository import Gtk, Adw
        return True
    except (ImportError, ValueError) as e:
        print("=" * 60)
        print("ERROR: GTK4 dependencies not found")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        print()
        print("Please install the required system packages:")
        print()
        print("Ubuntu/Debian:")
        print("    sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1")
        print()
        print("Fedora:")
        print("    sudo dnf install python3-gobject gtk4 libadwaita")
        print()
        print("Arch Linux:")
        print("    sudo pacman -S python-gobject gtk4 libadwaita")
        print()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mandarin Tone Coach - Learn Chinese tones with real-time feedback"
    )
    parser.add_argument(
        "--asr",
        choices=["mms", "whisper", "both", "none"],
        default="both",
        help="ASR backend for word recognition: mms, whisper, both, or none (default: both)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode (no CUDA/GPU)"
    )
    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    # Store settings in environment for app to read
    os.environ["TONE_COACH_ASR_BACKEND"] = args.asr
    if args.cpu:
        os.environ["TONE_COACH_DEVICE"] = "cpu"

    # Import and run application
    from src.gtk_ui import ToneCoachApp

    app = ToneCoachApp()
    # Pass empty argv to GTK since we handled args ourselves
    return app.run([sys.argv[0]])


if __name__ == "__main__":
    sys.exit(main())

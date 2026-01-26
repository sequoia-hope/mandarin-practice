#!/bin/bash
# Setup script for Mandarin Tone Coach
# Creates virtual environment and installs dependencies

set -e

echo "=========================================="
echo "  Mandarin Tone Coach - Setup"
echo "=========================================="
echo

# Check Python version
PYTHON=${PYTHON:-python3}
if ! command -v $PYTHON &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PY_VERSION ($PYTHON)"

# Check for required system packages (GTK4)
echo
echo "Checking GTK4 dependencies..."
if $PYTHON -c "import gi; gi.require_version('Gtk', '4.0')" 2>/dev/null; then
    echo "  GTK4 bindings: OK"
else
    echo "  GTK4 bindings: NOT FOUND for $PYTHON"
    echo

    # Try to find which Python has gi installed
    echo "Searching for Python with GTK bindings..."
    FOUND_PYTHON=""
    for py in python3.12 python3.11 python3.10 python3; do
        if command -v $py &> /dev/null; then
            if $py -c "import gi; gi.require_version('Gtk', '4.0')" 2>/dev/null; then
                FOUND_PYTHON=$py
                echo "  Found: $py has GTK4 bindings"
                break
            fi
        fi
    done

    if [ -n "$FOUND_PYTHON" ]; then
        echo
        echo "Run setup with the correct Python:"
        echo "  PYTHON=$FOUND_PYTHON ./setup.sh"
        exit 1
    fi

    echo
    echo "GTK4 Python bindings not found on any Python version."
    echo
    echo "Please install GTK4 system packages:"
    echo
    echo "  Ubuntu 24.04:"
    echo "    sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1"
    echo
    echo "  Ubuntu/Debian (older):"
    echo "    sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1"
    echo
    echo "  Fedora:"
    echo "    sudo dnf install python3-gobject gtk4 libadwaita"
    echo
    echo "  Arch Linux:"
    echo "    sudo pacman -S python-gobject gtk4 libadwaita"
    echo
    echo "If already installed, check your Python version:"
    echo "  $PYTHON --version"
    echo "  dpkg -l | grep python3-gi"
    echo
    echo "Then run this script again."
    exit 1
fi

# Create virtual environment
VENV_DIR="${VENV_DIR:-venv}"
echo
echo "Creating virtual environment in '$VENV_DIR'..."

if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists."
    read -p "  Recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        $PYTHON -m venv "$VENV_DIR" --system-site-packages
        echo "  Created fresh virtual environment."
    fi
else
    $PYTHON -m venv "$VENV_DIR" --system-site-packages
    echo "  Created virtual environment."
fi

# Activate and install
echo
echo "Installing dependencies..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip -q

# Install requirements
pip install -r requirements.txt

echo
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "To run the application:"
echo "  python run_gtk.py           # GTK GUI"
echo "  python run_gtk.py --cpu     # CPU-only mode"
echo "  python run_voice.py         # Voice menu"
echo "  python run_coach.py         # Conversational mode"
echo

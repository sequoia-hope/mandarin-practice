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

# Check for NVIDIA GPU
echo
echo "Checking for NVIDIA GPU..."
CPU_ONLY=false

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  Found: $GPU_NAME"
    echo
    read -p "Use GPU for acceleration? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        CPU_ONLY=true
        echo "  Will install CPU-only packages (smaller download)."
    else
        echo "  Will install GPU-accelerated packages."
    fi
else
    echo "  No NVIDIA GPU detected."
    CPU_ONLY=true
    echo "  Will install CPU-only packages (smaller download)."
fi

# Check for required system packages (GTK4)
echo
echo "Checking GTK4 dependencies..."
if $PYTHON -c "import gi; gi.require_version('Gtk', '4.0')" 2>/dev/null; then
    echo "  GTK4 bindings: OK"
else
    echo "  GTK4 bindings: NOT FOUND"
    echo
    echo "The GTK4 Python bindings must be installed as system packages."
    echo
    echo "Install them with:"
    echo
    echo "  Ubuntu/Debian:"
    echo "    sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1"
    echo
    echo "  Fedora:"
    echo "    sudo dnf install python3-gobject gtk4 libadwaita"
    echo
    echo "  Arch Linux:"
    echo "    sudo pacman -S python-gobject gtk4 libadwaita"
    echo
    echo "NOTE: If you're in an existing venv, deactivate it first and run"
    echo "this script from outside any virtual environment."
    echo
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

if [ "$CPU_ONLY" = true ]; then
    echo
    echo "Installing PyTorch (CPU-only, ~200MB)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

    echo
    echo "Installing remaining dependencies..."
    pip install -r requirements.txt
else
    echo
    echo "Installing dependencies with CUDA support (~2GB)..."
    pip install -r requirements.txt
fi

echo
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "To run the application:"
if [ "$CPU_ONLY" = true ]; then
echo "  python run_gtk.py --cpu     # GTK GUI (CPU mode)"
echo "  python run_voice.py --cpu   # Voice menu"
echo "  python run_coach.py --cpu   # Conversational mode"
echo
echo "Note: Using --cpu flag since GPU packages not installed."
else
echo "  python run_gtk.py           # GTK GUI"
echo "  python run_gtk.py --cpu     # Force CPU mode"
echo "  python run_voice.py         # Voice menu"
echo "  python run_coach.py         # Conversational mode"
fi
echo

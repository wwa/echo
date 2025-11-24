#!/bin/bash/

VENV_DIR="./vEcho"

if [ "$EUID" -ne 0 ]; then
  SUDO="sudo"
else
  SUDO=""
fi

if command -v apt >/dev/null 2>&1; then
  PM="apt"
elif command -v apt-get >/dev/null 2>&1; then
  PM="apt-get"
elif command -v pacman >/dev/null 2>&1; then
  PM="pacman"
else
  echo "Unsupported system: no apt, apt-get, or pacman found."
  exit 1
fi

echo "Detected package manager: $PM"

case "$PM" in
  apt|apt-get)
    $SUDO $PM update
    $SUDO $PM install -y \
      python3-tk \
      python3-dev \
      xclip \
      wl-clipboard \
      tesseract-ocr \
      espeak-ng
    ;;

  pacman)
    $SUDO pacman -Syu --needed \
      tk \
      xclip \
      wl-clipboard \
      tesseract \
      espeak-ng
    ;;
esac

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
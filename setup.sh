#!/bin/bash/

VENV_DIR="./vEcho"

sudo apt-get install python3-tk python3-dev xclip wl-clipboard
#sudo pacman -S tk xclip wl-clipboard

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
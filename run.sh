#!/usr/bin/env bash
set -e

VENV_DIR="vEcho"
LOG_DIR="logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/echo.log"

start_tail_tab() {
  # Try gnome-terminal in a new tab
  if command -v gnome-terminal >/dev/null 2>&1; then
    (
      gnome-terminal --tab -- bash -lc "echo 'Tailing $LOG_FILE'; tail -f \"$LOG_FILE\"; exec bash"
    ) & disown || echo "⚠️ Failed to open gnome-terminal tab for tail."

  # Try konsole in a new tab
  elif command -v konsole >/dev/null 2>&1; then
    (
      konsole --new-tab -e bash -c "echo 'Tailing $LOG_FILE'; tail -f \"$LOG_FILE\"; exec bash"
    ) & disown || echo "⚠️ Failed to open konsole tab for tail."

  # Fallback: x-terminal-emulator (may open a new window, not a tab)
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    (
      x-terminal-emulator -e bash -lc "echo Tailing $LOG_FILE; tail -f \"$LOG_FILE\"; exec bash"
    ) & disown || echo "⚠️ Failed to open x-terminal-emulator for tail."

  else
    echo "⚠️ Could not auto-open a terminal tab. Run this manually in another shell:"
    echo "    tail -f $LOG_FILE"
  fi
}

# Start tail in another tab/window, but don't block this script
start_tail_tab

# Now always start the venv + app in THIS console immediately
source "$VENV_DIR/bin/activate"
python ./Echo.py
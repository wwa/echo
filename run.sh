#!/usr/bin/env bash
set -e

VENV_DIR="vEcho"
LOG_DIR="logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$LOG_DIR"

IMPORTANT_LOG="$LOG_DIR/important.log"
LLM_LOG="$LOG_DIR/llm.log"
OTHER_LOG="$LOG_DIR/other.log"

start_tail_for() {
  local title="$1"
  local file="$2"

  if command -v gnome-terminal >/dev/null 2>&1; then
    (
      gnome-terminal --tab -- bash -lc "echo '$title'; echo 'Tailing $file'; tail -f \"$file\"; exec bash"
    ) & disown || echo "⚠️ Failed to open gnome-terminal tab for $title."

  elif command -v konsole >/dev/null 2>&1; then
    (
      konsole --new-tab -e bash -c "echo '$title'; echo 'Tailing $file'; tail -f \"$file\"; exec bash"
    ) & disown || echo "⚠️ Failed to open konsole tab for $title."

  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    (
      x-terminal-emulator -e bash -lc "echo '$title'; echo 'Tailing $file'; tail -f \"$file\"; exec bash"
    ) & disown || echo "⚠️ Failed to open x-terminal-emulator for $title."

  else
    echo "⚠️ Could not auto-open a terminal tab for '$title'. Run manually:"
    echo "    tail -f $file"
  fi
}

# ---------- three extra windows/tabs ----------
start_tail_for "IMPORTANT LOG (INFO+)" "$IMPORTANT_LOG"
start_tail_for "LLM LOG (requests/responses)" "$LLM_LOG"
start_tail_for "OTHER LOG (DEBUG+ app/tool)" "$OTHER_LOG"

# ---------- main app ----------
source "$VENV_DIR/bin/activate"
python ./Echo.py

#!/usr/bin/env bash
set -e

VENV_DIR="vEcho"
LOG_DIR="logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------
# LOAD .env INTO SHELL ENVIRONMENT
# ---------------------------------------------------
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

mkdir -p "$LOG_DIR"

IMPORTANT_LOG="$LOG_DIR/important.log"
LLM_LOG="$LOG_DIR/llm.log"
OTHER_LOG="$LOG_DIR/other.log"

SHOW_EXTRA_WINDOWS="${SHOW_EXTRA_WINDOWS:-true}"

# Array to store PIDs of extra log windows
EXTRA_PIDS=()

cleanup_extra_windows() {
  echo "🧹 Closing extra log windows..."
  for pid in "${EXTRA_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

# Trigger window cleanup
trap cleanup_extra_windows EXIT INT TERM

start_tail_for() {
  local title="$1"
  local file="$2"

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -c "echo '$title'; echo 'Tailing $file'; tail -f \"$file\"; exec bash" &
    EXTRA_PIDS+=($!)

  elif command -v konsole >/dev/null 2>&1; then
    konsole -e bash -c "echo '$title'; echo 'Tailing $file'; tail -f \"$file\"; exec bash" &
    EXTRA_PIDS+=($!)

  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -c "echo '$title'; echo 'Tailing $file'; tail -f \"$file\"; exec bash" &
    EXTRA_PIDS+=($!)

  else
    echo "⚠️ Could not auto-open a terminal for '$title'. Run manually:"
    echo "    tail -f $file"
  fi
}

# ---------------------------------------------------
# SHOW EXTRA WINDOWS
# ---------------------------------------------------
if [ "$SHOW_EXTRA_WINDOWS" = "true" ]; then
  start_tail_for "IMPORTANT LOG (INFO+)" "$IMPORTANT_LOG"
  start_tail_for "LLM LOG (requests/responses)" "$LLM_LOG"
  start_tail_for "OTHER LOG (DEBUG+ app/tool)" "$OTHER_LOG"
fi

# ---------------------------------------------------
# CHECK VIRTUAL ENVIRONMENT
# ---------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "❌ Virtualenv '$VENV_DIR' not found."
  exit 1
fi

source "$VENV_DIR/bin/activate"

echo "🚀 Starting Echo..."
python ./Echo.py

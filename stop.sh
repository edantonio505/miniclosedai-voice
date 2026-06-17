#!/usr/bin/env bash
#
# stop.sh — stop the MiniClosedAI voice service started by ./start.sh -d.
#
# Looks at the pidfile first (set by start.sh -d), then falls back to
# pgrep for any running uvicorn server:app on this directory's port so a
# foreground start.sh that was Ctrl-Z'd or otherwise disowned still gets
# cleaned up.
#
set -euo pipefail
cd "$(dirname "$0")"

PIDFILE="${VOICE_PIDFILE:-/tmp/voice.pid}"
PORT="${VOICE_PORT:-8090}"

found=""
if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    found="$pid"
  fi
  rm -f "$PIDFILE"
fi
if [[ -z "$found" ]]; then
  found=$(pgrep -f "uvicorn server:app.*${PORT}" | head -1 || true)
fi

if [[ -z "$found" ]]; then
  echo "Voice service not running."
  exit 0
fi

echo "Stopping voice service (pid $found)..."
kill "$found" 2>/dev/null || true
for _ in $(seq 1 10); do
  kill -0 "$found" 2>/dev/null || { echo "Stopped."; exit 0; }
  sleep 0.3
done
echo "Did not exit in 3s — sending SIGKILL."
kill -9 "$found" 2>/dev/null || true
echo "Stopped."

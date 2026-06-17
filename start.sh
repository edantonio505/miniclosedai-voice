#!/usr/bin/env bash
#
# start.sh — boot the MiniClosedAI voice service.
#
# Usage:
#   ./start.sh              # foreground, logs to terminal
#   ./start.sh -d           # background (daemon), logs to /tmp/voice.log
#   ./start.sh --port 8090  # custom port (default 8090)
#
# Requires:
#   ./setup.sh has been run once and ./env/ exists.
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${VOICE_PORT:-8090}"
DAEMON=0
LOG="${VOICE_LOG:-/tmp/voice.log}"
PIDFILE="${VOICE_PIDFILE:-/tmp/voice.pid}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--daemon) DAEMON=1; shift ;;
    --port)      PORT="$2"; shift 2 ;;
    --log)       LOG="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

[[ ! -d env ]] && { echo "venv not found at ./env — run ./setup.sh first" >&2; exit 1; }

# shellcheck disable=SC1091
source env/bin/activate

# Match the env vars the previous Docker entrypoint used so server.py code
# paths don't change. Defaults to GPU + medium.en — override in shell to tune.
export VOICE_PORT="$PORT"
export VOICE_VOICES_DIR="${VOICE_VOICES_DIR:-./voices}"
export VOICE_ASR_MODEL="${VOICE_ASR_MODEL:-medium.en}"
export VOICE_DEVICE="${VOICE_DEVICE:-auto}"
mkdir -p "$VOICE_VOICES_DIR"

CMD=(python -m uvicorn server:app --host 0.0.0.0 --port "$PORT")

if [[ "$DAEMON" == "1" ]]; then
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  disown
  echo "Voice service started (pid $(cat "$PIDFILE")) → log: $LOG"
  echo "Stop with: ./stop.sh"
else
  exec "${CMD[@]}"
fi

#!/usr/bin/env bash
#
# dev.sh — one-command bring-up for the miniclosedai-voice service.
#
# Mirrors miniclosedai/dev.sh: same verb set (up / down / restart / status /
# logs / help). Wraps the lower-level start.sh + stop.sh you already have.
# Bare-metal venv path is the default — the proven-working one on this
# Blackwell GB10 box, where the Docker CUDA wheel JITs unreliably.
#
# Usage:
#   ./dev.sh up           # generate cert if missing, start HTTPS voice service
#   ./dev.sh down         # stop the bare-metal voice service
#   ./dev.sh restart      # down + up
#   ./dev.sh status       # health probe, pid, cert SAN list, voices on disk
#   ./dev.sh logs         # tail /tmp/voice.log
#   ./dev.sh help         # this list
#
# For API-level operations (speak / transcribe / clone / list voices / connect
# URL) drive the running service with the ./vc terminal client — see the
# "Terminal CLI (vc)" section in README.md.
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${VOICE_PORT:-8090}"
LOG="${VOICE_LOG:-/tmp/voice.log}"
PIDFILE="${VOICE_PIDFILE:-/tmp/voice.pid}"
CERT_DIR=".devcerts"
CERT="${CERT_DIR}/dev-cert.pem"

# ── pretty output ───────────────────────────────────────────────────────────
c_blue=$'\e[1;34m'; c_green=$'\e[1;32m'; c_red=$'\e[1;31m'; c_yellow=$'\e[1;33m'; c_off=$'\e[0m'
step() { printf "\n%s▶ %s%s\n" "$c_blue"   "$1" "$c_off"; }
ok()   { printf   "%s✓ %s%s\n" "$c_green"  "$1" "$c_off"; }
warn() { printf   "%s! %s%s\n" "$c_yellow" "$1" "$c_off"; }
die()  { printf   "%s✗ %s%s\n" "$c_red"    "$1" "$c_off" >&2; exit 1; }

# ── helpers ─────────────────────────────────────────────────────────────────
voice_pid() {
  # Prefer the pidfile (set by start.sh -d), fall back to a pgrep so a
  # foreground start.sh started in another terminal is also seen as "alive".
  if [[ -f "$PIDFILE" ]]; then
    local p; p="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then echo "$p"; return; fi
  fi
  pgrep -f "uvicorn server:app.*${PORT}" | head -1
}
voice_alive()  { [[ -n "$(voice_pid)" ]]; }
voice_health() {
  # Try HTTPS first (the default since today); fall back to HTTP for the
  # --http escape hatch case so `status` works either way.
  curl -fsSk --max-time 2 "https://localhost:${PORT}/health" >/dev/null 2>&1 ||
  curl -fsS  --max-time 2 "http://localhost:${PORT}/health"  >/dev/null 2>&1
}
voice_scheme() {
  curl -fsSk --max-time 2 "https://localhost:${PORT}/health" >/dev/null 2>&1 && { echo https; return; }
  curl -fsS  --max-time 2 "http://localhost:${PORT}/health"  >/dev/null 2>&1 && { echo http;  return; }
  echo unknown
}
lan_ip() { ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || echo ''; }

# ── commands ────────────────────────────────────────────────────────────────
cmd_up() {
  if voice_alive && voice_health; then
    ok "Voice service already running (pid $(voice_pid))"
    cmd_status
    return
  fi
  if voice_alive && ! voice_health; then
    warn "Stale process at pid $(voice_pid) — restarting"
    ./stop.sh >/dev/null 2>&1 || true
    sleep 1
  fi
  # start.sh picks the scheme: HTTPS locally, auto-switched to HTTP on RunPod
  # (its edge proxy speaks plain HTTP to the container). Don't hardcode a scheme.
  step "Starting voice service on port ${PORT}"
  ./start.sh -d
  # Health-wait loop (cold model load can take 15-25 s on first launch).
  for _ in $(seq 1 40); do
    if voice_health; then
      ok "Voice service ready   $(curl -fsSk https://localhost:${PORT}/health 2>/dev/null ||
                                  curl -fsS  http://localhost:${PORT}/health  2>/dev/null)"
      cmd_status
      return
    fi
    sleep 1
  done
  warn "Did not become healthy in 40 s — check 'dev.sh logs'"
}

cmd_down() {
  if ! voice_alive; then
    ok "Voice service not running"
    return
  fi
  step "Stopping voice service (pid $(voice_pid))"
  ./stop.sh
  ok "Stopped"
}

cmd_restart() { cmd_down; cmd_up; }

cmd_status() {
  step "Voice service"
  if voice_alive; then
    local scheme; scheme="$(voice_scheme)"
    if voice_health; then
      ok "pid $(voice_pid)  ${scheme}://localhost:${PORT}/  ✓"
      local ip; ip="$(lan_ip)"
      [[ -n "$ip" ]] && ok "LAN access:        ${scheme}://${ip}:${PORT}/"
      [[ -n "${RUNPOD_POD_ID:-}" ]] && ok "RunPod proxy:      https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/"
    else
      warn "pid $(voice_pid)  but /health not responding"
    fi
  else
    warn "not running (start with: $0 up)"
  fi

  step "Self-signed dev cert"
  if [[ -f "$CERT" ]]; then
    local sans
    sans="$(openssl x509 -in "$CERT" -noout -text 2>/dev/null | awk '/Subject Alternative Name/{getline; print}' | sed 's/^ *//')"
    local exp
    exp="$(openssl x509 -in "$CERT" -noout -enddate 2>/dev/null | sed 's/notAfter=//')"
    ok "$CERT  expires: $exp"
    [[ -n "$sans" ]] && ok "SAN: $sans"
  else
    warn "no cert yet — first ./dev.sh up generates one"
  fi

  step "Voices on disk"
  if [[ -d voices ]]; then
    local count
    count="$(ls voices/*.wav voices/*.WAV voices/*.flac 2>/dev/null | wc -l)"
    if [[ "$count" -gt 0 ]]; then
      ls -1 voices/*.wav voices/*.WAV voices/*.flac 2>/dev/null \
        | sed 's|voices/||; s|^|  |'
    else
      warn "voices/ is empty"
    fi
  else
    warn "voices/ does not exist (will be created on first save)"
  fi
}

cmd_logs() {
  if [[ ! -f "$LOG" ]]; then
    die "Log file $LOG does not exist. Start the service with: $0 up"
  fi
  tail -f "$LOG"
}

case "${1:-up}" in
  up)             cmd_up ;;
  down)           cmd_down ;;
  restart)        cmd_restart ;;
  status)         cmd_status ;;
  logs)           cmd_logs ;;
  -h|--help|help) sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//; $d' ;;
  *)              die "Unknown command: $1 (try up | down | restart | status | logs | help)" ;;
esac

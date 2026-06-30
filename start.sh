#!/usr/bin/env bash
#
# start.sh — boot the MiniClosedAI voice service (HTTPS by default).
#
# Why HTTPS by default: the Voice Studio GUI uses navigator.getUserMedia for
# in-browser recording, and browsers ONLY expose that API in a secure context
# (https:// or http://localhost). To use the recorder from another machine on
# your LAN (e.g. https://192.168.0.110:8090/), the service has to speak TLS.
# A self-signed dev cert is good enough — the browser shows a one-time warning
# you click through.
#
# Usage:
#   ./start.sh              # foreground HTTPS, logs to terminal
#   ./start.sh -d           # background (daemon), logs to /tmp/voice.log
#   ./start.sh --port 8090  # custom port (default 8090)
#   ./start.sh --http       # plain HTTP (no TLS — recorder won't work from LAN)
#
# Requires:
#   ./setup.sh has been run once and ./env/ exists.
#   openssl (for cert generation on first run).
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${VOICE_PORT:-8090}"
DAEMON=0
HTTPS=1
LOG="${VOICE_LOG:-/tmp/voice.log}"
PIDFILE="${VOICE_PIDFILE:-/tmp/voice.pid}"
CERT_DIR=".devcerts"
CERT="${CERT_DIR}/dev-cert.pem"
KEY="${CERT_DIR}/dev-key.pem"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--daemon) DAEMON=1; shift ;;
    --port)      PORT="$2"; shift 2 ;;
    --log)       LOG="$2"; shift 2 ;;
    --http)      HTTPS=0; shift ;;
    -h|--help)
      sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

[[ ! -d env ]] && { echo "venv not found at ./env — run ./setup.sh first" >&2; exit 1; }

# Generate a self-signed dev cert if missing. Covers localhost + the host's
# primary LAN IP so the Voice Studio GUI works over both http://localhost:8090/
# AND https://<lan-ip>:8090/ without the browser blocking getUserMedia.
generate_cert() {
  command -v openssl >/dev/null || { echo "openssl not installed — install it first" >&2; exit 1; }
  mkdir -p "$CERT_DIR"
  local lan_ip
  lan_ip="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || echo '')"
  local san="DNS:localhost,DNS:voice.miniclosedai.dev,IP:127.0.0.1"
  [[ -n "$lan_ip" ]] && san="${san},IP:${lan_ip}"
  local cfg
  cfg=$(mktemp)
  cat > "$cfg" <<EOF
[req]
distinguished_name = req_dn
x509_extensions    = v3_req
prompt             = no
[req_dn]
CN = voice.miniclosedai.dev
O  = MiniClosedAI Voice (Development)
[v3_req]
subjectAltName = ${san}
basicConstraints = CA:FALSE
keyUsage         = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
EOF
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$KEY" -out "$CERT" \
    -days 825 -config "$cfg" >/dev/null 2>&1
  chmod 600 "$KEY"
  rm -f "$cfg"
  echo "Generated self-signed dev cert at ${CERT} (valid 825 days, SAN: ${san})"
}

if [[ "$HTTPS" == "1" ]]; then
  [[ ! -f "$CERT" || ! -f "$KEY" ]] && generate_cert
fi

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
if [[ "$HTTPS" == "1" ]]; then
  CMD+=(--ssl-certfile "$CERT" --ssl-keyfile "$KEY")
  SCHEME="https"
else
  SCHEME="http"
fi

if [[ "$DAEMON" == "1" ]]; then
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  disown
  echo "Voice service started (pid $(cat "$PIDFILE")) → log: $LOG"
  echo "  Local:    ${SCHEME}://localhost:${PORT}/"
  lan_ip="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || echo '')"
  [[ -n "$lan_ip" ]] && echo "  LAN:      ${SCHEME}://${lan_ip}:${PORT}/"
  echo "Stop with: ./stop.sh"
else
  exec "${CMD[@]}"
fi

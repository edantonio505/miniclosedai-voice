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
#   ./start.sh --http       # force plain HTTP (no TLS)
#   ./start.sh --https      # force HTTPS even on RunPod (rarely wanted)
#
# RunPod (and similar edge-proxy hosts) terminate TLS at their own proxy and
# forward to your container port over PLAIN HTTP. A service speaking HTTPS makes
# the proxy's HTTP request hit a TLS listener, so the public *.proxy.runpod.net
# URL fails to load. This script auto-detects RunPod (via $RUNPOD_POD_ID) and
# switches to HTTP so the clone-and-run path just works — no flag needed. The
# public URL is still https:// (the proxy adds TLS), so the browser mic recorder
# keeps working. Override the auto-detect with --http / --https.
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
HTTPS_SET=0          # did the user explicitly pass --http/--https?
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
    --http)      HTTPS=0; HTTPS_SET=1; shift ;;
    --https)     HTTPS=1; HTTPS_SET=1; shift ;;
    -h|--help)
      sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

[[ ! -d env ]] && { echo "venv not found at ./env — run ./setup.sh first" >&2; exit 1; }

# Auto-detect an edge-proxy host (RunPod) and default to HTTP. Their proxy
# terminates TLS and speaks plain HTTP to the container port, so an HTTPS
# service would make the public *.proxy.runpod.net URL unreachable. Skip this
# if the user explicitly chose a scheme with --http/--https.
RUNPOD="${RUNPOD_POD_ID:-}"
if [[ "$HTTPS_SET" == "0" && -n "$RUNPOD" ]]; then
  HTTPS=0
  echo "Detected RunPod (RUNPOD_POD_ID=$RUNPOD) — serving plain HTTP so the RunPod"
  echo "proxy can reach the service. Public URL stays HTTPS. Override with --https."
fi

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
  [[ -n "$RUNPOD" ]] && echo "  RunPod:   https://${RUNPOD}-${PORT}.proxy.runpod.net/   (expose port ${PORT} in the pod)"
  echo "Stop with: ./stop.sh"
else
  echo "Serving ${SCHEME}://0.0.0.0:${PORT}/"
  [[ -n "$RUNPOD" ]] && echo "  RunPod:   https://${RUNPOD}-${PORT}.proxy.runpod.net/   (expose port ${PORT} in the pod)"
  exec "${CMD[@]}"
fi

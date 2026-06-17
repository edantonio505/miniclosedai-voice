#!/usr/bin/env bash
#
# test.sh — wrapper that runs e2e_test.py against the local voice service.
#
# Use this before committing any change to miniclosedai-voice. It checks:
#   1. Critical imports + GPU access
#   2. /health, /voices, /speak, /speak/stream
#   3. ASR round-trip (TTS → /transcribe → fuzzy match)
#   4. Call mode WebRTC handshake (/call/configure + /webrtc/offer)
#
# Requires the server to already be running:
#   ./start.sh -d        (then ./test.sh)
#
# Pass-through flags go straight to e2e_test.py (e.g. --skip-call, --strict).
#
set -euo pipefail
cd "$(dirname "$0")"

[[ ! -d env ]] && { echo "venv not found — run ./setup.sh first" >&2; exit 1; }

# shellcheck disable=SC1091
source env/bin/activate
exec python e2e_test.py "$@"

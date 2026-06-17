#!/usr/bin/env bash
#
# setup.sh — one-time install for the MiniClosedAI voice service.
#
# Creates a Python venv at ./env/, detects the host's CUDA version, installs
# the right torch+torchaudio wheels from pytorch.org, then installs the rest
# of the deps (Chatterbox TTS, HF transformers, FastRTC, DeepFilterNet) at
# versions known to work together on NVIDIA hardware ranging from RTX 30/40/50
# desktops to GB10 / Jetson Orin.
#
# Re-running is safe: missing pieces get added, present ones are left alone.
#
# Usage:
#   ./setup.sh                  # auto-detect + install
#   ./setup.sh --cuda 12.8      # force a specific CUDA wheel channel
#   ./setup.sh --cpu            # CPU-only install (no torch GPU build)
#   ./setup.sh --python 3.12    # use a specific python interpreter
#
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR="env"
PYTHON_BIN=""
CUDA_OVERRIDE=""
CPU_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)   CUDA_OVERRIDE="$2"; shift 2 ;;
    --cpu)    CPU_ONLY=1; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

c_blue=$'\e[1;34m'; c_green=$'\e[1;32m'; c_red=$'\e[1;31m'; c_yellow=$'\e[1;33m'; c_dim=$'\e[2m'; c_off=$'\e[0m'
step() { printf "\n%s▶ %s%s\n" "$c_blue"   "$1" "$c_off"; }
ok()   { printf   "%s✓ %s%s\n" "$c_green"  "$1" "$c_off"; }
warn() { printf   "%s! %s%s\n" "$c_yellow" "$1" "$c_off"; }
die()  { printf   "%s✗ %s%s\n" "$c_red"    "$1" "$c_off" >&2; exit 1; }

# ─── 1. Pick a Python interpreter ────────────────────────────────────────
if [[ -z "$PYTHON_BIN" ]]; then
  # Prefer 3.12 (matches BCP_stuff/env which is the proven-working baseline),
  # fall back to 3.11. Anything older lacks the typing features chatterbox uses.
  for v in 3.12 3.11; do
    if command -v "python$v" >/dev/null 2>&1; then
      PYTHON_BIN="python$v"; break
    fi
  done
fi
[[ -z "$PYTHON_BIN" ]] && die "No python3.11+ found. Install python3.12 or pass --python /path/to/python."
ok "Using $(${PYTHON_BIN} --version) at $(command -v ${PYTHON_BIN})"

# ─── 2. Detect CUDA ──────────────────────────────────────────────────────
detect_cuda() {
  if [[ "$CPU_ONLY" == "1" ]]; then echo "cpu"; return; fi
  if [[ -n "$CUDA_OVERRIDE" ]]; then echo "cu${CUDA_OVERRIDE//./}"; return; fi
  command -v nvidia-smi >/dev/null 2>&1 || { echo "cpu"; return; }
  local v
  v=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1 | awk '{print $3}')
  if [[ -z "$v" ]]; then echo "cpu"; return; fi
  # Match against pytorch.org's actual wheel channels: cu124, cu126, cu128, cu130.
  # Pick the highest channel ≤ the host driver's CUDA version (forward-compatible).
  case "$v" in
    13.*|12.8*|12.9*) echo "cu130" ;;     # cu130 wheels work on driver≥CUDA 12.8
    12.4|12.5|12.6|12.7) echo "cu124" ;;
    11.*) echo "cu118" ;;
    *) echo "cu130" ;;
  esac
}
CUDA_CHANNEL=$(detect_cuda)
if [[ "$CUDA_CHANNEL" == "cpu" ]]; then
  warn "No CUDA detected — installing CPU-only PyTorch (voice will work, ASR/TTS will be slow)"
else
  ok "CUDA detected → installing torch wheels from pytorch.org/whl/${CUDA_CHANNEL}"
fi

# ─── 3. Create / refresh venv ────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  step "Creating venv at ./${VENV_DIR}/"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  ok "Venv created"
else
  ok "Venv already exists at ./${VENV_DIR}/"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --quiet --upgrade pip wheel setuptools
ok "pip / wheel / setuptools up to date"

# ─── 4. Install torch from the right channel ─────────────────────────────
step "Installing torch + torchaudio (${CUDA_CHANNEL})"
TORCH_VER=""
TORCHAUDIO_VER=""
case "$CUDA_CHANNEL" in
  cu130) TORCH_VER="torch==2.10.0";       TORCHAUDIO_VER="torchaudio==2.10.0" ;;
  cu128) TORCH_VER="torch>=2.7,<2.10";    TORCHAUDIO_VER="torchaudio>=2.7,<2.10" ;;
  cu124) TORCH_VER="torch>=2.4,<2.6";     TORCHAUDIO_VER="torchaudio>=2.4,<2.6" ;;
  cu118) TORCH_VER="torch>=2.1,<2.4";     TORCHAUDIO_VER="torchaudio>=2.1,<2.4" ;;
  cpu)   TORCH_VER="torch";               TORCHAUDIO_VER="torchaudio" ;;
esac
if [[ "$CUDA_CHANNEL" == "cpu" ]]; then
  pip install --quiet "$TORCH_VER" "$TORCHAUDIO_VER"
else
  pip install --quiet "$TORCH_VER" "$TORCHAUDIO_VER" \
    --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"
fi
ok "torch installed: $(python -c 'import torch; print(torch.__version__)')"

# ─── 5. Chatterbox with --no-deps (its torch==2.6 pin would downgrade us) ─
step "Installing chatterbox-tts==0.1.6 (--no-deps; sub-deps come from requirements.txt)"
pip install --quiet --no-deps chatterbox-tts==0.1.6
ok "chatterbox-tts installed"

# ─── 6. Remaining deps ───────────────────────────────────────────────────
step "Installing requirements.txt"
pip install --quiet -r requirements.txt
ok "requirements installed"

# ─── 7. Patch DeepFilterNet (if present) ─────────────────────────────────
# DeepFilterNet 0.5.6 imports torchaudio.backend.common.AudioMetaData, a path
# removed in torchaudio ≥ 2.10. We replace it with a stub since the symbol is
# only used in type annotations on code paths we don't exercise.
# Glob the venv directly — we can't `python -c "import df.io"` because that
# import is exactly what's broken.
DF_IO=$(find "$VENV_DIR" -path "*/df/io.py" -not -path "*/__pycache__/*" 2>/dev/null | head -1)
if [[ -n "$DF_IO" && -f "$DF_IO" ]]; then
  if grep -q "from torchaudio.backend.common import AudioMetaData" "$DF_IO" 2>/dev/null; then
    step "Patching DeepFilterNet's torchaudio.backend import"
    sed -i 's|from torchaudio.backend.common import AudioMetaData|class AudioMetaData: pass  # stub for torchaudio>=2.10|' "$DF_IO"
    ok "DeepFilterNet patched at $DF_IO"
  fi
fi

# ─── 8. Sanity check ─────────────────────────────────────────────────────
step "Verifying imports + GPU access"
python - <<'PY'
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
import torch
gpu = torch.cuda.is_available()
print(f"  torch:        {torch.__version__}")
print(f"  CUDA visible: {gpu}")
if gpu:
    print(f"  device:       {torch.cuda.get_device_name(0)}")
    try:
        (torch.randn(8, 8, device='cuda') @ torch.randn(8, 8, device='cuda')).sum().item()
        print("  GPU tensor op: OK")
    except Exception as e:
        print(f"  GPU tensor op: FAILED — {type(e).__name__}: {e}")
for name, mod in [
    ("transformers",  "transformers"),
    ("chatterbox",    "chatterbox.tts"),
    ("deepfilternet", "df.enhance"),
    ("fastrtc",       "fastrtc"),
]:
    try:
        __import__(mod)
        print(f"  {name:14s} import OK")
    except Exception as e:
        print(f"  {name:14s} FAIL — {type(e).__name__}: {str(e)[:200]}")
PY

cat <<EOF

${c_green}Setup complete.${c_off}  Next:
  ${c_dim}./start.sh${c_off}     start the voice server (port 8090)
  ${c_dim}./stop.sh${c_off}      stop the voice server

Re-run ${c_dim}./setup.sh${c_off} any time deps change.
EOF

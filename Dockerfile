# syntax=docker/dockerfile:1
#
# MiniClosedAI voice service — single-stage, multi-arch CUDA build.
#
# This Dockerfile produces the same image on amd64 and arm64:
#
#   * ASR: HuggingFace Whisper via transformers + PyTorch with CUDA.
#     Both architectures install the GPU torch wheel straight from
#     pytorch.org's cu124 index (no source compile, ~3 min builds).
#   * TTS: Piper. PyPI's `onnxruntime-gpu` wheel is available on amd64 only,
#     so x86_64 deploys get Piper on GPU; aarch64 falls back to CPU ONNX
#     gracefully (TTS first-chunk on CPU Piper is still ~100ms).
#
# Build:
#   docker build -t miniclosedai-voice:latest .
#
# Multi-arch push:
#   docker buildx build --platform=linux/amd64,linux/arm64 \
#       -t <registry>/miniclosedai-voice:latest --push .
#
# Run (host with NVIDIA driver + nvidia-container-toolkit):
#   docker run --rm --gpus all -p 8090:8090 \
#     -v voice_models:/root/.cache/huggingface \
#     -v voice_pipers:/voices \
#     miniclosedai-voice:latest

# CUDA 12.8 base — matches PyTorch's cu128 wheels and supports Blackwell sm_120/sm_121
# (NVIDIA GB10 etc.). cu124 + torch 2.5 errored with "no kernel image available".
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# PyTorch with CUDA 12.8 support — covers all current NVIDIA architectures
# including Blackwell consumer (sm_120) and GB10 (sm_121). Pulled from
# pytorch.org's cu128 index (has both linux/amd64 and linux/arm64 wheels
# since 2.7+). Pinning to >=2.7 to ensure Blackwell is supported.
RUN pip install --no-cache-dir 'torch>=2.7' 'torchaudio>=2.7' \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# onnxruntime-gpu has aarch64+CUDA wheels only on NVIDIA's Jetson index; the
# PyPI one is amd64-only. Try it; if the install fails, Piper falls back to
# its CPU ONNX provider (still ~100ms first chunk, fine for sentence streaming).
RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        pip install --no-cache-dir onnxruntime-gpu==1.20.1; \
    else \
        echo "[info] arm64 build — Piper TTS will use CPU ONNX provider"; \
    fi

COPY asr.py tts.py call.py server.py test_client.py ./

RUN mkdir -p /voices /root/.cache/huggingface

# Default to medium.en — the same model chatterbox uses and the sweet spot
# on a consumer GPU. Set VOICE_ASR_MODEL=tiny.en / small.en / large-v3 to
# trade off speed vs accuracy; VOICE_DEVICE=cpu to disable CUDA.
ENV VOICE_PORT=8090 \
    VOICE_VOICES_DIR=/voices \
    VOICE_ASR_MODEL=medium.en \
    VOICE_DEVICE=cuda

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8090/health || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]

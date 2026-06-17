# syntax=docker/dockerfile:1
#
# MiniClosedAI voice service — single-stage, multi-arch CUDA build.
#
# Mirrors the proven-working setup in BCP_stuff/env on this exact GB10:
#   * Base:    nvidia/cuda:13.3.0-cudnn-runtime (matches torch's cu130 wheels)
#   * Torch:   2.10.0+cu130 from pytorch.org's cu130 index (aarch64+amd64)
#   * ASR:     HuggingFace Whisper-medium.en (transformers pipeline on GPU)
#   * TTS:     Chatterbox 0.1.6 — installed with --no-deps to bypass its
#              torch==2.6.0 hard pin (the working install uses torch 2.10+).
#   * Denoise: DeepFilterNet (df.enhance) via ONNX runtime.
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

FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu22.04
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

# PyTorch 2.10.0+cu130 (matches BCP_stuff/env). cu130 is the only PyTorch
# build channel with native Blackwell sm_120/sm_121 support; older channels
# (cu124, cu128) either error or fall back to PTX JIT.
RUN pip install --no-cache-dir \
        torch==2.10.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu130

# Chatterbox-tts 0.1.6 hard-pins torch==2.6.0 and transformers==4.46.3 in its
# metadata, but the proven-working install at BCP_stuff/env runs it against
# torch 2.10.0+cu130 + transformers 4.49.0 with no runtime issues. Install
# chatterbox with --no-deps so pip doesn't downgrade torch back to CPU 2.6.0;
# the explicit deps in requirements.txt below satisfy what chatterbox needs.
RUN pip install --no-cache-dir --no-deps chatterbox-tts==0.1.6

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch deepfilternet 0.5.6 to work with torchaudio >= 2.10. Its io.py imports
# `torchaudio.backend.common.AudioMetaData`, a path that was removed in 2.10
# (the symbol moved to torchaudio.AudioMetaData). Upstream hasn't released a
# fix; this in-place edit keeps the install otherwise unchanged.
RUN sed -i 's|from torchaudio.backend.common import AudioMetaData|from torchaudio import AudioMetaData|' \
        /usr/local/lib/python3.11/dist-packages/df/io.py

# onnxruntime-gpu only ships amd64+CUDA wheels on PyPI. arm64 falls back to
# the CPU ONNX provider, which is fine for Piper (legacy) and DeepFilterNet's
# small enhancement model — both run in single-digit ms either way.
RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        pip install --no-cache-dir onnxruntime-gpu==1.20.1; \
    else \
        echo "[info] arm64 build — DeepFilterNet uses CPU ONNX provider"; \
    fi

COPY asr.py tts.py call.py server.py test_client.py ./

RUN mkdir -p /voices /root/.cache/huggingface

# Default to medium.en — same model chatterbox_fastrtc.py uses, sweet spot on
# a consumer NVIDIA. Set VOICE_ASR_MODEL=tiny.en / small.en / large-v3 to
# trade off speed vs accuracy; VOICE_DEVICE=cpu to disable CUDA.
ENV VOICE_PORT=8090 \
    VOICE_VOICES_DIR=/voices \
    VOICE_ASR_MODEL=medium.en \
    VOICE_DEVICE=cuda

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8090/health || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]

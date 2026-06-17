# MiniClosedAI Voice Service

Open-source voice microservice for MiniClosedAI. Browser ↔ FastRTC ↔ HuggingFace
Whisper (ASR) ↔ Chatterbox TTS (high-quality voice synthesis) ↔ DeepFilterNet
(noise suppression). Talks to MiniClosedAI as a `kind='voice'` backend; the URL
is registered in MiniClosedAI's Settings → Add endpoint.

## Quick start (any machine with NVIDIA GPU + CUDA driver)

```bash
git clone <this-repo>.git miniclosedai-voice
cd miniclosedai-voice
./setup.sh                  # detects CUDA, creates env/, installs everything
./start.sh                  # foreground server on :8090
# (or)
./start.sh -d               # daemonize, log to /tmp/voice.log
./stop.sh                   # stop the daemon
```

That's it. **No Docker, no nvidia-container-toolkit, no images to push.**
The setup script auto-detects your CUDA version and pulls matching torch
wheels from `pytorch.org/whl/cu{118,124,128,130}` so the same repo works on:

* GB10 / Grace Blackwell / DGX Spark (CUDA 13)
* RTX 50 / Hopper desktops (CUDA 12.8)
* RTX 30 / 40 / A100 / L4 (CUDA 12.4)
* Jetson Orin (CUDA 12.x)
* CPU-only fallback (`./setup.sh --cpu`)

## RunPod / cloud-GPU template

1. Pick any PyTorch / Ubuntu template with NVIDIA driver installed (most
   RunPod templates qualify).
2. Open a web terminal on the pod.
3. `git clone <this-repo>.git && cd miniclosedai-voice && ./setup.sh && ./start.sh -d`
4. Expose port 8090. Paste the public URL into MiniClosedAI's Settings.

First run downloads ~3 GB of torch + CUDA libs + the Whisper-medium.en model
and ~3 GB of Chatterbox weights. After that, restarts are instant.

## API

The service speaks the same endpoints MiniClosedAI's voice client expects:

```
GET  /health
GET  /voices
POST /transcribe       multipart audio → {text, language, segments}
POST /speak            JSON → audio/wav (one-shot)
POST /speak/stream     JSON → SSE chunked PCM
POST /call/configure   set per-call config (conv_id, miniclosedai_url, voice, lang)
POST /webrtc/offer     SDP offer → answer (mounted by FastRTC)
GET  /call/events/{id} SSE: transcript / chunk / end / error events
```

## Configuration

Environment variables (all optional):

| Var | Default | Description |
|---|---|---|
| `VOICE_PORT` | `8090` | TCP port |
| `VOICE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VOICE_ASR_MODEL` | `medium.en` | `tiny.en` / `small.en` / `medium.en` / `large-v3` |
| `VOICE_VOICES_DIR` | `./voices` | Where reference voice WAVs live |
| `VOICE_API_KEY` | _(unset)_ | Optional Bearer token for inbound auth |
| `VOICE_LOG` | `/tmp/voice.log` | Daemon log file when using `start.sh -d` |
| `VOICE_PIDFILE` | `/tmp/voice.pid` | Daemon pidfile |

## Testing

`test_client.py` ships in the repo — an aiortc-driven smoke test that
synthesizes a test phrase, drives the full call pipeline, and reports a
per-stage timing breakdown (audio→transcript / transcript→first LLM token /
first token→first audio).

```bash
source env/bin/activate
python test_client.py --url https://<miniclosedai-host>:8095 \
                      --conv-id 94 --phrase "Hello, can you hear me?"
```

## Requirements

* Python 3.11+ (3.12 preferred — matches the proven-working baseline)
* NVIDIA driver with CUDA 11.8 / 12.4 / 12.8 / 13.x (or CPU)
* ~6 GB free disk for the venv + models

No Docker, no root, no system packages required.

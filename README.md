# MiniClosedAI Voice Service

Self-hosted, GPU-accelerated voice microservice for [MiniClosedAI](https://github.com/edantonio505/miniclosedai).
Streams duplex audio between the browser and the bot via WebRTC.

```
Browser  ──WebRTC──►  FastRTC  ──►  Silero VAD  ──►  Whisper (ASR)  ──►  HTTP POST to MiniClosedAI's /chat/stream
                                                                              │
Browser  ◄──WebRTC──  Chatterbox Turbo TTS  ◄──  sentence-stream cleaner  ◄───┘
                       (per sentence, 4 CFM diffusion steps, fp16)
```

Talks to MiniClosedAI as a `kind='voice'` backend — register the URL in **Settings → + Add endpoint**.

---

## Quick start

```bash
git clone <this-repo>.git miniclosedai-voice
cd miniclosedai-voice
./setup.sh                  # detect CUDA, create env/, install everything
./start.sh -d               # daemon mode, log → /tmp/voice.log
# … point MiniClosedAI Settings at http://<host>:8090 …
./stop.sh                   # when you're done
```

That's it. **No Docker, no nvidia-container-toolkit, no images to push.**

The setup script reads `nvidia-smi`, picks the matching torch wheel channel
(`cu118` / `cu124` / `cu128` / `cu130` / `cpu`), and pip-installs everything.
The same repo runs on:

* GB10 / Grace Blackwell / DGX Spark (CUDA 13)
* RTX 50 / Hopper desktops (CUDA 12.8)
* RTX 30 / 40 / A100 / L4 / Jetson Orin (CUDA 12.4)
* CPU-only fallback (`./setup.sh --cpu`)

First run downloads ~3 GB of torch + CUDA libs, ~500 MB Whisper, ~3 GB
Chatterbox weights, ~50 MB DeepFilterNet — cached afterward so restarts are
fast (~10 s to `/health` ready, instant on subsequent boots).

---

## RunPod / cloud GPU template

1. Pick any RunPod (or other cloud) template with NVIDIA driver pre-installed.
2. Open a web terminal on the pod.
3. ```bash
   git clone <this-repo>.git miniclosedai-voice && cd miniclosedai-voice
   ./setup.sh && ./start.sh -d
   ```
4. Expose port 8090 in the pod's settings.
5. Paste the public URL into MiniClosedAI's **Settings → + Add endpoint** (kind=Voice).

---

## The three scripts

### `setup.sh` — one-time install

```
./setup.sh                  # auto-detect
./setup.sh --cuda 12.8      # force a CUDA wheel channel
./setup.sh --cpu            # CPU-only
./setup.sh --python 3.12    # specific interpreter
```

Re-running is safe. The script:

1. Picks Python 3.12 (preferred) or 3.11, creates `env/`.
2. Reads `nvidia-smi` to choose the right torch wheel index.
3. Installs torch + torchaudio first, then chatterbox-tts `--no-deps` (its
   `torch==2.6` hard pin is bypassed; the working set in `requirements.txt`
   substitutes).
4. Installs the rest of `requirements.txt`.
5. Patches `df/io.py` to work with torchaudio 2.10+ (one-line stub for
   `AudioMetaData`, which was removed upstream).
6. Verifies imports + GPU access.

### `start.sh` — boot the service

```
./start.sh                  # foreground
./start.sh -d               # background, log → /tmp/voice.log
./start.sh --port 9090      # custom port
```

Environment variables it reads (all optional):

| Var | Default | Description |
|---|---|---|
| `VOICE_PORT` | `8090` | TCP port |
| `VOICE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VOICE_ASR_MODEL` | `medium.en` | `tiny.en` / `small.en` / `medium.en` / `large-v3` |
| `VOICE_VOICES_DIR` | `./voices` | Where reference voice WAVs live |
| `VOICE_API_KEY` | _(unset)_ | Optional Bearer token for inbound auth |
| `VOICE_LOG` | `/tmp/voice.log` | Daemon log file |
| `VOICE_PIDFILE` | `/tmp/voice.pid` | Daemon pidfile |

### `stop.sh` — stop the service

```
./stop.sh
```

Looks at the pidfile first, falls back to `pgrep` for any orphaned uvicorn.

---

## HTTP API

The contract MiniClosedAI's `voice.py` client expects:

```
GET  /health                       cheap reachability probe
GET  /voices                       static voice catalog
POST /transcribe                   multipart audio → {text, language, segments}
POST /speak                        JSON → audio/wav (one-shot)
POST /speak/stream                 JSON → SSE chunked int16 PCM frames
POST /call/configure               set per-call conv_id + miniclosedai_url + voice
POST /webrtc/offer                 SDP offer → answer (mounted by FastRTC)
GET  /call/events/{webrtc_id}      SSE: {transcript|chunk|status|end|error} events
```

Call mode is the long-running WebRTC path; push-to-talk uses
`/transcribe` + `/speak/stream` separately.

---

## What's inside

| Layer | Library | Model | Notes |
|---|---|---|---|
| ASR | `transformers` + `torch` | `openai/whisper-medium.en` (default) | swap via `VOICE_ASR_MODEL`; English-only `.en` variants are ~3× faster than multilingual for the same size |
| TTS | `chatterbox-tts==0.1.6` (`tts_turbo` variant, `--no-deps`) | `ChatterboxTurboTTS.from_pretrained()` | token-streaming, fp16 transformer, **4** CFM diffusion steps (250× fewer than the default 1000), pattern lifted from `BCP_stuff/tts_server.py` |
| VAD + turn-taking | `fastrtc[vad]` | Silero VAD | `min_silence_duration_ms=300` (was 2000 default), `can_interrupt=False` to prevent speaker→mic echo from cancelling the bot mid-reply |
| Denoise | `deepfilternet==0.5.6` (sed-patched for torchaudio≥2.10) | DeepFilterNet ONNX | runs in single-digit ms on GPU |
| WebRTC | `aiortc` (via `fastrtc`) | – | host-network mode is fine; the proxy in MiniClosedAI keeps the browser on same-origin HTTPS |
| Web | `fastapi` + `uvicorn` | – | – |

---

## Performance

Measured on a single GB10 (Grace Blackwell, sm_121, PyTorch 2.10.0+cu130 PTX-JIT) with the included models warm:

### ASR (Whisper-small.en on GPU)

p50 **347 ms**, p95 **621 ms** across 8 varied test phrases. Accuracy 94 % avg
(0/8 below 50 %). The two lower scores were perfect transcriptions where
Whisper used numerals ("3.30 pm") instead of the spelled-out words ("three
thirty PM") in the prompt — semantically correct.

### TTS (Chatterbox Turbo on GPU)

First-audible-chunk timing via `/speak/stream`:

| Phrase | First chunk |
|---|---|
| "Hello." | 389 ms |
| "Hello, can you hear me?" | 587 ms |
| "Yes I can hear you clearly. How can I help you today?" | 892 ms |
| "Three plus four equals seven." | 736 ms |
| "The meeting is scheduled for next Tuesday at ten in the morning." | 914 ms |

Median first-chunk: **~700 ms**. Streams continuously after that — the user
hears continuous audio while the rest of the sentence is being generated.

### End-to-end call mode (per turn, warm)

| Stage | Time |
|---|---|
| audio → transcript (Whisper-small.en) | 100–400 ms |
| transcript → first LLM token | depends on bot (local 3B: ~500 ms; cloud / 20B+: 1–3 s) |
| first LLM token → first audio out | ~700 ms (first sentence completes + TTS first chunk) |

Total typical call-mode roundtrip: **2–4 s** with a small local LLM, **4–8 s**
with a 20B cloud model.

---

## Structured timing logs

Every call turn emits a single-line summary to `/tmp/voice.log` (the
`voice.call` logger):

```
INFO:voice.call:conv=94 [turn-start]      sr=48000 samples=144000 duration=3.00s peak=18130 rms=1943
INFO:voice.call:conv=94 [asr]             349 ms  → 'Can you hear me clearly?'
INFO:voice.call:conv=94 [llm]             first token in 2817 ms
INFO:voice.call:conv=94 [sentence #1]     2895 ms after LLM POST → 'Yes, I can understand you clearly.'
INFO:voice.call:conv=94 [tts #1]          first chunk in 3802 ms
INFO:voice.call:conv=94 [turn-done]       1 sentences | asr=349ms llm_ttft=2817ms first_sent=2895ms tts_first=3802ms total=10056ms | reply='Yes, I can understand you clearly. How can I help you today?'
```

Tail the timing logs during a live call:

```
tail -f /tmp/voice.log | grep voice.call
```

`peak` and `rms` on the `[turn-start]` line are useful for diagnosing audio
problems — if `peak=32768` every turn, the mic input is clipping and silero
will struggle to detect end-of-speech. If `peak < 2000`, the mic is too
quiet and silero may never fire at all.

---

## Voice cloning

Drop a 5–10 s clean speech WAV into `voices/` named `<id>.wav`. It's then
available at synthesis time by passing `voice=<id>` (the call backend reads
the chosen voice from `conversations.voice_settings.voice_id`).

The shipped `voices/default.wav` is the built-in fallback. To replace it,
overwrite that file; restart the service so Chatterbox re-loads its
conditioning.

---

## End-to-end test

### `./test.sh` — full smoke suite (run before every commit)

```bash
./start.sh -d                # service must be running
./test.sh                    # ~10 s, 15 assertions, color-coded pass/fail
```

Covers, in order:

1. **Imports + GPU** — torch CUDA, transformers, chatterbox, df, fastrtc, aiortc
2. **`/health`** — service reachable, model labels correct
3. **`/voices`** — catalog non-empty + well-formed
4. **`/speak`** — returns a valid WAV with non-trivial audio
5. **`/speak/stream`** — first chunk latency, chunks arrive over SSE
6. **ASR round-trip** — TTS output sent back through `/transcribe`,
   fuzzy-matched against the input phrase
7. **Call mode handshake** — `/call/configure` + `/webrtc/offer` accepts an
   aiortc-generated SDP and returns a valid answer

Flags:

```
./test.sh --skip-call       # skip the WebRTC stage (~3 s faster)
./test.sh --skip-roundtrip  # skip ASR round-trip
./test.sh --strict          # non-zero exit on warnings too (CI mode)
./test.sh --verbose         # full tracebacks on failure
```

Exit code is `0` on all-pass, `1` on any failure, `2` on warnings under
`--strict`. **Run this every time you add a feature** so a regression in
imports, audio I/O, or the call handshake catches you immediately.

### `test_client.py` — drives a real bot through call mode

A heavier test that goes all the way through the live LLM and back:

```bash
source env/bin/activate
python test_client.py \
    --url https://<miniclosedai-host>:8095 \
    --conv-id 94 \
    --phrase "Hello, can you hear me?" \
    --timeout 60
```

It synthesizes the phrase via the local `/speak`, pushes it through the
WebRTC call mode, collects the SSE events, and prints a per-stage timing
breakdown + pass/fail summary.

---

## Tuning knobs

### Fast / accurate / fastest

```bash
# Fastest English (lower accuracy on tricky phrases)
VOICE_ASR_MODEL=tiny.en ./start.sh -d

# Balanced (current default in start.sh examples)
VOICE_ASR_MODEL=small.en ./start.sh -d

# Best accuracy (slower)
VOICE_ASR_MODEL=medium.en ./start.sh -d

# Multilingual (Spanish, French, etc.)
VOICE_ASR_MODEL=large-v3 ./start.sh -d
```

### Echo handling

`can_interrupt=False` is hardcoded in `server.py` because the bot's TTS
audio bleeds back into the mic on laptops without AEC, causing silero to
fake a barge-in and cancel the reply. With this off you can't interrupt
the bot mid-reply. To re-enable barge-in (e.g., on a headset / mobile with
hardware AEC), edit `_ensure_stream()` in `server.py`.

### Mic level

The audio preprocessing in `server.py` applies a fixed gain (`_GAIN = 1`
by default — passthrough). If your mic chain delivers low signal
(`peak < 2000` in the `[turn-start]` logs), bump `_GAIN` to 2 or 4 to lift
the speech into silero's detection floor. If `peak = 32768` everywhere,
your mic is already loud — leave gain at 1.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `chatterbox.tts ImportError: PerthImplicitWatermarker is None` | `pkg_resources` missing (setuptools 81+) | `pip install 'setuptools<81'` (already in requirements.txt) |
| `df.enhance ImportError: cannot import 'AudioMetaData' from 'torchaudio'` | DeepFilterNet 0.5.6 expects torchaudio<2.10 API | `setup.sh` patches `df/io.py` automatically. Run it again. |
| Whisper warns `compute capability 12.1 not supported` | torch 2.10 doesn't have native sm_121 (Blackwell) kernels | PTX-JIT fallback works fine; ignore the warning |
| Silero "VAD speech chunks: []" forever | mic input clipping (peak=32768) | check `peak`/`rms` in `[turn-start]` log; lower `_GAIN` |
| Bot starts speaking then stops | speaker → mic echo triggering fake barge-in | `can_interrupt=False` should prevent this; if still happening, use a headset |
| Build pulls 3 GB of torch every rebuild (in Docker) | Docker layer cache busted | use `./setup.sh` instead — pip caches venv installs across rebuilds |

---

## Requirements

* Python **3.11+** (3.12 preferred — matches the proven-working baseline)
* NVIDIA driver with CUDA **11.8 / 12.4 / 12.8 / 13.x**, or CPU fallback
* ~**8 GB** free disk for the venv + Whisper + Chatterbox + DeepFilterNet
  model caches
* No root, no system packages, no Docker required

---

## Optional: Docker

A `Dockerfile` is included for users who prefer containers. The bash
install is recommended (faster builds, direct GPU access, no shim layer)
but the Docker path is functionally equivalent:

```bash
docker build -t miniclosedai-voice:latest .
docker run --rm --gpus all -p 8090:8090 \
  -v voice_models:/root/.cache/huggingface \
  -v voice_pipers:/voices \
  miniclosedai-voice:latest
```

---

## License

See `LICENSE`. Chatterbox-tts is MIT (Resemble AI). DeepFilterNet is
Apache-2.0. Whisper is MIT (OpenAI).

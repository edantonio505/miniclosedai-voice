"""server.py — MiniClosedAI voice service.

A small FastAPI app that wraps **faster-whisper** (ASR) + **Piper** (TTS)
behind the six-endpoint contract MiniClosedAI's voice client expects:

    GET  /health         — {ok, asr_model, tts_model, device, voices_loaded}
    GET  /voices         — {"en": [{id,name,gender}, ...], "es": [...]}
    POST /transcribe     — multipart audio → {text, language, segments}
    POST /speak          — JSON → audio/wav body (one-shot)
    POST /speak/stream   — JSON → SSE {chunk_b64, sample_rate} × N, then {done:true}
    POST /call/configure — JSON → set per-call config (conv_id, miniclosedai_url, …)
    POST /webrtc/offer   — SDP offer (mounted by FastRTC) → SDP answer; live duplex audio

The same image runs locally (`docker run`) or on a remote GPU pod (RunPod).
The user wires it into MiniClosedAI through Settings → Add endpoint with the
service's URL; no compose-level coupling, no shared filesystem.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import socket
from pathlib import Path

# WebRTC debugging — surface ICE state changes, peer connection lifecycle,
# and the handler's transcript / chunk events. Without this, fastrtc's
# default uvicorn config swallows every log below WARNING, so a stuck call
# looks identical to a working one from the outside.
logging.basicConfig(level=logging.INFO)
for name in ("aiortc", "aioice", "fastrtc", "voice.call"):
    logging.getLogger(name).setLevel(logging.DEBUG)

import re
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from asr import ASR
from tts import TTS, VOICE_CATALOG, scan_voice_catalog
from call import BotCallHandler

# ---------------------------------------------------------------------------
# Config (env vars)
# ---------------------------------------------------------------------------
ASR_MODEL = os.environ.get("VOICE_ASR_MODEL", "small")
TTS_VOICES_DIR = Path(os.environ.get("VOICE_VOICES_DIR", "/voices"))
DEVICE = os.environ.get("VOICE_DEVICE", "auto")
API_KEY = os.environ.get("VOICE_API_KEY", "").strip()
PORT = int(os.environ.get("VOICE_PORT", "8090"))

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
# The whisper model is ~500MB on disk; loading it adds a few seconds to the
# first request. We defer that work past container start so /health returns
# fast and Docker's healthcheck succeeds before models finish loading.
_asr: ASR | None = None
_tts: TTS | None = None
_load_lock = asyncio.Lock()


async def _get_asr() -> ASR:
    global _asr
    if _asr is None:
        async with _load_lock:
            if _asr is None:
                _asr = await asyncio.to_thread(ASR, ASR_MODEL, DEVICE)
    return _asr


async def _get_tts() -> TTS:
    global _tts
    if _tts is None:
        async with _load_lock:
            if _tts is None:
                # `auto` and `cuda` both enable GPU; TTS falls back to CPU
                # internally if torch.cuda.is_available() is False. The earlier
                # `== "cuda"` check forced auto to CPU and made TTS 20× slower
                # (1.2s → 22s on GB10 — Chatterbox Turbo's diffusion benefits
                # massively from CUDA fp16).
                _tts = TTS(voices_dir=TTS_VOICES_DIR, use_cuda=(DEVICE != "cpu"))
    return _tts


# ---------------------------------------------------------------------------
# Auth — optional Bearer token
# ---------------------------------------------------------------------------
def _require_auth(authorization: str | None = Header(None)) -> None:
    """If `VOICE_API_KEY` is set in the env, require a matching Bearer token."""
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid or missing Authorization header.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="MiniClosedAI Voice", docs_url="/docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpeakRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, max_length=4000)
    voice: str
    language: str
    speed: float | None = Field(None, ge=0.5, le=2.0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Cheap reachability probe. Doesn't touch models — safe to hit pre-warm."""
    return {
        "ok": True,
        "asr_model": ASR_MODEL,
        "tts_model": "chatterbox-turbo",
        "device": DEVICE,
        "voices_loaded": _tts is not None,
    }


def _lan_ip() -> str:
    """Best-effort primary LAN IP of this host (e.g. 192.168.0.110).

    Opens a throwaway UDP socket toward a public address so the kernel picks
    the interface it would actually route through, then reads back the local
    side. No packet is sent. Returns "" if it can't be determined (no network),
    so the caller can fall back to localhost.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return ""


@app.get("/api/connect-info")
def connect_info(request: Request):
    """The base URL to paste into MiniClosedAI (Settings → Add endpoint,
    Kind: voice). Mirrors the copy-able base_url miniclosedai-llm exposes for
    its model servers, but for this whole voice service rather than per-model.

    - base_url:     reachable from another machine on the LAN (or this host).
                    Scheme is whatever the GUI is being served over (https
                    behind the dev cert, http with --http); host is the LAN IP
                    so a miniclosedai running elsewhere can reach back, with
                    VOICE_PUBLIC_HOST / RUNPOD_POD_ID overrides for hosted pods.
    - alt_base_url: for a miniclosedai running as a Docker container on THIS
                    same host — it reaches the service via the host gateway.
    """
    scheme = request.url.scheme  # 'https' behind the dev cert, 'http' with --http
    host_override = (
        os.environ.get("VOICE_PUBLIC_HOST")
        or os.environ.get("PUBLIC_HOST")
        or os.environ.get("ADVERTISE_HOST")
    )
    pod = os.environ.get("RUNPOD_POD_ID")
    if pod and not host_override:
        base_url = f"https://{pod}-{PORT}.proxy.runpod.net"
    else:
        host = host_override or _lan_ip() or "localhost"
        base_url = f"{scheme}://{host}:{PORT}"
    return {
        "kind": "voice",
        "base_url": base_url,
        "alt_base_url": f"http://host.docker.internal:{PORT}",
        "auth_required": bool(API_KEY),
    }


@app.get("/voices")
def voices(_=Depends(_require_auth)):
    """Live catalog of available TTS voices, built by scanning VOICE_VOICES_DIR
    on every call. Shape matches what MiniClosedAI's voice.py reshapes for
    the per-bot Voice picker: `{lang: [{id, name, gender?}, ...], ...}`.

    Dynamic scan (rather than the old hardcoded `VOICE_CATALOG`) means voices
    cloned via the Voice Studio GUI (`POST /voices`) show up immediately —
    no service restart needed.
    """
    return scan_voice_catalog(TTS_VOICES_DIR)


# ---------------------------------------------------------------------------
# Voice cloning — POST + DELETE for the Voice Studio GUI
# ---------------------------------------------------------------------------
# The GUI records audio in the browser, encodes it as a WAV blob, and POSTs
# it here with a display name + language. The server validates the WAV,
# resamples to Chatterbox's reference rate (24000 Hz mono int16), and writes both the
# audio (`<id>.wav`) and a sidecar JSON (`<id>.json`) carrying the display
# metadata so subsequent GET /voices calls can show the friendly name.
#
# The same scan_voice_catalog() the GET path uses picks up the new file on
# its next invocation — no restart, no in-memory catalog to invalidate.

# 0.5 s minimum: anything shorter is almost certainly an accidental tap and
# would give Chatterbox no useful speaker conditioning. Long clips are NOT
# rejected — we auto-trim to the first _VOICE_MAX_DURATION_SEC seconds. Chatterbox
# only conditions on the leading slice of the reference (its own docs recommend
# 5-15 s), so the tail is wasted anyway; trimming beats making the user go
# hand-edit their file.
_VOICE_MIN_DURATION_SEC = 0.5
_VOICE_MAX_DURATION_SEC = 90.0   # trim cap, not a reject threshold
# Cap upload size so a runaway client can't DoS the disk. 20 MB comfortably fits
# a 90 s clip even at 96 kHz mono int16 (~17 MB); anything past that is a client
# bug or abuse, not a legitimate voice sample.
_VOICE_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Reserved id — the fallback voice every install ships with. Refused as a
# slug so a user can't clobber it via the GUI by naming a clone "default".
_RESERVED_VOICE_IDS = {"default"}

# Languages the Voice Studio dropdown advertises. Anything else (Spanish-only
# user cloning a Portuguese voice, etc.) is silently bucketed under the
# requested code; we don't gatekeep — Chatterbox itself is multilingual.
_KNOWN_LANGUAGES = ("en", "es")


def _slugify_voice_name(name: str) -> str:
    """`"Edgar's Voice!"` → `"edgars-voice"`. Lowercase ascii alnum + dashes,
    collapsed runs, trimmed, capped at 40 chars. Empty on input gives ""
    so the caller can 400."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:40]


def _unique_voice_id(voices_dir: Path, base: str) -> str:
    """Append `-2`, `-3`, … until no `<id>.wav` exists in `voices_dir`."""
    candidate = base
    n = 2
    while any((voices_dir / f"{candidate}.{ext}").exists() for ext in ("wav", "WAV", "flac")):
        candidate = f"{base}-{n}"
        n += 1
    return candidate


@app.post("/voices", status_code=201)
async def upload_voice(
    audio: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=60),
    language: str = Form("en"),
    _=Depends(_require_auth),
):
    """Clone a new voice from an uploaded WAV.

    Body: multipart/form-data
      audio:    WAV file (browser-recorded; ≥0.5 s, longer auto-trimmed to 90 s,
                mono or stereo, any SR)
      name:     human display name (shown in the MiniClosedAI dropdown)
      language: ISO-639-1 code (`en` / `es` — defaults to en)

    Returns: 201 {voice_id, name, language, duration_sec, sample_rate}

    The WAV is normalised on disk to 24000 Hz mono int16 (Chatterbox's reference
    rate) so the catalog stays uniform regardless of what the browser sent.
    """
    if not audio.filename:
        raise HTTPException(400, "Missing audio file.")
    if audio.content_type and not audio.content_type.startswith(("audio/wav", "audio/x-wav", "audio/wave")):
        # Permissive on the absence of content_type (some browsers omit it);
        # strict when present — we don't want to silently accept webm/mp3.
        raise HTTPException(415, f"Unsupported content_type {audio.content_type!r}; expected audio/wav.")

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty audio payload.")
    if len(raw) > _VOICE_MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload too large ({len(raw)} bytes; max {_VOICE_MAX_UPLOAD_BYTES}).")

    # Decode + validate the WAV via soundfile (round-trips PCM cleanly).
    try:
        data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as e:
        raise HTTPException(400, f"Could not decode WAV: {e}")
    if data.size == 0:
        raise HTTPException(400, "WAV decoded to zero samples.")
    duration_sec = data.shape[0] / float(sample_rate or 1)
    if duration_sec < _VOICE_MIN_DURATION_SEC:
        raise HTTPException(400, f"Recording too short ({duration_sec:.2f} s; min {_VOICE_MIN_DURATION_SEC} s).")
    # Auto-trim (don't reject) anything over the cap to the leading window.
    if duration_sec > _VOICE_MAX_DURATION_SEC:
        max_samples = int(_VOICE_MAX_DURATION_SEC * sample_rate)
        data = data[:max_samples]
        logging.getLogger("voice.upload").info(
            "upload %.1f s → trimmed to %.1f s", duration_sec, _VOICE_MAX_DURATION_SEC)
        duration_sec = data.shape[0] / float(sample_rate or 1)

    # Downmix to mono if needed, then resample to 24000 Hz with librosa —
    # the SAME resampler chatterbox's `prepare_conditionals` uses internally
    # (via `librosa.load(sr=S3GEN_SR=24000)`). librosa applies a polyphase
    # anti-aliasing filter; linear interpolation in the browser (or here)
    # introduces audible aliasing that the reference encoder reads as a
    # "darker" speaker timbre — making cloned voices sound dull / lower-
    # pitched than the source. 24000 Hz matches the proven RunPod handler's
    # brooke_voice_ref.wav natively (no resampling round-trip on the
    # chatterbox side). Note: chatterbox OUTPUTS at 22050, a DIFFERENT
    # number that's irrelevant to the reference-clip encoding path.
    import librosa
    mono = data.mean(axis=1).astype(np.float32)
    target_sr = 24_000
    if sample_rate != target_sr:
        mono = librosa.resample(mono, orig_sr=int(sample_rate), target_sr=target_sr)
    # Hard-clip to int16 range, dither-free (the source is already PCM).
    pcm_i16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)

    # Reserve a slug now; refuse "default" and any empty / unreserved-conflict.
    base = _slugify_voice_name(name)
    if not base:
        raise HTTPException(400, "Name must contain at least one letter or digit.")
    if base in _RESERVED_VOICE_IDS:
        raise HTTPException(400, f"`{base}` is a reserved voice id.")

    TTS_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    voice_id = _unique_voice_id(TTS_VOICES_DIR, base)

    wav_path = TTS_VOICES_DIR / f"{voice_id}.wav"
    sidecar = TTS_VOICES_DIR / f"{voice_id}.json"
    try:
        sf.write(str(wav_path), pcm_i16, target_sr, format="WAV", subtype="PCM_16")
        sidecar.write_text(
            json.dumps({
                "name": name.strip(),
                "language": (language or "en").lower(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_sample_rate": int(sample_rate),
                "duration_sec": round(duration_sec, 3),
            }, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        # Clean up partial write so a half-saved voice doesn't poison the catalog.
        for p in (wav_path, sidecar):
            try: p.unlink()
            except OSError: pass
        raise HTTPException(500, f"Could not write voice files: {e}")

    return {
        "voice_id": voice_id,
        "name": name.strip(),
        "language": (language or "en").lower(),
        "duration_sec": round(duration_sec, 3),
        "sample_rate": target_sr,
    }


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str, _=Depends(_require_auth)):
    """Remove a cloned voice from disk. Refuses `default` (the fallback that
    keeps the service functional). 404 if no such voice exists."""
    if voice_id in _RESERVED_VOICE_IDS:
        raise HTTPException(400, f"`{voice_id}` is reserved and cannot be deleted.")
    # Match the same extension priority order the catalog scanner uses.
    found = False
    for ext in ("wav", "WAV", "flac"):
        p = TTS_VOICES_DIR / f"{voice_id}.{ext}"
        if p.exists():
            try:
                p.unlink()
                found = True
            except OSError as e:
                raise HTTPException(500, f"Could not delete {p.name}: {e}")
    sidecar = TTS_VOICES_DIR / f"{voice_id}.json"
    if sidecar.exists():
        try:
            sidecar.unlink()
        except OSError:
            pass  # Audio is gone; orphan sidecar will be ignored by the scanner.
    if not found:
        raise HTTPException(404, f"No voice with id {voice_id!r}.")
    return {"ok": True, "voice_id": voice_id}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    _=Depends(_require_auth),
):
    """Multipart audio → JSON transcript. `language` is an optional hint;
    omit for auto-detect (faster-whisper's CLD is reasonable)."""
    asr = await _get_asr()
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio payload.")
    # Whisper is CPU/GPU-bound; offload to a worker thread so the event
    # loop stays responsive to /health probes during a long clip.
    return await asyncio.to_thread(asr.transcribe, data, language)


@app.post("/speak")
async def speak(req: SpeakRequest, _=Depends(_require_auth)):
    """One-shot synth — returns a single audio/wav body. Use /speak/stream for
    chunked playback over SSE."""
    tts = await _get_tts()
    chunks: list[bytes] = []
    sample_rate: int | None = None
    for chunk, sr in tts.synthesize_stream(req.text, req.voice, req.language, req.speed):
        chunks.append(chunk)
        sample_rate = sr
    pcm = b"".join(chunks)
    if not pcm or sample_rate is None:
        raise HTTPException(500, "TTS produced no audio.")
    audio_np = np.frombuffer(pcm, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


# ---------------------------------------------------------------------------
# Call mode — long-running duplex WebRTC via FastRTC
# ---------------------------------------------------------------------------
# The browser drives a real conversation:
#   1. POST /call/configure with {conv_id, miniclosedai_url, voice, language}
#   2. POST /webrtc/offer (mounted by FastRTC) with the SDP offer → SDP answer
#   3. Audio flows directly browser ↔ this service. After each detected pause,
#      our handler transcribes, calls MiniClosedAI's /chat/stream for the bot
#      reply, synthesizes via Piper, and yields audio back.
#   4. Transcript + reply tokens ride a WebRTC DataChannel as JSON events.
#
# v1 limit: ONE active call per voice service instance. Multi-concurrent
# requires scoping `_call_config` per WebRTC session id — punted to v3.

_call_config: dict = {
    "conv_id": None,
    "miniclosedai_url": "",
    "voice_id": "",
    "language": "en",
}

# Default voice per language — Piper voice ids from tts.py's VOICE_CATALOG.
_DEFAULT_VOICES = {"en": "en_US-amy-medium", "es": "es_MX-claude-high"}


def _default_voice(lang: str) -> str:
    return _DEFAULT_VOICES.get(lang, "en_US-amy-medium")


class CallConfigure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conv_id: int
    miniclosedai_url: str = Field(..., min_length=1)
    voice: str | None = None
    language: str | None = None


async def _prewarm_voice(voice: str) -> None:
    """Load the TTS model (if needed) and run prepare_conditionals for `voice`
    so the FIRST spoken sentence of the call isn't cold.

    Without this, only the `default` voice is warmed at startup (tts.py), and a
    call using a cloned voice (e.g. `ed2`) pays the ~hundreds-of-ms-to-seconds
    prepare_conditionals cost inside sentence #1 — the audio then lands after
    the whole reply has already streamed to the screen. Best-effort: any error
    just means the first turn warms lazily as before.
    """
    try:
        tts = await _get_tts()
        wav = tts._wav_for(voice) or tts._wav_for("default")
        if wav is None:
            return
        if voice != getattr(tts, "_current_voice", ""):
            await asyncio.to_thread(tts._switch_to, wav, voice)
            logging.getLogger("voice.call").info("conv prewarm: voice %r ready", voice)
    except Exception:
        logging.getLogger("voice.call").warning("voice prewarm failed", exc_info=True)


@app.post("/call/configure")
async def call_configure(req: CallConfigure, _=Depends(_require_auth)):
    """Set the per-call config the WebRTC handler will read on each turn.

    Must be called immediately before POSTing the SDP offer. Returns the
    resolved config so the browser knows which voice/language defaulted in.
    Kicks off a background warm of the chosen voice so the first turn is hot.
    """
    lang = (req.language or "en").lower()
    voice = req.voice or _default_voice(lang)
    _call_config["conv_id"] = req.conv_id
    _call_config["miniclosedai_url"] = req.miniclosedai_url
    _call_config["voice_id"] = voice
    _call_config["language"] = lang
    # Fire-and-forget: don't block the config response (the browser POSTs the
    # SDP offer right after) — the warm runs while the WebRTC handshake happens.
    asyncio.create_task(_prewarm_voice(voice))
    return {"ok": True, "conv_id": req.conv_id, "voice": voice, "language": lang}


@app.get("/call/events/{webrtc_id}")
async def call_events(webrtc_id: str):
    """SSE stream of AdditionalOutputs the handler yields during the call.

    FastRTC's Stream pushes `AdditionalOutputs(...)` into an internal queue
    keyed by `webrtc_id` — they are NOT delivered over the WebRTC DataChannel
    automatically. This endpoint forwards them as `data: {...}\\n\\n` frames so
    the browser can render `{transcript}` / `{chunk}` / `{end}` / `{error}`
    events in real time, alongside the audio track flowing over WebRTC.
    """
    stream = _ensure_stream()

    async def gen():
        try:
            async for outputs in stream.output_stream(webrtc_id):
                # AdditionalOutputs.args is a tuple of whatever the handler
                # passed to AdditionalOutputs(...) — usually one dict each.
                for arg in (getattr(outputs, "args", None) or ()):
                    if isinstance(arg, dict):
                        yield f"data: {json.dumps(arg)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _call_handler(audio):
    """The single global handler FastRTC drives on each VAD pause.

    Reads per-call config from `_call_config` (set by /call/configure). Builds
    a fresh BotCallHandler for each turn so per-call state stays local; the
    asr/tts singletons are reused, no extra model load.
    """
    if _call_config["conv_id"] is None or not _call_config["miniclosedai_url"]:
        return
    asr = await _get_asr()
    tts = await _get_tts()
    inst = BotCallHandler(
        asr, tts,
        _call_config["conv_id"],
        _call_config["miniclosedai_url"],
        _call_config["voice_id"],
        _call_config["language"],
    )
    async for ev in inst.respond(audio):
        yield ev


# Mount the FastRTC Stream at startup. Lazy so the import of fastrtc (which
# pulls aiortc + libsrtp) doesn't slow down /health's first response.
_stream = None


def _ensure_stream():
    global _stream
    if _stream is not None:
        return _stream
    from fastrtc import ReplyOnPause, Stream
    from fastrtc.reply_on_pause import AlgoOptions
    from fastrtc.pause_detection.silero import SileroVadOptions

    # ---- DeepFilterNet denoise + AGC -------------------------------------
    # silero VAD kept firing on background noise (AC, keyboard, fan). Cleaning
    # the audio BEFORE the VAD sees it removes the false positives, and the
    # cleaner SNR helps Whisper too. DeepFilterNet is a small ONNX-runnable
    # neural denoiser (~50 MB model) — replaces the earlier pyrnnoise attempt,
    # whose audiolab dep conflicts with aiortc's av<13 pin.
    _df_state = {"model": None, "df": None, "tried": False}
    def _get_df():
        if _df_state["tried"]:
            return _df_state["model"], _df_state["df"]
        _df_state["tried"] = True
        try:
            from df.enhance import init_df
            model, df, _ = init_df()
            _df_state["model"], _df_state["df"] = model, df
        except Exception:
            pass
        return _df_state["model"], _df_state["df"]

    def _denoise(pcm, sr):
        """Run DeepFilterNet on a 48 kHz int16 chunk. Returns int16 same len.
        Passes through unchanged on non-48k input or if DeepFilterNet didn't load."""
        if pcm is None or not hasattr(pcm, "astype") or sr != 48_000:
            return pcm
        model, df = _get_df()
        if model is None:
            return pcm
        try:
            import torch as _t
            f32 = pcm.astype(np.float32) / 32768.0
            x = _t.from_numpy(f32).unsqueeze(0)
            from df.enhance import enhance
            y = enhance(model, df, x)
            cleaned = y.squeeze().cpu().numpy()
            return np.clip(cleaned * 32768.0, -32768, 32767).astype(np.int16)
        except Exception:
            return pcm

    # IMPORTANT: previously 4× to rescue a very-quiet mic chain. The current
    # mic chain delivers peak~32768 (already at int16 ceiling), so 4× CLIPS
    # every sample and produces sustained noise that silero misreads as
    # continuous speech — the handler waits seconds before declaring pause.
    # Keep gain at 1 (passthrough). If you ever hit a quiet-mic setup again,
    # bump to 2 or 4 — but check logs for `peak=32767` clipping first.
    _GAIN = 1

    def _amplify(pcm):
        if pcm is None or not hasattr(pcm, "astype"):
            return pcm
        amp = pcm.astype(np.int32) * _GAIN
        np.clip(amp, -32768, 32767, out=amp)
        return amp.astype(np.int16)

    def _preprocess(pcm, sr):
        # Denoise first (kills background noise that fakes VAD), then gain
        # (boosts the quiet-mic case to silero's detection floor).
        return _amplify(_denoise(pcm, sr))

    async def _amplified_call_handler(audio):
        sr, pcm = audio
        async for ev in _call_handler((sr, _preprocess(pcm, sr))):
            yield ev

    # Monkey-patch ReplyOnPause.receive so VAD also sees preprocessed audio
    # (the handler-side preprocessing alone wouldn't help: VAD runs first and
    # decides whether to invoke the handler at all).
    _orig_receive = ReplyOnPause.receive
    def _gained_receive(self, frame):
        sr, pcm = frame
        return _orig_receive(self, (sr, _preprocess(pcm, sr)))
    ReplyOnPause.receive = _gained_receive

    _stream = Stream(
        handler=ReplyOnPause(
            _amplified_call_handler,
            # can_interrupt=False kills barge-in. We disable it because on a
            # laptop with speakers + open mic, the bot's TTS audio bleeds
            # back into the mic, silero scores it as "speech", and the active
            # reply gets cancelled mid-sentence — the "bot starts talking then
            # stops" symptom. Re-enable when the chain has proper AEC
            # (echo cancellation) on the browser-side getUserMedia constraints.
            can_interrupt=False,
            # Silero defaults to 2000ms of silence before declaring end-of-turn,
            # which is most of the perceived "took a really long time" lag.
            # 300ms is reliable for conversational turn-taking with a clean
            # mic chain. min_speech_duration_ms=100 catches short "yes"/"no".
            model_options=SileroVadOptions(
                min_silence_duration_ms=300,
                min_speech_duration_ms=100,
                speech_pad_ms=200,
            ),
            # AlgoOptions stays mostly default — chunk size 0.5s leaves enough
            # silero a 250ms-min window. speech_threshold=0.05 is permissive
            # so a quiet trailing word still counts as still-speaking.
            algo_options=AlgoOptions(
                audio_chunk_duration=0.5,
                started_talking_threshold=0.15,
                speech_threshold=0.05,
            ),
        ),
        modality="audio",
        mode="send-receive",
    )
    _stream.mount(app)
    return _stream


@app.on_event("startup")
async def _prewarm_models():
    # Whisper-small is ~500MB; loading it on the first VAD pause makes turn 1
    # of a call sluggish vs every subsequent turn. Pull both ASR and TTS into
    # memory AND run one real inference each so PyTorch's PTX-JIT kernels
    # compile and cache before the user's first utterance hits the pipeline.
    # Without the inference pass, the first transcribe still pays the JIT
    # tax (~2-3 s on sm_121 GB10).
    async def _warm():
        try:
            asr = await _get_asr()
            tts = await _get_tts()
            # Force a real Whisper forward pass on 1 s of zero-PCM. Fast on
            # GPU once kernels are cached, slow first time — which is exactly
            # what we want to take out of the call-mode critical path.
            import numpy as _np
            await asyncio.to_thread(
                asr.transcribe_array, _np.zeros(16000, dtype=_np.int16), 16000, "en",
            )
            # And one TTS forward pass so Chatterbox's t3 + s3gen JIT-cache
            # the kernels we use during synthesize_stream.
            for _ in tts.synthesize_stream("Hello.", "default", "en", None):
                break
        except Exception:
            import traceback
            traceback.print_exc()
    asyncio.create_task(_warm())


@app.post("/speak/stream")
async def speak_stream(req: SpeakRequest, _=Depends(_require_auth)):
    """SSE chunked synth — each frame is `{chunk_b64, sample_rate}` (int16 PCM
    base64-encoded), then a terminal `{done: true}`. MiniClosedAI's push-to-talk
    UI decodes the chunks straight into Web Audio for seamless playback.

    The Chatterbox Turbo `synthesize_stream` generator yields one chunk every
    ~75 speech tokens (~250-500 ms of GPU work per chunk). It's a SYNC generator,
    so iterating it inside this async handler would block the event loop for
    each chunk — and the SSE bytes for previously-yielded chunks would sit in
    FastAPI's buffer until the loop got idle time, producing the "I see the
    full paragraph then hear all the audio at once" symptom.
    We drive the sync generator from a worker thread and bridge each chunk
    back to the async caller via a thread-safe queue, so the SSE socket
    flushes every chunk the moment it's ready.
    """
    tts = await _get_tts()

    async def gen():
        chunk_q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()
        loop = asyncio.get_running_loop()

        def _run():
            try:
                for chunk, sr in tts.synthesize_stream(
                    req.text, req.voice, req.language, req.speed,
                ):
                    asyncio.run_coroutine_threadsafe(
                        chunk_q.put((chunk, sr)), loop,
                    )
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    chunk_q.put(("__error__", str(e))), loop,
                )
            finally:
                asyncio.run_coroutine_threadsafe(
                    chunk_q.put(SENTINEL), loop,
                )

        worker = asyncio.create_task(asyncio.to_thread(_run))
        try:
            while True:
                item = await chunk_q.get()
                if item is SENTINEL:
                    break
                if isinstance(item, tuple) and item[0] == "__error__":
                    yield f"data: {json.dumps({'error': item[1]})}\n\n"
                    return
                chunk, sr = item
                event = {
                    "chunk_b64": base64.b64encode(chunk).decode("ascii"),
                    "sample_rate": sr,
                }
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        finally:
            # If the client disconnected mid-stream, let the worker drain.
            await worker

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Mount FastRTC's WebRTC routes BEFORE the static catch-all below.
#
# Starlette resolves routes in registration order. `app.mount("/", StaticFiles)`
# matches every path prefix, so any route registered AFTER it never gets a
# look-in — and StaticFiles only supports GET, so POSTing to `/webrtc/offer`
# would return 405 Method Not Allowed instead of being handled by FastRTC.
#
# The previous design deferred `_ensure_stream()` to the FastAPI `startup`
# event "so fastrtc's heavy import doesn't slow /health". But the startup
# event registers FastRTC's routes AFTER the static mount in this module —
# wrong order. Eagerly building the Stream here at module-load fixes the
# ordering. The fastrtc import cost is paid once at process boot, not on
# every request, so the latency hit is at most a few hundred ms once.
#
# try/except so a missing fastrtc dep doesn't kill the whole service —
# /transcribe + /speak still work for push-to-talk users.
# ---------------------------------------------------------------------------
try:
    _ensure_stream()
except Exception:
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Static GUI — the "Voice Studio" page lives at `/`. Mounted LAST so all API
# routes above (/health, /voices, /speak, /transcribe, /call/*, /webrtc/offer)
# take routing precedence. `html=True` makes `/` serve `static/index.html`
# automatically; any unknown sub-path under `/` falls through to a 404 from
# StaticFiles (not from a Python handler), which is the desired behaviour.
#
# Skipped silently if the `static/` directory isn't present so a bare-bones
# `python -m uvicorn server:app` for API-only debugging still works.
# ---------------------------------------------------------------------------
class _NoCacheStatics(StaticFiles):
    """StaticFiles subclass that disables browser caching of the GUI assets.

    Without this, browsers hold onto the previous `app.js` / `style.css` /
    `index.html` aggressively — users see stale UI (empty paragraphs,
    missing buttons, old behaviour) after a server-side change and need to
    hard-refresh to pick up the new files. This is the same `_NoCacheStatics`
    pattern MiniClosedAI uses (app.py:3306 there) — local dev tool, no
    measurable cost at the bandwidth scale this service runs at.
    """
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", _NoCacheStatics(directory=str(_STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")

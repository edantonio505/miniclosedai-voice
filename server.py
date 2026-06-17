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
from pathlib import Path

# WebRTC debugging — surface ICE state changes, peer connection lifecycle,
# and the handler's transcript / chunk events. Without this, fastrtc's
# default uvicorn config swallows every log below WARNING, so a stuck call
# looks identical to a working one from the outside.
logging.basicConfig(level=logging.INFO)
for name in ("aiortc", "aioice", "fastrtc", "voice.call"):
    logging.getLogger(name).setLevel(logging.DEBUG)

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from asr import ASR
from tts import TTS, VOICE_CATALOG
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


@app.get("/voices")
def voices(_=Depends(_require_auth)):
    """The static catalog — same shape MiniClosedAI's voice.py reshapes for
    the per-bot Voice + Language dropdowns."""
    return VOICE_CATALOG


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


@app.post("/call/configure")
def call_configure(req: CallConfigure, _=Depends(_require_auth)):
    """Set the per-call config the WebRTC handler will read on each turn.

    Must be called immediately before POSTing the SDP offer. Returns the
    resolved config so the browser knows which voice/language defaulted in.
    """
    lang = (req.language or "en").lower()
    voice = req.voice or _default_voice(lang)
    _call_config["conv_id"] = req.conv_id
    _call_config["miniclosedai_url"] = req.miniclosedai_url
    _call_config["voice_id"] = voice
    _call_config["language"] = lang
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
async def _mount_call_stream():
    try:
        _ensure_stream()
    except Exception:
        # Don't kill the whole service if fastrtc fails to import — /transcribe
        # and /speak still work. The /call/configure endpoint will surface the
        # error if someone tries to use call mode.
        import traceback
        traceback.print_exc()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")

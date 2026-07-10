"""test_client.py — end-to-end call-mode test harness.

Runs inside the voice container (aiortc is already installed there). Drives
the same path the browser uses:

  1. Synthesize a test phrase via the local Piper TTS  →  WAV
  2. POST /api/conversations/{id}/call/configure       →  triggers LLM warmup
  3. Create RTCPeerConnection + DataChannel + audio track from the WAV
  4. POST /api/conversations/{id}/call/offer           →  SDP answer
  5. Stream the WAV into the WebRTC pipe; subscribe to /call/events SSE
  6. Collect transcript / chunk / end events + reply audio frames
  7. Print a JSON report (parsed by the host-side wrapper for the human view)

Run:
  python /app/test_client.py --url https://localhost:8095 --conv-id 100 \
        --phrase "Hello, can you hear me?" --timeout 30
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
import uuid
import wave
from dataclasses import dataclass, field, asdict

import httpx
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from av import AudioFrame


# ---------------------------------------------------------------------------
# Synthesize the test phrase via the local TTS so the test audio is a known
# clean signal at normal speech levels. Same path /speak uses, but we call
# the in-process TTS directly to avoid an extra HTTP hop.
# ---------------------------------------------------------------------------
async def synth_test_phrase(text: str, voice: str = "default", language: str = "en",
                            speak_url: str | None = None) -> str:
    """Generate `text` as a WAV via the local server's /speak endpoint.

    Reuses the server's GPU-loaded TTS instead of spinning up a second copy
    (Chatterbox on CPU takes minutes per sentence — unusable for tests).
    Tries HTTPS first (the bare-metal server runs TLS with a self-signed
    cert), then plain HTTP (legacy Docker). Falls back to a synthesized
    sine-burst at 16 kHz ONLY if both are unavailable — and says so loudly,
    because a beep never passes VAD/ASR and the run will read as a failure.
    """
    path = "/tmp/test_phrase.wav"
    audio = None
    sample_rate = 22050
    candidates = [speak_url] if speak_url else [
        "https://localhost:8090/speak", "http://localhost:8090/speak",
    ]
    for url in candidates:
        try:
            async with httpx.AsyncClient(timeout=120.0, verify=False) as c:
                resp = await c.post(url, json={
                    "text": text, "voice": voice, "language": language,
                })
                resp.raise_for_status()
                data = resp.content
                with wave.open(io.BytesIO(data)) as r:
                    sample_rate = r.getframerate()
                    audio = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)
                break
        except Exception:
            continue
    if audio is None:
        # Fallback — synthesize a 1-second 440Hz beep so the rest of the test
        # can still exercise WebRTC handshake + VAD. Won't transcribe (silero
        # rejects tones), so ASR/reply/TTS checks WILL fail — the timings are
        # the only useful output of a beep run.
        print("WARNING: /speak unreachable over https AND http — sending a "
              "beep; transcript/reply checks will fail.", file=sys.stderr)
        t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    # Pad with ~2s of trailing silence. Without it the WAV ends abruptly,
    # aiortc raises MediaStreamError instead of emitting silence frames, and
    # silero never sees the post-speech silence that triggers end-of-turn —
    # the handler never fires and the test reads as a false negative.
    trailing_silence = np.zeros(sample_rate * 2, dtype=np.int16)
    audio = np.concatenate([audio, trailing_silence])
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())
    return path


@dataclass
class TestReport:
    ok: bool = False
    error: str | None = None

    # Stage timings (ms, relative to wall clock start)
    t_synth_ms: float = 0
    t_configure_ms: float = 0
    t_offer_ms: float = 0
    t_ice_connected_ms: float = 0
    t_first_event_ms: float = 0
    t_first_transcript_ms: float = 0
    t_first_chunk_ms: float = 0
    t_first_audio_back_ms: float = 0
    t_end_ms: float = 0

    # Quality signals
    phrase_sent: str = ""
    transcript_received: str = ""
    transcript_match: bool = False  # fuzzy: sent ⊂ received or vice versa
    reply_text: str = ""
    reply_audio_frames: int = 0
    reply_audio_seconds: float = 0
    received_events: list[dict] = field(default_factory=list)
    audio_rms_sent: float = 0  # for diagnostic: did we send real audio?

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


async def run_test(
    miniclosedai_url: str,
    conv_id: int,
    phrase: str,
    timeout: float,
) -> TestReport:
    r = TestReport(phrase_sent=phrase)
    wall_t0 = time.perf_counter()

    # ---- 1. Synthesize the test phrase via local Piper -------------------
    try:
        wav_path = await synth_test_phrase(phrase)
        r.t_synth_ms = (time.perf_counter() - wall_t0) * 1000
        with wave.open(wav_path) as w:
            n = w.getnframes()
            pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
            r.audio_rms_sent = float(np.sqrt(np.mean(pcm ** 2)))
    except Exception as e:
        r.error = f"synth failed: {e}"
        return r

    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        # ---- 2. POST /call/configure --------------------------------------
        t_cfg = time.perf_counter()
        try:
            resp = await client.post(
                f"{miniclosedai_url}/api/conversations/{conv_id}/call/configure",
                json={},
            )
            resp.raise_for_status()
        except Exception as e:
            r.error = f"configure failed: {e}"
            return r
        r.t_configure_ms = (time.perf_counter() - t_cfg) * 1000

        # ---- 3. Set up RTCPeerConnection + tracks + DataChannel -----------
        pc = RTCPeerConnection()
        dc = pc.createDataChannel("text")  # required so FastRTC unblocks input
        player = MediaPlayer(wav_path)
        pc.addTrack(player.audio)

        audio_back_frames: list[AudioFrame] = []
        first_audio_back_t: list[float] = []

        @pc.on("track")
        def on_track(track):
            async def collect():
                while True:
                    try:
                        frame = await track.recv()
                    except Exception:
                        return
                    if not first_audio_back_t:
                        first_audio_back_t.append(time.perf_counter())
                    audio_back_frames.append(frame)
            asyncio.create_task(collect())

        ice_connected_t: list[float] = []
        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState == "connected" and not ice_connected_t:
                ice_connected_t.append(time.perf_counter())

        # ---- 4. POST /call/offer ------------------------------------------
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        # Wait briefly for ICE gathering — FastRTC doesn't accept trickle.
        for _ in range(60):  # ~3s
            if pc.iceGatheringState == "complete":
                break
            await asyncio.sleep(0.05)

        webrtc_id = str(uuid.uuid4())
        t_off = time.perf_counter()
        try:
            resp = await client.post(
                f"{miniclosedai_url}/api/conversations/{conv_id}/call/offer",
                json={
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                    "webrtc_id": webrtc_id,
                },
            )
            resp.raise_for_status()
            answer = resp.json()
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
        except Exception as e:
            r.error = f"offer failed: {e}"
            await pc.close()
            return r
        r.t_offer_ms = (time.perf_counter() - t_off) * 1000

        # ---- 5. Subscribe to events SSE in parallel -----------------------
        events: list[dict] = []
        end_received = asyncio.Event()

        async def consume_events():
            url = f"{miniclosedai_url}/api/conversations/{conv_id}/call/events/{webrtc_id}"
            try:
                async with client.stream("GET", url) as resp:
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        try:
                            ev = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        events.append({"t_ms": (time.perf_counter() - wall_t0) * 1000, **ev})
                        if ev.get("end") or ev.get("error"):
                            end_received.set()
                            return
            except Exception:
                end_received.set()

        events_task = asyncio.create_task(consume_events())

        # ---- 6. Wait for completion ---------------------------------------
        async def wait_for_end():
            try:
                await asyncio.wait_for(end_received.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

        await wait_for_end()
        # Drain a tiny bit more so trailing audio frames land in our counter.
        await asyncio.sleep(0.5)
        events_task.cancel()

        # ---- 7. Compute timings / quality --------------------------------
        r.received_events = events
        if events:
            r.t_first_event_ms = events[0]["t_ms"]
            tr = next((e for e in events if "transcript" in e), None)
            if tr:
                r.t_first_transcript_ms = tr["t_ms"]
                r.transcript_received = tr["transcript"]
            ch = next((e for e in events if "chunk" in e), None)
            if ch:
                r.t_first_chunk_ms = ch["t_ms"]
            chunks = [e["chunk"] for e in events if "chunk" in e]
            r.reply_text = "".join(chunks).strip()
            end_ev = next((e for e in reversed(events) if e.get("end") or e.get("error")), None)
            if end_ev:
                r.t_end_ms = end_ev["t_ms"]
        if ice_connected_t:
            r.t_ice_connected_ms = (ice_connected_t[0] - wall_t0) * 1000
        if first_audio_back_t:
            r.t_first_audio_back_ms = (first_audio_back_t[0] - wall_t0) * 1000

        r.reply_audio_frames = len(audio_back_frames)
        if audio_back_frames:
            total_samples = sum(
                getattr(f, "samples", 0) or f.to_ndarray().shape[-1]
                for f in audio_back_frames
            )
            sr = audio_back_frames[0].sample_rate or 24000
            r.reply_audio_seconds = total_samples / sr

        # Fuzzy transcript match — strip punctuation/case + token overlap ≥ 0.5
        sent_tokens = set(phrase.lower().translate(str.maketrans("", "", ".,?!")).split())
        got_tokens = set(r.transcript_received.lower().translate(str.maketrans("", "", ".,?!")).split())
        if sent_tokens and got_tokens:
            r.transcript_match = len(sent_tokens & got_tokens) >= max(2, len(sent_tokens) // 2)

        r.ok = (
            r.transcript_match
            and bool(r.reply_text)
            and r.reply_audio_frames > 0
        )

        await pc.close()
        await player.video.stop() if player.video else None

    return r


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="https://localhost:8095",
                   help="MiniClosedAI base URL (defaults to same-host HTTPS)")
    p.add_argument("--conv-id", type=int, default=100,
                   help="Conversation id to call (must be a registered bot)")
    p.add_argument("--phrase", default="Hello, can you hear me clearly?",
                   help="Test phrase to TTS and send through the WebRTC pipe")
    p.add_argument("--timeout", type=float, default=20.0,
                   help="Max seconds to wait for the bot's reply to complete")
    args = p.parse_args()

    report = await run_test(args.url, args.conv_id, args.phrase, args.timeout)
    print(report.to_json())
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    asyncio.run(main())

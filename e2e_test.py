"""e2e_test.py — voice service smoke + benchmark suite.

Run this BEFORE committing any change to miniclosedai-voice. Verifies the
full local pipeline end-to-end: GPU access, model loads, HTTP API, ASR
quality, TTS quality, and an aiortc-driven call-mode handshake. Each stage
exits early on the first hard failure so the surfaces you care about are
the first thing you see.

Usage:
    ./test.sh                # wrapper: activates ./env/ + runs this script
    env/bin/python e2e_test.py

Optional flags:
    --skip-call      skip the WebRTC call-mode stage (saves ~15s)
    --skip-roundtrip skip the synth → transcribe roundtrip stage
    --strict         non-zero exit on warnings too (CI mode)
    --verbose        print full tracebacks on failure

Targets the service at $VOICE_URL (default http://localhost:8090). The
service must be running (./start.sh -d).
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import sys
import time
import traceback
import wave
from dataclasses import dataclass, field
from typing import Callable

import httpx


# ─── pretty-print helpers ───────────────────────────────────────────────
R = "\x1b[31m"; G = "\x1b[32m"; Y = "\x1b[33m"; B = "\x1b[34m"; D = "\x1b[2m"; X = "\x1b[0m"
BOLD = "\x1b[1m"


def _h(title: str) -> None:
    print(f"\n{BOLD}{B}▶ {title}{X}")


def _ok(msg: str) -> None:
    print(f"  {G}✓{X} {msg}")


def _warn(msg: str) -> None:
    print(f"  {Y}!{X} {msg}")


def _fail(msg: str) -> None:
    print(f"  {R}✗{X} {msg}")


def _info(msg: str) -> None:
    print(f"    {D}{msg}{X}")


# ─── result tracking ────────────────────────────────────────────────────
@dataclass
class Suite:
    passed: list[str] = field(default_factory=list)
    warned: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    verbose: bool = False
    strict: bool = False

    def run(self, name: str, fn: Callable[[], None], *, fatal: bool = False) -> bool:
        try:
            fn()
        except _Skip as e:
            _warn(f"{name}: skipped — {e}")
            self.warned.append(name)
            return True
        except _Warn as e:
            _warn(f"{name}: {e}")
            self.warned.append(name)
            return True
        except AssertionError as e:
            self.failed.append((name, str(e)))
            _fail(f"{name}: {e}")
            if self.verbose:
                traceback.print_exc()
            return False
        except Exception as e:
            self.failed.append((name, f"{type(e).__name__}: {e}"))
            _fail(f"{name}: {type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()
            return False
        self.passed.append(name)
        _ok(name)
        return True

    def summary(self) -> int:
        total = len(self.passed) + len(self.warned) + len(self.failed)
        print()
        print(f"{BOLD}{'─' * 64}{X}")
        print(f"{BOLD}  passed:   {G}{len(self.passed):>3}{X}{BOLD}/{total}{X}")
        if self.warned:
            print(f"{BOLD}  warned:   {Y}{len(self.warned):>3}{X}{BOLD}/{total}{X}")
        if self.failed:
            print(f"{BOLD}  failed:   {R}{len(self.failed):>3}{X}{BOLD}/{total}{X}")
            print()
            for name, err in self.failed:
                print(f"  {R}✗{X} {name}: {err[:200]}")
        print(f"{BOLD}{'─' * 64}{X}")
        if self.failed:
            return 1
        if self.strict and self.warned:
            return 2
        return 0


class _Skip(Exception):
    pass


class _Warn(Exception):
    pass


# ─── HTTP/test helpers ─────────────────────────────────────────────────
def _client(url: str) -> httpx.Client:
    return httpx.Client(base_url=url, timeout=httpx.Timeout(120.0))


def _normalize(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum() or c.isspace()).strip()


def _fuzzy_score(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    if not na:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio() * 100


# ─── stages ─────────────────────────────────────────────────────────────
def stage_imports(s: Suite) -> None:
    _h("imports + GPU")

    def torch_cuda():
        import torch
        ver = torch.__version__
        ok = torch.cuda.is_available()
        if ok:
            _info(f"torch {ver}  CUDA={ok}  device={torch.cuda.get_device_name(0)}")
        else:
            _info(f"torch {ver}  CUDA=False (CPU mode)")
            raise _Warn("CUDA not available — running CPU-only is functional but slow")

    def gpu_op():
        import torch
        if not torch.cuda.is_available():
            raise _Skip("no CUDA")
        x = torch.randn(8, 8, device="cuda")
        (x @ x).sum().item()
        _info("ran a matmul on cuda without error")

    s.run("torch + CUDA", torch_cuda)
    s.run("GPU tensor op", gpu_op)
    s.run("transformers",     lambda: __import__("transformers"))
    s.run("chatterbox.tts",   lambda: __import__("chatterbox.tts"))
    s.run("chatterbox.tts_turbo", lambda: __import__("chatterbox.tts_turbo"))
    s.run("df.enhance",       lambda: __import__("df.enhance"))
    s.run("fastrtc",          lambda: __import__("fastrtc"))
    s.run("aiortc",           lambda: __import__("aiortc"))


def stage_health(s: Suite, url: str) -> dict:
    _h("HTTP /health")
    health = {}

    def get_health():
        nonlocal health
        with _client(url) as c:
            r = c.get("/health", timeout=5.0)
            assert r.status_code == 200, f"got {r.status_code}"
            health = r.json()
            assert health.get("ok") is True, health
            _info(
                f"asr_model={health.get('asr_model')}  "
                f"tts_model={health.get('tts_model')}  "
                f"device={health.get('device')}  "
                f"voices_loaded={health.get('voices_loaded')}"
            )

    s.run("reachable + healthy", get_health)
    return health


def stage_voices(s: Suite, url: str) -> None:
    _h("/voices catalog")

    def get_voices():
        with _client(url) as c:
            r = c.get("/voices")
            assert r.status_code == 200, f"got {r.status_code}"
            cat = r.json()
            assert isinstance(cat, dict) and cat, f"empty: {cat}"
            langs = list(cat.keys())
            n = sum(len(v) for v in cat.values())
            _info(f"{n} voices across {len(langs)} language(s): {', '.join(langs)}")
            for entries in cat.values():
                for e in entries:
                    assert "id" in e and "name" in e, f"bad entry: {e}"

    s.run("/voices returns a non-empty catalog", get_voices)


def stage_studio(s: Suite, url: str) -> None:
    """Voice Studio GUI + clone endpoints — exercises POST/DELETE /voices."""
    _h("Voice Studio (clone endpoints + static GUI)")

    def get_root():
        with _client(url) as c:
            r = c.get("/")
            assert r.status_code == 200, f"got {r.status_code}"
            assert "Voice Studio" in r.text, "GUI HTML not served"
    s.run("GET / serves the Voice Studio page", get_root)

    def reject_default_delete():
        with _client(url) as c:
            r = c.delete("/voices/default")
            assert r.status_code == 400, f"expected 400, got {r.status_code}"
    s.run("DELETE /voices/default is refused (reserved)", reject_default_delete)

    # Round-trip: upload a tiny synthetic WAV → list shows it → DELETE removes it.
    def upload_and_delete():
        # 1 s of 440 Hz sine at 22050 Hz mono — minimal valid WAV, easily
        # passes the 0.5 s minimum-duration check on the server.
        import math
        import struct
        import wave
        import io as _io
        sr = 22050
        n = sr  # 1 second
        amp = 0.25
        buf = _io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            samples = bytearray()
            for i in range(n):
                v = int(32767 * amp * math.sin(2 * math.pi * 440 * i / sr))
                samples += struct.pack("<h", v)
            w.writeframes(bytes(samples))
        wav_bytes = buf.getvalue()

        clone_name = "E2E Clone Test"
        with _client(url) as c:
            r = c.post(
                "/voices",
                files={"audio": ("clone.wav", wav_bytes, "audio/wav")},
                data={"name": clone_name, "language": "en"},
            )
            assert r.status_code == 201, f"upload failed: {r.status_code} {r.text}"
            payload = r.json()
            voice_id = payload["voice_id"]
            assert payload["name"] == clone_name
            assert payload["language"] == "en"
            assert payload["sample_rate"] == 22050
            assert 0.9 < payload["duration_sec"] < 1.2, payload

            # Catalog should now include the new voice under 'en'.
            cat = c.get("/voices").json()
            ids = [v["id"] for entries in cat.values() for v in entries]
            assert voice_id in ids, f"new voice missing: {ids}"

            # Delete + verify removal.
            dr = c.delete(f"/voices/{voice_id}")
            assert dr.status_code == 200, f"delete failed: {dr.status_code} {dr.text}"
            cat = c.get("/voices").json()
            ids = [v["id"] for entries in cat.values() for v in entries]
            assert voice_id not in ids, f"voice still in catalog after delete: {ids}"
    s.run("POST + DELETE /voices round-trip succeeds", upload_and_delete)


def stage_tts(s: Suite, url: str) -> bytes:
    _h("TTS via /speak (one-shot)")
    wav_bytes = b""

    def speak_short():
        nonlocal wav_bytes
        with _client(url) as c:
            t0 = time.perf_counter()
            r = c.post("/speak", json={
                "text": "Hello, this is a voice service smoke test.",
                "voice": "default", "language": "en",
            })
            ms = (time.perf_counter() - t0) * 1000
            assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"
            assert r.headers.get("content-type", "").startswith("audio/"), r.headers
            wav_bytes = r.content
            assert len(wav_bytes) > 44, f"too small: {len(wav_bytes)}"
            with wave.open(io.BytesIO(wav_bytes)) as w:
                sr, frames = w.getframerate(), w.getnframes()
                sec = frames / sr
                _info(f"WAV: {len(wav_bytes)/1024:.0f} KB  sr={sr}  {sec:.1f}s playback  synth={ms:.0f} ms")
                assert sec > 0.2, f"playback too short: {sec:.2f}s"
                if ms > 3000:
                    raise _Warn(f"synth was slow ({ms:.0f} ms) — first call cold-loads models")

    s.run("/speak returns valid WAV", speak_short)
    return wav_bytes


def stage_tts_stream(s: Suite, url: str) -> None:
    _h("TTS via /speak/stream (first-chunk latency)")

    def first_chunk_latency():
        with _client(url) as c:
            t0 = time.perf_counter()
            t_first = None
            n_chunks = 0
            with c.stream("POST", "/speak/stream", json={
                "text": "Streaming TTS first chunk timing check.",
                "voice": "default", "language": "en",
            }) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        try:
                            ev = json.loads(line[6:])
                        except Exception:
                            continue
                        if "chunk_b64" in ev:
                            if t_first is None:
                                t_first = time.perf_counter()
                            n_chunks += 1
                        if ev.get("done"):
                            break
            total = (time.perf_counter() - t0) * 1000
            first = (t_first - t0) * 1000 if t_first else 0
            _info(f"first chunk: {first:.0f} ms  total: {total:.0f} ms  chunks: {n_chunks}")
            assert n_chunks > 0, "no audio chunks arrived"
            if first > 1500:
                raise _Warn(f"first chunk slow ({first:.0f} ms) — model may still be warming")

    s.run("/speak/stream streams chunks", first_chunk_latency)


def stage_asr_roundtrip(s: Suite, url: str, wav: bytes) -> None:
    _h("ASR /transcribe (round-trip TTS → ASR)")

    if not wav:
        _warn("no WAV from TTS stage; skipping roundtrip")
        return

    def roundtrip():
        sent = "Hello, this is a voice service smoke test."
        with _client(url) as c:
            t0 = time.perf_counter()
            r = c.post("/transcribe",
                       files={"audio": ("smoke.wav", wav, "audio/wav")},
                       data={"language": "en"})
            ms = (time.perf_counter() - t0) * 1000
            assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"
            j = r.json()
            got = (j.get("text") or "").strip()
            acc = _fuzzy_score(sent, got)
            _info(f"sent: {sent!r}")
            _info(f"got:  {got!r}")
            _info(f"latency={ms:.0f} ms  accuracy={acc:.1f}%")
            assert got, "empty transcript"
            assert acc >= 50, f"low accuracy {acc:.1f}% — sent {sent!r}, got {got!r}"
            if acc < 80:
                raise _Warn(f"accuracy below 80% ({acc:.1f}%) — Whisper may have normalized numbers or punctuation")

    s.run("/transcribe round-trips TTS output", roundtrip)


def stage_call_mode(s: Suite, url: str) -> None:
    _h("Call mode (FastRTC WebRTC handshake)")

    def configure():
        with _client(url) as c:
            r = c.post("/call/configure", json={
                "conv_id": 1,
                "miniclosedai_url": "http://127.0.0.1:65535",  # bogus — we don't reach the LLM
                "voice": "default",
                "language": "en",
            })
            assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"
            _info(f"configured: {r.json()}")

    def webrtc_offer():
        import asyncio
        from aiortc import RTCPeerConnection
        from aiortc.mediastreams import AudioStreamTrack

        async def go():
            pc = RTCPeerConnection()
            pc.createDataChannel("text")  # FastRTC requires a DataChannel
            pc.addTrack(AudioStreamTrack())
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            for _ in range(60):
                if pc.iceGatheringState == "complete":
                    break
                await asyncio.sleep(0.05)
            async with httpx.AsyncClient(timeout=30.0) as ac:
                r = await ac.post(f"{url}/webrtc/offer", json={
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                    "webrtc_id": "smoke-" + str(int(time.time())),
                })
            await pc.close()
            assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"
            j = r.json()
            assert "sdp" in j and "type" in j, f"bad answer shape: {j.keys()}"
            assert "m=audio" in j["sdp"], "SDP answer missing audio m-section"
            _info(f"got valid SDP answer ({len(j['sdp'])} bytes)")

        asyncio.run(go())

    s.run("/call/configure accepts a config", configure)
    s.run("/webrtc/offer returns a valid SDP answer", webrtc_offer)


# ─── main ───────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default=os.environ.get("VOICE_URL", "http://localhost:8090"),
                   help="base URL of the running voice service")
    p.add_argument("--skip-call", action="store_true")
    p.add_argument("--skip-roundtrip", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="non-zero exit when there are warnings too (for CI)")
    p.add_argument("--verbose", action="store_true",
                   help="print full tracebacks on failure")
    args = p.parse_args()

    print(f"{BOLD}miniclosedai-voice — end-to-end smoke test{X}")
    print(f"{D}target: {args.url}{X}")

    s = Suite(verbose=args.verbose, strict=args.strict)

    stage_imports(s)
    if s.failed:
        print(f"\n{R}{BOLD}Aborting: imports failed — fix setup.sh first.{X}")
        return s.summary()

    stage_health(s, args.url)
    if s.failed:
        print(f"\n{R}{BOLD}Aborting: server not reachable. Run ./start.sh -d first.{X}")
        return s.summary()

    stage_voices(s, args.url)
    stage_studio(s, args.url)
    wav = stage_tts(s, args.url)
    stage_tts_stream(s, args.url)
    if not args.skip_roundtrip:
        stage_asr_roundtrip(s, args.url, wav)
    if not args.skip_call:
        stage_call_mode(s, args.url)

    return s.summary()


if __name__ == "__main__":
    sys.exit(main())

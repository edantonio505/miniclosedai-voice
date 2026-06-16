"""asr.py — HuggingFace Whisper wrapper for the MiniClosedAI voice service.

Replaces the earlier faster-whisper (CTranslate2) backend. Reasons for the
switch:

  * CTranslate2's PyPI wheel is CPU-only on aarch64, so getting GPU Whisper
    on Grace Blackwell / Jetson required a multi-stage Dockerfile that
    compiled CT2 from source (~15-minute first build).
  * PyTorch + transformers has aarch64+CUDA wheels available straight from
    `pip install`, so the same image builds in a few minutes on both arches
    with no source-compile gymnastics.
  * Matches the proven-working pattern in ``BCP_stuff/chatterbox_fastrtc.py``
    (HF pipeline + Whisper-medium.en + fp16 on CUDA).

Same two-method surface as before so server.py / call.py don't change:

  * ``transcribe(audio_bytes, language=None)`` — for any container the
    browser uploads (WebM/Opus, OGG/Opus, MP4, WAV, …). Decodes via the
    ``ffmpeg`` subprocess (more forgiving than PyAV for MediaRecorder's
    fragmented WebM output), then runs Whisper on the resulting float32 PCM.
  * ``transcribe_array(pcm, sample_rate, language=None)`` — used by the
    FastRTC call handler, which already has raw frames in memory. Resamples
    to 16 kHz if needed and skips the ffmpeg detour entirely.

Model size is picked via ``VOICE_ASR_MODEL`` (passed by server.py):

    tiny.en   — 39M params, fast, English-only
    small.en  — 244M params, decent on CPU
    medium.en — 769M params, GPU-recommended, the call-mode default
    large-v3  — 1.5B params, multilingual, GPU strongly recommended
    large-v3-turbo — distilled large-v3, ~5x faster, slight accuracy loss

Bare names like ``small`` are accepted too — they're auto-suffixed with
``.en`` for English (most voice-call traffic). Pass any HF model id with a
slash (e.g. ``distil-whisper/distil-medium.en``) to use it verbatim.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

import numpy as np
import torch
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# Whisper hallucinations that show up on near-silence / low-SNR audio. Same
# list chatterbox uses, plus a few we've hit ("RRRRRR" etc. are content-based
# repetitions caught by `_collapse_repeats` instead).
# ---------------------------------------------------------------------------
_WHISPER_HALLUCINATIONS = frozenset({
    "thank you.", "thanks for watching.", "thanks for watching!",
    "you", "thank you", "bye.", "bye!", "bye-bye.", "bye-bye!",
    ".", "!", "?", "...", "okay.", "okay!", "ok.", "ok!",
    "please subscribe.", "subscribe.", "like and subscribe.",
})


def _ffmpeg_decode_to_pcm(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode any ffmpeg-supported container to mono float32 PCM at target_sr.

    Robust against the "no-duration WebM" blobs MediaRecorder produces, which
    PyAV's matroska demuxer refuses to parse.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(target_sr),
            "pipe:1",
        ],
        input=audio_bytes, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[:400]
        raise RuntimeError(f"ffmpeg failed to decode the audio payload: {stderr}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _resolve_model_id(model_name: str) -> str:
    """Map the env-var-friendly short name to a full HF model id.

    ``medium.en`` → ``openai/whisper-medium.en``
    ``large-v3``  → ``openai/whisper-large-v3``
    ``distil-whisper/distil-medium.en`` → passthrough (has a slash already).
    """
    if "/" in model_name:
        return model_name
    return f"openai/whisper-{model_name}"


def _resolve_device(device: str) -> tuple[str, torch.dtype]:
    """Pick (device_str, torch_dtype). ``device`` may be 'auto' / 'cuda' / 'cpu'."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda", torch.float16
    return "cpu", torch.float32


def _normalize_pcm(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """int16/float32 PCM at any sample rate → mono float32 in [-1, 1] at 16 kHz."""
    if pcm.dtype == np.int16:
        audio = pcm.astype(np.float32) / 32768.0
    elif pcm.dtype == np.float32:
        audio = pcm
    else:
        audio = pcm.astype(np.float32)
        m = float(np.max(np.abs(audio)) or 1.0)
        if m > 1.5:
            audio = audio / m
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != 16000:
        # Linear interp is good enough for Whisper; the quality gap vs
        # librosa.resample is below Whisper's own noise floor on speech.
        ratio = 16000 / float(sample_rate)
        new_len = int(round(len(audio) * ratio))
        if new_len > 0:
            x_old = np.linspace(0, 1, num=len(audio), endpoint=False, dtype=np.float32)
            x_new = np.linspace(0, 1, num=new_len, endpoint=False, dtype=np.float32)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


def _clean_hallucination(text: str) -> str:
    """Filter known low-signal Whisper noise outputs."""
    if text.lower().strip() in _WHISPER_HALLUCINATIONS:
        return ""
    return text


class ASR:
    def __init__(self, model_name: str = "medium.en", device: str = "auto") -> None:
        device_str, dtype = _resolve_device(device)
        self.model_id = _resolve_model_id(model_name)
        self.device = device_str
        self.dtype = dtype
        # HF Whisper-large.* don't accept 'task'/'language' as english-only
        # constraints when they were trained on .en data. We store both fields
        # for the pipeline call below.
        self._is_english_only = self.model_id.endswith(".en")

        # Build the HF pipeline. `device=0` selects the first CUDA device when
        # device_str == 'cuda'; -1 keeps it on CPU. `torch_dtype=float16` on GPU
        # is the standard speedup over fp32 with negligible accuracy loss on
        # speech.
        self.pipe = hf_pipeline(
            task="automatic-speech-recognition",
            model=self.model_id,
            device=0 if device_str == "cuda" else -1,
            torch_dtype=dtype,
        )

    # ---- public API ----------------------------------------------------

    def transcribe(self, audio: bytes, language: str | None = None) -> dict[str, Any]:
        """Decode (via ffmpeg) + transcribe. Returns the legacy shape:
        ``{"text", "language", "segments"}``."""
        pcm = _ffmpeg_decode_to_pcm(audio, target_sr=16000)
        return self._run(pcm, language=language, want_segments=True)

    def transcribe_array(
        self,
        pcm,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        """Transcribe raw PCM already in memory (FastRTC call path).
        Returns the joined text string."""
        audio = _normalize_pcm(pcm, sample_rate)
        # Guard against tiny garbage clips — Whisper hallucinates on <300ms.
        if len(audio) < 16000 * 0.3:
            return ""
        out = self._run(audio, language=language, want_segments=False)
        return out["text"]

    # ---- internals -----------------------------------------------------

    def _run(self, audio_f32: np.ndarray, language: str | None, want_segments: bool) -> dict[str, Any]:
        generate_kwargs: dict[str, Any] = {}
        # English-only checkpoints don't take a language flag (and complain
        # if you pass one). Multilingual ones do.
        if not self._is_english_only and language:
            generate_kwargs["language"] = language

        result = self.pipe(
            {"sampling_rate": 16000, "raw": audio_f32},
            return_timestamps=want_segments,
            chunk_length_s=30,
            generate_kwargs=generate_kwargs or None,
        )
        text = (result.get("text") or "").strip()
        text = _clean_hallucination(text)
        if not want_segments:
            return {"text": text}
        chunks = result.get("chunks") or []
        return {
            "text": text,
            "language": language or ("en" if self._is_english_only else None),
            "segments": [
                {
                    "start": (c.get("timestamp") or (0, 0))[0],
                    "end": (c.get("timestamp") or (0, 0))[1],
                    "text": c.get("text", ""),
                }
                for c in chunks
            ],
        }

"""tts.py — Chatterbox TTS wrapper for the MiniClosedAI voice service.

Replaces the earlier Piper wrapper. Chatterbox gives much higher voice
quality at the cost of larger model + slower first-chunk latency. Same
two-method surface so server.py / call.py don't change:

    voices()           — static catalog (the `/voices` payload)
    synthesize_stream  — generator yielding (pcm16_chunk_bytes, sample_rate)

Chatterbox's generate() returns the full waveform at once (no native
streaming). For sentence-streamed playback we chunk the resulting tensor
into ~100ms pieces so call.py can pipe each chunk into FastRTC immediately
without buffering the whole sentence on the server.

Voice cloning: pass a path to a 5-10s reference WAV via the `voice_id` arg.
The catalog ships a couple of bundled references; users can drop additional
WAVs into the voices volume (mounted at /voices in the container) and pick
them by filename.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch


# Static catalog: voices the v1 ships. `id` is either a bundled reference
# WAV's filename (without extension) under VOICE_VOICES_DIR or the literal
# string "default" to use Chatterbox's built-in voice (no audio prompt).
VOICE_CATALOG: dict[str, list[dict]] = {
    "en": [
        {"id": "default",     "name": "Default (Chatterbox built-in)", "gender": "F"},
        # Add more by dropping `<id>.wav` into /voices and listing here.
    ],
    "es": [
        {"id": "default",     "name": "Default (Chatterbox built-in)", "gender": "F"},
    ],
}

# Chatterbox outputs 24 kHz mono float32 in [-1, 1]. We convert to int16
# (the format FastRTC and Piper both expected) so server.py / call.py
# don't need to know which TTS engine is behind the wrapper.
CHATTERBOX_SR = 24_000

# How many samples per yielded chunk. 100ms at 24 kHz = 2400 samples — small
# enough for snappy first-chunk delivery, large enough that we don't drown
# the WebRTC audio queue with tiny packets.
_CHUNK_SAMPLES = 2400


class TTS:
    def __init__(self, voices_dir: Path, use_cuda: bool = False) -> None:
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        self._model = None  # lazy — first generate() triggers ~3GB HF download

    @staticmethod
    def voices() -> dict[str, list[dict]]:
        return VOICE_CATALOG

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        # Late import so server startup doesn't pull torch + transformers on
        # the import path. The first synth call pays the model-load cost.
        from chatterbox.tts import ChatterboxTTS
        self._model = ChatterboxTTS.from_pretrained(device=self.device)
        return self._model

    def _resolve_voice_prompt(self, voice_id: str) -> str | None:
        """Map a voice id to a reference WAV path, or None for the default voice."""
        if not voice_id or voice_id == "default":
            return None
        # Look for `<voice_id>.wav` under the voices dir. If missing, fall
        # back to the default voice rather than crashing the call.
        candidate = self.voices_dir / f"{voice_id}.wav"
        return str(candidate) if candidate.exists() else None

    def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        language: str | None = None,
        speed: float | None = None,
    ) -> Iterator[tuple[bytes, int]]:
        """Yield (pcm16_chunk_bytes, sample_rate) tuples for the given text.

        Chatterbox synthesizes the whole sentence in one shot, so the
        "stream" here is the output tensor sliced into 100ms chunks. The
        first chunk lands ~500-1500 ms after the call (model inference time
        on GPU); subsequent chunks are immediate.
        """
        text = (text or "").strip()
        if not text:
            return

        model = self._ensure_model()
        prompt = self._resolve_voice_prompt(voice_id)

        kwargs = {}
        if prompt is not None:
            kwargs["audio_prompt_path"] = prompt

        wav = model.generate(text, **kwargs)
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().squeeze().to("cpu").float().numpy()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)

        # float32 [-1,1] → int16
        pcm_i16 = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)

        for start in range(0, len(pcm_i16), _CHUNK_SAMPLES):
            chunk = pcm_i16[start:start + _CHUNK_SAMPLES].tobytes()
            if chunk:
                yield chunk, CHATTERBOX_SR

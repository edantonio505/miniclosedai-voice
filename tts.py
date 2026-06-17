"""tts.py — Chatterbox Turbo TTS wrapper (streaming, fast-first-chunk).

The standard `ChatterboxTTS.generate()` runs the S3Gen vocoder for 1000
diffusion steps and emits the whole waveform at once — ~50 s per sentence.
That's wrong for a "call mode" pipeline.

The proven-working `BCP_stuff/tts_server.py` runs **ChatterboxTurboTTS** in
token-streaming mode with several speed knobs that together give a usable
first-chunk-in-300 ms profile:

  * ``N_CFM_STEPS = 4``  — vocoder runs 4 diffusion steps instead of 1000
                          (250× fewer ops, marginal quality cost)
  * ``t3.half()``        — fp16 weights on the transformer (~2× speedup)
  * ``inference_mode``   — no autograd, no gradient bookkeeping
  * ``TF32 matmul``      — tensor-core fast paths on Ampere+
  * Streaming token-by-token from T3, decoding ~75 tokens per chunk via S3Gen
    so the first audio frame leaves the server while the rest is still
    being generated
  * KV cache reuse so each decode step is incremental, not from-scratch

We port that pipeline here so server.py / call.py can keep the same surface:

    voices()           — static catalog (the `/voices` payload)
    synthesize_stream  — generator yielding (pcm16_chunk_bytes, sample_rate)

The CHATTERBOX_SR output rate is 22050 Hz — server.py and call.py already
handle arbitrary sample rates per-chunk (each yielded tuple carries its own
SR), so no caller changes are needed.

Voice selection: voice_id maps to a reference WAV under VOICE_VOICES_DIR. The
shipped default is `voices/default.wav` (bundled or symlinked from a 5–10 s
clean speech sample). Drop additional WAVs there to add voices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F


# Static catalog — the `/voices` payload. Each `id` MUST match a `<id>.wav`
# under VOICE_VOICES_DIR. `default` is the fallback when no match.
VOICE_CATALOG: dict[str, list[dict]] = {
    "en": [
        {"id": "default",  "name": "Default voice",  "gender": "F"},
    ],
    "es": [
        {"id": "default",  "name": "Default voice",  "gender": "F"},
    ],
}

# Chatterbox always emits 22050 Hz mono int16 audio.
CHATTERBOX_SR = 22_050

# How many speech tokens to accumulate before running the S3Gen vocoder and
# emitting an audio chunk. Lower = snappier first-chunk + smaller packets;
# higher = better intonation continuity inside one chunk. 75 matches the
# tts_server.py reference.
_CHUNK_TOKENS = 75

# Diffusion steps for the S3Gen vocoder per chunk. 4 is the speed/quality
# knee — ChatterboxTurbo's default is 1000, which is unusable for streaming.
# Bump to 6-8 if voice quality matters more than first-chunk latency.
_N_CFM_STEPS = 4

# Sampling params (taken from tts_server.py — known to give natural prosody).
_TEMPERATURE  = 0.9
_TOP_K        = 1000
_TOP_P        = 0.95
_REP_PENALTY  = 1.1
_EXAGGERATION = 0.7   # how much emotion the voice reference imposes


class TTS:
    """Chatterbox Turbo TTS, ported from tts_server.py's streaming pattern."""

    def __init__(self, voices_dir: Path, use_cuda: bool = False) -> None:
        # Tensor cores for the matmul-heavy T3 transformer.
        torch.set_float32_matmul_precision("high")

        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"

        # Late-bind chatterbox so import errors surface only on first use,
        # keeping `import tts` cheap when only voices() is needed.
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)

        # fp16 transformer for ~2× decode throughput on Ampere+.
        if self.device == "cuda":
            self._model.t3.half()

        # Lazy import of the silence token + utility used in the streaming
        # loop; isolated from __init__ so a stale model package surfaces clearly.
        from chatterbox.tts_turbo import punc_norm
        from chatterbox.models.s3gen.const import S3GEN_SIL
        self._punc_norm = punc_norm
        self._S3GEN_SIL = S3GEN_SIL

        self._current_voice = ""
        # If a default voice WAV is on disk, pre-warm with it so the first
        # synth call doesn't pay the prepare_conditionals cost.
        default_wav = self._wav_for("default")
        if default_wav is not None:
            self._switch_to(default_wav, "default")
            self._warmup()

    # ── public API ──────────────────────────────────────────────────────

    @staticmethod
    def voices() -> dict[str, list[dict]]:
        return VOICE_CATALOG

    def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        language: str | None = None,
        speed: float | None = None,
    ) -> Iterator[tuple[bytes, int]]:
        """Yield (pcm16_chunk_bytes, 22050) tuples as audio is generated.

        First chunk arrives in ~300 ms on GPU; subsequent chunks every
        ~150 ms while the LLM tokens are still streaming in. `speed` is
        ignored (Turbo's token rate is fixed; use length_scale post-hoc
        if you need pitch-preserving speedup).
        """
        text = (text or "").strip()
        if not text:
            return

        # Switch voice reference if needed.
        wav = self._wav_for(voice_id) or self._wav_for("default")
        if wav is None:
            raise RuntimeError(
                f"No voice WAV found for {voice_id!r} or 'default' under {self.voices_dir}/"
            )
        if voice_id != self._current_voice:
            self._switch_to(wav, voice_id)

        for chunk_f32 in self._stream_chunks_f32(text):
            pcm_i16 = np.clip(chunk_f32 * 32767.0, -32768, 32767).astype(np.int16)
            yield pcm_i16.tobytes(), CHATTERBOX_SR

    # ── internals ───────────────────────────────────────────────────────

    def _wav_for(self, voice_id: str) -> str | None:
        for ext in ("wav", "WAV", "flac"):
            p = self.voices_dir / f"{voice_id}.{ext}"
            if p.exists():
                return str(p)
        return None

    def _switch_to(self, wav_path: str, voice_id: str) -> None:
        self._model.prepare_conditionals(wav_path, exaggeration=_EXAGGERATION)
        if self.device == "cuda" and self._model.conds:
            self._model.conds.t3.speaker_emb = self._model.conds.t3.speaker_emb.half()
            self._model.conds.t3.emotion_adv = self._model.conds.t3.emotion_adv.half()
        self._current_voice = voice_id

    def _warmup(self) -> None:
        """Run one transformer forward to compile / cache CUDA graphs so the
        first real synth doesn't pay the JIT cost."""
        with torch.inference_mode():
            t3 = self._model.t3
            tok = self._model.tokenizer(
                self._punc_norm("Hello."),
                return_tensors="pt", padding=True, truncation=True,
            ).input_ids.to(self.device)
            start = t3.hp.start_speech_token * torch.ones_like(tok[:, :1])
            embeds, _ = t3.prepare_input_embeds(
                t3_cond=self._model.conds.t3,
                text_tokens=tok, speech_tokens=start, cfg_weight=0.0,
            )
            t3.tfmr(inputs_embeds=embeds, use_cache=True)

    def _stream_chunks_f32(self, text: str) -> Iterator[np.ndarray]:
        """Core streaming loop from BCP_stuff/tts_server.py:_stream_chunks.

        Yields float32 mono waveforms at CHATTERBOX_SR — the caller converts
        to int16 PCM bytes.
        """
        from transformers import (
            LogitsProcessorList, TemperatureLogitsWarper,
            TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor,
        )

        t3 = self._model.t3
        s3gen = self._model.s3gen
        conds = self._model.conds

        with torch.inference_mode():
            text_tokens = self._model.tokenizer(
                self._punc_norm(text),
                return_tensors="pt", padding=True, truncation=True,
            ).input_ids.to(self.device)

            logits_proc = LogitsProcessorList([
                TemperatureLogitsWarper(_TEMPERATURE),
                TopKLogitsWarper(_TOP_K),
                TopPLogitsWarper(_TOP_P),
                RepetitionPenaltyLogitsProcessor(_REP_PENALTY),
            ])

            speech_start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
            embeds, _ = t3.prepare_input_embeds(
                t3_cond=conds.t3, text_tokens=text_tokens,
                speech_tokens=speech_start, cfg_weight=0.0,
            )
            out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
            past_kv = out.past_key_values
            logits = t3.speech_head(out[0][:, -1:])
            processed = logits_proc(speech_start, logits[:, -1, :])
            cur_tok = torch.multinomial(F.softmax(processed, dim=-1), num_samples=1)
            all_toks = [cur_tok]
            pending: list[torch.Tensor] = []

            for _ in range(1000):  # hard cap so a bad sample can't run forever
                stop = bool(torch.all(cur_tok == t3.hp.stop_speech_token))
                val = int(cur_tok[0, 0].item())
                if val < 6561 and not stop:
                    pending.append(cur_tok[0])

                if (len(pending) >= _CHUNK_TOKENS or stop) and pending:
                    chunk = torch.cat(pending).to(self.device)
                    if stop:
                        sil = torch.tensor(
                            [self._S3GEN_SIL] * 3, dtype=torch.long, device=self.device
                        )
                        chunk = torch.cat([chunk, sil])
                    wav, _ = s3gen.inference(
                        speech_tokens=chunk, ref_dict=conds.gen,
                        n_cfm_timesteps=_N_CFM_STEPS,
                    )
                    audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    yield audio
                    pending = []

                if stop:
                    break

                embed = t3.speech_emb(cur_tok)
                out = t3.tfmr(inputs_embeds=embed, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                input_ids = torch.cat(all_toks[-100:], dim=1)
                logits = t3.speech_head(out[0])
                processed = logits_proc(input_ids, logits[:, -1, :])
                if torch.all(processed == -float("inf")):
                    break
                cur_tok = torch.multinomial(F.softmax(processed, dim=-1), num_samples=1)
                all_toks.append(cur_tok)

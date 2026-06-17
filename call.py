"""call.py — FastRTC handler for the "call the bot" mode.

A long-running WebRTC peer connection with VAD-based turn detection. On each
detected pause we transcribe the buffered audio, POST it to MiniClosedAI's
existing chat-streaming endpoint, and feed the reply to Piper — **one
sentence at a time** so the user starts hearing audio while the rest of the
reply is still generating.

Pipeline per turn:

    audio (since the last pause)  →  faster-whisper ASR                →  text
                                                                          ↓
    text  →  POST /api/conversations/{id}/chat/stream  (to MiniClosedAI)
                                                                          ↓
    chunked reply  →  sentence splitter  →  Piper TTS (per sentence)   →  audio frames
                          (text events ride in parallel via DataChannel queue)

Side-channel events (each `AdditionalOutputs({...})`) the browser renders:
  {status: 'transcribing'|'thinking'|'speaking'|'listening'}  — pipeline stage
  {transcript: "..."}                                          — user's text
  {chunk: "..."}                                               — assistant token
  {end: true}                                                  — turn finished
  {error: "..."}                                               — pipeline failure

FastRTC's ReplyOnPause class already handles barge-in: when the user starts
speaking during a yielded TTS stream, the handler's generator is cancelled
and the next utterance starts a fresh turn.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx
import numpy as np
from fastrtc import AdditionalOutputs, ReplyOnPause


# -- Sentence splitter -----------------------------------------------------

# Common abbreviations whose trailing dot must NOT be treated as a sentence
# boundary. Lower-cased; the lookup also lower-cases the candidate.
_ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "u.s.", "u.k.", "u.n.", "inc.", "ltd.", "co.", "no.",
    "fig.", "vol.", "ed.", "ch.", "pg.", "p.", "pp.", "approx.", "min.",
    "max.", "avg.", "incl.", "excl.", "rev.", "est.",
})

# Sentence terminator followed by whitespace + a likely next-sentence cue
# (uppercase letter / opening quote / paren / digit). Lookahead so we don't
# consume the next sentence's first character.
_SPLIT_PAT = re.compile(r"[.!?][\"')\]]?(\s+)(?=[A-Z\"'(0-9])")
# Soft boundary: terminator followed by a newline (LLM finished a paragraph).
_NEWLINE_PAT = re.compile(r"[.!?][\"')\]]?\n+")
# Trailing-word check to skip false positives like "Mr. Smith".
_LAST_WORD_PAT = re.compile(r"\S+$")


# ---------------------------------------------------------------------------
# TTS text cleaner — strip markdown, list markers, emojis, and other glyphs
# the LLM emits that Chatterbox (or any TTS) would read literally as
# "asterisk", "hash", "tilde", etc.
# Mirrors BCP_stuff/chatterbox_fastrtc.py:_clean_for_tts.
# ---------------------------------------------------------------------------
_CLEAN_BOLD_ITALIC = re.compile(r"\*{1,3}([^*]+?)\*{1,3}")    # **bold** → bold
_CLEAN_HEADER     = re.compile(r"^#{1,6}\s*", re.MULTILINE)   # ## Heading → Heading
_CLEAN_BULLET     = re.compile(r"^\s*[-*•·]\s+", re.MULTILINE)  # - item / * item
_CLEAN_LETTER_BULLET = re.compile(r"^\s*[A-Za-z]\)\s+", re.MULTILINE)  # A) item
_CLEAN_DIGIT_BULLET  = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)     # 1. item / 2) item
_CLEAN_BACKTICKS  = re.compile(r"`+")                          # `code` → code
_CLEAN_PIPES      = re.compile(r"\|")                          # | → drop (markdown tables)
_CLEAN_BRACKETS   = re.compile(r"[\[\]]")                      # [link](url) → link
_CLEAN_LINK_URL   = re.compile(r"\(https?://[^\s)]+\)")        # the (url) half of [text](url)
_CLEAN_TILDE_HEAVY = re.compile(r"~{1,3}([^~]+?)~{1,3}")        # ~~strike~~ → strike
_CLEAN_ANGLE_TAGS = re.compile(r"<[^>]+>")                     # <think>...</think> → drop
_CLEAN_NON_ASCII  = re.compile(r"[^\x00-\x7F]+")                # emojis, em-dashes, smart quotes
_CLEAN_MANY_BLANKS = re.compile(r"\n{2,}")
_CLEAN_WS         = re.compile(r"\s+")


def clean_for_tts(text: str) -> str:
    """Strip markdown / punctuation / emojis the TTS would mispronounce.

    Conservative: keeps normal sentence punctuation (.,!?:'"-) so prosody
    survives. Drops everything the user reported hearing literally
    (asterisks, hashes, backticks, bullets, code fences, HTML-ish tags).
    """
    if not text:
        return ""
    t = text
    # ``` code blocks ``` — drop entirely; speech of code is rarely useful.
    t = re.sub(r"```[\s\S]*?```", " ", t)
    # Strip markdown emphasis WHILE keeping the inner text.
    t = _CLEAN_BOLD_ITALIC.sub(r"\1", t)
    t = _CLEAN_TILDE_HEAVY.sub(r"\1", t)
    # Remove header / list markers (leave the content).
    t = _CLEAN_HEADER.sub("", t)
    t = _CLEAN_BULLET.sub("", t)
    t = _CLEAN_LETTER_BULLET.sub("", t)
    t = _CLEAN_DIGIT_BULLET.sub("", t)
    # Remove inline code marker but keep the word inside.
    t = _CLEAN_BACKTICKS.sub("", t)
    # Drop markdown link URL halves; keep the visible text.
    t = _CLEAN_LINK_URL.sub("", t)
    t = _CLEAN_BRACKETS.sub("", t)
    # Pipes, angle-bracket tags, emojis / non-ASCII glyphs.
    t = _CLEAN_PIPES.sub(" ", t)
    t = _CLEAN_ANGLE_TAGS.sub("", t)
    t = _CLEAN_NON_ASCII.sub("", t)
    # Collapse whitespace.
    t = _CLEAN_MANY_BLANKS.sub(" ", t)
    t = _CLEAN_WS.sub(" ", t)
    return t.strip()


def _next_sentence(buf: str, *, min_len: int = 20, force_max: int = 240) -> tuple[str | None, str]:
    """Pop the next complete sentence off the front of `buf`.

    Returns `(sentence, remaining)` — sentence is `None` if no complete one
    yet (and `remaining` == `buf`). Skips false positives like "Mr. Smith" or
    "e.g." by checking the trailing word, and refuses to split anything
    shorter than `min_len` characters. As a failsafe, force-splits at the
    last comma / space if `buf` reaches `force_max` chars with no terminator.
    """
    for m in _SPLIT_PAT.finditer(buf):
        end = m.end() - len(m.group(1))
        candidate = buf[:end].rstrip()
        if len(candidate) < min_len:
            continue
        lw = _LAST_WORD_PAT.search(candidate)
        if lw and lw.group().lower() in _ABBREVIATIONS:
            continue
        return candidate, buf[end:].lstrip()

    nl = _NEWLINE_PAT.search(buf)
    if nl and len((c := buf[:nl.end()].rstrip())) >= min_len:
        return c, buf[nl.end():]

    if len(buf) >= force_max:
        head = buf[:force_max]
        comma_idx = head.rfind(", ")
        if comma_idx >= min_len:
            return head[:comma_idx + 1], buf[comma_idx + 1:].lstrip()
        space_idx = head.rfind(" ")
        if space_idx >= min_len:
            return head[:space_idx], buf[space_idx:].lstrip()

    return None, buf

log = logging.getLogger("voice.call")


class BotCallHandler:
    """Per-call state + the async generator FastRTC calls on each pause.

    `asr` and `tts` are the module-level singletons from server.py; we don't
    own them. `conv_id`, `miniclosedai_url`, `voice_id`, `language` are the
    per-call config the browser sends with the offer.
    """

    def __init__(
        self,
        asr,
        tts,
        conv_id: int,
        miniclosedai_url: str,
        voice_id: str,
        language: str,
    ) -> None:
        self.asr = asr
        self.tts = tts
        self.conv_id = conv_id
        self.miniclosedai_url = miniclosedai_url.rstrip("/")
        self.voice_id = voice_id
        self.language = language

    async def _synthesize(self, text: str):
        """Synthesize one sentence in a worker thread; return all PCM chunks.

        Strips markdown / emojis / list bullets first — these read as
        "asterisk", "hash", etc. when handed to the TTS verbatim. Collecting
        per-sentence keeps the main async loop responsive; chunks then yield
        back into the FastRTC media track in order. Sentences are short
        enough (<240 chars) that the thread call returns in 150-400 ms.
        """
        text = clean_for_tts(text)
        if not text:
            return []
        def _run():
            out: list[tuple[bytes, int]] = []
            for pcm_chunk, sr in self.tts.synthesize_stream(
                text, self.voice_id, self.language,
            ):
                if pcm_chunk:
                    out.append((pcm_chunk, sr))
            return out
        return await asyncio.to_thread(_run)

    async def respond(self, audio):
        """Run one turn: ASR → chat (streamed) → sentence-by-sentence TTS.

        Each pipeline transition emits a `{status: ...}` event so the UI can
        show "transcribing / thinking / speaking / listening" feedback. TTS
        starts on the first complete sentence the LLM yields, so the user
        hears audio while later sentences are still being generated.
        """
        import time
        t_turn_start = time.perf_counter()
        timings: dict[str, float] = {}  # filled as each stage completes

        sample_rate, pcm = audio
        if hasattr(pcm, "flatten"):
            pcm = pcm.flatten()

        # diagnostic — what did silero hand us?
        n_samples = int(pcm.shape[-1]) if hasattr(pcm, "shape") else len(pcm)
        try:
            import numpy as _np
            peak = float(_np.abs(pcm.astype(_np.float32)).max())
            rms  = float(_np.sqrt(_np.mean(pcm.astype(_np.float32) ** 2)))
        except Exception:
            peak = rms = -1.0
        log.info(
            "conv=%s [turn-start] sr=%d samples=%d duration=%.2fs peak=%.0f rms=%.0f",
            self.conv_id, int(sample_rate), n_samples,
            n_samples / max(int(sample_rate), 1), peak, rms,
        )

        # 1. ASR
        yield AdditionalOutputs({"status": "transcribing"})
        t_asr = time.perf_counter()
        try:
            text = await asyncio.to_thread(
                self.asr.transcribe_array, pcm, int(sample_rate), self.language or None,
            )
        except Exception as e:
            log.exception("asr failed")
            yield AdditionalOutputs({"error": f"asr: {e}", "status": "listening"})
            return
        timings["asr_ms"] = (time.perf_counter() - t_asr) * 1000
        text = (text or "").strip()
        log.info("conv=%s [asr] %.0f ms  → %r", self.conv_id, timings["asr_ms"], text[:80])
        if not text:
            yield AdditionalOutputs({"status": "listening"})
            return

        yield AdditionalOutputs({"transcript": text})

        # 2. Stream chat reply; flush completed sentences to TTS as they form.
        yield AdditionalOutputs({"status": "thinking"})
        url = f"{self.miniclosedai_url}/api/conversations/{self.conv_id}/chat/stream"
        buffer = ""                          # accumulator for sentence detection
        spoke_anything = False               # whether any sentence has been synthesized
        full_reply: list[str] = []           # for logging / "the bot said X"
        t_llm_post = time.perf_counter()     # when we POSTed to chat/stream
        t_llm_first_tok: float | None = None
        t_first_sentence: float | None = None
        t_first_tts_chunk: float | None = None
        n_sentences = 0
        # The producer runs the LLM SSE reader + per-sentence TTS as a SEPARATE
        # task, and pushes events into an asyncio.Queue. This generator just
        # drains the queue and yields. Without this split the SSE socket isn't
        # read while await self._synthesize is running (~500-700 ms), so LLM
        # tokens pile up in the network buffer and then flush as a burst —
        # exactly the "text appears in stutters while audio plays smoothly"
        # symptom. Now text events forward as fast as the LLM emits them.
        out_q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async def _producer():
            nonlocal buffer, full_reply, n_sentences
            nonlocal t_llm_first_tok, t_first_sentence, t_first_tts_chunk
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(300.0, connect=10.0), verify=False,
                ) as client:
                    async with client.stream(
                        "POST", url,
                        # persist=False is the call-mode latency win — the
                        # persist=True path in MiniClosedAI launches a background
                        # task whose SSE buffer adds ~1.7s to TTFT. A call is
                        # ephemeral; we don't need refresh-resilient behavior.
                        json={"message": text, "persist": False, "include_history": True, "voice_mode": True},
                        headers={"Accept": "text/event-stream"},
                    ) as resp:
                        if resp.status_code >= 400:
                            detail = (await resp.aread()).decode(errors="replace")[:200]
                            msg = f"chat HTTP {resp.status_code}: {detail}"
                            log.warning("conv=%s %s", self.conv_id, msg)
                            await out_q.put(("err", msg))
                            return
                        sse_buf = ""
                        chat_done = False
                        async for raw in resp.aiter_text():
                            sse_buf += raw
                            parts = sse_buf.split("\n\n")
                            sse_buf = parts.pop()
                            for part in parts:
                                line = part.strip()
                                if not line.startswith("data:"):
                                    continue
                                try:
                                    ev = json.loads(line[5:].strip())
                                except json.JSONDecodeError:
                                    continue
                                if "error" in ev:
                                    await out_q.put(("err", ev["error"]))
                                    return
                                if "chunk" in ev:
                                    piece = ev["chunk"]
                                    full_reply.append(piece)
                                    buffer += piece
                                    if t_llm_first_tok is None:
                                        t_llm_first_tok = time.perf_counter()
                                        timings["llm_ttft_ms"] = (t_llm_first_tok - t_llm_post) * 1000
                                        log.info(
                                            "conv=%s [llm] first token in %.0f ms",
                                            self.conv_id, timings["llm_ttft_ms"],
                                        )
                                    # Forward the text IMMEDIATELY — the bubble
                                    # in the UI streams without waiting on TTS.
                                    await out_q.put(("text", piece))
                                    # Pump every complete sentence to TTS; the
                                    # synth blocks this producer for ~700 ms each
                                    # but the consumer keeps draining text events
                                    # the producer already queued.
                                    while True:
                                        sentence, buffer = _next_sentence(buffer)
                                        if not sentence:
                                            break
                                        n_sentences += 1
                                        if t_first_sentence is None:
                                            t_first_sentence = time.perf_counter()
                                            timings["first_sentence_ms"] = (t_first_sentence - t_llm_post) * 1000
                                            log.info(
                                                "conv=%s [sentence #%d] %.0f ms after LLM POST → %r",
                                                self.conv_id, n_sentences,
                                                timings["first_sentence_ms"], sentence[:80],
                                            )
                                        await out_q.put(("speaking", None))
                                        t_tts = time.perf_counter()
                                        try:
                                            chunks = await self._synthesize(sentence)
                                            if t_first_tts_chunk is None and chunks:
                                                t_first_tts_chunk = time.perf_counter()
                                                timings["tts_first_chunk_ms"] = (t_first_tts_chunk - t_tts) * 1000
                                                log.info(
                                                    "conv=%s [tts #%d] first chunk in %.0f ms",
                                                    self.conv_id, n_sentences,
                                                    timings["tts_first_chunk_ms"],
                                                )
                                            for pcm_chunk, out_sr in chunks:
                                                await out_q.put(("audio", (out_sr, pcm_chunk)))
                                        except Exception as e:
                                            log.exception("tts (mid-stream) failed: %r", e)
                                            await out_q.put(("err", f"tts: {e}"))
                                if ev.get("end"):
                                    chat_done = True
                                    break
                            if chat_done:
                                break
                # Trailing flush — the last sentence often has no terminator +
                # lookahead, so it stays in the buffer until the LLM stops.
                tail = buffer.strip()
                if tail:
                    await out_q.put(("speaking", None))
                    try:
                        for pcm_chunk, out_sr in await self._synthesize(tail):
                            await out_q.put(("audio", (out_sr, pcm_chunk)))
                    except Exception as e:
                        log.exception("tts (tail) failed: %r", e)
                        await out_q.put(("err", f"tts: {e}"))
            except Exception as e:
                log.exception("chat stream failed")
                await out_q.put(("err", f"chat: {e}"))
            finally:
                await out_q.put((SENTINEL, None))

        producer_task = asyncio.create_task(_producer())
        spoke_announced = False
        try:
            while True:
                kind, payload = await out_q.get()
                if kind is SENTINEL:
                    break
                if kind == "text":
                    yield AdditionalOutputs({"chunk": payload})
                elif kind == "speaking":
                    if not spoke_announced:
                        yield AdditionalOutputs({"status": "speaking"})
                        spoke_announced = True
                elif kind == "audio":
                    out_sr, pcm_chunk = payload
                    arr = np.frombuffer(pcm_chunk, dtype=np.int16)
                    yield (out_sr, arr.reshape(1, -1))
                elif kind == "err":
                    yield AdditionalOutputs({"error": payload})
        finally:
            if not producer_task.done():
                producer_task.cancel()

        # 4. Done — tell the UI to finalize the bubble and reset to listening.
        timings["total_ms"] = (time.perf_counter() - t_turn_start) * 1000
        reply_text = "".join(full_reply).strip()
        log.info(
            "conv=%s [turn-done] %d sentences | "
            "asr=%.0fms llm_ttft=%.0fms first_sent=%.0fms tts_first=%.0fms total=%.0fms "
            "| reply=%r",
            self.conv_id, n_sentences,
            timings.get("asr_ms", 0),
            timings.get("llm_ttft_ms", 0),
            timings.get("first_sentence_ms", 0),
            timings.get("tts_first_chunk_ms", 0),
            timings.get("total_ms", 0),
            reply_text[:100] + ("…" if len(reply_text) > 100 else ""),
        )
        yield AdditionalOutputs({"status": "listening"})
        yield AdditionalOutputs({"end": True})


def build_handler(asr, tts, conv_id, miniclosedai_url, voice_id, language) -> ReplyOnPause:
    """Construct a ReplyOnPause-wrapped instance for one call.

    FastRTC's ReplyOnPause does the VAD + buffering: it accumulates input
    audio frames, detects pauses via Silero VAD, and calls our `respond`
    generator with the buffered chunk. Barge-in support is built-in.
    """
    inst = BotCallHandler(asr, tts, conv_id, miniclosedai_url, voice_id, language)
    return ReplyOnPause(inst.respond)

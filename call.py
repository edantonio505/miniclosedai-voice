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
import uuid

import httpx
import numpy as np
from fastrtc import AdditionalOutputs, ReplyOnPause


# -- Relay mode ---------------------------------------------------------------
#
# Normally each turn POSTs to MiniClosedAI's /chat/stream — which requires this
# voice server to be able to reach MiniClosedAI. When the voice server runs
# remotely (RunPod, cloud) and MiniClosedAI sits on a private LAN, that reverse
# connection is impossible. Relay mode inverts the leg: the turn emits a
# `{turn_request: {turn_id, text}}` event on the /call/events channel (which
# MiniClosedAI is already reading — it proxies the SSE to the browser),
# MiniClosedAI runs the LLM itself, and streams the reply text back to us via
# POST /call/turn/{turn_id}. Every connection is outbound FROM MiniClosedAI,
# so any reachable voice server works — no tunnel, no public URL.
#
# `_RELAY_TURNS` maps a pending turn's id → the asyncio.Queue its reader is
# blocked on. server.py's /call/turn/{turn_id} endpoint feeds it.

_RELAY_TURNS: dict[str, asyncio.Queue] = {}

# How long the relay reader waits for the FIRST/next event before giving up —
# covers MiniClosedAI dying mid-turn or never having tapped the events stream.
_RELAY_TIMEOUT_S = 120.0


def get_relay_queue(turn_id: str) -> asyncio.Queue | None:
    return _RELAY_TURNS.get(turn_id)


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
# Terminator at the VERY END of the buffer (nothing after it yet). During
# streaming this means "sentence 1 is complete but the next token hasn't
# arrived" — we flush it now instead of waiting for the next sentence's first
# char (the _SPLIT_PAT lookahead) or for end-of-stream. This is the change that
# lets audio start the moment the first sentence finishes being written.
_END_TERM_PAT = re.compile(r"[.!?][\"')\]]?\s*$")
# Trailing-word check to skip false positives like "Mr. Smith".
_LAST_WORD_PAT = re.compile(r"\S+$")

# Minimum sentence length before we flush to TTS. The FIRST sentence of a turn
# uses a smaller floor so a short opener ("Yes.", "Sure!", "Got it.") starts
# playing immediately; later sentences use the larger floor to avoid chopping
# mid-thought. Both still honour the abbreviation / decimal guards.
_FIRST_MIN_LEN = 4
_MIN_LEN = 20


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

    # Eager end-of-buffer flush: a terminator sits at the end of the buffer with
    # nothing after it. Treat it as a complete sentence NOW rather than waiting
    # for the next token. Guard against decimals / list numerals ("3.14", "3.")
    # by refusing when the last word is purely digits, and keep the abbreviation
    # guard ("Dr.", "e.g.").
    if _END_TERM_PAT.search(buf):
        candidate = buf.rstrip()
        if len(candidate) >= min_len:
            lw = _LAST_WORD_PAT.search(candidate)
            word = lw.group().lower() if lw else ""
            core = word.rstrip(".!?\"')]")
            if word not in _ABBREVIATIONS and not core.isdigit():
                return candidate, ""

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
        miniclosedai_url: str | None,
        voice_id: str,
        language: str,
        relay: bool = False,
    ) -> None:
        self.asr = asr
        self.tts = tts
        self.conv_id = conv_id
        self.miniclosedai_url = (miniclosedai_url or "").rstrip("/")
        self.voice_id = voice_id
        self.language = language
        self.relay = relay

    async def _synthesize(self, text: str):
        """Yield (pcm_chunk_bytes, sample_rate) per TTS chunk as they form.

        Earlier this collected all chunks then returned a list — meaning the
        first audio frame couldn't leave the server until the WHOLE sentence
        had finished synthesizing (1-3 s on the GB10 PTX-JIT). Now it streams:
        Chatterbox Turbo yields one chunk every ~75 speech tokens (~250 ms),
        we push each chunk as soon as it appears, and the WebRTC track gets
        the first audible frame after just the first chunk completes.

        Strips markdown / emojis / list bullets first — these would read as
        "asterisk", "hash", etc. when handed to the TTS verbatim.
        """
        text = clean_for_tts(text)
        if not text:
            return

        # Drive the synchronous Chatterbox generator from a worker thread,
        # bridging chunks back to the async caller via a thread-safe queue.
        # This is the standard pattern for sync generators in asyncio: a
        # SENTINEL marks the end of the stream.
        chunk_q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()
        loop = asyncio.get_running_loop()

        def _run():
            try:
                for pcm_chunk, sr in self.tts.synthesize_stream(
                    text, self.voice_id, self.language,
                ):
                    if pcm_chunk:
                        # asyncio.Queue isn't thread-safe; route the put back
                        # through the loop so it lands on the right context.
                        asyncio.run_coroutine_threadsafe(
                            chunk_q.put((pcm_chunk, sr)), loop,
                        )
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    chunk_q.put(("__error__", e)), loop,
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
                    raise item[1]
                yield item
        finally:
            # If the consumer broke early, make sure the worker can finish.
            await worker

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
        # Two concurrent tasks, one output queue:
        #
        #   _llm_reader  ──reads SSE non-stop──►  text events to out_q
        #                                         sentence_q for the TTS worker
        #   _tts_worker  ──pops sentence──►  await synthesize  ──►  audio to out_q
        #
        # An earlier single-task version awaited synthesize inside the LLM
        # read loop. While that await ran (~500-700 ms per sentence), the
        # `async for raw in resp.aiter_text()` was suspended — incoming LLM
        # tokens piled up in MiniClosedAI's SSE buffer and then drained as
        # a burst when synthesize returned, producing the visible "text
        # stutters while audio plays smoothly" symptom. The two-task split
        # keeps the SSE socket continuously read regardless of TTS state,
        # so text events forward at the LLM's actual emission cadence.
        out_q: asyncio.Queue = asyncio.Queue()
        sentence_q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()
        # Relay-mode plumbing: a per-turn id + queue registered BEFORE the
        # turn_request event goes out, so the push endpoint can never race a
        # not-yet-registered turn.
        turn_id = uuid.uuid4().hex
        relay_q: asyncio.Queue = asyncio.Queue()
        if self.relay:
            _RELAY_TURNS[turn_id] = relay_q
        # Counter so the tts_worker can attach a stable index per sentence to
        # its first-chunk timing log. Producer increments; consumer reads.

        async def _ingest_piece(piece: str):
            """Handle one LLM text piece: forward to the UI immediately and
            pump completed sentences to the TTS worker. Shared by both the
            direct SSE reader and the relay reader — identical semantics."""
            nonlocal buffer, full_reply, n_sentences
            nonlocal t_llm_first_tok, t_first_sentence
            full_reply.append(piece)
            buffer += piece
            if t_llm_first_tok is None:
                t_llm_first_tok = time.perf_counter()
                timings["llm_ttft_ms"] = (t_llm_first_tok - t_llm_post) * 1000
                log.info(
                    "conv=%s [llm] first token in %.0f ms",
                    self.conv_id, timings["llm_ttft_ms"],
                )
            # Forward the text IMMEDIATELY — every token reaches the UI
            # bubble at the LLM's actual emission cadence, untouched by TTS.
            await out_q.put(("text", piece))
            # Pump complete sentences onto sentence_q; the tts_worker pops
            # them in parallel. The first sentence uses a smaller min length
            # so audio starts the instant it completes.
            while True:
                cur_min = _FIRST_MIN_LEN if n_sentences == 0 else _MIN_LEN
                sentence, buffer = _next_sentence(buffer, min_len=cur_min)
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
                await sentence_q.put((n_sentences, sentence))

        def _flush_tail_sync():
            """Queue whatever's still in the buffer (final fragment, often has
            no terminator). Returns the coroutine to await."""
            nonlocal n_sentences
            tail = buffer.strip()
            if tail:
                n_sentences += 1
                return sentence_q.put((n_sentences, tail))
            return None

        async def _llm_relay_reader():
            """Relay mode: MiniClosedAI runs the LLM and pushes the reply text
            to POST /call/turn/{turn_id}, which feeds `relay_q`. This reader
            just pops events — no outbound connection from the voice server at
            all (that's the point: works from RunPod/cloud into a LAN
            MiniClosedAI where the reverse dial is impossible)."""
            try:
                while True:
                    try:
                        ev = await asyncio.wait_for(relay_q.get(), timeout=_RELAY_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        await out_q.put((
                            "err",
                            "relay: no LLM data from MiniClosedAI within "
                            f"{int(_RELAY_TIMEOUT_S)}s — is its /call/events "
                            "stream still connected?",
                        ))
                        return
                    if not isinstance(ev, dict):
                        continue
                    if "error" in ev:
                        await out_q.put(("err", str(ev["error"])))
                        return
                    if "chunk" in ev and ev["chunk"]:
                        await _ingest_piece(str(ev["chunk"]))
                    if ev.get("end"):
                        break
                flush = _flush_tail_sync()
                if flush is not None:
                    await flush
            except Exception as e:
                log.exception("relay turn failed")
                await out_q.put(("err", f"chat: {e}"))
            finally:
                _RELAY_TURNS.pop(turn_id, None)
                # Signal end-of-sentences to the tts worker.
                await sentence_q.put((None, None))

        async def _llm_reader():
            """Read the LLM /chat/stream SSE socket continuously.

            Pushes text events directly to out_q (so the UI bubble sees them
            without TTS latency in the way) and queues complete sentences
            for the tts_worker. Never awaits synthesize itself — this is the
            critical property: as long as MiniClosedAI is sending tokens,
            this task is reading them.
            """
            nonlocal buffer, full_reply, n_sentences
            nonlocal t_llm_first_tok, t_first_sentence
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(300.0, connect=10.0), verify=False,
                ) as client:
                    async with client.stream(
                        "POST", url,
                        # persist=False is the call-mode latency win — the
                        # persist=True path in MiniClosedAI launches a
                        # background task whose SSE buffer adds ~1.7s to TTFT.
                        # Call mode persists asynchronously via the
                        # /voice/persist-call-turn endpoint at end-of-turn.
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
                                    await _ingest_piece(ev["chunk"])
                                if ev.get("end"):
                                    chat_done = True
                                    break
                            if chat_done:
                                break
                # Flush whatever's still in the buffer (final fragment, often
                # has no terminator).
                flush = _flush_tail_sync()
                if flush is not None:
                    await flush
            except Exception as e:
                log.exception("chat stream failed")
                await out_q.put(("err", f"chat: {e}"))
            finally:
                # Signal end-of-sentences to the tts worker.
                await sentence_q.put((None, None))

        async def _tts_worker():
            """Pop sentences off sentence_q, synthesize, push audio to out_q.

            Runs strictly in parallel to _llm_reader — synthesis time here
            doesn't gate the LLM SSE read at all. Each chunk is forwarded to
            out_q the moment Chatterbox emits it (no collect-then-yield), so
            the WebRTC track plays the first audio frame as soon as the first
            ~250 ms of speech has been synthesized rather than waiting for
            the whole sentence (1-3 s).
            """
            nonlocal t_first_tts_chunk
            try:
                while True:
                    idx, sentence = await sentence_q.get()
                    if idx is None:
                        break  # _llm_reader signalled end-of-sentences
                    await out_q.put(("speaking", None))
                    t_tts = time.perf_counter()
                    sentence_first_chunk_logged = False
                    try:
                        async for pcm_chunk, out_sr in self._synthesize(sentence):
                            if not sentence_first_chunk_logged:
                                sentence_first_chunk_logged = True
                                if t_first_tts_chunk is None:
                                    t_first_tts_chunk = time.perf_counter()
                                    timings["tts_first_chunk_ms"] = (t_first_tts_chunk - t_tts) * 1000
                                log.info(
                                    "conv=%s [tts #%d] first chunk in %.0f ms",
                                    self.conv_id, idx,
                                    (time.perf_counter() - t_tts) * 1000,
                                )
                            await out_q.put(("audio", (out_sr, pcm_chunk)))
                    except Exception as e:
                        log.exception("tts (sentence #%d) failed: %r", idx, e)
                        await out_q.put(("err", f"tts: {e}"))
            finally:
                # Whether we exited normally or via an exception, signal the
                # consumer that no more events are coming.
                await out_q.put((SENTINEL, None))

        if self.relay:
            # Announce the turn on the events channel — MiniClosedAI's events
            # proxy intercepts {turn_request}, runs the LLM, and streams the
            # reply back to POST /call/turn/{turn_id} (feeding relay_q above).
            yield AdditionalOutputs({"turn_request": {"turn_id": turn_id, "text": text}})
            llm_task = asyncio.create_task(_llm_relay_reader())
        else:
            llm_task = asyncio.create_task(_llm_reader())
        tts_task = asyncio.create_task(_tts_worker())
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
            # Cancel both tasks if they're still running (e.g. caller hung up).
            for t in (llm_task, tts_task):
                if not t.done():
                    t.cancel()
            # Belt-and-braces: drop the relay registration even if the reader
            # never ran (barge-in between registration and task start).
            _RELAY_TURNS.pop(turn_id, None)

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

        # 5. Fire-and-forget: write the turn into MiniClosedAI's conversation
        #    history. Decoupled from the LLM call (which uses persist=False
        #    for the latency win) so persistence costs zero TTFT but the
        #    conversation still records every turn — same shape as a normal
        #    /chat/stream turn would produce.
        # In relay mode MiniClosedAI ran the LLM itself and persists the turn
        # on its side — POSTing back would double-write (and needs the very
        # reverse connection relay mode exists to avoid).
        if text and reply_text and not self.relay:
            async def _persist():
                try:
                    persist_url = (
                        f"{self.miniclosedai_url}/api/conversations/"
                        f"{self.conv_id}/voice/persist-call-turn"
                    )
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(30.0, connect=5.0), verify=False,
                    ) as client:
                        r = await client.post(
                            persist_url,
                            json={"user": text, "assistant": reply_text},
                        )
                        if r.status_code >= 400:
                            log.warning(
                                "conv=%s persist-call-turn HTTP %d: %s",
                                self.conv_id, r.status_code,
                                r.text[:200].replace("\n", " "),
                            )
                except Exception as e:
                    log.warning("conv=%s persist-call-turn failed: %r", self.conv_id, e)
            asyncio.create_task(_persist())

        yield AdditionalOutputs({"status": "listening"})
        yield AdditionalOutputs({"end": True})


def build_handler(asr, tts, conv_id, miniclosedai_url, voice_id, language,
                  relay: bool = False) -> ReplyOnPause:
    """Construct a ReplyOnPause-wrapped instance for one call.

    FastRTC's ReplyOnPause does the VAD + buffering: it accumulates input
    audio frames, detects pauses via Silero VAD, and calls our `respond`
    generator with the buffered chunk. Barge-in support is built-in.
    """
    inst = BotCallHandler(asr, tts, conv_id, miniclosedai_url, voice_id, language,
                          relay=relay)
    return ReplyOnPause(inst.respond)

/* Voice Studio — vanilla JS, no build step. Drives:
 *   - Browser-side recording (getUserMedia + Web Audio → float32 buffer).
 *   - Client-side WAV encoder (downmix → resample → int16 PCM → WAV header).
 *   - Multipart upload to POST /voices.
 *   - Voices list from GET /voices, with delete + play-sample actions.
 *
 * Everything stays in this file; the page only needs `index.html` + `style.css`.
 */

const MAX_RECORD_SECONDS = 30;
// Sample rate hint — but the encoder NO LONGER resamples in the browser.
// We send WAVs at whatever rate AudioContext captured at (typically 48 kHz)
// and let the server handle resampling to 24 kHz with librosa's polyphase
// filter — the SAME resampler chatterbox itself uses internally.
// Crude linear interpolation in JS (with no anti-aliasing) was injecting
// aliasing artifacts; doing it server-side with librosa avoids that.
const TARGET_SAMPLE_RATE = 24000;
const SAMPLE_GREETING = "Hello! This is a quick sample of your cloned voice.";

// Reading scripts for the user to speak when recording. Each one runs ~8–12 s
// at conversational pace, covers a wide phoneme range so Chatterbox gets a
// phonetically diverse reference clip. The user sees these in the recorder
// card, can shuffle through them, and can switch language (EN ↔ ES) — the
// chosen language is remembered across visits via localStorage.
const READ_PROMPTS = {
  en: [
    "The quick brown fox jumps over the lazy dog. Bright stars shine above the silent meadow, while gentle music drifts through the evening air.",
    "Please call Stella. Ask her to bring these things with her from the store: six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
    "On a cold morning, the train arrived quietly at the station. Travelers stepped onto the platform with warm coats, eager to begin their journeys through the countryside.",
    "Hello, today I am recording a short voice sample. The weather is calm, the room is quiet, and I hope this clip captures my natural speaking voice clearly.",
    "A thousand tiny lights twinkled across the bay as the ferry pulled into the harbor, and the salty breeze carried the sound of distant laughter and music.",
  ],
  es: [
    "El veloz murciélago hindú comía feliz cardillo y kiwi. La cigüeña tocaba el saxofón detrás del palenque de paja, mientras la luna brillaba sobre el río tranquilo.",
    "Hola, hoy estoy grabando una muestra corta de mi voz. El clima está calmado, la habitación está silenciosa, y espero que este audio capture mi forma natural de hablar.",
    "El sol se asomaba entre las nubes mientras los pájaros cantaban en los árboles del parque. Una brisa suave traía el aroma del café recién hecho desde la cocina.",
    "Camino por la playa al atardecer, escuchando el sonido de las olas y sintiendo la arena tibia bajo los pies. Las gaviotas vuelan bajo, casi rozando el agua salada.",
    "En la ciudad nunca duerme la música. Por las calles se mezclan acordes de guitarra, voces alegres, risas en cafés y el ritmo constante de pasos sobre el adoquín.",
  ],
};
const LANG_KEY = "miniclosedai-voice:readPromptLang";   // localStorage key
let _readPromptLang = "en";   // mutated by setReadLang() / restored on boot

// ─────────────── DOM refs (single, named lookups so init is fast) ───────────
const els = {
  recordBtn:        document.getElementById("record-btn"),
  recordBtnLabel:   document.getElementById("record-btn-label"),
  recordTimer:      document.getElementById("record-timer"),
  recordError:      document.getElementById("record-error"),
  levelMeter:       document.getElementById("level-meter"),
  levelMeterBar:    document.getElementById("level-meter-bar"),
  uploadBtn:        document.getElementById("upload-btn"),
  uploadInput:      document.getElementById("upload-input"),
  saveCard:         document.getElementById("save-card"),
  playback:         document.getElementById("playback"),
  saveForm:         document.getElementById("save-form"),
  nameInput:        document.getElementById("name-input"),
  langInput:        document.getElementById("lang-input"),
  saveBtn:          document.getElementById("save-btn"),
  discardBtn:       document.getElementById("discard-btn"),
  saveError:        document.getElementById("save-error"),
  voicesList:       document.getElementById("voices-list"),
  voicesEmpty:      document.getElementById("voices-empty"),
  refreshBtn:       document.getElementById("refresh-btn"),
  toast:            document.getElementById("toast"),
  readPromptText:    document.getElementById("read-prompt-text"),
  readPromptShuffle: document.getElementById("read-prompt-shuffle"),
  readPromptLangEn:  document.getElementById("read-prompt-lang-en"),
  readPromptLangEs:  document.getElementById("read-prompt-lang-es"),
  saveCardLang:      document.getElementById("lang-input"),
};

// Index of the script currently shown WITHIN the current language pool.
// `setReadPrompt(undefined)` picks a different one at random so the shuffle
// button always changes something. Switching languages resets to index 0.
let _readPromptIdx = -1;
function setReadPrompt(idx) {
  const pool = READ_PROMPTS[_readPromptLang] || READ_PROMPTS.en;
  if (typeof idx !== "number") {
    if (pool.length <= 1) { idx = 0; }
    else {
      do { idx = Math.floor(Math.random() * pool.length); }
      while (idx === _readPromptIdx);
    }
  }
  _readPromptIdx = idx;
  if (els.readPromptText) els.readPromptText.textContent = pool[idx];
}

// Switch the active language for the read-aloud script + persist it.
// Side-effect: also pre-selects the matching option in the save-form's
// Language dropdown, because the user will almost always want their voice
// labeled with the language they just recorded in.
function setReadLang(lang) {
  if (lang !== "en" && lang !== "es") return;
  _readPromptLang = lang;
  try { localStorage.setItem(LANG_KEY, lang); } catch (_) {}
  // Toggle button styles + aria.
  if (els.readPromptLangEn) {
    els.readPromptLangEn.classList.toggle("active", lang === "en");
    els.readPromptLangEn.setAttribute("aria-pressed", String(lang === "en"));
  }
  if (els.readPromptLangEs) {
    els.readPromptLangEs.classList.toggle("active", lang === "es");
    els.readPromptLangEs.setAttribute("aria-pressed", String(lang === "es"));
  }
  // Default the save-form's language dropdown to match — easy correction
  // if user wants otherwise, but matches the common case.
  if (els.saveCardLang) els.saveCardLang.value = lang;
  // Reset to the first script in the new pool so the user sees a fresh
  // language-appropriate sentence immediately.
  setReadPrompt(0);
}

// ─────────────── Recorder state ─────────────────────────────────────────────
const recorder = {
  ctx: null,            // AudioContext
  stream: null,         // MediaStream from getUserMedia
  source: null,         // MediaStreamAudioSourceNode
  processor: null,      // ScriptProcessorNode (Web Audio fallback worklet path)
  chunks: [],           // collected Float32Arrays (mono mixdown)
  sampleRate: 0,        // ctx.sampleRate at capture time
  startedAt: 0,         // performance.now() of recording start
  timer: null,          // setInterval handle for the UI timer
  capturedWavBlob: null,// last finalized recording (Blob, audio/wav)
};

// ─────────────── Utility: timer formatting ─────────────────────────────────
function fmtTime(sec) {
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ─────────────── Toast ─────────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, kind = "ok") {
  els.toast.textContent = msg;
  els.toast.className = `toast is-${kind}`;
  els.toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 2400);
}

function setError(targetEl, msg) {
  if (!msg) {
    targetEl.hidden = true;
    targetEl.textContent = "";
    return;
  }
  targetEl.textContent = msg;
  targetEl.hidden = false;
}

// ─────────────── WAV encoder ────────────────────────────────────────────────
// Take an array of float32 mono chunks at `srcRate`, resample to `dstRate`,
// quantise to int16, and prepend a 44-byte WAV header. Returns a Blob the
// server's `sf.read()` accepts natively.
function encodeWav(chunks, srcRate, dstRate) {
  // 1. Concatenate.
  let totalLen = 0;
  for (const c of chunks) totalLen += c.length;
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const c of chunks) { merged.set(c, offset); offset += c.length; }

  // 2. Resample (linear interpolation — quality is plenty for speech).
  let resampled;
  if (srcRate === dstRate) {
    resampled = merged;
  } else {
    const newLen = Math.round(merged.length * (dstRate / srcRate));
    resampled = new Float32Array(newLen);
    const ratio = (merged.length - 1) / (newLen - 1);
    for (let i = 0; i < newLen; i++) {
      const idx = i * ratio;
      const lo = Math.floor(idx);
      const hi = Math.min(lo + 1, merged.length - 1);
      const t = idx - lo;
      resampled[i] = merged[lo] * (1 - t) + merged[hi] * t;
    }
  }

  // 3. Float32 [-1, 1] → int16. Clip to be safe.
  const i16 = new Int16Array(resampled.length);
  for (let i = 0; i < resampled.length; i++) {
    const s = Math.max(-1, Math.min(1, resampled[i]));
    i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  // 4. WAV header (RIFF / fmt  / data). 44 bytes; spec at
  //    http://soundfile.sapp.org/doc/WaveFormat/.
  const blockAlign = 2;          // 1 channel × 16 bit
  const byteRate = dstRate * blockAlign;
  const dataSize = i16.length * 2;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);
  function writeStr(off, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  }
  writeStr(0,  "RIFF");
  view.setUint32(4,  36 + dataSize, true);
  writeStr(8,  "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);          // fmt chunk size
  view.setUint16(20, 1, true);           // PCM
  view.setUint16(22, 1, true);           // mono
  view.setUint32(24, dstRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);          // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  new Int16Array(buf, 44).set(i16);
  return new Blob([buf], { type: "audio/wav" });
}

// ─────────────── Recorder lifecycle ─────────────────────────────────────────
async function startRecording() {
  setError(els.recordError, "");

  // Permission probe — clearest error message lives here, not deep inside
  // a media-stream failure later.
  if (!navigator.mediaDevices?.getUserMedia) {
    setError(els.recordError, "This browser doesn't support audio recording.");
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    setError(els.recordError, `Microphone access denied: ${e?.message || e}`);
    return;
  }

  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const source = ctx.createMediaStreamSource(stream);

  // We use ScriptProcessorNode (deprecated but universally supported, no
  // worklet boilerplate). Buffer size 4096 = ~85 ms at 48 kHz — fine for
  // recording-into-buffer; we don't need RT processing here.
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  source.connect(processor);
  // The processor needs a destination connection or it won't fire on some
  // browsers. Route to ctx.destination via a zero-gain so we don't echo back.
  const silence = ctx.createGain();
  silence.gain.value = 0;
  processor.connect(silence);
  silence.connect(ctx.destination);

  recorder.ctx = ctx;
  recorder.stream = stream;
  recorder.source = source;
  recorder.processor = processor;
  recorder.chunks = [];
  recorder.sampleRate = ctx.sampleRate;
  recorder.startedAt = performance.now();

  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    // Copy — input buffer is reused; storing the same ref repeatedly would
    // collapse to silence on every reuse.
    recorder.chunks.push(new Float32Array(input));

    // Cheap RMS for the level meter.
    let sumSq = 0;
    for (let i = 0; i < input.length; i++) sumSq += input[i] * input[i];
    const rms = Math.sqrt(sumSq / input.length);
    const pct = Math.min(100, Math.round(rms * 250));
    els.levelMeterBar.style.width = pct + "%";
  };

  // UI
  els.recordBtn.classList.add("is-recording");
  els.recordBtnLabel.textContent = "Stop recording";
  els.recordTimer.classList.add("is-recording");
  els.levelMeter.classList.add("is-active");
  recorder.timer = setInterval(() => {
    const elapsed = (performance.now() - recorder.startedAt) / 1000;
    els.recordTimer.textContent = fmtTime(elapsed);
    if (elapsed >= MAX_RECORD_SECONDS) {
      stopRecording().catch(() => {});
    }
  }, 250);
}

async function stopRecording() {
  if (!recorder.ctx) return;
  if (recorder.timer) { clearInterval(recorder.timer); recorder.timer = null; }

  try {
    recorder.processor.disconnect();
    recorder.source.disconnect();
    recorder.processor.onaudioprocess = null;
    for (const t of recorder.stream.getTracks()) t.stop();
    await recorder.ctx.close();
  } catch (e) { /* ignore — best-effort teardown */ }

  // UI reset
  els.recordBtn.classList.remove("is-recording");
  els.recordBtnLabel.textContent = "Start recording";
  els.recordTimer.classList.remove("is-recording");
  els.levelMeter.classList.remove("is-active");
  els.levelMeterBar.style.width = "0%";

  // Encode + preview.
  if (recorder.chunks.length === 0) {
    setError(els.recordError, "Captured zero audio frames — try again.");
    resetRecorderState();
    return;
  }
  // Encode at the source rate (no JS-side resample — server uses librosa).
  const blob = encodeWav(recorder.chunks, recorder.sampleRate, recorder.sampleRate);
  recorder.capturedWavBlob = blob;
  els.playback.src = URL.createObjectURL(blob);
  els.saveCard.hidden = false;
  els.nameInput.focus();

  resetRecorderState({ keepBlob: true });
}

function resetRecorderState({ keepBlob = false } = {}) {
  recorder.ctx = null;
  recorder.stream = null;
  recorder.source = null;
  recorder.processor = null;
  recorder.chunks = [];
  recorder.sampleRate = 0;
  recorder.startedAt = 0;
  recorder.timer = null;
  if (!keepBlob) {
    if (els.playback.src) URL.revokeObjectURL(els.playback.src);
    recorder.capturedWavBlob = null;
    els.playback.src = "";
  }
  els.recordTimer.textContent = "0:00";
}

// ─────────────── Uploader ───────────────────────────────────────────────────
// Parallel entry path to the recorder: pick an existing audio file, decode it
// via the Web Audio API (handles WAV/MP3/M4A/OGG/FLAC/WebM in modern browsers),
// re-encode to the same 22050 Hz mono int16 WAV the recorder produces, and
// drop the user into the EXACT same save form (Name + Language). The actual
// upload happens via `saveVoice()` — this function just stages the blob.
async function loadAudioFile(file) {
  setError(els.recordError, "");
  setError(els.saveError, "");
  if (!file) return;

  // Cheap guard: only let the AudioContext attempt formats it'll actually
  // decode. We still rely on decode-time failure for browser-specific quirks.
  const looksAudio = (file.type && file.type.startsWith("audio/")) ||
                     /\.(wav|mp3|m4a|aac|ogg|flac|webm)$/i.test(file.name || "");
  if (!looksAudio) {
    setError(els.recordError, `Not a recognised audio file: ${file.name}`);
    return;
  }
  // Cap upload size to match the recording cap (5 MB ≈ 113 s at 22 kHz mono).
  // Most 30 s clips in any of the supported formats are well under this.
  if (file.size > 10 * 1024 * 1024) {
    setError(els.recordError, `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB; max 10 MB).`);
    return;
  }

  // Decode → channels → encode pipeline. Surfaces decoder errors to the user
  // rather than dying silently inside the browser's audio stack.
  let audioBuffer;
  try {
    const arrayBuf = await file.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    audioBuffer = await ctx.decodeAudioData(arrayBuf);
    // Close the context so we don't keep a hardware resource for the rest of
    // the page's lifetime — we only needed it for one decode.
    try { await ctx.close(); } catch {}
  } catch (e) {
    setError(els.recordError, `Could not decode this file: ${e?.message || e}. Try WAV / MP3 / M4A / OGG.`);
    return;
  }

  // Validate duration against the same window the recording flow enforces, so
  // both entry paths converge on identical server-side semantics. The server's
  // ceiling is 35 s; we tighten to 30 s here so the user gets a clear UI hint
  // before the server can complain.
  const duration = audioBuffer.duration;
  if (duration < 0.5) {
    setError(els.recordError, `Audio is only ${duration.toFixed(2)} s — at least 0.5 s required.`);
    return;
  }
  if (duration > MAX_RECORD_SECONDS) {
    setError(els.recordError, `Audio is ${duration.toFixed(1)} s — trim it to ${MAX_RECORD_SECONDS} s or less.`);
    return;
  }

  // Downmix multi-channel sources to mono by averaging channels. Mono is what
  // Chatterbox actually conditions on; stereo just doubles the file size.
  const n = audioBuffer.length;
  const channels = audioBuffer.numberOfChannels;
  const mono = new Float32Array(n);
  if (channels === 1) {
    mono.set(audioBuffer.getChannelData(0));
  } else {
    const cs = [];
    for (let i = 0; i < channels; i++) cs.push(audioBuffer.getChannelData(i));
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let c = 0; c < channels; c++) sum += cs[c][i];
      mono[i] = sum / channels;
    }
  }

  // Reuse the recorder's encoder so the wire format is byte-identical.
  // Encode at the source rate (no JS-side resample — server uses librosa).
  const blob = encodeWav([mono], audioBuffer.sampleRate, audioBuffer.sampleRate);
  if (els.playback.src) URL.revokeObjectURL(els.playback.src);
  recorder.capturedWavBlob = blob;
  els.playback.src = URL.createObjectURL(blob);
  els.saveCard.hidden = false;

  // Pre-fill the Name input with a tidy version of the filename so the user
  // doesn't have to retype it: "edgar-sample.mp3" → "Edgar sample".
  const base = (file.name || "").replace(/\.[^.]+$/, "");
  if (base && !els.nameInput.value) {
    const friendly = base.replace(/[_-]+/g, " ").trim();
    els.nameInput.value = friendly.charAt(0).toUpperCase() + friendly.slice(1);
  }
  els.nameInput.focus();
}

function discardRecording() {
  if (els.playback.src) URL.revokeObjectURL(els.playback.src);
  els.playback.src = "";
  recorder.capturedWavBlob = null;
  els.saveCard.hidden = true;
  els.nameInput.value = "";
  els.langInput.value = "en";
  setError(els.saveError, "");
}

// ─────────────── Save (upload) ──────────────────────────────────────────────
async function saveVoice(e) {
  e.preventDefault();
  setError(els.saveError, "");
  if (!recorder.capturedWavBlob) {
    setError(els.saveError, "Nothing to save — record again.");
    return;
  }
  const name = els.nameInput.value.trim();
  if (!name) {
    setError(els.saveError, "Please give the voice a name.");
    els.nameInput.focus();
    return;
  }

  const fd = new FormData();
  fd.append("audio", recorder.capturedWavBlob, "recording.wav");
  fd.append("name", name);
  fd.append("language", els.langInput.value || "en");

  els.saveBtn.classList.add("is-busy");
  els.saveBtn.disabled = true;
  els.discardBtn.disabled = true;
  try {
    const r = await fetch("/voices", { method: "POST", body: fd });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${r.status}`);
    }
    const out = await r.json();
    toast(`Saved “${out.name}” (${out.duration_sec}s)`, "ok");
    discardRecording();
    await loadVoices();
  } catch (err) {
    setError(els.saveError, `Save failed: ${err.message || err}`);
  } finally {
    els.saveBtn.classList.remove("is-busy");
    els.saveBtn.disabled = false;
    els.discardBtn.disabled = false;
  }
}

// ─────────────── Voices list ────────────────────────────────────────────────
async function loadVoices() {
  els.voicesList.innerHTML = "";
  let catalog;
  try {
    const r = await fetch("/voices");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    catalog = await r.json();
  } catch (e) {
    toast(`Couldn't load voices: ${e.message || e}`, "error");
    return;
  }
  // Flatten + dedupe across languages (`default` lives under both en/es).
  const seen = new Set();
  const flat = [];
  for (const lang of Object.keys(catalog)) {
    for (const v of catalog[lang]) {
      if (seen.has(v.id)) continue;
      seen.add(v.id);
      flat.push({ ...v, language: lang });
    }
  }
  // built-in (`default`) on top, then alphabetical by name.
  flat.sort((a, b) => {
    if (a.id === "default" && b.id !== "default") return -1;
    if (b.id === "default" && a.id !== "default") return 1;
    return a.name.localeCompare(b.name);
  });
  els.voicesEmpty.hidden = flat.length > 0;
  for (const v of flat) els.voicesList.appendChild(renderVoiceRow(v));
}

function renderVoiceRow(v) {
  const li = document.createElement("li");
  li.className = "voice-row";

  const name = document.createElement("span");
  name.className = "name";
  name.textContent = v.name || v.id;
  const langPill = document.createElement("span");
  langPill.className = "lang-pill";
  langPill.textContent = v.language;
  name.appendChild(langPill);
  if (v.id === "default") {
    const builtin = document.createElement("span");
    builtin.className = "lang-pill builtin-pill";
    builtin.textContent = "built-in";
    name.appendChild(builtin);
  }

  const actions = document.createElement("div");
  actions.className = "actions";
  const playBtn = document.createElement("button");
  playBtn.type = "button";
  playBtn.className = "btn";
  playBtn.textContent = "▶ Sample";
  playBtn.title = "Synthesize a short sample of this voice";
  playBtn.addEventListener("click", () => playSample(v, playBtn));
  actions.appendChild(playBtn);

  if (v.id !== "default") {
    const del = document.createElement("button");
    del.type = "button";
    del.className = "btn btn-danger";
    del.textContent = "Delete";
    del.addEventListener("click", () => deleteVoice(v));
    actions.appendChild(del);
  }

  li.appendChild(name);
  li.appendChild(actions);
  return li;
}

async function playSample(v, btn) {
  btn.disabled = true;
  const origLabel = btn.textContent;
  btn.textContent = "Synthesizing…";
  try {
    const r = await fetch("/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: SAMPLE_GREETING,
        voice: v.id,
        language: v.language || "en",
      }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${r.status}`);
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play();
  } catch (e) {
    toast(`Sample failed: ${e.message || e}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = origLabel;
  }
}

async function deleteVoice(v) {
  if (!confirm(`Delete the voice “${v.name}”? This can't be undone.`)) return;
  try {
    const r = await fetch(`/voices/${encodeURIComponent(v.id)}`, { method: "DELETE" });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${r.status}`);
    }
    toast(`Deleted “${v.name}”`, "ok");
    await loadVoices();
  } catch (e) {
    toast(`Delete failed: ${e.message || e}`, "error");
  }
}

// ─────────────── Boot ──────────────────────────────────────────────────────
function bind() {
  els.recordBtn.addEventListener("click", async () => {
    if (recorder.ctx) await stopRecording();
    else await startRecording();
  });
  // The Upload-audio path mirrors the recorder: pick a file → decode →
  // re-encode → reveal the save form. Reuses the same `recorder.capturedWavBlob`
  // slot so saveVoice() doesn't need to care which path the audio came from.
  els.uploadBtn.addEventListener("click", () => {
    // If a recording is in progress, stop it first so the two paths don't fight.
    if (recorder.ctx) { stopRecording().catch(() => {}); return; }
    els.uploadInput.value = "";   // allow picking the same file twice in a row
    els.uploadInput.click();
  });
  els.uploadInput.addEventListener("change", async () => {
    const f = els.uploadInput.files && els.uploadInput.files[0];
    if (f) await loadAudioFile(f);
  });
  els.discardBtn.addEventListener("click", discardRecording);
  els.saveForm.addEventListener("submit", saveVoice);
  els.refreshBtn.addEventListener("click", loadVoices);
  if (els.readPromptShuffle) {
    // No-arg call → setReadPrompt picks a different random index.
    els.readPromptShuffle.addEventListener("click", () => setReadPrompt());
  }
  if (els.readPromptLangEn) els.readPromptLangEn.addEventListener("click", () => setReadLang("en"));
  if (els.readPromptLangEs) els.readPromptLangEs.addEventListener("click", () => setReadLang("es"));
  // Esc cancels recording / discards save-card.
  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (recorder.ctx) stopRecording().catch(() => {});
    else if (!els.saveCard.hidden) discardRecording();
  });
}

bind();
// Restore the user's preferred script language across visits, defaulting to
// English. setReadLang() also calls setReadPrompt(0), so we don't need a
// separate setReadPrompt() boot call.
try {
  const saved = localStorage.getItem(LANG_KEY);
  setReadLang(saved === "es" ? "es" : "en");
} catch (_) {
  setReadLang("en");
}
loadVoices();

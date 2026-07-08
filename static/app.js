/* Voice Studio — vanilla JS, no build step. Drives:
 *   - Browser-side recording (getUserMedia + Web Audio → float32 buffer).
 *   - Client-side WAV encoder (downmix → resample → int16 PCM → WAV header).
 *   - Multipart upload to POST /voices.
 *   - Voices list from GET /voices, with delete + play-sample actions.
 *
 * Everything stays in this file; the page only needs `index.html` + `style.css`.
 */

const MAX_RECORD_SECONDS = 30;
// Uploaded clips longer than this are auto-trimmed (not rejected) to the leading
// window before encoding — Chatterbox only conditions on the start of the
// reference, so the tail is wasted. Keep in sync with server _VOICE_MAX_DURATION_SEC.
const MAX_UPLOAD_SECONDS = 90;
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
  connectBaseUrl:    document.getElementById("connect-base-url"),
  connectCopy:       document.getElementById("connect-copy"),
  connectAltHint:    document.getElementById("connect-alt-hint"),
  connectAltBaseUrl: document.getElementById("connect-alt-base-url"),
  connectAuthHint:   document.getElementById("connect-auth-hint"),
  connectPromptBtn:  document.getElementById("connect-prompt-btn"),
  promptModal:       document.getElementById("prompt-modal"),
  promptMd:          document.getElementById("prompt-md"),
  promptCopy:        document.getElementById("prompt-copy"),
  promptClose:       document.getElementById("prompt-close"),
};

// Cached live data for the integration-prompt generator, so it can template the
// doc with the real base URL, auth requirement, and voice ids without a second
// network round-trip. Populated by loadConnectInfo() / loadVoices().
let _connectInfo = null;   // { base_url, alt_base_url, auth_required }
let _voiceCatalog = null;  // raw /voices response: { lang: [{id,name,gender?}] }

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
  // Cap the SOURCE file size. We auto-trim long clips after decode, but still
  // guard the raw upload so a giant file can't blow up decodeAudioData. 40 MB
  // holds several minutes of compressed audio (we only keep the first 90 s).
  if (file.size > 40 * 1024 * 1024) {
    setError(els.recordError, `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB; max 40 MB).`);
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

  // Only the lower bound is a hard error — too-short clips give Chatterbox no
  // usable conditioning. Too-long clips are fine: we auto-trim to the first
  // MAX_UPLOAD_SECONDS (matching the server) instead of making the user go
  // hand-edit their file.
  const duration = audioBuffer.duration;
  if (duration < 0.5) {
    setError(els.recordError, `Audio is only ${duration.toFixed(2)} s — at least 0.5 s required.`);
    return;
  }
  const willTrim = duration > MAX_UPLOAD_SECONDS;

  // Downmix multi-channel sources to mono by averaging channels. Mono is what
  // Chatterbox actually conditions on; stereo just doubles the file size.
  // Trim to the first MAX_UPLOAD_SECONDS while we're here so oversized uploads
  // never leave the browser.
  const maxSamples = Math.floor(MAX_UPLOAD_SECONDS * audioBuffer.sampleRate);
  const n = Math.min(audioBuffer.length, maxSamples);
  const channels = audioBuffer.numberOfChannels;
  const mono = new Float32Array(n);
  if (channels === 1) {
    mono.set(audioBuffer.getChannelData(0).subarray(0, n));
  } else {
    const cs = [];
    for (let i = 0; i < channels; i++) cs.push(audioBuffer.getChannelData(i));
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let c = 0; c < channels; c++) sum += cs[c][i];
      mono[i] = sum / channels;
    }
  }

  if (willTrim) {
    toast(`Audio was ${duration.toFixed(1)} s — trimmed to the first ${MAX_UPLOAD_SECONDS} s.`, "warn");
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
    _voiceCatalog = catalog;   // cache for the integration-prompt generator
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

  // Custom-text tester: a full-width accordion panel that stays collapsed until
  // "Test…" is clicked, then animates open to reveal a textarea + Play so the
  // user can hear this voice say anything. Uses the grid-template-rows 0fr→1fr
  // technique so it animates to the content's natural height with no JS measuring.
  const tester = document.createElement("div");
  tester.className = "voice-tester";              // collapsed by default (no .is-open)
  const testerInner = document.createElement("div");
  testerInner.className = "voice-tester-inner";   // overflow clip during the animation
  const testerBody = document.createElement("div");
  testerBody.className = "voice-tester-body";      // padding/border live here, hidden when collapsed

  const ta = document.createElement("textarea");
  ta.className = "voice-tester-input";
  ta.rows = 2;
  ta.maxLength = 4000;   // matches server /speak text max_length
  ta.placeholder = "Type text to hear in this voice…";
  ta.value = "The quick brown fox jumps over the lazy dog.";

  const testerActions = document.createElement("div");
  testerActions.className = "voice-tester-actions";
  const speakBtn = document.createElement("button");
  speakBtn.type = "button";
  speakBtn.className = "btn btn-primary btn-small";
  speakBtn.textContent = "▶ Play";
  speakBtn.title = "Synthesize the text above in this voice";
  speakBtn.addEventListener("click", () => synthAndPlay(v, ta.value, speakBtn));
  testerActions.appendChild(speakBtn);

  // Cmd/Ctrl+Enter in the textarea triggers Play — quick keyboard path.
  ta.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      synthAndPlay(v, ta.value, speakBtn);
    }
  });

  testerBody.appendChild(ta);
  testerBody.appendChild(testerActions);
  testerInner.appendChild(testerBody);
  tester.appendChild(testerInner);

  const testBtn = document.createElement("button");
  testBtn.type = "button";
  testBtn.className = "btn";
  testBtn.textContent = "Test…";
  testBtn.title = "Type custom text and hear it in this voice";
  testBtn.setAttribute("aria-expanded", "false");
  testBtn.addEventListener("click", () => {
    const open = tester.classList.toggle("is-open");
    testBtn.setAttribute("aria-expanded", String(open));
    testBtn.classList.toggle("is-active", open);
    if (open) ta.focus();
  });
  actions.appendChild(testBtn);

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
  li.appendChild(tester);
  return li;
}

// Synthesize `text` in voice `v` and play it back. Shared by the fixed-greeting
// "Sample" button and the custom-text "Test" panel. `btn` gets a busy state.
async function synthAndPlay(v, text, btn) {
  const clean = (text || "").trim();
  if (!clean) {
    toast("Enter some text to speak.", "error");
    return;
  }
  btn.disabled = true;
  const origLabel = btn.textContent;
  btn.textContent = "Synthesizing…";
  try {
    const r = await fetch("/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: clean,
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
    toast(`Speak failed: ${e.message || e}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = origLabel;
  }
}

function playSample(v, btn) {
  return synthAndPlay(v, SAMPLE_GREETING, btn);
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

// ─────────────── Connect to MiniClosedAI ───────────────────────────────────
// Fetches the base URL to register this voice service in MiniClosedAI and
// renders it into the copy-box. Falls back to the page's own origin if the
// /api/connect-info endpoint is unavailable (older server) so the Copy button
// still hands the user something usable.
async function loadConnectInfo() {
  let info;
  try {
    const r = await fetch("/api/connect-info");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    info = await r.json();
  } catch (_) {
    // Older server without the endpoint: the URL the browser is on is, at
    // worst, valid for a same-host MiniClosedAI.
    info = { base_url: location.origin, alt_base_url: "", auth_required: false };
  }
  _connectInfo = info;   // cache for the integration-prompt generator
  els.connectBaseUrl.textContent = info.base_url || location.origin;
  if (info.alt_base_url) {
    els.connectAltBaseUrl.textContent = info.alt_base_url;
    els.connectAltHint.hidden = false;
  }
  els.connectAuthHint.hidden = !info.auth_required;
}

els && els.connectCopy && els.connectCopy.addEventListener("click", () => {
  const url = els.connectBaseUrl.textContent;
  navigator.clipboard.writeText(url).then(
    () => toast("Copied base URL", "ok"),
    () => toast("Copy failed — select and copy manually", "error"));
});

// ─────────────── Integration-prompt modal ──────────────────────────────────
// Builds an "AI implementation prompt" (markdown) for wiring up TTS against this
// service, templated with the live base URL, auth requirement, and real voice
// ids from the caches. Safe to call before the caches populate — falls back to
// location.origin, no-auth, and the "default" voice.
function buildIntegrationPrompt() {
  const info = _connectInfo || {};
  const baseUrl = (info.base_url || location.origin).replace(/\/+$/, "");
  const authRequired = !!info.auth_required;

  // Real voice ids from the cached catalog, deduped across languages.
  const ids = [];
  if (_voiceCatalog) {
    for (const lang of Object.keys(_voiceCatalog)) {
      for (const v of (_voiceCatalog[lang] || [])) {
        if (v && v.id && !ids.includes(v.id)) ids.push(v.id);
      }
    }
  }
  if (!ids.includes("default")) ids.unshift("default");
  const exampleVoice = ids.includes("default") ? "default" : (ids[0] || "default");
  const voiceList = ids.map((id) => `- \`${id}\``).join("\n");

  // Auth fragments — only emitted when the service requires a key.
  const authHeaderCurl = authRequired
    ? `  -H "Authorization: Bearer <YOUR_API_KEY>" \\\n` : "";
  const authHeaderJs = authRequired
    ? `      "Authorization": "Bearer <YOUR_API_KEY>",\n` : "";
  const authSection = authRequired
    ? `## Auth
This service requires an API key. Send it on **every** request:

\`\`\`
Authorization: Bearer <YOUR_API_KEY>
\`\`\`

Store \`<YOUR_API_KEY>\` as a secret / environment variable — never hard-code it.
`
    : `## Auth
This service currently requires **no authentication**. If you later enable an API
key, add \`Authorization: Bearer <YOUR_API_KEY>\` to every request.
`;

  return `# Add text-to-speech (TTS) to my app

You are a coding assistant adding **text-to-speech** to my app. Implement a
client for the HTTP API described below. The examples are curl + JavaScript
\`fetch\`, but implement it idiomatically in whatever stack my project already
uses. Do not invent endpoints or fields — use exactly what is documented here.

## Service
- Base URL: \`${baseUrl}\`
- Audio format: PCM16, **mono**, **22050 Hz**.
- Synthesis: \`POST /speak\` (one-shot WAV) and \`POST /speak/stream\`
  (low-latency SSE streaming). Discover voices with \`GET /voices\`.

${authSection}
## Available voices
Fetch these at runtime from \`GET /voices\`; ids available right now:
${voiceList}

The \`default\` voice always exists — use it as a fallback.

## GET /voices — list voices
Returns voices grouped by language code:

\`\`\`json
{ "en": [ { "id": "default", "name": "Default voice", "gender": "F" } ],
  "es": [ { "id": "default", "name": "Default voice" } ] }
\`\`\`

\`\`\`bash
curl ${baseUrl}/voices${authRequired ? ` \\\n  -H "Authorization: Bearer <YOUR_API_KEY>"` : ""}
\`\`\`

## POST /speak — one-shot synthesis (returns a WAV file)
Request JSON body:

| field    | type   | required | notes |
|----------|--------|----------|-------|
| text     | string | yes      | 1–4000 characters |
| voice    | string | yes      | a voice id from /voices |
| language | string | yes      | e.g. "en" or "es" |
| speed    | float  | no       | 0.5–2.0, default 1.0 |

Any **extra/unknown field is rejected with HTTP 422** — send only these keys.
The response is raw \`audio/wav\` bytes (PCM16 mono @ 22050 Hz).

\`\`\`bash
curl -X POST ${baseUrl}/speak \\
${authHeaderCurl}  -H "Content-Type: application/json" \\
  -d '{"text":"Hello from my app!","voice":"${exampleVoice}","language":"en"}' \\
  --output speech.wav
\`\`\`

\`\`\`js
async function speak(text) {
  const res = await fetch("${baseUrl}/speak", {
    method: "POST",
    headers: {
${authHeaderJs}      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      voice: "${exampleVoice}",
      language: "en",
      // speed: 1.0, // optional, 0.5–2.0
    }),
  });
  if (!res.ok) throw new Error("TTS failed: HTTP " + res.status);
  const wavBlob = await res.blob();            // audio/wav
  const url = URL.createObjectURL(wavBlob);
  const audio = new Audio(url);
  audio.onended = () => URL.revokeObjectURL(url);
  await audio.play();
}
\`\`\`

## POST /speak/stream — streaming synthesis (Server-Sent Events)
Same JSON body as /speak, but the response is \`text/event-stream\`. Use it for
low-latency playback: start playing before the whole clip is synthesized.

Each frame is a \`data:\` line with JSON:
- Audio frame: \`{"chunk_b64":"<base64 int16 LE PCM mono>","sample_rate":22050}\`
- Completion:  \`{"done":true}\`
- Error:       \`{"error":"<message>"}\`  (a single frame; stop and surface it)

\`chunk_b64\` decodes to **raw little-endian int16 PCM samples** — mono, 22050 Hz,
**no WAV header**. Convert to Float32 in [-1, 1] to feed the Web Audio API (or
write a WAV header yourself if you need a file).

\`\`\`bash
curl -N -X POST ${baseUrl}/speak/stream \\
${authHeaderCurl}  -H "Content-Type: application/json" \\
  -d '{"text":"Streaming hello!","voice":"${exampleVoice}","language":"en"}'
\`\`\`

\`\`\`js
async function speakStream(text, onPcm) {
  const res = await fetch("${baseUrl}/speak/stream", {
    method: "POST",
    headers: {
${authHeaderJs}      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text, voice: "${exampleVoice}", language: "en" }),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split("\\n\\n");
    buf = frames.pop();                  // keep the trailing partial frame
    for (const frame of frames) {
      const line = frame.split("\\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const msg = JSON.parse(line.slice(5).trim());
      if (msg.error) throw new Error(msg.error);
      if (msg.done) return;
      // base64 -> int16 PCM -> Float32 [-1, 1]
      const bytes = Uint8Array.from(atob(msg.chunk_b64), (c) => c.charCodeAt(0));
      const pcm16 = new Int16Array(bytes.buffer);
      const f32 = Float32Array.from(pcm16, (s) => s / 32768);
      onPcm(f32, msg.sample_rate);       // e.g. schedule into an AudioContext
    }
  }
}
\`\`\`

## Requirements for your implementation
1. Add a reusable \`speak(text, voice?, language?)\` function plus a streaming
   variant for long text.
2. Load available voices from \`GET /voices\`; default to the \`default\` voice.
3. Handle errors: non-2xx from /speak, and \`{"error":...}\` frames from the
   stream. Surface a user-visible message; never fail silently.
4. Keep the base URL${authRequired ? " and API key" : ""} in configuration, not hard-coded in call sites.
`;
}

let _prevFocus = null;   // element to restore focus to when the modal closes

function openPromptModal() {
  els.promptMd.textContent = buildIntegrationPrompt();   // textContent = literal, XSS-safe
  _prevFocus = document.activeElement;
  els.promptModal.hidden = false;
  els.promptCopy.focus();          // land focus on the primary action
}

function closePromptModal() {
  if (els.promptModal.hidden) return;
  els.promptModal.hidden = true;
  if (_prevFocus && document.contains(_prevFocus)) _prevFocus.focus();
  _prevFocus = null;
}

els && els.promptCopy && els.promptCopy.addEventListener("click", () => {
  navigator.clipboard.writeText(els.promptMd.textContent).then(
    () => toast("Copied integration prompt", "ok"),
    () => toast("Copy failed — select the text and copy manually", "error"));
});

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
  // Integration-prompt modal: open from the connect card, close via the X or a
  // click on the dimmed backdrop (but not the dialog itself).
  els.connectPromptBtn && els.connectPromptBtn.addEventListener("click", openPromptModal);
  els.promptClose && els.promptClose.addEventListener("click", closePromptModal);
  els.promptModal && els.promptModal.addEventListener("click", (e) => {
    if (e.target === els.promptModal) closePromptModal();
  });
  // Esc closes the modal first, else cancels recording / discards save-card.
  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!els.promptModal.hidden) { closePromptModal(); return; }
    if (recorder.ctx) stopRecording().catch(() => {});
    else if (!els.saveCard.hidden) discardRecording();
  });
}

// ─────────────── Theme toggle ───────────────────────────────────────────────
// Cycles System → Light → Dark. The active mode is persisted to localStorage
// and re-applied before paint via a tiny inline script in index.html (so the
// page never flashes the wrong theme on load). In "system" mode we honour
// prefers-color-scheme; in explicit "light"/"dark" the user override wins.
const THEME_KEY = "miniclosedai-voice:theme";
const THEME_MODES = ["system", "light", "dark"];

function systemPrefersDark() {
  return matchMedia("(prefers-color-scheme: dark)").matches;
}

function effectiveDark(mode) {
  if (mode === "dark") return true;
  if (mode === "light") return false;
  return systemPrefersDark();   // system
}

function applyTheme(mode) {
  document.documentElement.classList.toggle("dark", effectiveDark(mode));
  // Show only the icon corresponding to the active mode.
  const ids = { system: "theme-icon-system", light: "theme-icon-light", dark: "theme-icon-dark" };
  for (const m of THEME_MODES) {
    const el = document.getElementById(ids[m]);
    if (el) el.style.display = (m === mode) ? "" : "none";
  }
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.title = `Theme: ${mode}`;
    btn.setAttribute("aria-label", `Theme: ${mode}. Click to cycle.`);
  }
}

function initTheme() {
  let mode;
  try { mode = localStorage.getItem(THEME_KEY) || "system"; } catch { mode = "system"; }
  if (!THEME_MODES.includes(mode)) mode = "system";
  applyTheme(mode);

  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const cur = (function() {
        try { return localStorage.getItem(THEME_KEY) || "system"; } catch { return "system"; }
      })();
      const next = THEME_MODES[(THEME_MODES.indexOf(cur) + 1) % THEME_MODES.length];
      try { localStorage.setItem(THEME_KEY, next); } catch (_) {}
      applyTheme(next);
    });
  }
  // While in "system" mode, react to live OS theme changes.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    try {
      const m = localStorage.getItem(THEME_KEY) || "system";
      if (m === "system") applyTheme("system");
    } catch (_) {}
  });
}

bind();
initTheme();
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
loadConnectInfo();

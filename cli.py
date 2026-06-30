#!/usr/bin/env python3
"""vc — terminal client for miniclosedai-voice.

Everything the Voice Studio GUI does, from the shell: check status, print the
connect URL to register in MiniClosedAI, list / clone / remove voices, synthesize
speech to a WAV, and transcribe audio. It's a thin HTTP client over the *same*
endpoints the browser GUI uses, so the two stay in live sync (clone a voice here →
it shows in the GUI and in every bot's TTS dropdown, and vice-versa).

Dependency-free: standard library only (argparse + urllib + json + ssl) — runs
under any python3, no venv required, so it can be copied to another machine and
pointed at a remote service (see `### Remote / agent access` in the README). Talks
to the service at $VOICE_URL (default https://localhost:$VOICE_PORT, 8090, falling
back to http:// automatically). Sends Authorization: Bearer $VOICE_API_KEY when
that env var is set. TLS verification is OFF by default (the dev cert is
self-signed); set VOICE_VERIFY=1 to enforce it.

Run `vc <command> -h` for per-command help. Common commands:
    vc status | url | voices
    vc speak "hello world" [--voice default --language en --out out.wav --play]
    vc transcribe clip.wav
    vc clone sample.wav --name "Edgar"   vc rm <voice_id>
    vc serve
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXIT_OK, EXIT_ERR, EXIT_UNREACHABLE = 0, 1, 2


# --------------------------------------------------------------------- config
def _load_dotenv() -> dict:
    env = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_DOTENV = _load_dotenv()


def cfg(name: str, default: str = "") -> str:
    return os.environ.get(name) or _DOTENV.get(name) or default


def _candidates() -> list[str]:
    """Base URLs to try, in order. A pinned VOICE_URL wins; otherwise probe
    https first (the dev-cert default) then http (the `--http` escape hatch and
    the Docker container, which serves plain http)."""
    url = cfg("VOICE_URL")
    if url:
        return [url.rstrip("/")]
    port = cfg("VOICE_PORT", "8090")
    return [f"https://localhost:{port}", f"http://localhost:{port}"]


# Resolved working base, set once by require_service() so every later call reuses
# the scheme that actually answered (no repeated https→http fallback per request).
_BASE: str | None = None


def base_url() -> str:
    return _BASE or _candidates()[0]


def _headers(extra: dict | None = None) -> dict:
    h = {"Accept": "application/json"}
    key = cfg("VOICE_API_KEY")
    if key:
        h["Authorization"] = f"Bearer {key}"
    if extra:
        h.update(extra)
    return h


def _ssl_ctx():
    """Unverified context for the self-signed dev cert (the default). The voice
    service is a user-trusted LAN endpoint; VOICE_VERIFY=1 opts back into
    standard verification."""
    if cfg("VOICE_VERIFY") == "1":
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# --------------------------------------------------------------------- ANSI
_TTY = sys.stdout.isatty()


def c(text, color):
    if not _TTY:
        return text
    codes = {"dim": "2", "red": "31", "green": "32", "yellow": "33",
             "blue": "34", "cyan": "36", "bold": "1"}
    return f"\033[{codes[color]}m{text}\033[0m"


# --------------------------------------------------------------------- HTTP
class ApiError(Exception):
    def __init__(self, status, detail):
        self.status = status
        self.detail = detail
        msg = detail.get("message") if isinstance(detail, dict) else detail
        super().__init__(msg if isinstance(msg, str) else json.dumps(msg))


class Unreachable(Exception):
    pass


def _request(method, path, *, data=None, headers=None, timeout=60, base=None):
    b = (base or base_url()).rstrip("/")
    url = path if path.startswith("http") else b + path
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=_headers(headers))
    ctx = _ssl_ctx() if url.startswith("https") else None
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except ValueError:
            detail = body
        raise ApiError(e.code, detail)
    except (urllib.error.URLError, ConnectionError, socket.timeout, OSError) as e:
        raise Unreachable(str(e))


def api_get(path, timeout=30):
    with _request("GET", path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def api_post(path, obj=None, timeout=120):
    data = json.dumps(obj or {}).encode()
    with _request("POST", path, data=data,
                  headers={"Content-Type": "application/json"}, timeout=timeout) as r:
        body = r.read().decode()
        return json.loads(body) if body else None


def api_delete(path, timeout=60):
    with _request("DELETE", path, timeout=timeout) as r:
        body = r.read().decode()
        return json.loads(body) if body else None


def api_multipart(path, fields, files, timeout=300):
    """POST multipart/form-data, built by hand so we stay stdlib-only (no
    requests/httpx). `files` maps name → (filename, bytes, content_type)."""
    boundary = "----vc" + base64.b16encode(os.urandom(8)).decode()
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fn, content, ctype) in files.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    with _request("POST", path, data=body,
                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                  timeout=timeout) as r:
        out = r.read().decode()
        return json.loads(out) if out else None


def post_binary(path, obj, timeout=300) -> bytes:
    """POST JSON, return the raw response body (used for /speak's WAV)."""
    data = json.dumps(obj).encode()
    with _request("POST", path, data=data,
                  headers={"Content-Type": "application/json"}, timeout=timeout) as r:
        return r.read()


def post_sse(path, obj, timeout=300):
    """POST JSON, yield each `data: {...}` SSE frame as a dict (for /speak/stream)."""
    data = json.dumps(obj).encode()
    with _request("POST", path, data=data,
                  headers={"Content-Type": "application/json",
                           "Accept": "text/event-stream"}, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except ValueError:
                continue


# --------------------------------------------------------------------- helpers
def die(msg, code=EXIT_ERR):
    print(c("error:", "red") + " " + msg, file=sys.stderr)
    sys.exit(code)


def require_service():
    """Probe /health across candidate base URLs; cache the one that answers.

    Exits 2 (unreachable) with a friendly hint when nothing answers, matching
    the sibling CLIs' `require_daemon()`."""
    global _BASE
    for cand in _candidates():
        try:
            with _request("GET", "/health", timeout=8, base=cand) as r:
                _BASE = cand
                return json.loads(r.read().decode())
        except ApiError as e:
            if e.status in (401, 403):
                die("unauthorized — set VOICE_API_KEY to match the service.",
                    EXIT_UNREACHABLE)
            _BASE = cand  # reachable, just an odd /health — let the caller see it
            raise
        except Unreachable:
            continue
    die(f"voice service not running at {_candidates()[0]} — start it:  "
        f"./dev.sh up   (or  ./vc serve)", EXIT_UNREACHABLE)


def _lan_ip() -> str:
    """Best-effort primary LAN IP (no packet sent). "" if undeterminable."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return ""


def connect_info() -> dict:
    """The {kind, base_url, alt_base_url, auth_required} to register in
    MiniClosedAI. Prefers the server's /api/connect-info; falls back to deriving
    it from the working base (localhost → LAN IP) for older services that lack
    the endpoint."""
    try:
        return api_get("/api/connect-info")
    except ApiError as e:
        if e.status != 404:
            raise
    u = urllib.parse.urlsplit(base_url())
    host = u.hostname or "localhost"
    if host in ("localhost", "127.0.0.1"):
        host = _lan_ip() or host
    port = u.port or int(cfg("VOICE_PORT", "8090"))
    return {
        "kind": "voice",
        "base_url": f"{u.scheme}://{host}:{port}",
        "alt_base_url": f"http://host.docker.internal:{port}",
        "auth_required": bool(cfg("VOICE_API_KEY")),
    }


def flat_voices() -> list[dict]:
    """Flatten the /voices catalog ({lang: [{id,name,gender?}]}) to a list with
    a `lang` field on each entry."""
    cat = api_get("/voices")
    out = []
    for lang, items in cat.items():
        for v in items:
            out.append({"lang": lang, **v})
    return out


def resolve_voice(key: str) -> str:
    """Forgiving voice-id match: exact → prefix → name substring (like the
    sibling CLIs' model-id matching)."""
    voices = flat_voices()
    ids = [v["id"] for v in voices]
    if key in ids:
        return key
    pref = sorted({i for i in ids if i.startswith(key)})
    if len(pref) == 1:
        return pref[0]
    byname = sorted({v["id"] for v in voices if key.lower() in v.get("name", "").lower()})
    if len(byname) == 1:
        return byname[0]
    if not voices:
        die("no voices yet — clone one:  vc clone <audio> --name NAME")
    cands = pref or byname or sorted(set(ids))
    die(f"'{key}' didn't match one voice. Candidates: {', '.join(cands)}")


def _table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(c(line, "bold"))
    for r in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(r)))


_AUDIO_CTYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".mp4": "audio/mp4", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".flac": "audio/flac", ".webm": "audio/webm",
}


def _audio_ctype(p: Path) -> str:
    return _AUDIO_CTYPES.get(p.suffix.lower(), "application/octet-stream")


def _play(path: str):
    """Play a WAV through the first available system player. A no-op (with a
    hint) when none is installed — never an error, so headless use is fine."""
    for tool, argv in (("ffplay", [path]), ("paplay", [path]),
                       ("aplay", [path]), ("afplay", [path])):
        exe = shutil.which(tool)
        if not exe:
            continue
        args = [exe]
        if tool == "ffplay":
            args += ["-nodisp", "-autoexit", "-loglevel", "quiet"]
        args += argv
        try:
            subprocess.run(args, check=False)
        except Exception as e:
            print(c(f"(playback via {tool} failed: {e})", "dim"))
        return
    print(c("(no audio player found — install ffmpeg/alsa to use --play)", "dim"))


# --------------------------------------------------------------------- commands
def cmd_status(args):
    h = require_service()
    info = {}
    try:
        info = connect_info()
    except ApiError:
        pass
    count = 0
    try:
        count = len({v["id"] for v in flat_voices()})
    except ApiError:
        pass
    if args.json:
        print(json.dumps({"health": h, "connect": info, "voice_count": count}, indent=2))
        return
    ok = h.get("ok")
    print(f"{c('status', 'dim')}     {c('ready', 'green') if ok else c('degraded', 'yellow')}")
    print(f"{c('asr', 'dim')}        {h.get('asr_model')}")
    print(f"{c('tts', 'dim')}        {h.get('tts_model')}")
    print(f"{c('device', 'dim')}     {h.get('device')}  (models_loaded={h.get('voices_loaded')})")
    if info.get("base_url"):
        print(f"{c('base_url', 'dim')}   {c(info['base_url'], 'cyan')}")
    print(f"{c('voices', 'dim')}     {count}")


def cmd_url(args):
    require_service()
    info = connect_info()
    print(f"Kind:     {info.get('kind', 'voice')}")
    print(f"Base URL: {c(info['base_url'], 'cyan')}")
    if info.get("alt_base_url"):
        print(c(f"(same-host Docker miniclosedai? use {info['alt_base_url']})", "dim"))
    if info.get("auth_required"):
        print(c("(auth on — set this endpoint's API key to match VOICE_API_KEY)", "dim"))
    print(c("register in MiniClosedAI: Settings → Backends → Add endpoint (Kind: voice)", "dim"))


def cmd_voices(args):
    require_service()
    cat = api_get("/voices")
    if args.json:
        print(json.dumps(cat, indent=2))
        return
    rows = []
    for lang in sorted(cat):
        for v in cat[lang]:
            rows.append([lang, v["id"], v.get("name", ""), v.get("gender", "") or ""])
    if not rows:
        print(c("no voices yet — clone one:  vc clone <audio> --name NAME", "dim"))
        return
    _table(rows, ["LANG", "ID", "NAME", "GENDER"])


def cmd_clone(args):
    require_service()
    p = Path(args.audio)
    if not p.exists():
        die(f"file not found: {args.audio}")
    ctype = "audio/wav" if p.suffix.lower() == ".wav" else _audio_ctype(p)
    if not ctype.startswith("audio/wav"):
        print(c(f"note: the service accepts WAV for cloning; sending {p.name} "
                f"as {ctype} may be rejected (convert with ffmpeg first).", "yellow"))
    try:
        out = api_multipart("/voices",
                            {"name": args.name, "language": args.language},
                            {"audio": (p.name, p.read_bytes(), ctype)})
    except ApiError as e:
        die(str(e))
    print(f"cloned {c(out['voice_id'], 'bold')}  "
          f"({out.get('language')}, {out.get('duration_sec')}s @ {out.get('sample_rate')}Hz)")
    print(c(f"try it:  vc speak \"hello\" --voice {out['voice_id']} "
            f"--language {out.get('language', 'en')}", "dim"))


def cmd_rm(args):
    require_service()
    vid = resolve_voice(args.voice_id)
    try:
        api_delete(f"/voices/{vid}")
    except ApiError as e:
        die(str(e))
    print(f"removed {c(vid, 'bold')}")


def cmd_speak(args):
    require_service()
    voice = resolve_voice(args.voice)
    out = Path(args.out)
    payload = {"text": args.text, "voice": voice, "language": args.language}
    if args.speed is not None:
        payload["speed"] = args.speed

    if args.stream:
        # Assemble the SSE PCM chunks into a WAV with the stdlib wave module.
        pcm = bytearray()
        sample_rate = None
        for ev in post_sse("/speak/stream", payload):
            if ev.get("error"):
                die(f"speak failed: {ev['error']}")
            if "chunk_b64" in ev:
                pcm += base64.b64decode(ev["chunk_b64"])
                sample_rate = ev.get("sample_rate", sample_rate)
            if ev.get("done"):
                break
        if not pcm or not sample_rate:
            die("stream produced no audio.")
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # int16
            w.setframerate(int(sample_rate))
            w.writeframes(bytes(pcm))
    else:
        wav = post_binary("/speak", payload)
        if not wav:
            die("speak produced no audio.")
        out.write_bytes(wav)

    print(str(out))
    if args.play:
        _play(str(out))


def cmd_transcribe(args):
    require_service()
    p = Path(args.audio)
    if not p.exists():
        die(f"file not found: {args.audio}")
    fields = {"language": args.language} if args.language else {}
    try:
        res = api_multipart("/transcribe", fields,
                            {"audio": (p.name, p.read_bytes(), _audio_ctype(p))})
    except ApiError as e:
        die(str(e))
    if args.json:
        print(json.dumps(res, indent=2))
        return
    print((res.get("text") or "").strip())


def cmd_serve(args):
    dev = ROOT / "dev.sh"
    if not dev.exists():
        die("dev.sh not found")
    os.execv("/bin/bash", ["bash", str(dev), "up"])


# --------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(
        prog="vc", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    for name in ("status", "info"):
        s = sub.add_parser(name, help="service health + connect URL + voice count")
        s.add_argument("--json", action="store_true")
        s.set_defaults(fn=cmd_status)

    s = sub.add_parser("url", help="base URL to register in MiniClosedAI (Kind: voice)")
    s.set_defaults(fn=cmd_url)

    for name in ("voices", "ls"):
        s = sub.add_parser(name, help="list available voices")
        s.add_argument("--json", action="store_true")
        s.set_defaults(fn=cmd_voices)

    s = sub.add_parser("clone", help="clone a voice from a WAV sample")
    s.add_argument("audio", help="path to a WAV clip (0.5–35 s)")
    s.add_argument("--name", required=True, help="display name shown in the dropdown")
    s.add_argument("--language", default="en", help="en | es (default: en)")
    s.set_defaults(fn=cmd_clone)

    s = sub.add_parser("rm", help="remove a cloned voice")
    s.add_argument("voice_id")
    s.set_defaults(fn=cmd_rm)

    s = sub.add_parser("speak", help="synthesize text to a WAV file")
    s.add_argument("text")
    s.add_argument("--voice", default="default", help="voice id (default: default)")
    s.add_argument("--language", default="en", help="en | es (default: en)")
    s.add_argument("--speed", type=float, help="0.5–2.0 speed multiplier")
    s.add_argument("--out", default="speech.wav", help="output WAV path (default: speech.wav)")
    s.add_argument("--play", action="store_true", help="play after saving (needs ffplay/aplay)")
    s.add_argument("--stream", action="store_true", help="use the streaming /speak/stream endpoint")
    s.set_defaults(fn=cmd_speak)

    s = sub.add_parser("transcribe", help="transcribe an audio file to text")
    s.add_argument("audio", help="path to an audio file (any ffmpeg-readable format)")
    s.add_argument("--language", help="language hint, e.g. en (default: auto-detect)")
    s.add_argument("--json", action="store_true", help="full {text, language, segments}")
    s.set_defaults(fn=cmd_transcribe)

    sub.add_parser("serve", help="start the service (runs ./dev.sh up)").set_defaults(fn=cmd_serve)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return EXIT_OK
    try:
        args.fn(args)
        return EXIT_OK
    except Unreachable:
        die(f"voice service not running at {base_url()} — start it:  "
            f"./dev.sh up   (or  ./vc serve)", EXIT_UNREACHABLE)
    except ApiError as e:
        die(str(e))
    except KeyboardInterrupt:
        return EXIT_OK
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

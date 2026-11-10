"""The wire protocol between the app (the client, a weak laptop) and `peresmeshnik serve` (the strong box): one WebSocket
over TLS per side, so the two sides drop and reconnect on their own. Both ends import this module; a change here is a
change of the contract (VERSION).

    wss://<host>:47000/v1/side/them     them -> me: the call's audio up; subtitles, latency and my language's voice down
    wss://<host>:47000/v1/side/me       me -> them: my microphone up; subtitles and my voice in their language down
    wss://<host>:47000/v1/pair          pairing: a one-time code from `serve` -> a device token

The server's certificate is self-signed; the client pins its SHA-256 (`fingerprint`) at pairing and checks it before it
sends anything. The token goes only in `hello`, never in a URL or a header (they end up in logs).

A side session, in order:
1. client: `hello` (text) - VERSION, the device token, the side, the languages, and for a voice the server must speak
   with (me -> them: my confirmed profile only, checked on the client) its `{"sha256", "bytes"}`, else null;
2. server: `send_voice` when it has no voice with that sha256 (a warm session keeps one) -> the client sends the profile
   folder as a zip (`pack_voice`) in binary messages of at most CHUNK bytes, `bytes` in all; the server `unpack_voice`s it;
3. server: `ready` once its engines are up (the models load: seconds to ~30 s for a clone), or `error` and it closes;
4. both ways until the client closes:
   - up, binary: the capture's audio as it comes, int16 LE mono at 16 kHz, any length (`pcm_bytes`);
   - up, text: `interrupt` (barge-in: I talk over this side's voice) -> the server drops the voice not sent yet and
     answers `interrupted`; voice that was on its way before that is dropped by the client;
   - down, text: `caption` and `latency` (engines.Caption / Latency, the server's clock: from the clause end it heard),
     `state` (a tray line of the server's side, 'note: ...' / 'error: ...'), `error`;
   - down, binary: speech, a u32 LE sample rate + int16 LE mono PCM, at most SPEECH_MS a piece: playback starts on the
     first piece, not after a whole sentence (BENCH remote-0: the network then adds one RTT to the first sound).
Pairing: client `pair` {nonce, mac = pair_mac(code, fp, nonce)} -> server `paired` {token, mac = pair_mac(code, fp, nonce,
token)}: each proves it knows the code for this certificate; a man in the middle with his own certificate fails both.

Liveness is the WebSocket's own ping (the client: every PING_S, no pong in PONG_S -> the connection is dead).
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import struct
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from peresmeshnik.engines import Caption, Latency, Speech

VERSION = 1
PORT = 47000
SIDES = ('them', 'me')
PAIR_PATH = '/v1/pair'
SPEECH_MS = 100  # the server's voice goes down in pieces this long
CHUNK = 1 << 20  # a voice upload's binary message at most; the server's max message size (the audio is ~1 KB a frame)
VOICE_MAX = 64 << 20  # a voice profile, zipped
PING_S, PONG_S = 1.0, 3.0
# error codes: the client does not try again by itself after these (a new pairing, an update, other languages);
# after 'busy' (another session is on), 'internal' or a lost connection it does
PERMANENT = frozenset({'version', 'auth', 'hello', 'lang', 'voice'})
ERRORS = PERMANENT | {'busy', 'internal', 'code', 'attempts'}  # 'code', 'attempts': pairing


class ProtocolError(Exception):
    """A message that breaks the protocol; `code` goes to the other end in `error`."""

    def __init__(self, code: str, text: str):
        super().__init__(text)
        self.code = code


def side_path(side: str) -> str:
    return f'/v1/side/{side}'


def message(type_: str, **kw) -> str:
    return json.dumps({'type': type_, **kw}, ensure_ascii=False)


def error(code: str, text: str) -> str:
    assert code in ERRORS, code
    return message('error', code=code, text=text)


# --- hello ---------------------------------------------------------------------------------------------------------

def hello(token: str, side: str, src: str, tgt: str, voice: dict | None = None, app: str = '') -> str:
    """`voice`: {'sha256': hex, 'bytes': n} of `pack_voice`'s zip, or None (the server's stock voice)."""
    return message('hello', version=VERSION, app=app, token=token, side=side, src=src, tgt=tgt, voice=voice)


def parse_hello(text: str | bytes, side: str) -> dict:
    """The server's check of the first message on /v1/side/<side>; raises ProtocolError. The token is checked by the
    caller (pairing), after the version: an old client hears 'version', not 'auth'."""
    from peresmeshnik.config import LANGS
    try:
        h = json.loads(text)
    except (TypeError, ValueError):
        raise ProtocolError('hello', 'the first message is not a hello') from None
    if not isinstance(h, dict) or h.get('type') != 'hello':
        raise ProtocolError('hello', 'the first message is not a hello')
    if h.get('version') != VERSION:
        raise ProtocolError('version', f'protocol {h.get("version")} here, the server speaks {VERSION}: update the app on '
                                       'both computers to the same version')
    if not isinstance(h.get('token'), str) or not h['token']:
        raise ProtocolError('auth', 'no device token: pair this computer with the server')
    if h.get('side') != side:
        raise ProtocolError('hello', f'hello for side {h.get("side")!r} on /v1/side/{side}')
    if h.get('src') not in LANGS or h.get('tgt') not in LANGS or h['src'] == h['tgt']:
        raise ProtocolError('lang', f'languages {h.get("src")!r} -> {h.get("tgt")!r}')
    v = h.get('voice')
    if v is not None and not (isinstance(v, dict) and isinstance(v.get('sha256'), str) and len(v['sha256']) == 64
                              and type(v.get('bytes')) is int and 0 < v['bytes'] <= VOICE_MAX):
        raise ProtocolError('hello', 'a bad voice in hello')
    return h


# --- events and audio ----------------------------------------------------------------------------------------------

def pcm_bytes(pcm: np.ndarray) -> bytes:
    """Float32 [-1, 1] (or int16) -> int16 LE."""
    if pcm.dtype != np.int16:
        pcm = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    return pcm.astype('<i2', copy=False).tobytes()


def pcm_from(b: bytes, offset: int = 0) -> np.ndarray:
    """int16 LE -> float32 [-1, 1]."""
    return np.frombuffer(b, '<i2', offset=offset).astype(np.float32) / 32768


def event(e: Caption | Latency) -> str:
    if isinstance(e, Latency):
        return message('latency', clause=e.clause, what=e.what, s=e.s)
    return message('caption', original=e.original, translation=e.translation, final=e.final, clause=e.clause)


def speech(e: Speech) -> list[bytes]:
    """A Speech event -> binary messages of at most SPEECH_MS each."""
    n = max(1, e.sample_rate * SPEECH_MS // 1000)
    head = struct.pack('<I', e.sample_rate)
    return [head + pcm_bytes(e.pcm[i:i + n]) for i in range(0, len(e.pcm), n)]


def decode(m: str | bytes) -> Caption | Latency | Speech | dict:
    """A message from the server: an engines event, or a control message as a dict ('ready', 'state', 'error', ...)."""
    if isinstance(m, bytes):
        if len(m) < 4 or len(m) % 2:
            raise ProtocolError('internal', f'a speech message of {len(m)} bytes')
        return Speech(pcm_from(m, 4), struct.unpack_from('<I', m)[0])
    d = json.loads(m)
    if d.get('type') == 'caption':
        return Caption(d['original'], d['translation'], d['final'], d['clause'])
    if d.get('type') == 'latency':
        return Latency(d['clause'], d['what'], d['s'])
    return d


# --- the voice profile ---------------------------------------------------------------------------------------------

def pack_voice(folder: Path) -> bytes:
    """A voice profile folder (voice.json, reference.wav, the engines' caches) -> one zip, files in name order (so the
    same profile gives the same sha256 and a warm session keeps it). Subfolders are kept, links are not followed."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not p.is_symlink():
                info = zipfile.ZipInfo(p.relative_to(folder).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, p.read_bytes())
    data = buf.getvalue()
    if len(data) > VOICE_MAX:
        raise ValueError(f'the voice profile is {len(data) >> 20} MB zipped, at most {VOICE_MAX >> 20} MB')
    return data


def voice_ref(data: bytes) -> dict:
    """hello's `voice` for `pack_voice`'s zip."""
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def unpack_voice(data: bytes, dest: Path) -> None:
    """The server's side of `pack_voice`, into an empty folder: no absolute names, no '..', at most VOICE_MAX unpacked
    (a zip bomb), voice.json must be there. Raises ProtocolError('voice', ...)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            if sum(i.file_size for i in infos) > VOICE_MAX:
                raise ProtocolError('voice', 'the voice profile unpacks too big')
            names = []
            for i in infos:
                p = PurePosixPath(i.filename)
                if i.is_dir():
                    continue
                if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename:
                    raise ProtocolError('voice', f'a bad name in the voice profile: {i.filename!r}')
                names.append((i, p))
            if 'voice.json' not in {p.as_posix() for _, p in names}:
                raise ProtocolError('voice', 'no voice.json in the voice profile')
            for i, p in names:
                out = dest.joinpath(*p.parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(i))  # at most its file_size (summed above), CRC checked
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        raise ProtocolError('voice', f'the voice profile does not unpack: {e}') from None


# --- pairing -------------------------------------------------------------------------------------------------------

def fingerprint(der: bytes) -> str:
    """SHA-256 of a certificate (DER), hex: what the client pins and `serve` shows."""
    return hashlib.sha256(der).hexdigest()


def norm_code(code: str) -> str:
    """The pairing code as typed: case, dashes and spaces do not matter; 0 and 1 read as O and I (base32 has neither)."""
    return code.upper().replace('-', '').replace(' ', '').replace('0', 'O').replace('1', 'I')


def pair_mac(code: str, fp: str, nonce: str, token: str = '') -> str:
    """The client's proof (no token) or the server's (with the token it hands out) that it knows the code, bound to this
    certificate and this pairing's nonce."""
    who = b'server' if token else b'client'
    msg = b'\0'.join((b'peresmeshnik-pair', who, fp.encode(), nonce.encode(), token.encode()))
    return hmac.new(norm_code(code).encode(), msg, hashlib.sha256).hexdigest()


def pair(code: str, fp: str, nonce: str, name: str, app: str = '') -> str:
    """The client's message on /v1/pair; `nonce`: 32 random bytes, hex; `name`: this computer, for `serve --list`."""
    return message('pair', version=VERSION, app=app, name=name, nonce=nonce, mac=pair_mac(code, fp, nonce))


def paired(code: str, fp: str, nonce: str, token: str) -> str:
    """The server's answer to a good `pair`: the device token (32 random bytes, hex) and its proof."""
    return message('paired', token=token, mac=pair_mac(code, fp, nonce, token))

"""`peresmeshnik serve`: this computer translates the sides of a weak one (the app with a side's engine 'remote', its
remote/client.RemoteDirection), over the protocol in remote/protocol.py.

    peresmeshnik serve [--host ADDR] [--port 47000] [--public] [--log] [--code] [--fetch en-ru [--accept-licenses]]
    peresmeshnik serve --list | --revoke NAME

One TLS WebSocket per side (its own certificate, pairing.cert, kept in <data dir>/server); /v1/pair hands out device
tokens for the one-time code `serve` shows with its address (`https://<address>:<port>/#<code>`), made at the start when
no device is paired yet or with --code. A side's session: hello (the version first, then the token: pairing.Pairing),
the voice it speaks with when the server has not got it (a zip in pieces, its sha256 and size checked, unpacked into a
folder of its own, 0700), then the side's engines - the app's own `Side` and its local path (`Side._local`: the clone or
a stock voice, the stock voice when the clone cannot, the Piper runner started again when it dies), its input, output
and subtitles the connection (`Net`) - then `ready`, and audio up, subtitles, latency, state and voice down until the
client closes; `interrupt` drops the voice not sent yet, `interrupted` says so. No echo gate and no live voice here:
the client has them (its LiveSwitch wraps the stream; none here, `_Unswitched`).

- one session: one paired computer with its two sides; another one hears 'busy' while any side of the first is on, and
  takes the server when they are all off (their engines go);
- warm: a side's engines stay `warm_s` after its client left; a hello for the same side with the same languages and the
  same voice (sha256) gets them back at once, without loading the models (a reconnect after a drop: the clause the drop
  cut is closed by a pause and its voice dropped, the client sends it again);
- the voice: only from this connection's hello, so "me -> them" speaks only with the voice that came on its own side,
  and a contact's voice only ever on 'them'; nothing is shared between devices. Its folder goes when the side's session
  ends, and every folder left behind (a crash) when the server starts;
- the log keeps numbers: the clause text and translation that pipeline.py logs are taken out (`NoText`), with --log
  (`[server] log_text`) they stay. Neither sound nor text is written to disk;
- headless: neither Qt nor a sound device is touched (no PySide6, no sounddevice);
- guarded: only private addresses without --public, no browser (an Origin header is refused), the token checked right
  after hello and before anything else, at most MAX_UNAUTH connections without one, /v1/pair only while a code is open,
  a device unpaired by another process (`--revoke`) cut off within REVOKE_S, "me -> them" only with my own voice
  confirmed live (`not_mine`), nothing but SAFE files kept of an uploaded profile (no pickle is ever loaded from it).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import errno
import hashlib
import json
import logging
import shutil
import socket
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from peresmeshnik import __version__
from peresmeshnik.app import Side
from peresmeshnik.engines import SAMPLE_RATE, Speech
from peresmeshnik.remote import pairing, protocol
from peresmeshnik.remote.protocol import ProtocolError

log = logging.getLogger('peresmeshnik.server')
HELLO_S = 10.0  # the first message comes this soon after the connection, else it is closed
VOICE_S = 120.0  # a voice upload (at most protocol.VOICE_MAX) takes at most this long
PING_S = 5.0  # a client gone without a reset (Wi-Fi) is noticed in twice this: its side goes warm, another computer may come
KEY = {'them': 'them_to_me', 'me': 'me_to_them'}
MAX_UNAUTH = 8  # connections at once that have not shown a device token yet (each may hold a message of protocol.CHUNK)
REVOKE_S = 5.0  # a device unpaired from another process (`serve --revoke`) loses its session within this
SAFE = {'.json', '.wav', '.flac', '.ogg', '.npy', '.npz'}  # what is kept of an uploaded profile: no pickle (xtts.pt,
# qwen3.pt are torch.load'ed: the engines compute them here again)
TEXT = ('text', 'translation')  # what pipeline.py logs that was said


def not_mine(meta: dict) -> str:
    """'' for my own voice confirmed at the microphone (voice.json v2: kind 'me', consent by a live reading), else why
    not: only that may speak as me (M-20), whatever the client says."""
    # ponytail: S3's voice.why_not_mine is the same rule; this goes when stream S3 lands in main
    if meta.get('version', 1) < 2:
        return 'a profile of an older app: update the app on the laptop'
    if meta.get('kind') != 'me':
        return 'it is not your own voice'
    if (meta.get('consent') or {}).get('how') not in ('mic-live', 'import+mic-match'):
        return 'it is not confirmed as your own voice yet'
    return ''


class NoText(logging.Filter):
    """pipeline.py's per-clause JSON line without what was said and its translation: only the numbers stay."""

    def filter(self, r: logging.LogRecord) -> bool:
        try:
            d = json.loads(r.getMessage())
        except (TypeError, ValueError):
            return True
        if isinstance(d, dict) and any(k in d for k in TEXT):
            r.msg, r.args = json.dumps({k: v for k, v in d.items() if k not in TEXT}, ensure_ascii=False), ()
        return True


class Conn:
    """One client connection: what goes down is queued here (from any thread) and sent in order by `run`. Speech is
    tagged with the barge-in count it was made under: after an interrupt the speech queued before it is not sent."""

    def __init__(self, ws, loop: asyncio.AbstractEventLoop):
        self.ws, self.loop, self.epoch = ws, loop, 0
        self.authed = False  # its hello's token was checked (Server.unauth)
        self.q: asyncio.Queue = asyncio.Queue()

    def put(self, data, speech: bool = False) -> None:
        try:
            self.loop.call_soon_threadsafe(self.q.put_nowait, (data, self.epoch if speech else None))
        except RuntimeError:  # the server's loop is gone
            pass

    def end(self) -> None:
        """What is queued goes, then the connection closes."""
        try:
            self.loop.call_soon_threadsafe(self.q.put_nowait, None)
        except RuntimeError:
            pass

    async def run(self) -> None:
        from websockets.exceptions import ConnectionClosed
        try:
            while (item := await self.q.get()) is not None:
                data, epoch = item
                if epoch is None or epoch == self.epoch:
                    await self.ws.send(data)
            await self.ws.close()
        except ConnectionClosed:
            pass


class Net:
    """A side's input, output and subtitles (app.Side's `inp`, `out`, `overlay`): the connection it has now, none while
    it is warm (what it says then is dropped)."""

    def __init__(self):
        self.conn: Conn | None = None
        self._frames = None

    # inp
    def start(self, on_frames) -> None:
        self._frames = on_frames

    def feed(self, pcm: np.ndarray) -> None:  # the server's loop: audio from the client
        if self._frames is not None:
            self._frames(pcm)

    def stop(self) -> None:
        pass

    # out
    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        if (c := self.conn) is not None:
            for m in protocol.speech(Speech(pcm, sample_rate)):
                c.put(m, speech=True)

    def flush(self) -> None:
        if (c := self.conn) is not None:
            c.epoch += 1

    # overlay
    def show(self, e) -> None:
        if (c := self.conn) is not None:
            c.put(protocol.event(e))

    latency = show

    def state(self, text: str) -> None:  # Side's on_error: a tray line on the client
        if (c := self.conn) is not None:
            c.put(protocol.message('state', text=text))


class _Unswitched:
    """A side's local direction as it is. app.Side wraps 'me' in audio/mic.LiveSwitch unless the direction switches to
    the live voice itself; on the server nothing goes live (the client's own LiveSwitch does), so it says it does."""
    supports_live = True

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def set_live(self, on: bool) -> None:
        pass


class ServerSide(Side):
    """app.Side as it is, its local path without the live-voice switch."""

    def _local(self, s):
        d = super()._local(s)
        return d if isinstance(d, _Unswitched) else _Unswitched(d)

    stock = False  # the clone could not speak: a stock voice does

    def _fallback(self, s, why):  # the clone cannot speak (also mid-call, Side._clone_died): the stock voice
        try:
            d = _Unswitched(super()._fallback(s, why))
        except RuntimeError as e:  # zh, ar: no stock voice either; trying again does not cure that
            raise ProtocolError('lang', str(e)) from None
        self.stock = True
        return d


class Slot:
    """A side's session: its engines (built once, kept while warm), its voice folder, the connection it has now."""

    def __init__(self, side: str, device: str, key: tuple):
        self.side, self.device, self.key = side, device, key
        self.sha = ''  # sha256 of the device token it came with: unpaired meanwhile -> the session goes
        self.refused = ''  # the voice sent is not mine to speak with: a stock voice, and why
        self.net = Net()
        self.conn: Conn | None = None
        self.task: asyncio.Future | None = None  # -> the ServerSide, built on a thread
        self.folder: Path | None = None
        self.expire: asyncio.TimerHandle | None = None
        self.notes: tuple[str, ...] = ()  # state lines for the client after `ready`
        self.voice = ''  # 'clone' | 'stock': what it speaks with
        self.dead = self.failed = False

    @property
    def engines(self) -> ServerSide | None:
        t = self.task
        return t.result() if t is not None and t.done() and not t.cancelled() and t.exception() is None else None


class Server:
    """`folder`: the certificate, paired.json and the voices' folders (<data dir>/server); `settings`: the server's own
    (its [local] models and threads, [server])."""

    def __init__(self, folder: Path, settings, host: str | list[str] = '127.0.0.1', port: int = protocol.PORT,
                 log_text: bool = False, warm_s: float | None = None, public: bool = False):
        for h in host if isinstance(host, list) else [host]:  # never '' / 0.0.0.0 / a public one unless --public
            pairing.check_bind(h, public)  # ValueError
        self.folder, self.settings, self.host, self.port = folder, settings, host, port
        self.log_text = log_text
        self.warm_s = settings.server.warm_s if warm_s is None else warm_s
        self.crt, self.key, self.fp = pairing.cert(folder)
        self.pairing = pairing.Pairing(folder, self.fp)
        self.voices = folder / 'voices'
        self.slots: dict[str, Slot] = {}
        self.started = threading.Event()
        self.on_start = None  # called once it listens (serve: the pairing line, never for a port another server holds)
        self.ports: list[int] = []
        self.device = ''
        self._stops: list[threading.Thread] = []
        self.unauth = 0  # connections without a checked token now (MAX_UNAUTH)
        self._filter = NoText()
        self._stop: asyncio.Event | None = None
        self.lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def clean(self) -> None:
        """Voice folders a crashed server left behind go; the folder of them is this user's only (0700)."""
        self.voices.mkdir(parents=True, exist_ok=True)
        self.voices.chmod(0o700)
        for p in self.voices.iterdir():
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    async def serve(self) -> None:
        """Until `stop()` (or cancelled: Ctrl+C); every side's engines are stopped and their voices removed at the end."""
        import ssl

        from websockets.asyncio.server import serve
        self._loop, self._stop, self.lock = asyncio.get_running_loop(), asyncio.Event(), asyncio.Lock()
        if not self.log_text:
            logging.getLogger('peresmeshnik.latency').addFilter(self._filter)
        if (wl := logging.getLogger('websockets')).getEffectiveLevel() < logging.INFO:
            wl.setLevel(logging.INFO)  # its DEBUG logs every frame: hello's token, pair's nonce and mac
        try:
            import signal
            self._loop.add_signal_handler(signal.SIGTERM, self._stop.set)  # docker stop: the voices go too
        except (NotImplementedError, AttributeError, RuntimeError, ValueError):  # Windows; not the main thread
            pass
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(self.crt, self.key)
        try:
            from peresmeshnik.languages import gpu
            self.device = 'cuda' if await asyncio.to_thread(gpu) else 'cpu'
            async with serve(self._handler, self.host, self.port, ssl=ctx, compression=None, max_size=protocol.CHUNK,
                             ping_interval=PING_S, ping_timeout=PING_S, close_timeout=1, server_header=None,
                             origins=[None]) as srv:  # no Origin: not a browser (a web page could burn the code)
                self.ports = sorted({s.getsockname()[1] for s in srv.sockets})
                self.clean()  # once the port is ours (not another server's voices), before any session (on the loop)
                self.started.set()
                if self.on_start:
                    self.on_start()
                watch = asyncio.create_task(self._revoked())
                await self._stop.wait()
                watch.cancel()
        finally:
            for slot in list(self.slots.values()):
                self._drop(slot)
            for t in self._stops:
                t.join(20)
            logging.getLogger('peresmeshnik.latency').removeFilter(self._filter)

    def stop(self) -> None:
        """From another thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)

    # connections ----------------------------------------------------------------------------------
    async def _handler(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed
        # small messages (pong, captions, the voice's pieces) not held back: asyncio sets it on Linux only (CI: not on
        # Windows or macOS), BENCH remote-0
        ws.transport.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        path = ws.request.path
        side = next((x for x in protocol.SIDES if path == protocol.side_path(x)), None)
        if side is None and path != protocol.PAIR_PATH:
            await ws.close(1008, 'not found')
            return
        if self.unauth >= MAX_UNAUTH:  # many connections that showed no token yet: not one more
            await ws.close(1013, 'too many connections')
            return
        if side is None:
            self.unauth += 1
            try:
                await self._pair(ws)
            except (TimeoutError, ConnectionClosed):
                pass
            finally:
                self.unauth -= 1
            return
        self.unauth += 1
        conn = Conn(ws, self._loop)
        sender = asyncio.create_task(conn.run())
        try:
            if (slot := await self._open(conn, side)) is not None:
                await self._stream(conn, slot)
        except ProtocolError as e:
            log.info('%s: refused: %s (%s)', side, e.code, e)
            conn.put(protocol.error(e.code, str(e)))
        except (TimeoutError, ConnectionClosed):
            pass
        except Exception as e:  # a bug: the client hears it, the server goes on
            log.exception('%s: the session failed', side)
            conn.put(protocol.error('internal', f'{type(e).__name__}: {e}'))
        finally:
            if not conn.authed:
                self.unauth -= 1
            self._left(conn)
            conn.end()
            try:
                await asyncio.wait_for(sender, 5)
            except (TimeoutError, ConnectionClosed):
                ws.transport.abort()

    async def _pair(self, ws) -> None:
        """/v1/pair: only while a code is open (one use, pairing.CODE_S); the device paired is said on the terminal."""
        p = self.pairing
        if not p.code or p.used or p.clock() > p.until or p.tries >= pairing.CODE_TRIES:
            await ws.send(protocol.error('code', 'no pairing code is open on the server: start it with --code for one'))
            return
        m = await asyncio.wait_for(ws.recv(), HELLO_S)
        a = await asyncio.to_thread(p.handle, m)  # paired.json is written with an fsync
        await ws.send(a)
        if json.loads(a)['type'] == 'paired':
            name = ''.join(c for c in json.loads(m)['name'].strip()[:pairing.NAME_MAX] if c.isprintable())
            print(f'paired: {name} from {ws.remote_address[0]}', flush=True)  # the terminal, not the log

    async def _revoked(self) -> None:
        """Every REVOKE_S: a device unpaired meanwhile (`serve --revoke`, another process) loses its sessions, warm too."""
        while True:
            await asyncio.sleep(REVOKE_S)
            if not self.slots:
                continue
            try:
                known = {d['sha256'] for d in await asyncio.to_thread(self.pairing.devices)}
            except (OSError, ValueError, KeyError, TypeError):  # being rewritten, or broken: the next round
                continue
            for slot in [s for s in self.slots.values() if s.sha not in known]:
                log.info('%s: its computer was unpaired: the session ends', slot.side)
                if slot.conn is not None:
                    slot.conn.put(protocol.error('auth', 'this computer was removed from the server: pair it again '
                                                         '(tray menu, Server…)'))
                self._drop(slot)

    async def _open(self, conn: Conn, side: str) -> Slot | None:
        """hello, auth, one session, the voice, the engines -> the slot after `ready`; None: another connection of the
        same side took it meanwhile."""
        h = protocol.parse_hello(await asyncio.wait_for(conn.ws.recv(), HELLO_S), side)
        device = await asyncio.to_thread(self.pairing.check, h['token'])  # an fsync: not on the loop
        if device is None:
            raise ProtocolError('auth', 'this computer is not paired with the server, or it was removed from it: pair it '
                                        'again (tray menu, Server…)')
        conn.authed = True
        self.unauth -= 1
        sha = hashlib.sha256(h['token'].encode()).hexdigest()
        v = h['voice']
        key = (h['src'], h['tgt'], v and v['sha256'])
        async with self.lock:  # who has the server, and this side's slot: one hello at a time
            if any(s.conn is not None and s.device != device for s in self.slots.values()):
                raise ProtocolError('busy', 'the server is translating for another computer now')
            for s in [s for s in self.slots.values() if s.device != device]:  # all off: the last one's engines go
                self._drop(s)
            slot = self.slots.get(side)
            if slot is not None and (slot.key != key or slot.failed or slot.task is None):  # other languages or voice
                self._drop(slot)
                slot = None
            warm = slot is not None
            if warm:
                if slot.expire:
                    slot.expire.cancel()
                if slot.conn is not None:  # the client's new connection for this side: the old one is dead to it (a
                    slot.conn.ws.transport.abort()  # Wi-Fi drop without a reset: this server has not noticed yet)
                    self._cut(slot)
            else:
                slot = self.slots[side] = Slot(side, device, key)
            slot.conn, slot.sha = conn, sha
        log.info('%s: %s -> %s for %s, %s, %s', side, h['src'], h['tgt'], device, 'its voice' if v else 'a stock voice',
                 'warm' if warm else 'loading')
        try:
            if not warm:
                if v:
                    try:
                        why = await self._voice(conn, slot, v)
                    finally:  # dropped meanwhile (another hello of this side): its _drop ran before this folder was
                        if slot.dead and slot.folder is not None:  # made or while it was unpacked: it goes here
                            await asyncio.to_thread(shutil.rmtree, slot.folder, True)
                    if why:  # not my confirmed voice: it goes, and a stock voice speaks (as in the app: M-20)
                        await asyncio.to_thread(shutil.rmtree, slot.folder, True)
                        slot.folder = None
                        slot.refused = f'note: me → them: a stock voice speaks: the voice sent cannot speak for you ({why})'
                if slot.conn is not conn:
                    return None
                slot.task = asyncio.ensure_future(asyncio.to_thread(self._build, slot, h))
                slot.task.add_done_callback(lambda t: t.cancelled() or t.exception())  # retrieved: no warning if alone
            await asyncio.shield(slot.task)
        except BaseException:
            mine = slot.conn is conn
            if mine:
                slot.conn = None  # not ended by _drop: the error goes first (_handler)
            if mine or slot.task is None or (slot.task.done() and slot.engines is None):
                self._drop(slot)  # its voice did not come, or the engines did not start: nothing to keep warm
            raise
        if slot.conn is not conn:
            return None
        conn.put(protocol.message('ready', warm=warm, device=self.device, voice=slot.voice, app=__version__))
        for n in slot.notes:
            conn.put(protocol.message('state', text=n))
        slot.net.conn = conn
        return slot

    async def _voice(self, conn: Conn, slot: Slot, v: dict) -> str:
        """The voice of this hello, unpacked into the slot's own folder; '' or why it may not speak as me (side 'me')."""
        conn.put(protocol.message('send_voice'))
        data = bytearray()
        try:
            async with asyncio.timeout(VOICE_S):
                while len(data) < v['bytes']:
                    m = await conn.ws.recv()
                    if not isinstance(m, bytes):
                        raise ProtocolError('voice', 'a text message in the middle of the voice profile')
                    if len(data) + len(m) > v['bytes']:  # never more than hello said
                        raise ProtocolError('voice', 'more of the voice profile than hello said')
                    data += m
        except TimeoutError:
            raise ProtocolError('internal', f'the voice profile did not come in {VOICE_S:g} s') from None
        if len(data) != v['bytes'] or (await asyncio.to_thread(hashlib.sha256, data)).hexdigest() != v['sha256']:
            raise ProtocolError('voice', 'the voice profile arrived damaged (its size or sha256 is not the one in hello)')
        slot.folder = Path(tempfile.mkdtemp(prefix=f'{slot.side}-', dir=self.voices))  # 0700
        meta = await asyncio.to_thread(self._unpack, data, slot.folder)
        return not_mine(meta) if slot.side == 'me' else ''

    @staticmethod
    def _unpack(data: bytearray, folder: Path) -> dict:
        """protocol.unpack_voice, only the SAFE files kept, and a voice.json the app can read (a broken one is refused
        once, not retried forever) -> its meta."""
        from peresmeshnik import voice
        protocol.unpack_voice(bytes(data), folder)
        for f in [f for f in folder.rglob('*') if f.is_file() and f.suffix.lower() not in SAFE]:
            f.unlink()
        try:
            return voice.load(folder.name, folder.parent).meta
        except (OSError, ValueError) as e:
            raise ProtocolError('voice', f'the voice profile does not read: {e}') from None

    def _build(self, slot: Slot, h: dict) -> ServerSide:
        """The side's engines, on a thread (seconds; a clone ~30 s). Raises ProtocolError('lang') when this server cannot
        run the pair (no model for it, or not downloaded)."""
        from peresmeshnik import languages
        from peresmeshnik.models import ModelMissing
        from peresmeshnik.voice import STOCK
        s = copy.deepcopy(self.settings)
        s.them_to_me = s.me_to_them = 'local'
        s.forced_cloud = []
        if slot.side == 'them':
            s.their_lang, s.my_lang = h['src'], h['tgt']
        else:
            s.my_lang, s.their_lang = h['src'], h['tgt']
            if slot.folder is not None:  # voice.load(name) joins the voices' folder with it: an absolute path stays
                s.local.voice = str(slot.folder)
            else:  # no voice in hello: the stock one, not a profile of this computer's
                s.local.voice, s.local.clone = '', STOCK
        pair = languages.direction(s, KEY[slot.side])
        if not pair.local:
            raise ProtocolError('lang', f'this server cannot translate {h["src"]} -> {h["tgt"]}: {"; ".join(pair.why)}')
        said = []  # what the side says while it starts (the clone fell back to a stock voice): after `ready`
        try:
            side = ServerSide(slot.side, s, slot.net, slot.net, slot.net, on_error=said.append)
        except ModelMissing as e:
            raise ProtocolError('lang', f'the server is missing a model for {h["src"]} -> {h["tgt"]}: {e}') from None
        side.on_error = slot.net.state
        slot.notes = (*(f'note: {n}' for n in pair.notes), *filter(None, [slot.refused]), *said)
        slot.voice = 'stock' if side.stock else pair.voice
        side.on_failure = lambda why, e: self._loop.call_soon_threadsafe(self._failed, slot, why)
        side.start()
        return side

    async def _stream(self, conn: Conn, slot: Slot) -> None:
        async for m in conn.ws:
            if slot.conn is not conn:
                return
            if isinstance(m, bytes):
                if len(m) % 2 == 0:
                    slot.net.feed(protocol.pcm_from(m))
                continue
            try:
                kind = json.loads(m).get('type')
            except (ValueError, AttributeError):
                continue
            if kind == 'interrupt':
                slot.engines.interrupt()  # the pipeline drops what is not said yet, Net.flush what is not sent
                conn.put(protocol.message('interrupted'))

    def _left(self, conn: Conn) -> None:
        """A connection ended: its side stays warm."""
        for slot in list(self.slots.values()):
            if slot.conn is not conn:
                continue
            self._cut(slot)
            slot.expire = self._loop.call_later(self.warm_s, self._expire, slot)
            log.info('%s: the client left; its engines stay %g s', slot.side, self.warm_s)

    def _cut(self, slot: Slot) -> None:
        """The side's connection is gone: a pause closes the clause it cut, and that clause's voice is not said; the
        client sends the clause again on its next connection (client.KEEP_S), so it is heard whole, once."""
        slot.conn = slot.net.conn = None
        if (side := slot.engines) is not None:
            side.inp.feed(np.zeros(SAMPLE_RATE // 2, np.float32))
            side.interrupt()

    def _expire(self, slot: Slot) -> None:
        if slot.conn is None:
            log.info('%s: no client for %g s: its engines go', slot.side, self.warm_s)
            self._drop(slot)

    def _failed(self, slot: Slot, why: str) -> None:
        """The side's engines died (app.Side.on_failure): the client hears `internal` and connects again; the engines are
        built anew (seconds, a clone ~30 s), so its quick reconnect may time out and its side goes local or down until
        its next try (cloud/fallback.py retry_s)."""
        slot.failed = True
        if slot.conn is not None:
            slot.conn.put(protocol.error('internal', why))
        self._drop(slot)

    def _drop(self, slot: Slot) -> None:
        """The slot's engines stop and its voice goes (on a thread: stopping waits for the engines' threads)."""
        if slot.dead:
            return
        slot.dead = True
        if self.slots.get(slot.side) is slot:
            del self.slots[slot.side]
        if slot.expire:
            slot.expire.cancel()
        if slot.conn is not None:
            slot.conn.end()
        slot.conn = slot.net.conn = None

        def stop():
            if (side := slot.engines) is not None:
                try:
                    side.stop()
                except Exception:
                    log.exception('%s: stopping its engines', slot.side)
            if slot.folder is not None:
                shutil.rmtree(slot.folder, ignore_errors=True)

        def go(_=None):
            t = threading.Thread(target=stop, name=f'stop-{slot.side}', daemon=True)
            self._stops = [x for x in self._stops if x.is_alive()] + [t]
            t.start()
        if slot.task is None or slot.task.done():
            go()
        else:  # still loading: stopped once it is up
            slot.task.add_done_callback(go)


# --- the command ---------------------------------------------------------------------------------------------------

def hosts(host: str) -> list[str]:
    """'' = this computer's private addresses (pairing.addresses) and loopback (ssh -L)."""
    return [host] if host else [*pairing.addresses(), '127.0.0.1']


def url(host: str, port: int, code: str) -> str:
    return f'https://{f"[{host}]" if ":" in host else host}:{port}/#{code}'


def fetch(s, pairs: list[str], accept: bool = False) -> int:
    """`serve --fetch en-ru`: the models a pair needs both ways round (either computer's language may be 'mine'), each
    side local, as `peresmeshnik models fetch` gets them (its license screen, --accept-licenses): a headless server or a
    container gets its models before it starts. 0: all there."""
    from peresmeshnik import config, languages, models
    ids = []
    for p in pairs:
        a, _, b = p.partition('-')
        if a not in config.LANGS or b not in config.LANGS or a == b:
            print(f'--fetch {p}: two different languages of {", ".join(config.LANGS)}, like en-ru', file=sys.stderr)
            return 2
        for mine, theirs in ((a, b), (b, a)):
            t = copy.deepcopy(s)
            t.my_lang, t.their_lang, t.them_to_me, t.me_to_them, t.forced_cloud = mine, theirs, 'local', 'local', []
            ids += languages.needed_models(t)
    return models.cli(['fetch', *dict.fromkeys(ids), *(['--accept-licenses'] if accept else [])])


def cli(argv=None) -> int:
    from peresmeshnik import config
    s = config.load()
    d = s.server
    ap = argparse.ArgumentParser(prog='peresmeshnik serve', description='Translate for a weaker computer: it pairs with '
                                 'the line shown here (its tray menu, Server…), then its sides set to "remote" run here.')
    ap.add_argument('--host', default=d.host, help='the address to listen on (default: this computer\'s private ones '
                                                   'and 127.0.0.1)')
    ap.add_argument('--port', type=int, default=d.port)
    ap.add_argument('--public', action='store_true', default=d.public, help='allow a public address (the internet)')
    ap.add_argument('--log', action='store_true', default=d.log_text,
                    help='the log keeps what was said and its translation too (default: numbers only)')
    ap.add_argument('--code', '--pair', action='store_true', help='a new pairing code (one use, 10 minutes)')
    ap.add_argument('--list', action='store_true', help='the paired computers')
    ap.add_argument('--fetch', metavar='SRC-TGT', action='append', default=[],
                    help='download the models of this language pair (both ways round) first, then serve; again for more')
    ap.add_argument('--accept-licenses', action='store_true', help='with --fetch: accept restricted licenses without asking')
    ap.add_argument('--revoke', metavar='NAME', help='unpair a computer: its token stops working')
    a = ap.parse_args(argv)
    folder = config.data_dir() / 'server'
    if a.list or a.revoke:
        p = pairing.Pairing(folder)
        if a.revoke:
            ok = p.revoke(a.revoke)
            print(f'{a.revoke}: removed' if ok else f'{a.revoke}: not paired', flush=True)
            return 0 if ok else 1
        for dev in p.devices():
            print(f'{dev["name"]}\tpaired {dev["paired"]}\tlast seen {dev["last"]}')
        return 0
    try:
        for h in hosts(a.host):
            pairing.check_bind(h, a.public)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if a.fetch and (rc := fetch(s, a.fetch, a.accept_licenses)):
        return rc
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    logging.getLogger('websockets').setLevel(logging.WARNING)
    srv = Server(folder, s, hosts(a.host), a.port, a.log, public=a.public)
    print(f'Peresmeshnik server {__version__}, certificate SHA-256 {srv.fp}', flush=True)

    def pair_line():  # once it listens: a server already on the port keeps its own code
        code = srv.pairing.new_code()
        print('Pair a computer (its tray menu, Server…) with this line, in 10 minutes, once:', flush=True)
        shown = [x for h in hosts(a.host) for x in (pairing.addresses() if h in ('0.0.0.0', '::') else [h])]
        for h in [h for h in shown if h != '127.0.0.1'] or shown:
            print(f'  {url(h, a.port, code)}', flush=True)
    if a.code or not srv.pairing.devices():
        srv.on_start = pair_line
    try:
        asyncio.run(srv.serve())
    except KeyboardInterrupt:
        pass
    except OSError as e:  # the port is taken, or the address is not this computer's
        busy = e.errno in (errno.EADDRINUSE, getattr(errno, 'WSAEADDRINUSE', -1))
        print(f'cannot serve: {e}' + ('\nA server already runs here: stop it first (a new pairing code: start it again '
                                      'with --code)' if busy else ''), file=sys.stderr)
        return 1
    return 0

"""RemoteDirection: one side translated on another computer (`peresmeshnik serve`), behind engines.Direction, so the app
keeps its capture, output, echo gates, live voice and barge-in, and only the heavy part (VAD, ASR, MT, TTS and the clone)
moves out (ANALYSIS-remote). It is the "cloud" of a `cloud/fallback.Fallback`: it ends its events with CloudDown when
the server cannot serve now, and the side goes local (or says why it cannot) until a new one is up.

- connect: TCP, TLS and the WebSocket in CONNECT_S at most; the server's certificate must be the one pinned at pairing
  (its SHA-256), else nothing is sent - not even the token; then `hello` (protocol.py), the voice when the server asks,
  and `ready` once its engines are up (READY_S: a first session loads the models). `ready` (an Event) and `ready_at` as
  GeminiDirection's: a fallback's probe is used when it is set;
- liveness: a WebSocket ping every PING_S; no pong in PONG_S -> the connection is lost (a Wi-Fi without a reset is not
  noticed otherwise). A lost connection gets one new one (its `ready` in RECONNECT_READY_S: the server's warm session),
  which hears again the input since the end of the last clause the server finished (KEEP_S at most): a clause cut by
  the drop is translated whole. A new one that fails, or SHORT_MAX connections in a row lost within SHORT_S of their
  `ready` (a server that crashes on each, a Wi-Fi that cannot hold one) -> CloudDown;
- too slow -> CloudDown: a frame of the call would reach the server `stall_s / 2` after it was said (an uplink too slow
  for the call), or the server hung: a phrase went up (this side's own gate, as GeminiDirection's: Silero VAD, >= 1 s
  of speech) and nothing at all came down `stall_s` after it began (its ASR), or a final subtitle came and no voice
  `stall_s` after it (its voice). Not from the phrase's start to its voice: a long clause is voiced after it ends (cut
  at 5 s, then ~1.3-1.9 s of synthesis: BENCH);
- the server refuses (`error`) -> ServerError, a CloudDown with the protocol's `code`; `permanent` ones (an unknown
  token, another protocol version, languages the server has no models for) are not cured by trying again;
- voice: the server sends it in pieces of protocol.SPEECH_MS; a piece that comes when the playback has run dry gets
  `jitter_ms` of silence before it, a cushion against the next piece being late (BENCH remote-0: no underruns at any
  hold; 40 ms by default);
- latency: the server's `caption` / `voice` are measured on its clock from the clause end it heard; here they get the
  last ping's RTT on top (the way up and the way down). `latency` keeps ('remote.voice', s) and ('remote.caption', s);
  `said_until` (for the fallback) is the end of the last clause whose voice began, from its `tts_first` (the server's
  `voice` also holds its own play queue); `rtt` is the last ping's, `server` the server's `ready` (what it runs);
- barge-in: `interrupt()` -> the server drops the voice it has not sent; what was on its way is dropped here until the
  server says `interrupted` (HUSH_S at most);
- a server that floods: messages up to protocol.CHUNK, a speech rate of 8-48 kHz, at most EVENTS_MAX events not taken
  yet (else CloudDown: the app does not keep up).

Threads: one asyncio thread; `feed()` and `interrupt()` only hand on; `events()` blocks on a queue; `close()` cancels.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import queue
import ssl
import threading
import time

import numpy as np

from peresmeshnik import __version__
from peresmeshnik.cloud.gemini import BLOCK, SLOW_MIN_BLOCKS, CloudDown, make_gate
from peresmeshnik.engines import Caption, Latency, Speech
from peresmeshnik.remote import protocol

log = logging.getLogger(__name__)
CONNECT_S = 2.0  # TCP + TLS + WebSocket: ~5 ms on a LAN, ~0.3-0.9 s to a far server (BENCH remote-0)
READY_S = 120.0  # a new session: the server loads ASR, MT and the voice (a clone ~30 s)
RECONNECT_READY_S = 5.0  # after a drop: the server's warm session answers at once
KEEP_S = 3.0  # input kept for a new connection: the clause the drop cut, or the start of a phrase while connecting
SETTLE_S = 0.45  # a clause's final subtitle comes this long after its speech ended (fallback.SETTLE_S)
SHORT_S, SHORT_MAX = 5.0, 3  # as GeminiDirection's: a connection lost this soon after `ready`, this many times in a row
HUSH_S = 2.0  # an interrupt the server does not answer: the voice comes through again after this
EVENTS_MAX = 4096  # events not taken yet (~7 min of voice in 100 ms pieces)
JITTER_MS = 40
_RESUMED = object()  # an interrupt is over: the server answered it, or the connection it was for is gone


class ServerError(CloudDown):
    """The server refused: `code` is protocol's (`permanent`: trying again does not help)."""

    def __init__(self, code: str, text: str):
        if code == 'auth' and 'pair' not in text:
            text += ': pair this computer again (tray menu, Server…)'
        super().__init__(text)
        self.code = code

    @property
    def permanent(self) -> bool:
        return self.code in protocol.PERMANENT


class RemoteDirection:
    """`address`: 'host:port' (IPv6 in brackets); `fp`: the server certificate's SHA-256 pinned at pairing; `voice`: the
    voice profile the server speaks with (protocol.pack_voice), None for its stock voice; `gate`: what tells a phrase
    began (gemini.make_gate(): Silero VAD, an energy gate without its model)."""
    supports_live = False

    def __init__(self, address: str, side: str, src: str, tgt: str, token: str, fp: str, voice: bytes | None = None,
                 jitter_ms: int = JITTER_MS, stall_s: float = 6.0, on_state=lambda text: None, gate=None):
        assert side in protocol.SIDES, side
        self.address, self.side, self.src, self.tgt, self.token, self.fp = address, side, src, tgt, token, fp
        self.voice, self.jitter_ms, self.stall_s, self.on_state = voice, jitter_ms, stall_s, on_state
        self.gate = gate or make_gate()  # before the thread: a model error comes out here
        self.ready = threading.Event()
        self.ready_at = 0.0
        self.server: dict = {}
        self.rtt = 0.0
        self.said_until = 0.0  # monotonic, this side's clock: the end of the last clause whose voice began
        self.latency: collections.deque = collections.deque(maxlen=256)
        self._events: queue.Queue = queue.Queue(EVENTS_MAX)
        self._hush, self._hush_lock, self._hush_at = 0, threading.Lock(), 0.0  # interrupts not answered yet, the last's time
        self._play_end = 0.0  # events thread: when the speech handed out so far has played
        # loop thread:
        self._in: asyncio.Queue = asyncio.Queue()  # (time, pcm) of feed(); pcm None: an interrupt
        self._kept: collections.deque = collections.deque()  # (time, pcm) of the last KEEP_S
        self._final_at = 0.0  # when the last final subtitle came
        self._done_until = 0.0  # the end of the last clause whose subtitle came, this side's clock (its `caption` latency)
        self._heard_at = 0.0  # when the server's pipeline last said anything
        self._voice_at = 0.0  # when the last piece of voice came
        self._phrase_at, self._voiced = None, 0  # the oldest phrase sent up with nothing down since, its 100 ms of speech
        self._final_wait = None  # the oldest final subtitle with no voice since
        self._buf = np.zeros(0, np.int16)  # the gate's input, in BLOCKs
        self._lost, self._fatal, self._ws = '', None, None
        self._unanswered = 0  # interrupts sent on this connection, not answered yet
        self._ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._ssl.check_hostname, self._ssl.verify_mode = False, ssl.CERT_NONE  # self-signed: pinned by fingerprint
        self._ssl.minimum_version = ssl.TLSVersion.TLSv1_2
        self._loop = asyncio.new_event_loop()
        self._task = self._loop.create_task(self._main())
        self._thread = threading.Thread(target=self._run, name=f'remote-{side}', daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:  # closed before it started
            pass
        finally:
            self._loop.close()
            self._end(None)

    # Direction ------------------------------------------------------------------------------------
    def feed(self, pcm: np.ndarray) -> None:
        try:
            self._loop.call_soon_threadsafe(self._in.put_nowait, (time.monotonic(), pcm))
        except RuntimeError:  # closed
            pass

    def events(self):
        while (e := self._events.get()) is not None:
            if isinstance(e, BaseException):
                raise e
            if e is _RESUMED:
                with self._hush_lock:
                    self._hush = max(0, self._hush - 1)
                continue
            if isinstance(e, Speech):
                with self._hush_lock:
                    if self._hush and time.monotonic() - self._hush_at > HUSH_S:  # the server never answered
                        self._hush = 0
                    hushed = self._hush
                if hushed:  # on its way before the interrupt
                    continue
                e = self._jitter(e)
            yield e

    def interrupt(self) -> None:
        with self._hush_lock:
            self._hush += 1
            self._hush_at = time.monotonic()
        self._play_end = 0.0
        try:
            self._loop.call_soon_threadsafe(self._in.put_nowait, (time.monotonic(), None))
        except RuntimeError:
            pass

    def close(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._task.cancel)
        except RuntimeError:  # already closed
            pass
        self._thread.join(5)

    def _jitter(self, e: Speech) -> Speech:
        now, dur = time.monotonic(), len(e.pcm) / e.sample_rate
        if now >= self._play_end and self.jitter_ms:  # the playback ran dry: a cushion before this piece
            hold = np.zeros(e.sample_rate * self.jitter_ms // 1000, np.float32)
            e, dur, self._play_end = Speech(np.concatenate([hold, e.pcm]), e.sample_rate), dur + len(hold) / e.sample_rate, now
        self._play_end = max(self._play_end, now) + dur
        return e

    # loop thread ----------------------------------------------------------------------------------
    async def _main(self):
        try:
            keep, short = await self._open(READY_S), 0
            while True:
                lost = await self._serve(keep)
                short = short + 1 if time.monotonic() - self.ready_at < SHORT_S else 0  # a long one lost: not a short one
                if short >= SHORT_MAX:
                    raise CloudDown(f'the connection to the server was lost {short} times right after it opened ({lost})')
                log.info('%s: the connection to the server was lost (%s): connecting again', self.side, lost)
                try:
                    keep = await self._open(RECONNECT_READY_S)
                except CloudDown as e:
                    if isinstance(e, ServerError) and e.permanent:  # a revoked token, ...: as it is
                        raise
                    raise CloudDown(f'the connection to the server was lost ({lost}) and a new one failed: {e}') from None
        except asyncio.CancelledError:  # close()
            pass
        except Exception as e:  # CloudDown, or a bug: the side hears it
            self._end(e)
        finally:
            if self._ws is not None:
                try:
                    await asyncio.wait_for(self._ws.close(), 1)  # a polite close: the server sees the user stopped
                except BaseException:  # noqa: B036 - close() cancelled it, or the server is gone: ending anyway
                    pass
                self._ws.transport.abort()  # no socket left behind whatever happened (a loop that ends does not close it)

    async def _open(self, ready_s: float) -> asyncio.Task:
        """Connect, check the certificate, hello, the voice, `ready`; returns the running keepalive."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed
        if self._ws is not None:  # the lost one: its socket goes, no polite close
            self._ws.transport.abort()
        self._lost, self._fatal, self._ws = '', None, None
        try:
            ws = await connect(f'wss://{self.address}{protocol.side_path(self.side)}', ssl=self._ssl, compression=None,
                               proxy=None, open_timeout=CONNECT_S, ping_interval=None, close_timeout=1,
                               max_size=protocol.CHUNK, user_agent_header=None)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # refused, no route, timeout, TLS, not a Peresmeshnik server
            raise CloudDown(f'cannot reach the server {self.address}: {type(e).__name__}: {e}'.rstrip(': ')) from None
        self._ws = ws
        der = ws.transport.get_extra_info('ssl_object').getpeercert(binary_form=True)
        if protocol.fingerprint(der) != self.fp:
            raise ServerError('auth', f'the server at {self.address} is not the one this computer was paired with (its '
                                      'certificate changed): pair again')
        keep = asyncio.create_task(self._keepalive(ws))  # the server may load its models a while: pings meanwhile
        try:
            await ws.send(protocol.hello(self.token, self.side, self.src, self.tgt,
                                         protocol.voice_ref(self.voice) if self.voice else None, app=__version__))
            async with asyncio.timeout(ready_s):
                while True:
                    m = protocol.decode(await ws.recv())
                    if not isinstance(m, dict):  # nothing is translated before `ready`
                        continue
                    if m.get('type') == 'ready':
                        break
                    if m.get('type') == 'send_voice' and self.voice:
                        for i in range(0, len(self.voice), protocol.CHUNK):
                            await ws.send(self.voice[i:i + protocol.CHUNK])
                    elif m.get('type') == 'state':
                        self._state(m)
                    elif m.get('type') == 'error':
                        raise ServerError(m.get('code', 'internal'), f'the server: {m.get("text", "")}')
        except TimeoutError:
            keep.cancel()
            raise CloudDown(f'the server was not ready in {ready_s:.0f} s') from None
        except ConnectionClosed as e:
            keep.cancel()
            raise CloudDown(f'the server closed the connection ({self._lost or e})') from None
        except BaseException:
            keep.cancel()
            raise
        self.server = m
        self._heard_at, self._phrase_at, self._final_wait = time.monotonic(), None, None  # a cut phrase goes up again
        for _ in range(self._unanswered):  # the old connection's: this one has no voice queued
            self._put(_RESUMED)
        self._unanswered = 0
        self.ready_at = time.monotonic()
        self.ready.set()
        log.info('%s: the server is ready: %s', self.side, {k: v for k, v in m.items() if k != 'type'})
        return keep

    async def _serve(self, keep: asyncio.Task) -> str:
        """Until the connection is lost (-> why); CloudDown when it is too slow or the server refuses."""
        from websockets.exceptions import ConnectionClosed
        ws = self._ws
        since = max(self._done_until or self._final_at - SETTLE_S, time.monotonic() - KEEP_S)
        while not self._in.empty():  # what came while connecting
            self._keep(*self._in.get_nowait())
        again = [pcm for t, pcm in self._kept if t >= since]
        tasks = {asyncio.create_task(self._receive(ws)), asyncio.create_task(self._send(ws, again)), keep}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._fatal:
            raise self._fatal
        for t in done:
            if not t.cancelled() and (e := t.exception()) is not None and not isinstance(e, ConnectionClosed):
                raise e
        return self._lost or 'the server closed it'

    def _keep(self, t: float, pcm) -> None:
        if pcm is None:  # an interrupt while connecting: nothing to stop on the new connection
            self._put(_RESUMED)
            return
        self._kept.append((t, pcm))
        while t - self._kept[0][0] > KEEP_S:
            self._kept.popleft()

    async def _send(self, ws, again: list) -> None:
        for pcm in again:  # the clause the drop cut (or the phrase begun while connecting), from its start
            await ws.send(protocol.pcm_bytes(pcm))
        while True:
            t, pcm = await self._in.get()
            if pcm is None:
                self._unanswered += 1
                await ws.send(protocol.message('interrupt'))
                continue
            self._keep(t, pcm)
            self._phrase(t, pcm)
            late = self.stall_s / 2 - (time.monotonic() - t)
            try:
                if late <= 0:
                    raise TimeoutError
                await asyncio.wait_for(ws.send(protocol.pcm_bytes(pcm)), late)
            except TimeoutError:
                self._fatal = CloudDown(f'the network is too slow for the call: the audio reaches the server more than '
                                        f'{self.stall_s / 2:g} s late')
                return

    async def _receive(self, ws) -> None:
        async for m in ws:
            e = protocol.decode(m)
            if isinstance(e, dict):
                if e.get('type') == 'interrupted' and self._unanswered:
                    self._unanswered -= 1
                    self._put(_RESUMED)
                elif e.get('type') == 'state':
                    self._state(e)
                elif e.get('type') == 'error':
                    err = ServerError(e.get('code', 'internal'), f'the server: {e.get("text", "")}')
                    if err.permanent:
                        self._fatal = err
                    self._lost = str(err)
                    return
                continue
            self._heard_at = time.monotonic()
            if isinstance(e, Speech):
                self._voice_at = self._heard_at
            elif isinstance(e, Caption) and e.final:
                self._final_at = self._heard_at
                if self._final_wait is None or self._voice_at >= self._final_wait:
                    self._final_wait = self._heard_at
            elif isinstance(e, Latency) and e.what in ('caption', 'voice', 'tts_first'):
                end = self._heard_at - e.s - self.rtt  # its clause ended (as fed here) then: the way up, the server, the way down
                if e.what == 'tts_first':  # its voice begins (`voice` holds the server's play queue too)
                    self.said_until = end
                else:
                    e = Latency(e.clause, e.what, round(e.s + self.rtt, 3))
                    self.latency.append((f'remote.{e.what}', e.s))
                    if e.what == 'caption':
                        self._done_until = end
            if not self._put(e):
                return

    async def _keepalive(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed
        while True:
            try:
                self.rtt = await asyncio.wait_for(await ws.ping(), protocol.PONG_S)
            except TimeoutError:
                self._lost = f'no answer from the server in {protocol.PONG_S:g} s'
                ws.transport.abort()
                return
            except ConnectionClosed:
                return
            now = time.monotonic()
            if (at := self._phrase_at) is not None and self._voiced >= SLOW_MIN_BLOCKS and self._heard_at < at \
                    and now - at > self.stall_s:
                self._fatal = CloudDown(f'the server stopped translating: nothing from it {self.stall_s:g} s after a phrase '
                                        'began')
                return
            if (at := self._final_wait) is not None and self._voice_at < at and now - at > self.stall_s:
                self._fatal = CloudDown(f'the server stopped speaking: no voice {self.stall_s:g} s after a subtitle')
                return
            await asyncio.sleep(protocol.PING_S)

    def _phrase(self, t: float, pcm) -> None:
        """This side's gate on what goes up: the oldest phrase with nothing down since it began, and its speech."""
        x = pcm if pcm.dtype == np.int16 else (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
        self._buf = np.concatenate([self._buf, x.ravel()])
        n = 0
        while len(self._buf) - n >= BLOCK:
            ev = self.gate.update(self._buf[n:n + BLOCK], t)
            n += BLOCK
            if ev == 'start' and (self._phrase_at is None or self._heard_at >= self._phrase_at):
                self._phrase_at, self._voiced = t, 0
            if self._phrase_at is not None and self.gate.on and self.gate.quiet == 0:
                self._voiced += 1
        self._buf = self._buf[n:]

    def _put(self, e) -> bool:
        """An event for the side (loop thread); False: the side does not take them (EVENTS_MAX), the connection goes."""
        try:
            self._events.put_nowait(e)
            return True
        except queue.Full:
            self._fatal = CloudDown(f'the app does not keep up with the server: {EVENTS_MAX} events not taken')
            if self._ws is not None:
                self._ws.transport.abort()
            return False

    def _end(self, e) -> None:
        """The last event (an error, or None): it gets in even past EVENTS_MAX (the oldest one goes)."""
        while True:
            try:
                self._events.put_nowait(e)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._events.get_nowait()

    def _state(self, m: dict) -> None:
        kind, _, text = str(m.get('text', '')).partition(': ')
        if kind not in ('note', 'error'):
            kind, text = 'note', str(m.get('text', ''))
        self.on_state(f'{kind}: server: {text}')

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
  which hears again the input since the last clause the server finished (KEEP_S at most): a clause cut by the drop is
  translated whole. A new one that fails -> CloudDown;
- too slow -> CloudDown: a frame of the call would reach the server `stall_s / 2` after it was said (an uplink too slow
  for the call), or a phrase is open (a partial subtitle came) and the server has sent nothing for `stall_s`;
- the server refuses (`error`) -> ServerError, a CloudDown with the protocol's `code`; `permanent` ones (an unknown
  token, another protocol version, languages the server has no models for) are not cured by trying again;
- voice: the server sends it in pieces of protocol.SPEECH_MS; a piece that comes when the playback has run dry gets
  `jitter_ms` of silence before it, a cushion against the next piece being late (BENCH remote-0: no underruns at any
  hold; 40 ms by default);
- latency: the server's `caption` / `voice` are measured on its clock from the clause end it heard; here they get the
  last ping's RTT on top (the way up and the way down). `latency` keeps ('remote.voice', s) and ('remote.caption', s)
  for the fallback; `rtt` is the last one, `server` the server's `ready` (what it runs);
- barge-in: `interrupt()` -> the server drops the voice it has not sent; what was on its way is dropped here until the
  server says `interrupted`.

Threads: one asyncio thread; `feed()` and `interrupt()` only hand on; `events()` blocks on a queue; `close()` cancels.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import queue
import ssl
import threading
import time

import numpy as np

from peresmeshnik import __version__
from peresmeshnik.cloud.gemini import CloudDown
from peresmeshnik.engines import Caption, Latency, Speech
from peresmeshnik.remote import protocol

log = logging.getLogger(__name__)
CONNECT_S = 2.0  # TCP + TLS + WebSocket: ~5 ms on a LAN, ~0.3-0.9 s to a far server (BENCH remote-0)
READY_S = 120.0  # a new session: the server loads ASR, MT and the voice (a clone ~30 s)
RECONNECT_READY_S = 5.0  # after a drop: the server's warm session answers at once
KEEP_S = 3.0  # input kept for a new connection: the clause the drop cut, or the start of a phrase while connecting
SETTLE_S = 0.45  # a clause's final subtitle comes this long after its speech ended (fallback.SETTLE_S)
JITTER_MS = 40
_RESUMED = object()  # an interrupt is over: the server answered it, or the connection it was for is gone


class ServerError(CloudDown):
    """The server refused: `code` is protocol's (`permanent`: trying again does not help)."""

    def __init__(self, code: str, text: str):
        super().__init__(text)
        self.code = code

    @property
    def permanent(self) -> bool:
        return self.code in protocol.PERMANENT


class RemoteDirection:
    """`address`: 'host:port' (IPv6 in brackets); `fp`: the server certificate's SHA-256 pinned at pairing; `voice`: the
    voice profile the server speaks with (protocol.pack_voice), None for its stock voice."""
    supports_live = False

    def __init__(self, address: str, side: str, src: str, tgt: str, token: str, fp: str, voice: bytes | None = None,
                 jitter_ms: int = JITTER_MS, stall_s: float = 6.0, on_state=lambda text: None):
        assert side in protocol.SIDES, side
        self.address, self.side, self.src, self.tgt, self.token, self.fp = address, side, src, tgt, token, fp
        self.voice, self.jitter_ms, self.stall_s, self.on_state = voice, jitter_ms, stall_s, on_state
        self.ready = threading.Event()
        self.ready_at = 0.0
        self.server: dict = {}
        self.rtt = 0.0
        self.said_until = 0.0  # monotonic, this side's clock: the end of the last clause whose voice began
        self.latency: collections.deque = collections.deque(maxlen=256)
        self._events: queue.Queue = queue.Queue()
        self._hush, self._hush_lock = 0, threading.Lock()  # interrupts the server has not answered yet
        self._play_end = 0.0  # events thread: when the speech handed out so far has played
        # loop thread:
        self._in: asyncio.Queue = asyncio.Queue()  # (time, pcm) of feed(); pcm None: an interrupt
        self._kept: collections.deque = collections.deque()  # (time, pcm) of the last KEEP_S
        self._final_at = 0.0  # when the last final subtitle came
        self._heard_at = 0.0  # when the server's pipeline last said anything
        self._open_clause = False  # a partial subtitle came and its final not yet
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
            self._events.put(None)

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
                    self._hush -= 1
                continue
            if isinstance(e, Speech):
                if self._hush:  # on its way before the interrupt
                    continue
                e = self._jitter(e)
            yield e

    def interrupt(self) -> None:
        with self._hush_lock:
            self._hush += 1
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
            keep = await self._open(READY_S)
            while True:
                lost = await self._serve(keep)
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
            self._events.put(e)
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
        self._open_clause, self._heard_at = False, time.monotonic()
        for _ in range(self._unanswered):  # the old connection's: this one has no voice queued
            self._events.put(_RESUMED)
        self._unanswered = 0
        self.ready_at = time.monotonic()
        self.ready.set()
        log.info('%s: the server is ready: %s', self.side, {k: v for k, v in m.items() if k != 'type'})
        return keep

    async def _serve(self, keep: asyncio.Task) -> str:
        """Until the connection is lost (-> why); CloudDown when it is too slow or the server refuses."""
        from websockets.exceptions import ConnectionClosed
        ws = self._ws
        since = max(self._final_at - SETTLE_S, time.monotonic() - KEEP_S)
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
            self._events.put(_RESUMED)
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
                    self._events.put(_RESUMED)
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
            if isinstance(e, Caption):
                self._open_clause = not e.final
                if e.final:
                    self._final_at = self._heard_at
            elif isinstance(e, Latency) and e.what in ('caption', 'voice'):
                e = Latency(e.clause, e.what, round(e.s + self.rtt, 3))
                self.latency.append((f'remote.{e.what}', e.s))
                if e.what == 'voice':  # its clause ended (as fed here) this long before now: the way up, the server, the way down
                    self.said_until = self._heard_at - e.s
            self._events.put(e)

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
            if self._open_clause and time.monotonic() - self._heard_at > self.stall_s:
                self._fatal = CloudDown(f'the server stopped translating: nothing from it {self.stall_s:g} s into a phrase')
                return
            await asyncio.sleep(protocol.PING_S)

    def _state(self, m: dict) -> None:
        kind, _, text = str(m.get('text', '')).partition(': ')
        if kind not in ('note', 'error'):
            kind, text = 'note', str(m.get('text', ''))
        self.on_state(f'{kind}: server: {text}')

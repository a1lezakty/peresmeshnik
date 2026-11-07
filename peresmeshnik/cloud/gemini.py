"""GeminiDirection: one translation direction over Gemini Live (docs/ARCHITECTURE.md §3, §4).

- voice: `gemini-3.5-live-translate-preview`, a continuous interpreter (24 kHz speech out). It
  never gets `audio_stream_end`: that cuts the end of the translation (stage-0 bench);
- subtitles (optional): `gemini-3.5-transcribe-live`; a speech gate sends `audio_stream_end` at
  each 300 ms pause, which brings the final text from ~1.6 s to ~0.65 s after the speaker stops;
- a connection lives ~10 min (transcribe: 10 min of audio per session). After `reconnect_s`, at
  the next quiet moment (at once on GoAway or a dropped connection) a new connection opens - the
  translator resumes its session with the last handle and runs with context compression, so it
  is not capped at 15 min. Input switches between two 100 ms blocks: nothing is lost or sent
  twice. The old connection is drained until its output is quiet and the new one's events are
  held back until then, so the end of a translation is not cut and the order is kept;
- latency per link goes to the `peresmeshnik.latency` logger and to `self.latency`.

Threads: one asyncio thread per direction; `feed()` only enqueues; `close()` ends the thread.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from peresmeshnik.config import Cloud
from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech

OUT_RATE = 24_000
BLOCK = SAMPLE_RATE // 10  # 100 ms, as the Live API recommends
BACKLOG = 300  # blocks (30 s) kept while a dropped connection is replaced
LANG = {'en': 'en-US', 'ru': 'ru-RU'}
log = logging.getLogger('peresmeshnik.latency')
warn = logging.getLogger(__name__).warning


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a.astype(np.float32) ** 2)))


class _Gate:
    """Speech from block energy: above an absolute floor (-70 dBFS), 3x the noise (a slowly rising
    minimum) and -26 dB from a fast-decaying peak (speaker levels differ a lot). 'pause' after 300 ms
    below it (transcribe gets audio_stream_end), 'end' of the utterance after 1 s.
    ponytail: energy gate; feed S1's Silero VAD events once app.py wires both."""
    def __init__(self):
        self.peak, self.noise, self.on, self.quiet, self.last_voiced = 0.0, 0.0003, False, 0, 0.0

    def update(self, x: np.ndarray, t: float) -> str | None:
        rms = _rms(x) / 32768
        self.peak = max(rms, self.peak * 0.98)  # half-life ~3.4 s
        self.noise = rms if rms < self.noise else self.noise * 1.005  # rises ~0.4 dB/s
        if rms > max(0.0003, 3 * self.noise, 0.05 * self.peak):
            self.quiet, self.last_voiced = 0, t
            if not self.on:
                self.on = True
                return 'start'
        elif self.on:
            self.quiet += 1
            if self.quiet == 3:
                return 'pause'
            if self.quiet == 10:
                self.on = False
                return 'end'
        return None


@dataclass
class _Utt:
    """One speech run seen by the gate; marks say which latencies were logged for it."""
    start: float
    fresh: bool = True  # the translator was quiet when it started: its first audio is measurable
    pause: float | None = None  # last 300 ms pause
    end: float | None = None  # 1 s of silence after it
    marks: set = field(default_factory=set)


class _Conn:
    def __init__(self, gen, cm, session):
        self.gen, self.cm, self.session, self.started = gen, cm, session, time.monotonic()
        self.task: asyncio.Task | None = None
        self.last_voiced = 0.0  # end of the last voiced audio chunk (arrival + length)
        self.waiting = 0.0  # first speech block sent since the last server message
        self.unanswered = 0  # speech blocks sent since the last server message
        self.failed = False  # a send failed: treat as dropped


class _Link:
    """One Live model (translate or transcribe) over successive connections."""
    stall_s = {'translate': 10.0, 'transcribe': 6.0}  # speech sent, no server message since: the session is dead

    def __init__(self, d: GeminiDirection, kind: str):
        self.d, self.kind = d, kind
        self.model = d.cfg.translate_model if kind == 'translate' else d.cfg.transcribe_model
        self.conn: _Conn | None = None
        self.next: asyncio.Task | None = None
        self.retiring: list[tuple[asyncio.Task, _Conn]] = []
        self.backlog: list[bytes] = []
        self.gen = 0  # connections are numbered 1, 2, ...
        self.out_gen = 1  # events of this connection go out now; newer ones wait in `held`
        self.held: dict[int, list] = {}
        self.closed: set[int] = set()
        self.handle: str | None = None
        self.go_away = False
        self.src_text = ''  # translate: source transcript since the last translated piece

    def config(self):
        from google.genai import types
        if self.kind == 'transcribe':
            return types.LiveConnectConfig(
                response_modalities=['TEXT'],
                input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[LANG.get(self.d.src, self.d.src)]))
        return types.LiveConnectConfig(
            response_modalities=['AUDIO'],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(target_language_code=self.d.tgt),
            session_resumption=types.SessionResumptionConfig(handle=self.handle),
            context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()))

    async def connect(self) -> _Conn:
        for attempt in range(6):
            t0 = time.monotonic()
            try:
                cm = self.d.client.aio.live.connect(model=self.model, config=self.config())
                session = await cm.__aenter__()
                break
            except Exception as e:  # quota, network, expired handle: retry fresh, then give up loudly
                if attempt == 5:
                    raise
                warn('%s: connect failed (%s), retry', self.kind, str(e)[:200])
                self.handle = None if attempt else self.handle
                await asyncio.sleep(min(30, 2 ** attempt))
        self.gen += 1
        c = _Conn(self.gen, cm, session)
        self.d.mark(f'cloud.{self.kind}.connect', time.monotonic() - t0)
        c.task = asyncio.create_task(self.recv(c))
        c.task.add_done_callback(lambda t: t.cancelled() or t.exception())  # dropped: due() notices
        return c

    async def start(self):
        self.conn = await self.connect()

    def due(self, quiet: bool) -> bool:
        c, now = self.conn, time.monotonic()
        if self.next:
            return False
        if c.task.done() or c.failed or self.go_away:
            return True
        if c.unanswered >= 20 and now - c.waiting > self.stall_s[self.kind]:  # >= 2 s of speech: not just a noise burst
            warn('%s: no answer %.0f s after speech, reconnecting', self.kind, now - c.waiting)
            return True  # seen live: a transcribe session went silent for 84 s, then closed with 1011
        age = now - c.started
        calm = quiet and now - c.last_voiced > 1.0
        return age >= self.d.cfg.reconnect_s and (calm or age >= self.d.cfg.reconnect_s + 45)

    async def _send(self, blk: bytes):
        from google.genai import types
        await self.conn.session.send_realtime_input(audio=types.Blob(data=blk, mime_type=f'audio/pcm;rate={SAMPLE_RATE}'))

    async def send(self, blk: bytes, quiet: bool, voiced: bool):
        if self.due(quiet):
            self.go_away = False
            self.next = asyncio.create_task(self.connect())
        if self.next and self.next.done():
            if self.next.exception() is not None:  # still no connection: the next block tries again
                warn('%s: reconnect failed (%s)', self.kind, str(self.next.exception())[:200])
                self.next = None
            else:
                old, self.conn, self.next = self.conn, self.next.result(), None  # switch between two blocks
                self.d.mark(f'cloud.{self.kind}.reconnect_after', time.monotonic() - old.started)
                self.retiring.append((asyncio.create_task(self.retire(old)), old))
        c = self.conn
        self.backlog.append(blk)
        if c.task.done() or c.failed:  # dropped: keep the audio (up to 30 s) for the next connection
            del self.backlog[:-BACKLOG]
            await asyncio.sleep(0)  # let the new connection finish even when blocks come in a burst
            return
        while self.backlog:  # what waited for this connection goes first, in order
            try:
                await self._send(self.backlog[0])
            except Exception as e:
                warn('%s: send failed (%s)', self.kind, str(e)[:200])
                c.failed = True
                return
            self.backlog.pop(0)
        if voiced:
            c.unanswered += 1
            c.waiting = c.waiting or time.monotonic()

    async def stream_end(self):
        if self.kind == 'transcribe' and not (self.conn.task.done() or self.conn.failed):  # never to the translator
            try:
                await self.conn.session.send_realtime_input(audio_stream_end=True)
            except Exception as e:
                warn('%s: audio_stream_end failed (%s)', self.kind, str(e)[:200])

    async def retire(self, c: _Conn, drain_max: float = 8.0):
        """Keep listening to a replaced connection until its output is quiet, then close it."""
        t0 = time.monotonic()
        while not c.task.done() and time.monotonic() - t0 < drain_max and time.monotonic() - max(c.last_voiced, t0) < 1.5:
            await asyncio.sleep(0.25)  # bounded: a few seconds once per reconnect
        await self.close_conn(c)

    async def close_conn(self, c: _Conn):
        c.task.cancel()
        try:
            await c.cm.__aexit__(None, None, None)
        except Exception:
            pass
        await asyncio.gather(c.task, return_exceptions=True)
        self.closed.add(c.gen)
        while self.out_gen in self.closed:  # the next connection's events may go out now
            self.out_gen += 1
            for e in self.held.pop(self.out_gen, ()):
                self.d.emit(e)

    async def close(self):
        for t, c in self.retiring:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
            if c.gen not in self.closed:
                await self.close_conn(c)
        if self.next:
            self.next.cancel()
            await asyncio.gather(self.next, return_exceptions=True)
            if not self.next.cancelled() and self.next.exception() is None:  # it connected just now
                await self.close_conn(self.next.result())
        if self.conn:
            await self.close_conn(self.conn)

    def out(self, c: _Conn, e):
        if c.gen <= self.out_gen:
            self.d.emit(e)
        else:
            self.held.setdefault(c.gen, []).append(e)

    async def recv(self, c: _Conn):
        while True:
            got = False
            async for m in c.session.receive():
                got = True
                self.on_message(c, m, time.monotonic())
            if not got:
                return

    def on_message(self, c: _Conn, m, now: float):
        c.waiting, c.unanswered = 0.0, 0
        u = m.session_resumption_update
        if u and u.resumable and u.new_handle:
            self.handle = u.new_handle
        if m.go_away and c is self.conn:
            self.go_away = True
            logging.getLogger(__name__).info('%s: GoAway after %.0f s, time left %s', self.kind, now - c.started, m.go_away.time_left)
        sc = m.server_content
        if not sc:
            return
        for p in (sc.model_turn.parts if sc.model_turn else None) or ():
            if not (p.inline_data and p.inline_data.data):
                continue
            a = np.frombuffer(p.inline_data.data, '<i2')
            if _rms(a) > 200:
                c.last_voiced = now + len(a) / OUT_RATE
                self.d.on_audio(now, c.last_voiced)
            elif c.last_voiced and now - c.last_voiced > 1.0:
                self.d.on_quiet(c.last_voiced)
            if a.any():  # the translator streams zeros through silence
                self.out(c, Speech(a.astype(np.float32) / 32768, OUT_RATE))
        if self.kind == 'translate':
            if sc.input_transcription and sc.input_transcription.text:
                self.src_text += sc.input_transcription.text
            if sc.output_transcription and sc.output_transcription.text:
                self.out(c, Caption(self.src_text.strip(), sc.output_transcription.text.strip(), True))
                self.src_text = ''
            return
        if sc.interim_input_transcription and sc.interim_input_transcription.text:
            self.d.on_text(now, final=False)
            self.out(c, Caption(sc.interim_input_transcription.text.strip()))
        if sc.input_transcription and sc.input_transcription.text:
            self.d.on_text(now, final=True)
            self.out(c, Caption(sc.input_transcription.text.strip(), final=True))


class GeminiDirection:
    """Direction (engines.py) over Gemini Live: Speech from the translator, Caption from both models."""

    def __init__(self, src: str, tgt: str, key: str | None = None, cfg: Cloud | None = None, client=None):
        self.src, self.tgt, self.cfg = src, tgt, cfg or Cloud()
        if client is None:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version='v1beta'))  # Live is v1beta-only
        self.client = client
        self.latency: list[tuple[str, float]] = []
        self._events: queue.Queue = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._in: asyncio.Queue = asyncio.Queue()
        self._utts: list[_Utt] = []
        self._voiced_until = 0.0  # end of the last voiced translated chunk
        self._segment = False  # transcribe: a segment is open (partials came, no final yet)
        self._thread = threading.Thread(target=self._run, name=f'gemini-{src}-{tgt}', daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    # Direction ------------------------------------------------------------------------------------
    def feed(self, pcm: np.ndarray) -> None:
        """16 kHz mono float32 (or int16), any length; returns at once."""
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._in.put_nowait, (time.monotonic(), pcm))

    def events(self):
        while (e := self._events.get()) is not None:
            if isinstance(e, BaseException):
                raise e
            yield e

    def close(self) -> None:
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._in.put_nowait, None)
            self._thread.join(15)

    # loop thread ----------------------------------------------------------------------------------
    def emit(self, e):
        self._events.put(e)

    def mark(self, name: str, s: float):
        self.latency.append((name, s))
        log.info('%s %d ms', name, round(s * 1000))

    def _oldest(self, now: float, mark: str, since: str = 'start') -> _Utt | None:
        for u in self._utts:
            t = getattr(u, since)
            if mark not in u.marks and t is not None and t < now:
                u.marks.add(mark)
                return u
        return None

    def on_audio(self, now, until):
        self._voiced_until = until
        u = self._oldest(now, 'audio')
        if u and u.fresh:
            self.mark('cloud.translate.first_audio_after_start', now - u.start)

    def on_quiet(self, last_voiced):
        done = [u for u in self._utts if u.end is not None and u.end < last_voiced and 'done' not in u.marks]
        for u in done:
            u.marks.add('done')
        if done:
            self.mark('cloud.translate.done_after_end', last_voiced - done[-1].end)

    def on_text(self, now, final):
        if not final and not self._segment and (u := self._oldest(now, 'partial')):
            self.mark('cloud.transcribe.first_partial_after_start', now - u.start)
        self._segment = not final
        if final and (u := self._oldest(now, 'final', 'pause')):
            self.mark('cloud.transcribe.final_after_pause', now - u.pause)

    async def _main(self):
        links = [_Link(self, 'translate')] + ([_Link(self, 'transcribe')] if self.cfg.subtitles else [])
        try:
            await asyncio.gather(*(lk.start() for lk in links))
            gate, buf = _Gate(), np.zeros(0, np.int16)
            while (item := await self._in.get()) is not None:
                t, pcm = item
                if pcm.dtype != np.int16:
                    pcm = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
                buf = np.concatenate([buf, pcm.ravel()])
                n = 0
                while len(buf) - n >= BLOCK:
                    blk, n = buf[n:n + BLOCK], n + BLOCK
                    ev = gate.update(blk, t)
                    if ev == 'start':
                        self._utts = [u for u in self._utts[-8:] if not {'audio', 'done'} <= u.marks]
                        self._utts.append(_Utt(t - 0.1, fresh=t - self._voiced_until > 1.0))
                    for lk in links:
                        await lk.send(blk.tobytes(), not gate.on, gate.on and gate.quiet == 0)
                    if ev == 'pause':
                        self._utts[-1].pause = gate.last_voiced
                        for lk in links:
                            await lk.stream_end()
                    if ev == 'end':
                        self._utts[-1].end = gate.last_voiced
                buf = buf[n:]
            if links[0].conn:
                self.on_quiet(links[0].conn.last_voiced)
        except Exception as e:
            self._events.put(e)
        finally:
            for lk in links:
                await lk.close()
            self._events.put(None)

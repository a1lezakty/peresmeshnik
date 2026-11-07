"""GeminiDirection against a fake Live API: every block is sent once and in order across
reconnects, translated audio keeps its order and loses nothing, the translator never gets
audio_stream_end, resumption handles are passed on."""
import asyncio
import queue

import numpy as np
from google.genai import types

from peresmeshnik.cloud.gemini import BLOCK, GeminiDirection
from peresmeshnik.config import Cloud
from peresmeshnik.engines import Caption, Speech


class FakeSession:
    def __init__(self, live, model, n):
        self.live, self.model, self.n, self.q, self.k = live, model, n, asyncio.Queue(), 0
        self.mute = n in live.mute  # a session that died on the server: takes audio, never answers

    async def send_realtime_input(self, audio=None, audio_stream_end=None):
        await asyncio.sleep(0)  # a real socket send yields to the loop
        if self.n in self.live.broken:
            raise ConnectionError('socket closed')
        if audio_stream_end:
            self.live.stream_ends.append(self.model)
            return
        blk = np.frombuffer(audio.data, '<i2')
        self.live.sent.append((self.model, self.n, int(blk[0])))
        if self.mute:
            return
        if self.model == 'translate':  # echo: quiet but nonzero "speech" tagged with the block index
            out = np.full(BLOCK, 100, '<i2')
            out[0] = blk[0]
            self.k += 1
            await self.q.put(types.LiveServerMessage(
                server_content=types.LiveServerContent(model_turn=types.Content(parts=[
                    types.Part(inline_data=types.Blob(data=out.tobytes(), mime_type='audio/pcm'))]))))
            await self.q.put(types.LiveServerMessage(session_resumption_update=types.LiveServerSessionResumptionUpdate(
                new_handle=f'h{self.n}.{self.k}', resumable=True)))
        elif blk.std() > 1000:
            await self.q.put(types.LiveServerMessage(server_content=types.LiveServerContent(
                input_transcription=types.Transcription(text='hello'))))

    async def receive(self):
        while (m := await self.q.get()) is not None:
            yield m


class FakeLive:
    def __init__(self):
        self.sent, self.stream_ends, self.handles, self.n, self.mute, self.broken = [], [], [], 0, set(), set()

    def connect(self, model, config):
        live = self

        class Cm:
            async def __aenter__(self):
                live.n += 1
                if config.session_resumption:
                    live.handles.append(config.session_resumption.handle)
                self.s = FakeSession(live, model, live.n)
                return self.s

            async def __aexit__(self, *a):
                await self.s.q.put(None)
        return Cm()


class FakeClient:
    def __init__(self):
        self.live = FakeLive()
        self.aio = self


def collect(d, n_speech, timeout=10):
    got = []
    while sum(isinstance(e, Speech) for e in got) < n_speech:
        got.append(d._events.get(timeout=timeout))
    return got


def test_reconnects_keep_order_and_lose_nothing():
    c = FakeClient()
    c.live.broken = {3, 7}  # these connections fail on the first send: their blocks go to the next one
    d = GeminiDirection('en', 'ru', cfg=Cloud(translate_model='translate', reconnect_s=0.0), client=c)
    n = 40
    for i in range(n + 10):  # quiet input (the gate stays off), so a reconnect is due at every block
        blk = np.ones(BLOCK, np.int16)
        blk[0] = i  # blocks after n: the stream goes on, as a capture device does
        d.feed(blk[:700])  # odd frame sizes: the direction re-blocks to 100 ms
        d.feed(blk[700:])
    got = collect(d, n)
    d.close()
    sent = [i for model, _, i in c.live.sent]
    assert sent[:n] == list(range(n)) and len(set(sent)) == len(sent)  # each block once, in order
    conns = {k for _, k, _ in c.live.sent}
    assert len(conns) > 5  # it really reconnected
    heard = [round(e.pcm[0] * 32768) for e in got if isinstance(e, Speech)]
    assert heard[:n] == list(range(n))  # old connections drained first, nothing lost
    hs = c.live.handles  # each connection resumes with the newest handle the server gave (broken ones gave none)
    gens = [int(h[1:].split('.')[0]) for h in hs[2:]]
    assert hs[0] is None and len(hs) > 5 and all(hs[2:]) and gens == sorted(gens) and not set(gens) & c.live.broken
    assert not c.live.stream_ends  # never audio_stream_end to the translator


def test_subtitles_get_stream_end_at_pauses():
    c = FakeClient()
    d = GeminiDirection('en', 'ru', cfg=Cloud(translate_model='translate', transcribe_model='transcribe', subtitles=True), client=c)
    loud = (np.sin(np.arange(BLOCK) / 3) * 8000).astype(np.int16)
    for _ in range(10):
        d.feed(loud)
    for _ in range(15):
        d.feed(np.ones(BLOCK, np.int16))
    got = collect(d, 25)
    caps = []
    try:
        while True:
            caps.append(d._events.get(timeout=1))
    except queue.Empty:
        pass
    d.close()
    assert c.live.stream_ends == ['transcribe']  # one pause, only the transcriber is told
    assert any(isinstance(e, Caption) and e.final and e.original == 'hello' for e in got + caps)


def test_silent_session_is_replaced(monkeypatch):
    from peresmeshnik.cloud import gemini
    monkeypatch.setattr(gemini._Link, 'stall_s', {'translate': 0.3, 'transcribe': 0.3})
    c = FakeClient()
    c.live.mute = {1}  # the first connection never answers
    d = GeminiDirection('en', 'ru', cfg=Cloud(translate_model='translate'), client=c)
    loud = (np.sin(np.arange(BLOCK) / 3) * 8000).astype(np.int16)
    import time
    for i in range(30):
        loud[0] = i
        d.feed(loud.copy())
        time.sleep(0.05)  # real time, so the stall shows
    got = collect(d, 3)
    d.close()
    assert c.live.n >= 2 and {k for _, k, _ in c.live.sent} >= {1, 2}
    assert all(isinstance(e, Speech) for e in got if not isinstance(e, Caption))

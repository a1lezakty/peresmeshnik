"""The cloud side's local fallback against a fake Gemini Live server (tests/fake_gemini.py: a real WebSocket over TLS for
the real google-genai client): the connection drops, the server refuses with an error or the quota, it is too slow ->
the side speaks locally from the clause the cloud had not said, one message; the cloud answers again -> back at the
start of a phrase. Local engines are the fakes of test_pipeline (no models). Switch times are printed (BENCH, stage 5)."""
import threading
import time

import numpy as np
import pytest
from fake_gemini import FakeGemini
from test_pipeline import FRAME, EnergyVad, UpperMt, WordAsr, silence, tone

from peresmeshnik.cloud import fallback
from peresmeshnik.cloud.fallback import Fallback
from peresmeshnik.cloud.gemini import GeminiDirection, _Gate
from peresmeshnik.config import Cloud
from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech
from peresmeshnik.pipeline import LocalDirection


class TextTts:
    """Says each word as 0.1 s of a tone and remembers what it said."""
    sample_rate = 8000
    said: list

    def __init__(self):
        TextTts.said = self.said = []

    def synth(self, text, lang, voice=None):
        self.said.append(text)
        for _ in text.split():
            yield np.full(800, 0.3, np.float32)


class Run:
    def __init__(self, fake: FakeGemini, after_s: float = 6.0, make_tts=TextTts, subtitles=False, voice_note='', **cloud):
        self.fake, self.states, self.got = fake, [], []
        self.fb = Fallback(
            'them', 'ru', subtitles,
            make_cloud=lambda: GeminiDirection('en', 'ru', cfg=Cloud(fallback_after_s=after_s, **cloud), client=fake.client(),
                                               gate=_Gate(), fail_fast=True),
            make_local=lambda tts: LocalDirection(EnergyVad(), WordAsr(), UpperMt(), tts, 'en', 'ru'),
            make_tts=make_tts, preload=True, on_state=self.states.append, voice_note=voice_note)
        assert self.fb.cloud.ready.wait(10)
        self.t = threading.Thread(target=self._consume, daemon=True)
        self.t.start()

    def _consume(self):
        for e in self.fb.events():
            self.got.append((time.monotonic(), e))

    def say(self, pcm):
        """Feed at real-time pace, as a capture does."""
        t0 = time.perf_counter()
        for i in range(0, len(pcm), FRAME):
            time.sleep(max(0.0, t0 + i / SAMPLE_RATE - time.perf_counter()))
            self.fb.feed(pcm[i:i + FRAME])

    def speech(self, since=0.0, rate=None):
        return [(t, e) for t, e in self.got if isinstance(e, Speech) and t >= since and (rate is None or e.sample_rate == rate)]

    def close(self):
        self.fb.close()
        self.t.join(10)
        self.fake.close()


@pytest.fixture
def fake():
    f = FakeGemini(lag=0.3)
    yield f
    f.close()


def cloud_voice(r, since=0.0):  # the fake translator's tone (cloud speech), not the local TextTts
    return [t for t, e in r.speech(since) if float(np.abs(e.pcm).max(initial=0)) > 0.1 and not np.allclose(e.pcm, e.pcm[0])]


def test_dropped_connection_goes_local_without_repeating(fake):
    r = Run(fake)
    try:
        r.say(np.concatenate([tone(1.2), silence(2.5)]))  # phrase A: the cloud says it
        assert cloud_voice(r), 'the cloud did not speak'
        r.say(tone(0.6))  # phrase B begins ...
        fake.down = True
        fake.drop()  # ... the network goes
        t_drop = time.monotonic()
        r.say(np.concatenate([tone(0.6), silence(1.5)]))
        local = [t for t, e in r.speech(t_drop) if np.allclose(e.pcm, e.pcm[0])]
        notes = [s for s in r.states if s.startswith('note')]
        print(f'drop -> local: decided {r.fb.switched[0][1] - t_drop:.2f} s, first local voice {local[0] - t_drop:.2f} s')
        assert len(notes) == 1 and 'the cloud failed' in notes[0] and r.fb.mode == 'local', r.states
        assert r.fb.switched[0][1] - t_drop < 2.0
        assert r.fb.probe is None and r.fb.retry_at > time.monotonic() + 20  # no try back before 30 s
        n = len(r.states)
        r.fb.close()
        assert len(r.states) == n  # closing says nothing
        assert TextTts.said == ['W0 W1 W2'], TextTts.said  # B (its three words) locally; A not again
    finally:
        r.close()


def test_the_stock_voice_instead_of_the_clone_says_why(fake, monkeypatch):
    """H-09 (review of PR #22): me -> them falls back to a stock voice because the clone is not ready: the first fallback
    says why the voice is not the user's own (once)."""
    from pathlib import Path

    from peresmeshnik import config, languages, models
    from peresmeshnik.local import clone
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(models, 'path', lambda i: Path('/models') / i)
    monkeypatch.setattr(clone, 'voice_cmd', lambda: ['voice-runner'])
    s, made = config.Settings(), {}
    s.me_to_them, s.my_lang, s.their_lang = 'cloud', 'ru', 'en'  # the clone chosen, no voice enrolled
    monkeypatch.setattr(fallback, 'Fallback', lambda *a, **k: made.update(k))
    fallback.make('me', s, 'key')
    note = made['voice_note']
    assert 'a stock voice speaks for you' in note and 'enrolled' in note and made['preload'], made
    s.local.voice = 'mine'  # ready: the clone itself, nothing to say
    fallback.make('me', s, 'key')
    assert made['voice_note'] == ''
    r = Run(fake, voice_note=note)  # the real Fallback (imported above)
    try:
        r.say(np.concatenate([tone(1.2), silence(1.0)]))
        fake.down = True
        fake.drop()
        r.say(np.concatenate([tone(0.6), silence(1.5)]))
        notes = [x for x in r.states if x.startswith('note')]
        assert len(notes) == 1 and notes[0].endswith('; ' + note), r.states
    finally:
        r.close()


def test_too_slow_says_the_muted_phrase_locally(fake):
    fake.lag = 30.0  # the translation never comes in time
    r = Run(fake, after_s=2.0)
    try:
        t0 = time.monotonic()
        r.say(np.concatenate([tone(1.2), silence(1.5)]))  # its clause is translated (muted) before the cloud is judged
        r.say(silence(1.0))
        notes = [s for s in r.states if s.startswith('note')]
        print(f'too slow -> local: decided {r.fb.switched[0][1] - t0:.2f} s after the phrase began (limit 2 s)')
        assert len(notes) == 1 and 'no translation' in notes[0], r.states
        assert TextTts.said == ['W0 W1 W2'], TextTts.said  # the phrase the cloud had not said, said once
    finally:
        r.close()


def test_quota_error_goes_local_with_the_reason(fake):
    r = Run(fake)
    try:
        r.say(np.concatenate([tone(0.6), silence(1.0)]))
        fake.refuse = (1011, 'RESOURCE_EXHAUSTED: You exceeded your current quota')
        fake.drop()
        t_drop = time.monotonic()
        r.say(np.concatenate([tone(1.2), silence(1.0)]))
        print(f'quota -> local: decided {r.fb.switched[0][1] - t_drop:.2f} s')
        notes = [s for s in r.states if s.startswith('note')]
        assert len(notes) == 1 and 'quota' in notes[0], r.states
    finally:
        r.close()


def test_back_to_the_cloud_at_a_phrase_start(fake, monkeypatch):
    monkeypatch.setattr(fallback, 'RETRY_S', 0.0)
    r = Run(fake)
    try:
        r.say(np.concatenate([tone(0.6), silence(1.0)]))
        fake.down = True
        fake.drop()
        r.say(np.concatenate([tone(1.2), silence(1.0)]))  # local
        assert r.fb.mode == 'local'
        fake.down = False
        n = fake.connections
        r.say(np.concatenate([tone(1.2), silence(1.0)]))  # its start: a new session opens, nothing sent
        assert fake.connections == n + 1 and r.fb.mode == 'local'
        said = list(TextTts.said)
        t_back = time.monotonic()
        r.say(np.concatenate([tone(1.2), silence(2.0)]))  # its start: back to the cloud
        back = [t for w, t in r.fb.switched if w == 'back']
        print(f'back to the cloud: {back[0] - t_back:.2f} s after the phrase began (its first partial subtitle)')
        assert r.fb.mode == 'cloud' and [s for s in r.states if 'back in the cloud' in s], r.states
        assert TextTts.said == said, 'the phrase after the switch was said locally too'
        assert cloud_voice(r, back[0]), 'the cloud did not say the phrase'
        first = [i for k, i in fake.inputs if k == fake.connections]
        assert len(first) >= 30, len(first)  # the phrase from before its first partial: >= 1.2 s of its speech
    finally:
        r.close()


def test_a_cloud_back_but_still_too_slow_waits_longer_each_time(fake, monkeypatch):
    """The cloud answers again but stays too slow: each time it goes local again without a word, the pause before the
    next try back doubles (not every RETRY_S, with two tray messages each)."""
    monkeypatch.setattr(fallback, 'RETRY_S', 0.1)
    fake.lag = 30.0  # connects, never translates in time
    r = Run(fake, after_s=1.5)
    try:
        pauses, end = [], time.monotonic() + 60
        while len(pauses) < 3 and time.monotonic() < end:
            downs = sum(w == 'down' for w, _ in r.fb.switched)
            r.say(np.concatenate([tone(1.2), silence(1.0)]))
            if sum(w == 'down' for w, _ in r.fb.switched) > downs:
                pauses.append(round(r.fb.retry_s, 3))
        assert pauses == [0.1, 0.2, 0.4], (pauses, r.fb.switched, r.states)
    finally:
        r.close()


class SlowTts(TextTts):
    """Each word takes 0.6 s to synthesize: a clause is still being said when the next phrase begins."""
    def synth(self, text, lang, voice=None):
        self.said.append(text)
        for _ in text.split():
            time.sleep(0.6)
            yield np.full(800, 0.3, np.float32)


@pytest.mark.parametrize('subtitles', [False, True], ids=['local-subtitles', 'cloud-subtitles'])
def test_back_to_the_cloud_while_a_local_clause_is_said(fake, monkeypatch, subtitles):
    """The side goes back to the cloud at a phrase start while the clause before it is still being said locally: the
    rest of that clause still reaches the output."""
    monkeypatch.setattr(fallback, 'RETRY_S', 0.0)
    r = Run(fake, make_tts=SlowTts, subtitles=subtitles)
    try:
        r.say(np.concatenate([tone(0.6), silence(1.0)]))
        fake.lag = 30.0 if subtitles else fake.lag  # cloud subtitles: the local path is made at the fallback
        fake.down = True
        fake.drop()
        r.say(np.concatenate([tone(1.2), silence(3.0)]))  # local
        assert r.fb.mode == 'local', r.states
        fake.down, fake.lag = False, 0.3
        t0 = time.monotonic()
        r.say(np.concatenate([tone(1.2), silence(0.5)]))  # its start: a session opens; it is said locally, slowly
        r.say(np.concatenate([tone(1.2), silence(3.0)]))  # its start: back to the cloud while the one before is said
        back = [t for w, t in r.fb.switched if w == 'back']
        assert back and r.fb.mode == 'cloud', r.states
        local = [(t, e) for t, e in r.speech(t0) if float(np.abs(e.pcm).max(initial=0)) > 0.1 and np.allclose(e.pcm, e.pcm[0])]
        assert any(t > back[0] for t, _ in local), 'the test did not catch a clause going on through the switch'
        n = sum(len(e.pcm) for _, e in local)
        assert n >= 0.9 * 3 * 0.1 * fallback.Muted.sample_rate, f'{n} samples of the local clause reached the output'
    finally:
        r.close()


def test_no_local_voice_says_why(fake):
    def missing():
        raise RuntimeError('piper-en_US-john-medium is not downloaded')
    r = Run(fake, make_tts=missing)
    try:
        r.say(silence(0.3))
        fake.down = True
        fake.drop()
        r.say(np.concatenate([tone(1.2), silence(1.0)]))
        errs = [s for s in r.states if s.startswith('error')]
        assert len(errs) == 1 and 'no local path here' in errs[0] and 'not downloaded' in errs[0], r.states
        assert r.fb.mode == 'down'
        assert any(isinstance(e, Caption) and e.final for _, e in r.got), 'the local subtitles stopped'
    finally:
        r.close()


def test_cloud_subtitles_make_the_local_path_at_the_fallback(fake):
    r = Run(fake, subtitles=True)
    try:
        assert r.fb.local is None
        fake.lag = 30.0
        fake.down = True
        fake.drop()
        r.say(np.concatenate([tone(1.2), silence(1.5)]))
        assert r.fb.mode == 'local' and r.fb.local is not None, r.states
        assert TextTts.said == ['W0 W1 W2'], TextTts.said  # the kept input from the cut on reached the new local path
    finally:
        r.close()


def test_stock_voice_falls_back_to_piper():
    """[local] clone = piper (voice.STOCK, an 8 GB Mac): me -> them falls back to the stock Piper voice, not the clone."""
    from peresmeshnik import config
    s = config.Settings()
    s.local.clone, s.local.voice = 'piper', ''
    why = fallback.voice_why('me', s)
    assert fallback.stock('me', s) and 'NVIDIA' not in why and 'enrolled' not in why, why
    # the clone chosen: test_a_clone_not_ready_falls_back_to_the_stock_voice


def test_a_cloud_me_side_passes_latency_through_the_live_switch(monkeypatch):
    """me -> them in the cloud: the side wraps the Fallback in LiveSwitch ("translation <-> live voice"); the cloud's
    Latency(voice_end) goes through to the overlay, the side does not die on it."""
    from peresmeshnik import app, config
    from peresmeshnik.audio.mic import LiveSwitch
    from peresmeshnik.engines import Latency

    class Cloud:  # a Fallback as Side._cloud returns it: no live voice of its own
        supports_live = False

        def feed(self, pcm):
            pass

        def events(self):
            yield Caption('Привет.', 'Hello.', True, 0)
            yield Latency(-1, 'voice_end', 3.4)
            yield Speech(np.full(2400, 0.3, np.float32), 24_000)

        def close(self):
            pass
    monkeypatch.setattr(app.Side, '_cloud', lambda self, s: Cloud())
    s = config.Settings()
    s.me_to_them = 'cloud'
    side = app.Side('me', s, None, None, None)
    sw = side.directions[0][0]
    assert isinstance(sw, LiveSwitch)
    got = list(sw.events())  # the side's consumer reads these
    assert [type(e).__name__ for e in got] == ['Caption', 'Latency', 'Speech'], got


def test_a_stalled_network_goes_local_in_seconds(fake, monkeypatch):
    """H-04: the network is gone without a reset (Wi-Fi lost): a send never returns. The side goes local in seconds,
    not after the TCP timeout (~15 min), and turning translation off does not wait for the socket."""
    import asyncio

    from peresmeshnik.cloud import gemini
    r = Run(fake)
    try:
        r.say(np.concatenate([tone(1.2), silence(1.0)]))
        real = gemini._Link._send

        async def stalled(self, blk):
            if fake.down:
                await asyncio.Event().wait()  # the socket's buffer is full and nothing drains it
            await real(self, blk)
        monkeypatch.setattr(gemini._Link, '_send', stalled)
        fake.down = True  # no new connection either
        t_stall = time.monotonic()
        r.say(np.concatenate([tone(2.0), silence(2.5)]))
        print(f'stall -> local: decided {r.fb.switched[0][1] - t_stall:.2f} s' if r.fb.switched else 'stall: still cloud')
        assert r.fb.mode == 'local' and r.fb.switched[0][1] - t_stall < 4.5, r.states
        t = time.monotonic()
        r.fb.close()
        assert time.monotonic() - t < 2.0
    finally:
        r.close()


def test_a_slow_uplink_goes_local(fake, monkeypatch):
    """H-05: the uplink carries less than the call needs (a phone in a bad spot): the input would lag behind the call
    without bound. The side goes local once the cloud gets it fallback_after_s / 2 late."""
    import asyncio

    from peresmeshnik.cloud import gemini
    real = gemini._Link._send

    async def slow(self, blk):
        await asyncio.sleep(0.2)  # 100 ms of audio takes 200 ms to send
        await real(self, blk)
    r = Run(fake, after_s=2.0)
    try:
        monkeypatch.setattr(gemini._Link, '_send', slow)
        t0 = time.monotonic()
        r.say(np.concatenate([tone(1.0), silence(1.0)] * 3))
        notes = [s for s in r.states if s.startswith('note')]
        assert len(notes) == 1 and 'uplink' in notes[0] and r.fb.switched[0][1] - t0 < 3.5, r.states
    finally:
        r.close()


def test_a_link_that_connects_later_does_not_make_the_call_late(fake, monkeypatch):
    """H-05, the review: with cloud subtitles the translator connects 1.5 s after the transcriber (a lost SYN, a slow
    TLS handshake): the audio said meanwhile waited for both, it is not "fallback_after_s / 2 late" on a healthy link."""
    import asyncio

    from peresmeshnik.cloud import gemini
    real = gemini._Link.connect

    async def connect(self):
        if self.kind == 'translate':
            await asyncio.sleep(1.5)
        return await real(self)
    monkeypatch.setattr(gemini._Link, 'connect', connect)
    d = GeminiDirection('en', 'ru', cfg=Cloud(fallback_after_s=2.0, subtitles=True), client=fake.client(), gate=_Gate(),
                        fail_fast=True)
    got = []
    t = threading.Thread(target=lambda: got.extend(_catch(d.events)), daemon=True)
    t.start()
    try:
        pcm = np.concatenate([tone(1.0), silence(1.5)])
        t0 = time.perf_counter()
        for i in range(0, len(pcm), FRAME):  # from the start, as a capture does: before both links are up
            time.sleep(max(0.0, t0 + i / SAMPLE_RATE - time.perf_counter()))
            d.feed(pcm[i:i + FRAME])
        assert d.ready.is_set() and not [e for e in got if isinstance(e, BaseException)], got
    finally:
        d.close()


def _catch(events):
    try:
        yield from events()
    except BaseException as e:
        yield e


def test_a_cloud_that_closes_each_session_at_once_goes_local(monkeypatch):
    """M-11: the server accepts a session and closes it right away (the quota told after the setup): a few tries, then
    the side goes local, not 5 new sessions a second for ever."""
    import asyncio
    import json

    class ClosesAfterSetup(FakeGemini):
        closing = False

        async def _serve(self, ws):
            if not self.closing:
                return await super()._serve(ws)
            json.loads(await ws.recv())
            self.connections += 1
            await ws.send(json.dumps({'setupComplete': {}}))
            await asyncio.sleep(0.05)
            await ws.close(1011, 'You exceeded your current quota')
    f = ClosesAfterSetup(lag=0.3)
    r = Run(f)
    try:
        r.say(silence(0.5))
        f.closing = True
        f.drop()
        n0, t0 = f.connections, time.monotonic()
        r.say(silence(3.0))
        assert r.fb.mode == 'local' and f.connections - n0 <= 4, (f.connections - n0, r.states)
        assert 'quota' in r.states[-1] and r.fb.switched[0][1] - t0 < 2.0, r.states
    finally:
        r.close()


def test_short_sessions_not_in_a_row_keep_the_cloud():
    """M-11 (review of PR #22): a session lost right after it opened now and then, with sessions that lived their
    time (a planned switch, reconnect_s) in between, is not "3 in a row": the side stays in the cloud."""
    import asyncio
    import json

    class ClosesSome(FakeGemini):
        closes: set = set()

        async def _serve(self, ws):
            if self.connections + 1 not in self.closes:
                return await super()._serve(ws)
            json.loads(await ws.recv())
            self.connections += 1
            await ws.send(json.dumps({'setupComplete': {}}))
            await asyncio.sleep(0.05)
            await ws.close(1011, 'Internal error')
    f = ClosesSome(lag=0.3)
    r = Run(f, reconnect_s=1.0)
    try:
        r.say(silence(0.5))
        f.closes = {f.connections + 1, f.connections + 3, f.connections + 5}  # lost at once, lives, lost, lives...
        n0 = f.connections
        f.drop()
        r.say(silence(5.0))
        assert f.connections - n0 >= 6 and r.fb.mode == 'cloud', (f.connections - n0, r.states)
    finally:
        r.close()


class _Cloud:
    """A fail_fast cloud direction's stand-in: up at once; `made` / `closed` list them."""
    made: list = []
    closed: list = []

    def __init__(self):
        import queue
        self.ready, self.q, self.latency = threading.Event(), queue.Queue(), []
        _Cloud.made.append(self)
        self.ready.set()

    def feed(self, pcm):
        pass

    def events(self):
        while (e := self.q.get()) is not None:
            if isinstance(e, BaseException):
                raise e
            yield e

    def close(self):
        _Cloud.closed.append(self)
        self.q.put(None)


def test_turning_off_while_local_leaves_no_cloud_probe():
    """M-12: translation off while the side speaks locally after a cloud fault: the local path's last partial subtitle
    (a new clause, queued as it closes) must not open a new cloud session that nobody closes."""
    import queue

    from peresmeshnik.cloud.gemini import CloudDown

    class Local:
        def __init__(self, tts):
            self.mt, self.q = None, queue.Queue()

        def warm_up(self):
            pass

        def feed(self, pcm):
            pass

        def events(self):
            while (e := self.q.get()) is not None:
                yield e

        def close(self):  # the listen worker's last partial is still on its way
            self.q.put(Caption('and then', clause=7))
            time.sleep(0.3)
            self.q.put(None)
    _Cloud.made, _Cloud.closed = [], []
    fb = Fallback('them', 'ru', False, make_cloud=_Cloud, make_local=Local, make_tts=TextTts)
    t = threading.Thread(target=lambda: list(fb.events()), daemon=True)
    t.start()
    _Cloud.made[0].q.put(CloudDown('network'))
    for _ in range(50):
        if fb.mode == 'local':
            break
        time.sleep(0.02)
    fb.retry_at = 0.0  # the 30 s since the fault have passed
    fb.close()
    t.join(2)
    time.sleep(0.5)
    assert fb.mode == 'local' and not [c for c in _Cloud.made if c not in _Cloud.closed]


def test_turning_off_while_the_local_path_loads_closes_it(fake):
    """M-12: cloud subtitles, the cloud fails and the local path is being made (models load) when translation is turned
    off: the new path is closed, its workers end."""
    made = []

    def slow_local(tts):
        time.sleep(1.0)  # ASR and MT load
        d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), tts, 'en', 'ru')
        made.append(d)
        return d
    fb = Fallback('them', 'ru', True, make_cloud=lambda: GeminiDirection(
        'en', 'ru', cfg=Cloud(), client=fake.client(), gate=_Gate(), fail_fast=True), make_local=slow_local, make_tts=TextTts)
    assert fb.cloud.ready.wait(10)
    t = threading.Thread(target=lambda: list(fb.events()), daemon=True)
    t.start()
    fake.down = True
    fake.drop()
    for _ in range(50):  # the next block finds the connection lost; _cloud_failed makes the local path
        fb.feed(tone(0.1))
        time.sleep(0.02)
    fb.close()
    t.join(5)
    time.sleep(1.5)
    assert len(made) == 1 and not [w.name for w in made[0]._threads if w.is_alive()]


def test_turning_off_while_the_voice_preloads_leaves_no_runner():
    """L-19: off right after on: the stock voice being preloaded (a piper-runner process) is closed when it is up."""
    class SlowTts:
        sample_rate, live = 22050, []

        def __init__(self):
            time.sleep(0.5)  # the runner starts
            SlowTts.live.append(self)

        def close(self):
            SlowTts.live.remove(self)
    _Cloud.made, _Cloud.closed = [], []
    fb = Fallback('me', 'en', True, make_cloud=_Cloud, make_local=None, make_tts=SlowTts, preload=True)
    fb.close()
    time.sleep(1.0)
    assert SlowTts.live == [] and fb.muted._tts is None


def test_a_clone_not_ready_falls_back_to_the_stock_voice(monkeypatch):
    """H-09: me -> them in the cloud with the clone chosen, but no voice enrolled yet (or no voice pack, no model): when
    the cloud fails, the language's stock voice speaks; the side is down only when there is neither."""
    from pathlib import Path

    from peresmeshnik import config, languages, models
    from peresmeshnik.local import clone
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)  # the clone keeps up here (languages.clone_why)
    monkeypatch.setattr(models, 'path', lambda i: Path('/models') / i)  # every model downloaded
    monkeypatch.setattr(clone, 'voice_cmd', lambda: ['voice-runner'])  # the voice pack is installed
    s = config.Settings()
    s.me_to_them, s.my_lang, s.their_lang = 'cloud', 'ru', 'en'
    assert fallback.voice_why('me', s) == '' and fallback.stock('me', s)  # no voice enrolled: the stock en voice
    s.local.voice = 'mine'
    assert fallback.voice_why('me', s) == '' and not fallback.stock('me', s)  # the clone
    s.local.voice, s.their_lang = '', 'zh'  # no stock voice for Chinese: nothing to fall back to
    assert 'enrolled' in fallback.voice_why('me', s)
    s.local.voice, s.their_lang, asked = 'mine', 'en', []  # the app's one answer "the clone can speak here" is asked

    def clone_why(model_id, lang='', has_gpu=None):
        asked.append((model_id, lang))
        return 'your cloned voice needs an NVIDIA GPU'
    monkeypatch.setattr(languages, 'clone_why', clone_why, raising=False)  # stream B's, merged after this one
    assert fallback.voice_why('me', s) == '' and fallback.stock('me', s) and asked[-1] == (s.local.clone, 'en')

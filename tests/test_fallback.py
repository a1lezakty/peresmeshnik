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
    def __init__(self, fake: FakeGemini, after_s: float = 6.0, make_tts=TextTts, subtitles=False):
        self.fake, self.states, self.got = fake, [], []
        self.fb = Fallback(
            'them', 'ru', subtitles,
            make_cloud=lambda: GeminiDirection('en', 'ru', cfg=Cloud(fallback_after_s=after_s), client=fake.client(),
                                               gate=_Gate(), fail_fast=True),
            make_local=lambda tts: LocalDirection(EnergyVad(), WordAsr(), UpperMt(), tts, 'en', 'ru'),
            make_tts=make_tts, preload=True, on_state=self.states.append)
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
    s.local.clone = config.Settings().local.clone
    assert not fallback.stock('me', s) and 'enrolled' in fallback.voice_why('me', s)


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

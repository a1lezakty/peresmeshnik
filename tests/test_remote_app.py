"""The remote server in the app (package 3): the settings, a side on the server with its local fallback (cloud/fallback.py
with RemoteDirection against tests/fake_remote.py), the clause the drop cut heard again locally, "down" -> the call's own
sound passes until the server is back, barge-in reaching the server, the voice that may go up (M-20)."""
import json
import threading
import time

import numpy as np
import pytest
from fake_remote import FakeServer
from test_fallback import TextTts
from test_pipeline import FRAME, EnergyVad, UpperMt, WordAsr, silence, tone
from test_remote_client import TOKEN, until
from test_ui import FakeIn, FakeOut

from peresmeshnik import app, config, languages
from peresmeshnik.cloud import fallback
from peresmeshnik.cloud.fallback import Fallback
from peresmeshnik.engines import SAMPLE_RATE, Caption, Latency
from peresmeshnik.pipeline import LocalDirection
from peresmeshnik.remote import protocol
from peresmeshnik.remote.client import RemoteDirection


@pytest.fixture
def srv():
    s = FakeServer()
    yield s
    s.close()


def test_settings(tmp_path):
    p = tmp_path / 'settings.toml'
    p.write_text('them_to_me = "remote"\nme_to_them = "remote"\n[remote]\njitter_ms = 80\n[server]\nport = 47100\n')
    s = config.load(p)
    assert (s.them_to_me, s.me_to_them, s.remote.jitter_ms, s.remote.retry_s, s.server.port) == ('remote', 'remote', 80, 5.0, 47100)
    assert config.problem == ''
    config.save(s, p)
    assert config.load(p).remote.jitter_ms == 80
    p.write_text('them_to_me = "remot"\n')  # a typo: local, never another computer or the cloud
    assert config.load(p).them_to_me == 'local'
    assert languages.needed_models(s) == ['silero-vad']  # both sides on the server: nothing to download here


class Run:
    def __init__(self, srv, local=True, retry_s=0.3):
        self.states, self.modes, self.got = [], [], []

        def make_local(tts):
            if not local:
                raise RuntimeError('model asr-en is not downloaded')
            return LocalDirection(EnergyVad(), WordAsr(), UpperMt(), tts, 'en', 'ru')
        self.fb = Fallback('them', 'ru', True,
                           make_cloud=lambda: RemoteDirection(srv.address, 'them', 'en', 'ru', TOKEN, srv.fp, jitter_ms=0),
                           make_local=make_local, make_tts=TextTts, on_state=self.states.append,
                           label=('the server', 'on the server'), retry_s=retry_s, on_mode=self.modes.append)
        assert self.fb.cloud.ready.wait(5)
        self.t = threading.Thread(target=lambda: [self.got.append(e) for e in self.fb.events()], daemon=True)
        self.t.start()

    def say(self, pcm):
        t0 = time.perf_counter()
        for i in range(0, len(pcm), FRAME):
            time.sleep(max(0.0, t0 + i / SAMPLE_RATE - time.perf_counter()))
            self.fb.feed(pcm[i:i + FRAME])

    def captions(self):
        return [e for e in self.got if isinstance(e, Caption) and e.final]

    def close(self):
        self.fb.close()
        self.t.join(10)


def test_the_server_fails_and_the_cut_clause_is_said_locally(srv):
    r = Run(srv)
    try:
        r.say(np.concatenate([tone(2.0), silence(0.5)]))  # clause A (4 words): the server voices it
        srv.send(protocol.event(Caption('a a a a', 'A A A A', True, 0)), protocol.event(Latency(0, 'voice', 0.5)))
        until(lambda: r.captions())
        r.say(tone(1.0))  # clause B (2 words) ...
        srv.refuse = ('internal', 'restarting')
        srv.drop()  # ... the server goes, and does not come back at once
        r.say(silence(1.5))
        why = lambda: f'mode {r.fb.mode}, captions {r.captions()}, said {TextTts.said}, states {r.states}'  # noqa: E731
        until(lambda: r.fb.mode == 'local' and len(r.captions()) >= 2, 10, why)  # a slow runner (macOS, 3 cores)
        assert len(r.captions()) == 2, why()
        b = r.captions()[1].translation
        assert len(b.split()) == 2 and TextTts.said == [b]  # B, heard again locally and said; A not said twice
        note = [s for s in r.states if 'the server failed' in s]
        assert len(note) == 1 and 'translating locally until it answers again' in note[0] and r.modes == ['local']
    finally:
        r.close()


def test_down_then_back_on_the_server(srv):
    r = Run(srv, local=False)
    try:
        srv.refuse = ('busy', 'another session')
        srv.drop()
        until(lambda: r.fb.mode == 'down')
        assert r.modes == ['down']
        assert [s for s in r.states if s.startswith('error') and 'you hear the call as it is until the server answers again'
                in s], r.states
        srv.refuse = None  # the server is back: tried again retry_s after, on the input that keeps coming
        t = time.monotonic()
        while r.fb.mode != 'cloud':
            r.fb.feed(silence(0.032))
            time.sleep(0.03)
            assert time.monotonic() - t < 5, r.states
        assert r.modes == ['down', 'cloud'] and 'note: them → me: back on the server' in r.states
        srv.send(protocol.event(Caption('hi', 'привет', True, 3)))
        until(lambda: any(c.translation == 'привет' for c in r.captions()))
        r.fb.interrupt()  # barge-in reaches the server
        until(lambda: {'type': 'interrupt'} in srv.conns[-1].texts)
    finally:
        r.close()


@pytest.mark.parametrize('name', ['them', 'me'])
def test_a_side_down_passes_the_call_and_takes_it_back(srv, monkeypatch, tmp_path, name):
    """Down: them -> me plays the call as it is in the headphones; me -> them sends my own voice (LiveSwitch); the frames
    still reach the side, which finds the server again and translates."""
    from peresmeshnik.remote import pairing
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path))  # no local models: no local path
    monkeypatch.setattr(pairing, 'load', lambda path=None: {'address': srv.address, 'fp': srv.fp, 'token': TOKEN})
    s = config.Settings()
    s.them_to_me = s.me_to_them = 'remote'
    s.remote.retry_s = 0.3
    inp, out, states = FakeIn(), FakeOut(), []
    side = app.Side(name, s, inp, out, type('O', (), {'show': lambda self, c: None})(), on_error=states.append)
    side.start()
    sw = side.directions[0][0]
    fb = getattr(sw, 'inner', sw)
    try:
        assert fb.cloud.ready.wait(5)
        frame = np.zeros(512, np.float32)
        inp.cb(frame)
        assert not side.auto and out.played == []
        srv.refuse = ('internal', 'restarting')
        srv.drop()
        until(lambda: side.auto)
        if name == 'them':
            assert out.flushed == 1
            inp.cb(frame)
            until(lambda: (512, 16000) in out.played)  # the call as it is, in the headphones
        else:
            assert sw.live  # my own voice to the call, crossfaded
        srv.refuse = None
        t = time.monotonic()
        while side.auto:  # the frames still reach the side: it finds the server again
            inp.cb(frame)
            time.sleep(0.03)
            assert time.monotonic() - t < 5, states
        assert fb.mode == 'cloud' and not side.live
        assert out.flushed == 2 if name == 'them' else not sw.live
    finally:
        side.stop()


def test_a_revoked_device_is_not_tried_again(srv):
    r = Run(srv, local=False, retry_s=0.1)
    try:
        srv.refuse = ('auth', 'this computer is not paired here (any more): pair again')
        srv.drop()
        until(lambda: r.fb.mode == 'down')
        n = len(srv.hellos)
        for _ in range(20):  # 0.6 s of input: ~5 tries at 0.1 s doubling, were it not permanent
            r.fb.feed(silence(0.032))
            time.sleep(0.03)
        assert len(srv.hellos) == n and r.fb.probe is None
        assert [x for x in r.states if 'pair again' in x and x.startswith('error')], r.states
    finally:
        r.close()


def test_not_paired_says_so(monkeypatch):
    from peresmeshnik.remote import pairing
    monkeypatch.setattr(pairing, 'load', lambda path=None: None)
    s = config.Settings()
    s.me_to_them = 'remote'
    with pytest.raises(RuntimeError, match='not paired'):
        app.Side('me', s, None, None, None)


def test_only_my_confirmed_voice_goes_up(tmp_path, monkeypatch):
    from peresmeshnik import voice
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path)
    s = config.Settings()
    s.local.voice = 'v'
    (tmp_path / 'v').mkdir()
    for meta, ok in (({'name': 'v'}, True),  # v1: recorded by the user in the wizard
                     ({'version': 2, 'kind': 'me', 'consent': {'how': 'mic-live'}}, True),
                     ({'version': 2, 'kind': 'me', 'consent': {'how': 'import+mic-match'}}, True),
                     ({'version': 2, 'kind': 'contact', 'consent': {'how': 'call-saved'}}, False),
                     ({'version': 2, 'kind': 'me'}, False)):
        (tmp_path / 'v' / 'voice.json').write_text(json.dumps(meta))
        if ok:
            assert fallback.my_voice(s)
        else:
            with pytest.raises(RuntimeError, match='not your own confirmed voice'):
                fallback.my_voice(s)
    s.local.clone = voice.STOCK
    assert fallback.my_voice(s) is None

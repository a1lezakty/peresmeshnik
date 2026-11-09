"""The voice pack's runner (voice-runner/voice_runner.py) and its client (local/clone.py VoicePackClone), with a stub
engine (tests/voice_stub.py) instead of XTTS: the protocol, barge-in, errors, a runner that dies or never starts, its
idle CPU; make_clone's choice; the app's fallback when the clone cannot speak. No torch, no coqui-tts, no GPU."""
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import psutil
import pytest

from peresmeshnik.engines import Voice
from peresmeshnik.local import clone
from peresmeshnik.local.clone import RUNNER, CloneUnavailable, VoicePackClone, make_clone, voice_cmd

STUB = [sys.executable, str(RUNNER), '--engine', str(Path(__file__).with_name('voice_stub.py')), '--device', 'cpu']


@pytest.fixture
def pack(tmp_path):
    p = VoicePackClone(tmp_path / 'xtts-v2', cmd=STUB)
    yield p
    p.close()


@pytest.fixture
def me(tmp_path):
    (tmp_path / 'me').mkdir()
    return Voice(name='me', engine='xtts', path=tmp_path / 'me')


def say(p, text, v):
    return np.concatenate(list(p.synth(text, 'en', v)))


def test_protocol(pack, me):
    assert pack.sample_rate == 24_000 and pack.info == {'device': 'cpu', 'model': str(pack.info['model']), 'torch': 'stub',
                                                        'cuda': False}
    x = say(pack, 'hello', me)
    assert x.dtype == np.float32 and len(x) == 3 * 2400 and np.all(x == np.float32(0.05))
    assert pack.similarity(me, np.full(2400, 0.5, np.float32), 24_000) == pytest.approx(0.05)  # the PCM got there
    assert pack.latents(me) == me.path / 'xtts.pt' and (me.path / 'xtts.pt').read_text() == '1'
    pack.latents(me)
    assert (me.path / 'xtts.pt').read_text() == '1'  # cached
    assert pack.stats()['torch'] == 'stub'


def test_enroll_conditions_again(pack, tmp_path):
    from test_clone import speech_like
    v = pack.enroll(speech_like(35), 16_000, root=tmp_path / 'voices')
    assert v.engine == 'xtts' and (v.path / 'reference.wav').exists() and (v.path / 'xtts.pt').read_text() == '1'
    v = pack.enroll(speech_like(35, seed=1), 16_000, root=tmp_path / 'voices')
    assert (v.path / 'xtts.pt').read_text() == '2'  # a new recording, a new conditioning


def test_barge_in_stops_the_runner_quickly(pack, me):
    """The pipeline closes synth() mid-sentence: the runner stops after its current chunk, the next request starts
    clean (none of the old audio), all without restarting it (the model loads in ~10 s)."""
    pid = pack._p.pid
    gen = pack.synth('slow', 'en', me)
    next(gen), next(gen)
    t = time.perf_counter()
    gen.close()
    assert time.perf_counter() - t < 0.5  # 100 chunks of 50 ms would be 5 s
    t = time.perf_counter()
    x = say(pack, 'hi', me)
    assert len(x) == 3 * 2400 and np.all(x == np.float32(0.02)) and time.perf_counter() - t < 0.5
    assert pack._p.pid == pid and pack._p.poll() is None


def test_an_error_is_a_reply(pack, me):
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        say(pack, 'boom', me)
    assert len(say(pack, 'hi', me)) == 3 * 2400  # it goes on
    with pytest.raises(RuntimeError, match='unknown request'):
        pack._send({'cmd': 'dance'})
        pack._reply()


def test_a_dead_runner_raises_instead_of_hanging(pack, me):
    with pytest.raises(RuntimeError, match=r'the voice pack exited \(3\)'):
        say(pack, 'die', me)
    t = time.perf_counter()
    with pytest.raises(RuntimeError, match='exited'):
        say(pack, 'hi', me)
    with pytest.raises(RuntimeError, match='exited'):
        pack.similarity(me, np.zeros(10, np.float32), 24_000)
    assert time.perf_counter() - t < 1
    pack.close()


@pytest.mark.parametrize('cmd, why', [
    ([sys.executable, str(RUNNER), '--engine', STUB[3]], 'no CUDA GPU for the voice clone'),  # CUDA expected
    (STUB, 'FileNotFoundError'),  # the stub's 'broken' model folder
    ([sys.executable, '-c', 'import sys; sys.exit(4)'], r'exited \(4\)'),
    (['/nonexistent/python.exe'], 'did not start'),
])
def test_startup_failure_is_clear(tmp_path, cmd, why):
    model = tmp_path / ('broken' if 'FileNotFound' in why else 'xtts-v2')
    with pytest.raises(CloneUnavailable, match=why):
        VoicePackClone(model, cmd=cmd)


def test_a_runner_that_never_answers_is_killed(tmp_path):
    t = time.perf_counter()
    with pytest.raises(CloneUnavailable, match='no answer in 1 s'):
        VoicePackClone(tmp_path, cmd=[sys.executable, '-c', 'import time; time.sleep(60)'], start_s=1)
    assert time.perf_counter() - t < 5


def test_the_real_engine_without_coqui_says_so(tmp_path):
    """The runner's default engine is local/xtts.py; in this venv TTS (or torch, on the Windows runner) is missing, and the
    start header says that."""
    with pytest.raises(CloneUnavailable, match="No module named '(TTS|torch)'"):
        VoicePackClone(tmp_path, cmd=[sys.executable, str(RUNNER), '--device', 'cpu'])


def test_runner_sleeps_between_requests(pack, me):
    say(pack, 'hello', me)
    p = psutil.Process(pack._p.pid)
    time.sleep(0.5)
    t0 = sum(p.cpu_times()[:2])
    time.sleep(3)
    assert sum(p.cpu_times()[:2]) - t0 < 0.02  # the app's idle watchdog counts its children (0.13 s per 60 s)


def test_make_clone_chooses(monkeypatch, tmp_path):
    import peresmeshnik.config as config
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path)
    monkeypatch.setitem(sys.modules, 'torch', None)  # a base install: no torch in this process
    assert voice_cmd() is None
    with pytest.raises(CloneUnavailable, match='needs the voice pack'):
        make_clone(tmp_path)
    pack = tmp_path / 'voice-pack'
    pack.mkdir()
    (pack / 'python.exe').write_text('')
    assert voice_cmd() is None  # half installed: no .complete yet
    with pytest.raises(CloneUnavailable, match='install did not finish'):
        make_clone(tmp_path)
    from peresmeshnik.local import voicepack
    (pack / '.complete').write_text(voicepack.identity() + '\n')  # this app's pack (voice-runner/pack-windows.json)
    assert voice_cmd() == [str(pack / 'python.exe'), str(RUNNER)] and RUNNER.exists()
    monkeypatch.setenv('PERESMESHNIK_VOICE_CMD', ' '.join(STUB))
    p = make_clone(tmp_path / 'xtts-v2')
    try:
        assert isinstance(p, VoicePackClone) and p.info['device'] == 'cpu'
    finally:
        p.close()


# --- the app: the clone cannot speak -> the stock voice (never the cloud: local was chosen)

class FakeDirection:
    def __init__(self, tts=None, voice=None, error=None):
        self.tts, self.voice, self._error, self.closed, self.warm = tts, voice, error, False, 0

    def warm_up(self):
        self.warm += 1
        if self.tts == 'oom':
            raise RuntimeError('CUDA out of memory')

    def events(self):
        return iter(())

    def feed(self, pcm):
        pass

    def close(self):
        self.closed = True


def me_side(monkeypatch, tmp_path, clone_error=None, key=True):
    from peresmeshnik import app, config, models, pipeline, voice
    s = config.Settings()
    s.me_to_them, s.local.voice = 'local', 'me'
    monkeypatch.setattr(models, 'path', lambda model_id: tmp_path)
    monkeypatch.setattr(voice, 'load', lambda name, root=None: Voice(name, 'xtts', tmp_path))
    made = []

    def make_local(src, tgt, settings=None, tts=None, voice=None, **k):
        made.append(tts)
        return FakeDirection(tts, voice)
    monkeypatch.setattr(pipeline, 'make_local', make_local)

    def make(model_dir, threads=2):
        if isinstance(clone_error, Exception):
            raise clone_error
        return clone_error or 'clone'
    monkeypatch.setattr(clone, 'make_clone', make)
    side = app.Side.__new__(app.Side)
    side.name, side.src, side.tgt, side.engine, side.settings, side.live = 'me', s.my_lang, s.their_lang, 'local', s, False
    side._lock, side._stopped, side.directions, side.errors = threading.Lock(), False, [], []
    side.on_error = side.errors.append
    cloud = types.SimpleNamespace(supports_live=True, close=lambda: None, set_live=lambda on: None)

    def no_key(s):
        raise RuntimeError('no Gemini API key')
    side._cloud = (lambda s: cloud) if key else no_key
    return side, s, made, cloud


def test_clone_unavailable_falls_back_to_the_stock_voice(monkeypatch, tmp_path):
    side, s, made, _ = me_side(monkeypatch, tmp_path, CloneUnavailable('no CUDA GPU'), key=False)
    d = side._local(s)
    assert made == [None] and d.warm == 1 and side.engine == 'local'  # make_local's own Piper voice
    assert side.errors == ['error: me: the voice clone is off (no CUDA GPU); a stock voice speaks instead']


def test_clone_out_of_vram_at_warm_up_falls_back_to_the_stock_voice_not_the_cloud(monkeypatch, tmp_path):
    side, s, made, cloud = me_side(monkeypatch, tmp_path, 'oom')  # a cloud key is there: still not the cloud
    d = side._local(s)
    assert d is not cloud and side.engine == 'local' and made == ['oom', None]
    assert side.errors == ['error: me: the voice clone is off (CUDA out of memory); a stock voice speaks instead']


def test_clone_that_dies_mid_call_is_replaced(monkeypatch, tmp_path):
    from peresmeshnik.audio.mic import LiveSwitch
    side, s, made, cloud = me_side(monkeypatch, tmp_path, key=False)
    dead = FakeDirection('clone', voice=Voice('me', 'xtts', tmp_path), error=RuntimeError('the voice pack exited (-11)'))
    side.directions = [(LiveSwitch(dead), True, True)]
    side._consume(side.directions[0][0], True, True)  # its events end at once: the worker died
    assert dead.closed and side.errors == ['error: me: the voice clone is off (the voice pack exited (-11)); '
                                           'a stock voice speaks instead']
    new = side.directions[0][0]
    assert isinstance(new, LiveSwitch) and new.inner.tts is None and new.inner.voice is None  # no second fallback
    new.close()

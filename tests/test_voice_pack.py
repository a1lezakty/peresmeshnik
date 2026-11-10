"""The voice pack's runner (voice-runner/voice_runner.py) and its client (local/clone.py VoicePackClone), with a stub
engine (tests/voice_stub.py) instead of XTTS: the protocol, barge-in, errors, a runner that dies, hangs or never starts,
its stderr, its idle CPU; make_clone's choice; the app's fallback when the clone cannot speak. No torch, no coqui-tts,
no GPU."""
import json
import os
import re
import subprocess
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


@pytest.fixture(autouse=True)
def data(monkeypatch, tmp_path):
    """The runner's log (<data dir>/logs/voice-runner.log) in the test's folder."""
    import peresmeshnik.config as config
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path / 'data')
    return tmp_path / 'data'


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
    """L-43: one clause the runner cannot say (a number num2words cannot read, CUDA out of memory on a long one) is
    left out, the clone goes on; only errors in a row (MAX_ERRORS) end it, and the side falls back to the stock voice."""
    assert list(pack.synth('boom', 'en', me)) == []  # left out, no error
    assert len(say(pack, 'hi', me)) == 3 * 2400  # it goes on
    for _ in range(VoicePackClone.MAX_ERRORS - 1):
        assert list(pack.synth('boom', 'en', me)) == []
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        say(pack, 'boom', me)
    with pytest.raises(RuntimeError, match='unknown request'):
        pack._send({'cmd': 'dance'})
        pack._reply()


def test_the_clone_in_this_process_leaves_one_clause_out_too(monkeypatch, me):
    """L-43 (review of PR #22): XTTS in this process (Linux from source) as the voice pack: one clause it cannot say is
    left out, the clone goes on; MAX_ERRORS in a row end it."""
    from peresmeshnik.local import xtts

    def synth(self, text, lang, voice=None):
        if text == 'boom':
            raise RuntimeError('CUDA out of memory')
        yield np.full(3, 0.5, np.float32)
    monkeypatch.setattr(xtts.Engine, 'synth', synth)
    c = object.__new__(clone.XttsClone)  # no model: its synthesis is the stand-in above
    assert list(c.synth('boom', 'en', me)) == [] and len(list(c.synth('hi', 'en', me))) == 1
    for _ in range(VoicePackClone.MAX_ERRORS - 1):
        assert list(c.synth('boom', 'en', me)) == []
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        list(c.synth('boom', 'en', me))


def test_a_runner_dying_after_an_error_is_the_clones_death(pack, me):
    """L-43, the review: the runner died (or was killed by the timeout) right after it said a clause failed: that is the
    clone gone, said now (the side falls back), not one clause lost and the death seen only at the next one."""
    wait = pack._wait
    pack._wait = lambda read: (pack._p.kill(), b'')[1] if read == pack._p.stdout.readline else wait(read)  # died then
    with pytest.raises(RuntimeError, match='exited'):
        list(pack.synth('boom', 'en', me))


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


def test_a_runner_that_stops_answering_is_killed(tmp_path, me):
    """A hung CUDA call: no reply chunk in reply_s -> the runner is killed and the call raises (the direction's worker
    dies and the app's fallback runs) instead of the voice going silent for good."""
    p = VoicePackClone(tmp_path / 'xtts-v2', cmd=STUB, reply_s=1)
    try:
        t = time.perf_counter()
        with pytest.raises(RuntimeError, match=r'did not answer in 1 s'):
            say(p, 'hang', me)
        assert time.perf_counter() - t < 5 and p._p.poll() is not None
    finally:
        p.close()


def test_stderr_nobody_reads_does_not_hide_the_reason(tmp_path):
    """A windowed .exe has no stderr: its child gets a pipe nobody reads. The runner's traceback there must not kill it
    before its {"error"} line."""
    r, w = os.pipe()
    os.close(r)
    p = subprocess.Popen([*STUB, '--model-dir', str(tmp_path / 'broken')], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=w)
    os.close(w)
    out, _ = p.communicate(timeout=60)
    assert json.loads(out.splitlines()[0])['error'].startswith('FileNotFoundError'), out


def test_the_runner_log(tmp_path, data):
    """Its stderr goes to <data dir>/logs/voice-runner.log, new at each start; a runner that exits without a reason
    of its own is named by the log's last lines."""
    with pytest.raises(CloneUnavailable, match='FileNotFoundError'):
        VoicePackClone(tmp_path / 'broken', cmd=STUB)
    log = data / 'logs' / 'voice-runner.log'
    assert 'Traceback' in log.read_text()
    crash = [sys.executable, '-c', 'import sys; print("torch DLL not found", file=sys.stderr); sys.exit(4)']
    with pytest.raises(CloneUnavailable, match=r'did not start: exited \(4\): torch DLL not found$'):
        VoicePackClone(tmp_path, cmd=crash)
    assert log.read_text().strip() == 'torch DLL not found'


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
    with monkeypatch.context() as m, pytest.raises(CloneUnavailable, match='needs the voice pack'):
        m.setattr(sys, 'platform', 'linux')  # a source install: torch and coqui-tts are the user's
        make_clone(tmp_path)
    with monkeypatch.context() as m, pytest.raises(CloneUnavailable, match=r'not installed: install it in the setup '
                                                                          r'wizard \(Your voice\)'):
        m.setattr(sys, 'platform', 'win32')  # the app: the next step, not the module that is missing
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


def test_voice_cmd_on_windows_drops_the_quotes(monkeypatch):
    cmd = r'"C:\Program Files\py\python.exe" C:\vr\voice_runner.py --device cpu'
    monkeypatch.setattr(clone, 'os', types.SimpleNamespace(name='nt', environ={'PERESMESHNIK_VOICE_CMD': cmd}))
    assert voice_cmd() == [r'C:\Program Files\py\python.exe', r'C:\vr\voice_runner.py', '--device', 'cpu']


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


def me_side(monkeypatch, tmp_path, clone_error=None, key=True, meta=None):
    from peresmeshnik import app, config, languages, models, pipeline, voice
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)  # an NVIDIA GPU: the clone is chosen (clone_why)
    monkeypatch.setattr(languages, 'pack_here', lambda: True)  # and its voice pack is installed
    s = config.Settings()
    s.me_to_them, s.local.voice = 'local', 'me'
    monkeypatch.setattr(models, 'path', lambda model_id: tmp_path)
    (tmp_path / 'reference.wav').write_bytes(b'my recording')
    sha = __import__('hashlib').sha256(b'my recording').hexdigest()  # voice.json v2: mine, confirmed on this recording
    meta = {'kind': 'me', 'consent': {'how': 'mic-live', 'ref_sha256': sha}} if meta is None else meta
    monkeypatch.setattr(voice, 'load', lambda name, root=None: Voice(name, 'xtts', tmp_path, meta))
    made = []

    def make_local(src, tgt, settings=None, tts=None, voice=None, **k):
        made.append(tts)
        return FakeDirection(tts, voice)
    monkeypatch.setattr(pipeline, 'make_local', make_local)

    def make(model_id, model_dir, threads=2):
        if isinstance(clone_error, Exception):
            raise clone_error
        return clone_error or 'clone'
    monkeypatch.setattr(clone, 'make_engine', make)
    side = app.Side.__new__(app.Side)
    side.name, side.src, side.tgt, side.engine, side.settings, side.live = 'me', s.my_lang, s.their_lang, 'local', s, False
    side._lock, side.directions, side.errors = threading.Lock(), [], []
    side.on_error = side.errors.append
    cloud = types.SimpleNamespace(supports_live=True, close=lambda: None, set_live=lambda on: None)

    def no_key(s):
        raise RuntimeError('no Gemini API key')
    side._cloud = (lambda s: cloud) if key else no_key

    def api_key(c):
        if not key:
            raise SystemExit('no Gemini API key')  # as cloud.api_key says it
        return 'key'
    monkeypatch.setattr('peresmeshnik.cloud.api_key', api_key)
    return side, s, made, cloud


def test_clone_unavailable_falls_back_to_the_stock_voice(monkeypatch, tmp_path):
    side, s, made, _ = me_side(monkeypatch, tmp_path, CloneUnavailable('no CUDA GPU'), key=False)
    d = side._local(s)
    assert made == [None] and d.warm == 1 and side.engine == 'local'  # make_local's own Piper voice
    assert side.errors == ['note: me → them: a stock voice speaks instead of your cloned voice (no CUDA GPU)']  # L-29


def test_clone_out_of_vram_at_warm_up_falls_back_to_the_stock_voice_not_the_cloud(monkeypatch, tmp_path):
    side, s, made, cloud = me_side(monkeypatch, tmp_path, 'oom')  # a cloud key is there: still not the cloud
    d = side._local(s)
    assert d is not cloud and side.engine == 'local' and made == ['oom', None]
    assert side.errors == ['note: me → them: a stock voice speaks instead of your cloned voice (CUDA out of memory). With '
                           'your Gemini key you can switch "Me → them" to cloud in the settings']  # said, not done


class SlowTts:
    """A runner on a card that cannot keep up: 0.1 s of speech per 0.15 s (RTF 1.5)."""
    sample_rate = 24_000

    def synth(self, text, lang, voice=None, speed=1.0):
        for _ in range(4):
            time.sleep(0.15)
            yield np.zeros(self.sample_rate // 10, np.float32)

    def close(self):
        pass


def test_a_clone_slower_than_real_time_gives_way_to_the_stock_voice(monkeypatch, tmp_path):
    """Its real-time factor, measured at warm-up, above MAX_CLONE_RTF (a GTX 10xx, say): me -> them has no lag limit,
    so the stock voice speaks from the start, with the reason, instead of a clone that falls further behind."""
    from peresmeshnik import app
    slow = SlowTts()
    side, s, made, cloud = me_side(monkeypatch, tmp_path, slow, key=False)
    d = side._local(s)
    assert made == [slow, None] and d.warm == 1  # the clone made and dropped, the stock voice warmed up
    m = re.search(r'this computer is too slow for the clone: (\d+\.\d) s of synthesis per second', side.errors[0])
    assert len(side.errors) == 1 and m and float(m[1]) > app.MAX_CLONE_RTF, side.errors  # 1.5, more on a slow runner
    assert app.clone_rtf(slow, 'de', None) > app.MAX_CLONE_RTF  # a language without a test phrase: the English one


def test_clone_that_dies_mid_call_is_replaced(monkeypatch, tmp_path):
    from peresmeshnik.audio.mic import LiveSwitch
    side, s, made, cloud = me_side(monkeypatch, tmp_path, key=False)

    def _speak():  # the error as the pipeline's speak worker records it (app.spoke: the TTS, not ASR or MT)
        raise RuntimeError('the voice pack exited (-11)')
    try:
        _speak()
    except RuntimeError as e:
        error = e
    dead = FakeDirection('clone', voice=Voice('me', 'xtts', tmp_path), error=error)
    side.directions = [(LiveSwitch(dead), True, True)]
    side._consume(side.directions[0][0], True, True)  # its events end at once: the worker died
    assert dead.closed and side.errors == ['note: me → them: a stock voice speaks instead of your cloned voice '
                                           '(the voice pack exited (-11))',
                                           'error: me → them: its events ended by themselves']  # the stock voice's: a failure
    new = side.directions[0][0]
    assert isinstance(new, LiveSwitch) and new.inner.tts is None and new.inner.voice is None  # no second fallback
    new.close()


def test_an_asr_error_mid_call_is_no_clone_death(monkeypatch, tmp_path):
    """The worker that died is ASR's (or MT's): not the clone, so no stock voice; the side fails and restarts (S3)."""
    from peresmeshnik.audio.mic import LiveSwitch
    side, s, made, cloud = me_side(monkeypatch, tmp_path, key=False)
    failures = []
    side.on_failure = lambda why, e: failures.append(why)

    def _listen():
        raise RuntimeError('onnxruntime: bad allocation')
    try:
        _listen()
    except RuntimeError as e:
        error = e
    dead = FakeDirection('clone', voice=Voice('me', 'xtts', tmp_path), error=error)
    side.directions = [(LiveSwitch(dead), True, True)]
    side._consume(side.directions[0][0], True, True)
    assert not dead.closed and side.errors == [] and made == []
    assert failures == ['the translation failed (RuntimeError: its events ended by themselves)']


def test_no_stock_voice_to_fall_back_to_says_why(monkeypatch, tmp_path):
    """M-24: they speak Chinese (no stock voice with a usable license), me -> them local, the clone cannot speak here: the
    side says so and how out (the cloud), not "unknown model ''" with the registry's 200 ids."""
    import pytest

    from peresmeshnik import models
    side, s, made, _ = me_side(monkeypatch, tmp_path, CloneUnavailable('no CUDA GPU'), key=False)
    side.tgt = 'zh'
    with pytest.raises(RuntimeError) as e:
        side._local(s)
    assert 'Chinese' in str(e.value) and 'cloud' in str(e.value) and 'unknown model' not in str(e.value), e.value
    with pytest.raises(models.ModelMissing) as e:
        models.get('')
    assert len(str(e.value)) < 40


def test_settings_of_0_3_0_without_a_voice_speak_in_the_stock_voice(monkeypatch, tmp_path):
    """FIXES 3: 0.3.0 said "No recording is needed" where the clone could not speak (a Mac, no NVIDIA GPU) and kept
    clone = 'xtts-v2' without a voice; stage 8 picks Pocket TTS there. No voice, or its weights not downloaded: the side
    starts in the stock voice with the reason, as in 0.3.0, not an error."""
    from peresmeshnik import languages, models
    side, s, made, _ = me_side(monkeypatch, tmp_path, key=False)
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)  # a Mac: Pocket TTS on the CPU
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')
    s.local.voice = ''
    assert languages.direction(s, 'me_to_them').tts == 'pocket-tts-en'
    d = side._local(s)
    assert made == [None] and d.warm == 1 and side.errors == [
        'note: me → them: a stock voice speaks instead of your cloned voice (no voice enrolled (setup wizard, "Your voice"))']

    def missing(model_id):
        raise models.ModelMissing(f'{model_id} is not downloaded')
    monkeypatch.setattr(models, 'path', missing)
    s.local.voice, side.errors[:] = 'me', []
    d = side._local(s)
    assert made == [None, None] and 'pocket-tts-en is not downloaded' in side.errors[0]


class OrtFail(Exception):
    """onnxruntime's own errors (Fail, InvalidGraph, RuntimeException) derive from Exception, not RuntimeError."""


def test_onnxruntime_errors_give_the_stock_voice(monkeypatch, tmp_path):
    """FIXES 4: an onnxruntime error making Pocket TTS (its graphs, out of memory) is CloneUnavailable; one at warm-up
    (the first phrase) falls back to the stock voice too, not an error of the side."""
    from peresmeshnik.local import clone, pocket

    def broken(*a, **k):
        raise OrtFail('[ONNXRuntime] Fail: bad allocation')
    monkeypatch.setattr(pocket, 'PocketTts', broken)
    with pytest.raises(CloneUnavailable, match='Pocket TTS did not start: .*bad allocation'):
        clone.make_engine('pocket-tts-en', tmp_path)

    class Warm(FakeDirection):
        def warm_up(self):
            raise OrtFail('[ONNXRuntime] RuntimeException: out of memory')
    from peresmeshnik import pipeline
    side, s, made, _ = me_side(monkeypatch, tmp_path, key=False)
    real = pipeline.make_local
    monkeypatch.setattr(pipeline, 'make_local', lambda *a, tts=None, **k: Warm(tts) if tts == 'clone' else real(*a, tts=tts, **k))
    d = side._local(s)
    assert isinstance(d, FakeDirection) and not isinstance(d, Warm) and 'out of memory' in side.errors[0]


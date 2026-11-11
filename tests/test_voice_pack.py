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



def live(state):
    """A voice made during the call, as voices/live.py's LiveProfile: no folder, the engine's state."""
    return types.SimpleNamespace(name='them', engine='xtts', path=None, state=state)


def test_a_voice_from_a_sample_speaks_without_a_folder(pack, me):
    """voices/live.py: condition(pcm, sr) gives the runner's id of the voice's state; a synth by it speaks in that voice
    (the stub says the sample's mean), a profile folder as before."""
    a, b = (pack.condition(np.full(16_000, x, np.float32), 16_000) for x in (0.25, 0.5))
    assert isinstance(a, str) and a != b
    assert np.all(say(pack, 'hello', live(a)) == np.float32(0.25)) and len(say(pack, 'hello', live(b))) == 3 * 2400
    assert np.all(say(pack, 'hello', live(b)) == np.float32(0.5)) and np.all(say(pack, 'hello', me) == np.float32(0.05))
    gen = pack.synth('slow', 'en', live(a))  # barge-in as with a folder
    next(gen)
    gen.close()
    assert np.all(say(pack, 'hi', live(a)) == np.float32(0.25))


def test_a_state_the_runner_no_longer_has_is_a_clause_error(pack, tmp_path, caplog):
    """The runner keeps its last MAX_STATES (one a synth used stays); an older one, or one of a runner before (a
    restart), is a LookupError of that synth (voices/live.py makes it again), not one of the errors counted to its
    death: the clone goes on."""
    limit = int(re.search(r'^MAX_STATES = (\d+)', RUNNER.read_text(), re.M)[1])
    a, b = (pack.condition(np.full(2400, x, np.float32), 24_000) for x in (0.25, 0.5))
    say(pack, 'hi', live(a))  # used: newer than b now
    for _ in range(limit - 1):
        pack.condition(np.full(2400, 0.1, np.float32), 24_000)
    assert np.all(say(pack, 'hi', live(a)) == np.float32(0.25))
    with pytest.raises(LookupError, match='no voice state'):  # b: dropped
        list(pack.synth('hi', 'en', live(b)))
    assert len(say(pack, 'hi', live(a))) == 3 * 2400  # it goes on
    other = VoicePackClone(tmp_path / 'xtts-v2', cmd=STUB, log=tmp_path / 'other.log')
    try:
        with pytest.raises(LookupError):  # another runner's id: not a voice there
            list(other.synth('hi', 'en', live(a)))
    finally:
        other.close()


def test_a_new_call_on_warm_engines_leaves_nothing_of_the_last_in_the_runner(pack):
    """FIXES-s2 r3 1: the runner keeps the states condition() made (Qwen3's with the raw sample); a new call on warm
    engines (voices/live.py reset(), the server's warm side) makes it forget them all: the last call's voices are not
    there to speak with any more, and a new one is made as before."""
    import test_live
    from test_live import AmpSpeaker, speech

    from peresmeshnik.voices import live as lv
    v = lv.LiveVoice(lambda: pack, AmpSpeaker, 'xtts-v2', 'en')
    a = [v.offer(speech(1.2, 0.3, i), 'a', 0, 1.2) for i in range(3)]
    a[-1].known.wait(5)
    test_live.time.sleep(0.5)  # its voice is made after the clause is known
    old = v.people[0].voice
    assert old is not None and len(say(pack, 'hi', old)) == 3 * 2400  # the runner speaks in it
    v.reset()
    b = [v.offer(speech(1.2, 0.1, i), 'b', v._since + 1, v._since + 2.2) for i in range(3)]
    b[-1].known.wait(5)
    test_live.time.sleep(0.5)
    with pytest.raises(LookupError):  # gone from the runner
        list(pack.synth('hi', 'en', old))
    assert len(v.people) == 1 and len(say(pack, 'hi', v.people[0].voice)) == 3 * 2400  # B's, made anew
    v.close()


def test_a_contacts_recording_speaks_from_their_first_clause_and_goes_from_the_runner_when_deleted(pack, monkeypatch,
                                                                                                    tmp_path):
    """BRIEF-s2-keep-nvidia: a contact kept as their speech; the next call's runner makes their voice from it as soon as
    it is up, so their first clause speaks in it. Deleted in Voices… during the call: forgotten at once, here and in
    the runner (Qwen3's state holds the sample), while the voice of someone else in the call stays."""
    import test_live
    from test_live import AmpSpeaker, speech

    from peresmeshnik import models, voice
    from peresmeshnik.voices import live as lv
    root = tmp_path / 'voices'
    monkeypatch.setattr(voice, 'voices_dir', lambda: root)
    anna = voice.save_them('Anna', 'en', 12.0, {voice.SPEAKER: (test_live.B, models.get(voice.SPEAKER).fingerprint)},
                           root, (np.full(9 * 16_000, 0.25, np.float32), 16_000))
    v = lv.LiveVoice(lambda: pack, AmpSpeaker, 'xtts-v2', 'en', lv.kept('xtts-v2'))
    first = v.offer(speech(1.2, 0.1), 'b', 0, 1.2)  # Anna (B) speaks first
    assert first.known.wait(10) and first.person.kept
    kept = first.person.voice
    assert np.allclose(say(pack, 'hi', kept), 0.25, atol=1e-3)  # the stub speaks the mean of its sample
    a = [v.offer(speech(1.2, 0.3, i), 'a', 0, 1.2) for i in range(3)]  # someone else, made in this call
    a[-1].known.wait(5)
    test_live.time.sleep(0.5)  # its voice is made after the clause is known
    other = v.people[-1].voice
    voice.discard(anna.name, root)
    v.deleted()
    after = v.offer(speech(1.2, 0.1, 5), 'b', 0, 1.2)
    assert after.known.wait(5) and not after.person.kept  # gone: someone new
    with pytest.raises(LookupError):  # gone from the runner
        list(pack.synth('hi', 'en', kept))
    assert len(say(pack, 'hi', other)) == 3 * 2400  # the other voice of the call stays
    v.close()


def test_the_runner_forgets_the_states_named_and_an_empty_list_forgets_none(pack):
    """FIXES-s2-39 3: {"cmd": "forget", "states": [...]} drops those only; [] none (not all, as without "states")."""
    a, b = (pack.condition(np.full(2400, x, np.float32), 24_000) for x in (0.25, 0.5))
    pack.forget_states([])
    assert np.all(say(pack, 'hi', live(a)) == np.float32(0.25)) and np.all(say(pack, 'hi', live(b)) == np.float32(0.5))
    pack.forget_states([a])
    with pytest.raises(LookupError):
        list(pack.synth('hi', 'en', live(a)))
    assert np.all(say(pack, 'hi', live(b)) == np.float32(0.5))
    pack.forget_states()
    with pytest.raises(LookupError):
        list(pack.synth('hi', 'en', live(b)))


def test_a_voice_pushed_out_of_the_runner_is_made_again_from_their_speech(pack, monkeypatch, tmp_path):
    """FIXES-s2-39 2: the runner keeps its last MAX_STATES voices; a contact's pushed out by more of them speaks their
    next clause in their voice again (made from their recording), not silence nor the end of their voice."""
    import test_live
    from test_live import AmpSpeaker, FakeStock, speech

    from peresmeshnik import models, voice
    from peresmeshnik.voices import live as lv
    limit = int(re.search(r'^MAX_STATES = (\d+)', RUNNER.read_text(), re.M)[1])
    root = tmp_path / 'voices'
    voice.save_them('Anna', 'en', 12.0, {voice.SPEAKER: (test_live.B, models.get(voice.SPEAKER).fingerprint)}, root,
                    (np.full(9 * 16_000, 0.25, np.float32), 16_000))
    v = lv.LiveVoice(lambda: pack, AmpSpeaker, 'xtts-v2', 'en', lv.kept('xtts-v2', root))
    tts = lv.TheirVoice(FakeStock(), v)
    c = v.offer(speech(1.2, 0.1), 'b', 0, 1.2)
    assert np.allclose(np.concatenate(list(tts.synth('hi', 'en', c))), 0.25, atol=1e-3)
    for _ in range(limit + 1):
        pack.condition(np.full(2400, 0.1, np.float32), 24_000)
    c = v.offer(speech(1.2, 0.1, 1), 'b', 0, 1.2)
    out = np.concatenate(list(tts.synth('hi again', 'en', c)))
    assert len(out) and np.allclose(out, 0.25, atol=1e-3) and not v.dead and tts.stock.said == []
    v.close()


def test_condition_from_another_thread_waits_for_the_stream(pack, me):
    """FIXES addon 1: voices/live.py conditions the other person's voice on its own thread while the speak thread reads a
    synth's stream from the same runner: one request at a time on the pipe (condition waits for the stream's end), both
    replies whole, instead of a JSON line read as a chunk's length (a hang, the runner killed)."""
    import threading
    gen = pack.synth('hello', 'en', me)
    first = next(gen)  # the stream is open: its other chunks are still in the pipe
    got = []
    t = threading.Thread(target=lambda: got.append(pack.condition(np.full(16_000, 0.25, np.float32), 16_000)))
    t.start()
    t.join(0.5)
    assert t.is_alive() and not got  # waits for the stream
    rest = list(gen)
    t.join(10)
    assert not t.is_alive() and isinstance(got[0], str)
    assert len(first) + sum(map(len, rest)) == 3 * 2400 and np.all(np.concatenate([first, *rest]) == np.float32(0.05))
    assert np.all(say(pack, 'hi', live(got[0])) == np.float32(0.25))  # the voice it made speaks


class FastTts:
    """A clone that keeps up: 0.1 s of speech at once."""
    sample_rate = 24_000

    def __init__(self, name):
        self.name = name

    def synth(self, text, lang, voice=None, speed=1.0):
        yield np.full(self.sample_rate // 10, 0.1, np.float32)

    def close(self):
        pass


def test_qwen3_that_does_not_start_gives_way_to_xtts(monkeypatch, tmp_path):
    """FIXES addon 6: the addon's Qwen3-TTS chosen, its runner does not start (CloneUnavailable): the next engine,
    XTTS-v2, speaks my cloned voice, said why; not the stock voice."""
    from peresmeshnik import languages, models
    side, s, made, _ = me_side(monkeypatch, tmp_path, key=False)
    monkeypatch.setattr(languages, 'addon_here', lambda: True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(8.0, (572, 83), 8.6))
    monkeypatch.setattr(models, 'is_fetched', lambda m, base=None: True)
    monkeypatch.setattr('peresmeshnik.voice.mine', lambda name, root=None: Voice(name, 'xtts', tmp_path))
    assert languages.direction(s, 'me_to_them').tts == 'qwen3-tts'
    tried = []

    def make(model_id, model_dir, threads=2):
        tried.append(model_id)
        if model_id == 'qwen3-tts':
            raise CloneUnavailable('the voice pack: ModuleNotFoundError: faster_qwen3_tts')
        return FastTts(model_id)
    monkeypatch.setattr(clone, 'make_engine', make)
    d = side._local(s)
    assert tried == ['qwen3-tts', 'xtts-v2'] and made[0].name == 'xtts-v2' and d.warm == 1
    assert side.errors == ['note: me → them: xtts-v2 speaks your cloned voice: qwen3-tts did not start (the voice pack: '
                           'ModuleNotFoundError: faster_qwen3_tts)']


def test_the_runner_sends_no_telemetry(monkeypatch):
    """C: the runner alone (no app before it) called onnxruntime's telemetry from Qwen3's tokenizer; the variable on its
    import (Linux, macOS), the call before the engine's code (Windows' TraceLogging does not read it)."""
    env = {k: v for k, v in os.environ.items() if k != 'ORT_DISABLE_TELEMETRY'}
    code = ('import importlib.util, os, sys; s = importlib.util.spec_from_file_location("r", sys.argv[1]); '
            's.loader.exec_module(importlib.util.module_from_spec(s)); print(os.environ.get("ORT_DISABLE_TELEMETRY"))')
    assert subprocess.run([sys.executable, '-I', '-c', code, str(RUNNER)], env=env, capture_output=True, text=True,
                          check=True).stdout.strip() == '1'
    import importlib.util
    spec = importlib.util.spec_from_file_location('voice_runner_t', RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import onnxruntime as rt
    calls = []
    monkeypatch.setattr(rt, 'disable_telemetry_events', lambda: calls.append(1))
    m.no_telemetry()
    assert calls == [1]

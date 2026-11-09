"""macOS call sound, no models: the process tap helper (or its honest refusal without permission), the original muted
while tapped, BlackHole 16ch for macOS < 14.4 (capture, and the pass-through while translation is off), the
headphones through PortAudio and their reopening. Needs BlackHole 2ch and 16ch (CI: brew); BlackHole 2ch stands in
for the speakers / headphones (the runner has no real output), its input side is what they would play.

PERESMESHNIK_TCC=denied|granted: the CI step wrote that answer for "System Audio Recording" into TCC.db of the runner's
VM; the tests that need the other answer skip. PERESMESHNIK_REQUIRE_MACOS=1 (CI): a missing BlackHole, tap helper or
PERESMESHNIK_TCC fails the test instead of skipping it, so a green job means the tests ran.
"""
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS')
TCC = os.environ.get('PERESMESHNIK_TCC', '')
SPEAKERS = 'BlackHole 2ch'
SR = 48000


def blackhole() -> bool:
    if sys.platform != 'darwin':
        return False
    from peresmeshnik.audio import coreaudio
    names = {d['name'] for d in coreaudio.devices()}
    return {'BlackHole 2ch', 'BlackHole 16ch'} <= names


REQUIRE = os.environ.get('PERESMESHNIK_REQUIRE_MACOS') == '1'


def need(ok: bool, why: str) -> None:
    if not ok:
        (pytest.fail if REQUIRE else pytest.skip)(why)


@pytest.fixture
def bh():
    need(blackhole(), 'needs BlackHole 2ch and 16ch')


@pytest.fixture
def helper_ok():
    need(helper_built(), 'needs the tap helper: PERESMESHNIK_TAP or native/macos/tap/build (swiftc)')


@pytest.fixture
def granted():
    need(TCC in ('denied', 'granted'), 'needs PERESMESHNIK_TCC (the TCC.db row of the CI step)')
    if TCC != 'granted':
        pytest.skip('PERESMESHNIK_TCC=granted: the runner allows System Audio Recording')


@pytest.fixture
def denied():
    need(TCC in ('denied', 'granted'), 'needs PERESMESHNIK_TCC (the TCC.db row of the CI step)')
    if TCC != 'denied':
        pytest.skip('PERESMESHNIK_TCC=denied: the runner denies System Audio Recording')


needs_blackhole = pytest.mark.usefixtures('bh')


def helper_built() -> bool:
    try:
        from peresmeshnik.audio.capture_macos import helper
        return bool(helper())
    except RuntimeError:
        return False


needs_helper = pytest.mark.usefixtures('helper_ok')


def tone(seconds: float, hz: float = 440.0, sr: int = SR) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * hz * np.arange(int(seconds * sr)) / sr)).astype(np.float32)


def peak_hz(x: np.ndarray, sr: int) -> float:
    x = x[:sr]  # 1 s: 1 Hz bins
    return float(np.abs(np.fft.rfft(x * np.hanning(len(x)))).argmax() * sr / len(x))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def record(device: str, seconds: float) -> np.ndarray:
    """What `device` plays, at SR (recorded at the device's own rate: two rates on one device break each other)."""
    import sounddevice as sd

    from peresmeshnik.audio.coreaudio import stream_args
    from peresmeshnik.audio.resample import Resampler
    args = stream_args(device, output=False)  # as the app's streams (coreaudio.BLOCK)
    sr = args.pop('samplerate')
    x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, device=device, dtype='float32', blocking=True, **args)[:, 0]
    return Resampler(sr, SR)(x)


def play(x: np.ndarray, device: str, blocking: bool = False) -> None:
    import sounddevice as sd

    from peresmeshnik.audio.coreaudio import stream_args
    from peresmeshnik.audio.resample import Resampler
    args = stream_args(device)
    sr = args.pop('samplerate')
    sd.play(Resampler(SR, sr)(x), sr, device=device, blocking=blocking, **args)


def set_default_output(name: str) -> str:
    """Make `name` the system's default output; returns the one before (tests only)."""
    import ctypes

    from peresmeshnik.audio import coreaudio as ca
    before = ca.default()
    dev = ctypes.c_uint32(ca.device_id(name))
    err = ca._lib().AudioObjectSetPropertyData(ca.SYSTEM, ca.Address(ca.DEFAULT_OUTPUT, ca.GLOBAL, 0), 0, None, 4, ctypes.byref(dev))
    assert err == 0 and ca.default() == name, (err, ca.default())
    return before


def until(cond, s: float = 5.0) -> bool:
    end = time.monotonic() + s
    while not (ok := bool(cond())) and time.monotonic() < end:
        time.sleep(0.05)
    return ok


def set_rate(name: str, hz: int) -> None:
    """The device's own rate, as Audio MIDI Setup sets it (tests only)."""
    import ctypes

    from peresmeshnik.audio import coreaudio as ca
    v = ctypes.c_double(hz)
    err = ca._lib().AudioObjectSetPropertyData(ca.device_id(name), ca.Address(ca.RATE, ca.GLOBAL, 0), 0, None, 8, ctypes.byref(v))
    assert err == 0 and until(lambda: ca.rate(name) == hz), (err, ca.rate(name))


@pytest.fixture
def afplay(tmp_path):
    """A process playing a 440 Hz tone (60 s) on the default output, which is BlackHole 2ch (the 'speakers')."""
    import soundfile as sf
    before = set_default_output(SPEAKERS)
    sf.write(tmp_path / 'tone.wav', tone(60), SR)
    p = subprocess.Popen(['afplay', str(tmp_path / 'tone.wav')])
    time.sleep(1.0)  # its audio object appears once it plays
    yield p
    p.kill()
    p.wait()
    if before:
        set_default_output(before)


class Frames:
    def __init__(self):
        self.got, self.lock = [], threading.Lock()

    def __call__(self, x):
        with self.lock:
            self.got.append(x)

    def pcm(self) -> np.ndarray:
        with self.lock:
            return np.concatenate(self.got) if self.got else np.zeros(0, np.float32)


@needs_helper
@needs_blackhole
def test_helper_answers_check_and_lists_the_playing_process(afplay):
    from peresmeshnik.audio import capture_macos as cm
    code, why = cm.check()
    print(f'check: {code} {why!r}')
    assert code == {'denied': 4, 'granted': 0}.get(TCC, code)
    if code == 4:
        assert 'System Audio Recording' in why
    procs = cm.processes()
    assert any(p['pid'] == afplay.pid and p['playing'] for p in procs), procs


def test_call_app_keys_fold_helpers_and_webkit():
    from peresmeshnik.audio import capture_macos as cm
    assert cm.app_of('com.google.Chrome.helper.renderer') == 'com.google.Chrome'
    assert cm.app_of('us.zoom.xos') == 'us.zoom.xos'
    assert cm.app_of('com.apple.WebKit.GPU') == cm.WEBKIT  # Safari's media; the tap's --bundle prefix takes it


@needs_helper
@needs_blackhole
@pytest.mark.usefixtures('denied')
def test_without_permission_the_tap_says_why(afplay):
    from peresmeshnik.audio import capture_macos as cm
    ended = []
    tap = cm.ProcessTap(pid=afplay.pid, on_end=ended.append)
    frames = Frames()
    tap.start(frames)
    for _ in range(50):
        if ended:
            break
        time.sleep(0.1)
    tap.stop()
    assert ended and 'System Audio Recording' in ended[0] and 'System Settings' in ended[0], ended
    assert not len(frames.pcm())
    from peresmeshnik import config
    from peresmeshnik.app import Controller  # the app refuses the them -> me side with the same reason
    with pytest.raises(RuntimeError, match='System Audio Recording'):
        Controller(overlay=None, on_state=print)._input('them', config.Settings())


@needs_helper
@needs_blackhole
@pytest.mark.usefixtures('granted')
def test_tap_hears_the_process_and_mutes_it_for_the_speakers(afplay):
    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.engines import SAMPLE_RATE
    before = record(SPEAKERS, 1.0)
    assert rms(before) > 0.1 and abs(peak_hz(before, SR) - 440) <= 2, 'afplay is not heard on the speakers'
    frames, ended = Frames(), []
    tap = cm.ProcessTap(pid=afplay.pid, on_end=ended.append)
    tap.start(frames)
    try:
        for _ in range(50):  # the header comes once the tap runs
            if len(frames.pcm()) > SAMPLE_RATE // 2:
                break
            time.sleep(0.1)
        during = record(SPEAKERS, 1.5)
        time.sleep(0.5)
        got = frames.pcm()
    finally:
        tap.stop()
    after = record(SPEAKERS, 1.0)
    print(f'tap: rate {tap.rate}, {len(got) / SAMPLE_RATE:.2f} s, rms {rms(got):.3f}, {peak_hz(got[-SAMPLE_RATE:], SAMPLE_RATE):.0f} Hz; '
          f'speakers rms before {rms(before):.3f} during {rms(during[SR // 2:]):.4f} after {rms(after[SR // 2:]):.3f}; {tap.status}')
    assert not ended, ended
    assert rms(got[-SAMPLE_RATE:]) > 0.1 and abs(peak_hz(got[-SAMPLE_RATE:], SAMPLE_RATE) - 440) <= 2
    assert rms(during[SR // 2:]) < 0.01, 'the original still plays while tapped'
    assert rms(after[SR // 2:]) > 0.1, 'the original did not come back after the tap'


@needs_helper
@needs_blackhole
@pytest.mark.usefixtures('granted')
def test_a_killed_helper_lets_the_original_through(afplay):
    """Core Audio removes a private tap and its aggregate device with the process that made them: the call is heard
    again even when the helper could not clean up (SIGKILL), and the app is told how to get it back if not. CI runs it
    first, on a fresh coreaudiod: late in the set the app stayed muted in 9 of 10 tries (stage 5, BENCH)."""
    import signal

    from peresmeshnik.audio import capture_macos as cm
    frames, ended = Frames(), []
    tap = cm.ProcessTap(pid=afplay.pid, on_end=ended.append)
    tap.start(frames)
    try:
        assert until(lambda: len(frames.pcm()) > 8000), tap.status
        during = record(SPEAKERS, 1.0)
        os.kill(tap._proc.pid, signal.SIGKILL)
        assert until(lambda: ended), 'no on_end'
        after = record(SPEAKERS, 1.5)
    finally:
        tap.stop()
    print(f'killed helper: speakers rms during {rms(during[SR // 2:]):.4f}, after {rms(after[SR // 2:]):.3f}; {ended}')
    assert rms(during[SR // 2:]) < 0.01
    assert rms(after[SR // 2:]) > 0.1, 'the original stayed muted after the helper was killed'
    assert '-9' in ended[0] and 'killall coreaudiod' in ended[0], ended


@needs_helper
@needs_blackhole
@pytest.mark.usefixtures('granted')
def test_tap_keeps_the_pitch_when_the_device_changes_its_rate(afplay):
    """m5: the call app's device changes its rate (speakers at 48 kHz -> AirPods in a call, 16/24 kHz; here BlackHole
    2ch 48 -> 44.1 kHz): the voice keeps its pitch. On the runner the tap keeps its own format (Core Audio converts);
    if the tap's own rate changes, the helper exits with 7 and is started again at the new one."""
    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.audio import coreaudio as ca
    from peresmeshnik.engines import SAMPLE_RATE
    frames, ended = Frames(), []
    tap = cm.ProcessTap(pid=afplay.pid, on_end=ended.append)
    old = ca.rate(SPEAKERS)
    tap.start(frames)
    try:
        assert until(lambda: tap.rate), tap.status
        first = tap.rate
        set_rate(SPEAKERS, 44100 if old != 44100 else 48000)
        new = ca.rate(SPEAKERS)
        time.sleep(1.0)  # a restart takes ~0.3 s
        n = len(frames.pcm())
        time.sleep(1.5)
        got = frames.pcm()[n:]
    finally:
        tap.stop()
        set_rate(SPEAKERS, old)
    print(f'rate change: device {old} -> {new} Hz, tap {first} -> {tap.rate} Hz; after it '
          f'{len(got) / SAMPLE_RATE:.2f} s, {peak_hz(got, SAMPLE_RATE):.0f} Hz, rms {rms(got):.3f}; {tap.status}')
    assert not ended, ended
    assert tap.rate in (first, new), tap.rate
    assert len(got) > SAMPLE_RATE and rms(got) > 0.1 and abs(peak_hz(got, SAMPLE_RATE) - 440) <= 2


@needs_blackhole
def test_blackhole_16ch_capture_and_pass_through(tmp_path, monkeypatch):
    """macOS < 14.4: the call app plays into BlackHole 16ch; translating, the app records it; off, it passes it on to
    the headphones, and only while the call app plays (with no call nothing runs)."""
    import soundfile as sf

    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.engines import SAMPLE_RATE
    frames, ended = Frames(), []
    cap = cm.BlackHoleIn(on_end=ended.append)
    cap.start(frames)
    play(tone(1.5), cm.CALL_DEVICE, blocking=True)
    cap.stop()
    got = frames.pcm()
    assert rms(got) > 0.1 and abs(peak_hz(got[np.argmax(np.abs(got) > 0.1):], SAMPLE_RATE) - 440) <= 2

    # off: the pass-through runs only while the call app plays into BlackHole 16ch (Core Audio says so)
    import psutil
    monkeypatch.setattr(cm.PassThrough, 'ZERO_S', 1.0)
    me = psutil.Process()

    def cpu_over(seconds: float) -> float:
        t0, c0 = time.monotonic(), sum(me.cpu_times()[:2])
        time.sleep(seconds)
        return (sum(me.cpu_times()[:2]) - c0) / (time.monotonic() - t0)
    sf.write(tmp_path / 'tone.wav', tone(7), SR)
    before = set_default_output(cm.CALL_DEVICE)  # afplay is the call app with BlackHole 16ch as its speaker
    through = cm.PassThrough(lambda exclude='': SPEAKERS, on_end=ended.append)
    calls = []
    try:
        through.start()
        idle = cpu_over(3.0)
        idle_opened = through.opened
        calls.append(subprocess.Popen(['afplay', str(tmp_path / 'tone.wav')]))
        assert until(lambda: through.opened == 1), 'the call app plays, the pass-through did not open'
        time.sleep(0.5)
        heard = record(SPEAKERS, 1.0)
        busy = cpu_over(3.0)
        calls[0].wait(10)
        assert until(lambda: not through.output, 8), 'the call app stopped, the pass-through stayed open'
        n = through.opened
        calls.append(subprocess.Popen(['afplay', str(tmp_path / 'tone.wav')]))  # the next call
        assert until(lambda: through.opened > n), 'the next call did not open it'
        time.sleep(0.5)
        again = record(SPEAKERS, 1.0)
    finally:
        through.stop()
        for c in calls:
            c.kill()
        if before:
            set_default_output(before)
    print(f'pass-through: CPU {idle * 100:.2f} % of a core with no call (opened {idle_opened}), {busy * 100:.2f} % '
          f'during one; speakers rms {rms(heard):.3f}, {peak_hz(heard, SR):.0f} Hz; next call rms {rms(again):.3f}; '
          f'opened {through.opened} times')
    assert idle_opened == 0 and idle < 0.005
    assert rms(heard) > 0.1 and abs(peak_hz(heard, SR) - 440) <= 2
    assert rms(again) > 0.1
    assert not ended, ended


@needs_blackhole
def test_headphones_play_and_reopen_after_a_stream_error():
    from peresmeshnik.app import Headphones
    from peresmeshnik.audio.out import SoundDeviceOut
    ended = []
    out = SoundDeviceOut(22050, SPEAKERS, on_end=ended.append)
    out.play(tone(2.0, sr=22050), 22050)
    heard = record(SPEAKERS, 1.0)
    out.flush()
    out.stop()
    assert rms(heard) > 0.1 and abs(peak_hz(heard, SR) - 440) <= 2
    assert not ended

    states = []
    hp = Headphones(SPEAKERS, on_state=states.append)
    hp.play(tone(0.5, sr=22050), 22050)
    first = hp._out
    first._stream.abort()  # the model of a dead device: the stream ends by itself
    for _ in range(30):
        if hp._out is None:
            break
        time.sleep(0.1)
    assert hp._out is None and states == ['error: the headphones went away; reopening'], states
    hp.play(tone(2.0, sr=22050), 22050)
    assert hp._out is not None and hp._out is not first and hp._out.alive
    heard = record(SPEAKERS, 1.0)
    hp.stop()
    assert states[-1].startswith('translating (the headphones are back'), states
    assert rms(heard) > 0.1


TONE_APP = 'io.github.peresmeshnik.test-tone'
PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>CFBundleIdentifier</key><string>{id}</string><key>CFBundleExecutable</key>
<string>tone</string><key>CFBundlePackageType</key><string>APPL</string></dict></plist>"""


@pytest.fixture(scope='module')
def tone_app(tmp_path_factory):
    """A call app with a helper, as Chrome or Electron apps are: Tone.app (TONE_APP) and Tone Helper.app (TONE_APP.helper)
    inside it, built from tests/macos/tone.swift. Returns {bundle id: its executable}."""
    import shutil
    need(bool(shutil.which('swiftc')), 'needs swiftc')
    root = tmp_path_factory.mktemp('tone')
    apps = {TONE_APP: root / 'Tone.app', TONE_APP + '.helper': root / 'Tone.app' / 'Contents' / 'Frameworks' / 'Tone Helper.app'}
    exe = {}
    for bundle_id, app in apps.items():
        (app / 'Contents' / 'MacOS').mkdir(parents=True)
        (app / 'Contents' / 'Info.plist').write_text(PLIST.format(id=bundle_id))
        exe[bundle_id] = app / 'Contents' / 'MacOS' / 'tone'
    src = os.path.join(os.path.dirname(__file__), 'macos', 'tone.swift')
    subprocess.run(['swiftc', '-O', src, '-o', str(exe[TONE_APP])], check=True, timeout=300)
    shutil.copy(exe[TONE_APP], exe[TONE_APP + '.helper'])
    return exe


@needs_helper
@needs_blackhole
@pytest.mark.usefixtures('granted')
def test_tap_by_bundle_waits_for_the_app_and_takes_its_helpers(tone_app):
    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.engines import SAMPLE_RATE
    before = set_default_output(SPEAKERS)  # stereo, as a call app's speaker (a mono tap of 16 channels is 1/8 as loud)
    frames, ended, procs = Frames(), [], []
    tap = cm.ProcessTap(TONE_APP, on_end=ended.append)
    try:
        tap.start(frames)
        time.sleep(1.0)
        assert not len(frames.pcm()) and 'waiting' in tap.status, tap.status  # not playing yet: the helper waits
        procs.append(subprocess.Popen([str(tone_app[TONE_APP]), '440']))
        time.sleep(2.5)
        first = frames.pcm()[-SAMPLE_RATE:]
        procs.append(subprocess.Popen([str(tone_app[TONE_APP + '.helper']), '660']))  # starts later: joins the tap
        time.sleep(2.5)
        both = frames.pcm()[-SAMPLE_RATE:]
        listed = cm.apps()  # the tray's menu: one row for the app and its helper
    finally:
        tap.stop()
        for p in procs:
            p.kill()
        if before:
            set_default_output(before)
    spectrum = np.abs(np.fft.rfft(both * np.hanning(len(both))))
    print(f'bundle tap: first {peak_hz(first, SAMPLE_RATE):.0f} Hz rms {rms(first):.3f}; then 440 Hz {spectrum[440]:.0f}, '
          f'660 Hz {spectrum[660]:.0f}; {tap.status}')
    assert not ended, ended
    assert [k for k, _ in listed].count(TONE_APP) == 1 and not any(k.endswith('.helper') for k, _ in listed), listed
    assert rms(first) > 0.05 and abs(peak_hz(first, SAMPLE_RATE) - 440) <= 2
    assert spectrum[660] > 0.3 * spectrum[440], 'the helper process did not join the tap'


RECORDER = '''
import signal, sys, threading, numpy as np, sounddevice as sd
from peresmeshnik.audio.coreaudio import stream_args
args, got, done = stream_args(sys.argv[1], output=False), [], threading.Event()
sr = args['samplerate']
signal.signal(signal.SIGTERM, lambda *_: done.set())
with sd.InputStream(device=sys.argv[1], channels=1, dtype='float32', callback=lambda x, *_: got.append(x[:, 0].copy()), **args):
    print('recording', flush=True)
    done.wait(float(sys.argv[2]))
x = np.concatenate(got)
np.save(sys.argv[3], np.stack([x, np.full_like(x, sr)]))
'''


def recorder(device: str, seconds: float, path) -> subprocess.Popen:
    """Records `device` in a process of its own (a PortAudio refresh in this one would end the recording), until
    `seconds` or SIGTERM."""
    p = subprocess.Popen([sys.executable, '-c', RECORDER, device, str(seconds), str(path)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == 'recording'
    return p


def recorded(proc: subprocess.Popen, path, stop: bool = False) -> tuple[np.ndarray, int]:
    if stop:
        proc.terminate()
    assert proc.wait(60) == 0
    x, sr = np.load(path)
    return x, int(sr[0])


def longest_quiet_s(x: np.ndarray, sr: int, level: float = 0.02) -> float:
    """The longest run of near-silence between the first and the last loud sample."""
    loud = np.flatnonzero(np.abs(x) > level)
    if len(loud) < 2:
        return len(x) / sr
    return float(np.diff(loud).max() / sr)


@needs_blackhole
def test_devices_refresh_keeps_the_streams_with_a_short_gap(tmp_path):
    """(Answer 19:40 (5)) the saved headphones are back while translating: PortAudio is refreshed with every stream
    of the process suspended and resumed, the engines untouched. A tone goes BlackHole 16ch -> BlackHoleIn ->
    Headphones (BlackHole 2ch) of the wizard's test phrase (a Controller(test=True)): the main controller owns neither
    stream, the registry has them."""
    import soundfile as sf

    from peresmeshnik import config
    from peresmeshnik.app import Controller
    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.engines import SAMPLE_RATE
    before = set_default_output(cm.CALL_DEVICE)  # afplay (another process: not touched by our refresh) is the call app
    sf.write(tmp_path / 'tone.wav', tone(8), SR)
    call = subprocess.Popen(['afplay', str(tmp_path / 'tone.wav')])
    states = []
    ctl = Controller(overlay=None, on_state=states.append)
    s = config.Settings()
    s.ui.headphones = SPEAKERS
    test = Controller(overlay=None, on_state=states.append, them_file=os.path.join(DATA, 'call-en.ogg'), settings=s, test=True)
    hp = test._output('them', s)  # the test phrase's own headphones, as the wizard plays it
    cap = cm.BlackHoleIn()
    q = __import__('queue').Queue()  # as the app: the capture callback only hands on, a thread plays (a Side)

    def player():
        while (pcm := q.get()) is not None:
            hp.play(pcm, SAMPLE_RATE)
    t_play = threading.Thread(target=player, daemon=True)
    t_play.start()
    try:
        time.sleep(0.5)
        cap.start(q.put)
        rec = recorder(SPEAKERS, 6.0, tmp_path / 'heard.npy')
        time.sleep(3.0)
        ctl.on = True
        phases = ctl.submit(ctl._refresh_devices).result()
        x, sr = recorded(rec, tmp_path / 'heard.npy')
    finally:
        ctl.on = False
        cap.stop()
        q.put(None)
        t_play.join(5)
        hp.stop()
        call.terminate()
        call.wait(5)
        if before:
            set_default_output(before)
    gap = longest_quiet_s(x, sr)
    print(f'devices refresh: {phases} ms, longest gap heard {gap * 1000:.0f} ms; {states[-1:]}')
    assert phases['streams'] >= 2 and not phases['failed'], phases
    assert rms(x) > 0.05, 'the tone did not come through'
    assert gap <= 0.8, f'{gap:.2f} s of silence'  # the aim is 0.5 s (BENCH); a stop is quantised by Pa_Sleep(100)


FAKE = os.path.join(os.path.dirname(__file__), 'macos', 'fakedevice.swift')


@pytest.fixture(scope='module')
def fake_exe(tmp_path_factory):
    import shutil
    need(bool(shutil.which('swiftc')), 'needs swiftc')
    exe = tmp_path_factory.mktemp('fake') / 'fakedevice'
    subprocess.run(['swiftc', '-O', FAKE, '-o', str(exe)], check=True, timeout=300)
    return exe


@pytest.fixture
def fakedevice(fake_exe):
    """make(name, over) -> a public aggregate device `name` over `over`, made by another process (as headphones that
    are plugged in); unplug(p) destroys it. Core Audio tells the app through its own notifications."""
    from peresmeshnik.audio import coreaudio as ca
    made = []

    def make(name: str, over: str = SPEAKERS) -> subprocess.Popen:
        p = subprocess.Popen([str(fake_exe), name, over], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        made.append(p)
        assert p.stdout.readline().startswith('made '), p.wait(5)
        assert until(lambda: ca.device_id(name) is not None), f'{name} did not appear'
        return p
    yield make
    for p in made:
        unplug(p)


def unplug(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.stdin.close()
        p.wait(10)


PHONES, SPEAKERS2 = 'Test Phones', 'Test Speakers'


@needs_blackhole
def test_headphones_follow_the_saved_device_as_it_goes_and_comes_back(fakedevice):
    """M6 through Core Audio's own notifications: the saved headphones (Test Phones, over BlackHole 2ch) are not there,
    the default (Test Speakers, over BlackHole 16ch) plays; they come (Arrival, a PortAudio refresh, the output moves
    there with the speech not played yet), they go (Watch: on_end, the default again)."""
    from peresmeshnik.app import Controller, Headphones
    from peresmeshnik.audio import coreaudio as ca
    fakedevice(SPEAKERS2, 'BlackHole 16ch')
    ca.refresh_portaudio()  # PortAudio sees the devices of its last refresh
    before = set_default_output(SPEAKERS2)
    states = []
    ctl = Controller(overlay=None, on_state=print)
    hp = Headphones(PHONES, states.append, on_back=lambda: ctl.submit(ctl._refresh_devices))
    stop = threading.Event()

    def speak():  # the translation's voice, chunk by chunk
        x = tone(0.1, sr=22050)
        while not stop.is_set():
            hp.play(x, 22050)
            time.sleep(0.1)
    t = threading.Thread(target=speak, daemon=True)
    t.start()
    try:
        assert until(lambda: hp._out is not None) and hp._out.device == SPEAKERS2, states
        phones = fakedevice(PHONES)
        t0 = time.monotonic()
        assert until(lambda: hp._out is not None and hp._out.device == PHONES, 10), (hp._out and hp._out.device, states)
        moved = time.monotonic() - t0
        time.sleep(0.3)
        heard = record(SPEAKERS, 1.0)  # what Test Phones plays
        unplug(phones)
        t0 = time.monotonic()
        assert until(lambda: any('went away' in x for x in states)), states
        noticed = time.monotonic() - t0
        assert until(lambda: hp._out is not None and hp._out.device == SPEAKERS2), (hp._out and hp._out.device, states)
    finally:
        stop.set()
        t.join(5)
        hp.stop()
        if before:
            set_default_output(before)
    print(f'saved headphones: back on them in {moved:.2f} s (rms {rms(heard):.3f}), unplugging noticed in '
          f'{noticed:.2f} s; {states}')
    assert rms(heard) > 0.05


@needs_blackhole
def test_headphones_unknown_to_portaudio_play_on_the_default_meanwhile(fakedevice):
    """M2: Core Audio has the saved headphones, PortAudio does not see them even after the refresh it was asked for
    (a Bluetooth profile settling), or nobody can refresh: the default plays meanwhile, not silence."""
    from peresmeshnik.app import Headphones
    from peresmeshnik.audio import coreaudio as ca
    fakedevice(SPEAKERS2, 'BlackHole 16ch')
    ca.refresh_portaudio()
    before = set_default_output(SPEAKERS2)
    phones = fakedevice(PHONES)  # after PortAudio's refresh: unknown to it
    asked, states = [], []
    hp = Headphones(PHONES, states.append, on_back=lambda: asked.append(1))  # a refresh that does not help
    hp.retry_s = 0.1
    hp2 = Headphones(PHONES, states.append)
    try:
        hp.play(tone(0.3, sr=22050), 22050)
        first = hp._out
        time.sleep(0.2)
        hp.play(tone(0.3, sr=22050), 22050)
        hp2.play(tone(0.3, sr=22050), 22050)
        devices = [h._out and h._out.device for h in (hp, hp2)]
        waits, asked_first = hp._arrival is not None, list(asked)
        # they change (out and in again), still unknown to PortAudio after the "refresh": the waiting Arrival fires,
        # the output stays on a device that plays (N4) and waits for the next change
        unplug(phones)
        phones = fakedevice(PHONES)
        assert until(lambda: len(asked) == 2), 'the waiting Arrival did not fire'
        assert until(lambda: hp._arrival is not None), 'not waiting again'
        stayed = hp._out and hp._out.device
    finally:
        hp.stop()
        hp2.stop()
        if before:
            set_default_output(before)
    print(f'unknown to PortAudio: refresh asked {len(asked)}, then {devices}; {states}')
    assert first is None and asked_first == [1]  # the first chunk: a refresh is asked, nothing plays yet
    assert devices == [SPEAKERS2, SPEAKERS2] and waits  # then the default, and back to them when they change
    assert stayed == SPEAKERS2


PLAYER = '''
import signal, sys, numpy as np, sounddevice as sd
from peresmeshnik.audio.coreaudio import stream_args
# SIGTERM ends it cleanly (PortAudio closes the stream at exit): killed while playing, its last 1024-frame buffer could
# stay looping in BlackHole 16ch for the rest of the runner's session (seen: 423 Hz, every later test hears it)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
args = stream_args(sys.argv[1])
sr = args.pop('samplerate')
x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(float(sys.argv[2]) * sr)) / sr)).astype(np.float32)
sd.play(x, sr, device=sys.argv[1], blocking=True, **args)
'''


def player(device: str, seconds: float) -> subprocess.Popen:
    """A call app playing a 440 Hz tone into `device`, in a process of its own (whatever the default output)."""
    return subprocess.Popen([sys.executable, '-c', PLAYER, device, str(seconds)])


@needs_blackhole
def test_pass_through_plays_on_the_default_while_portaudio_does_not_see_the_saved_headphones(fakedevice, monkeypatch):
    """M2 / N2 (macOS < 14.4, translation off): the saved headphones are in Core Audio, PortAudio does not see them
    (a Bluetooth profile settling: hidden here): one refresh, the default plays, no refresh loop; when they change and
    PortAudio sees them, the waiting Arrival moves the pass-through to them."""
    from peresmeshnik import config
    from peresmeshnik.app import Controller
    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.audio import coreaudio as ca
    from peresmeshnik.audio import out as out_mod
    monkeypatch.setenv('PERESMESHNIK_CALL_PATH', 'blackhole')
    fakedevice(SPEAKERS2)
    ca.refresh_portaudio()
    before = set_default_output(SPEAKERS2)
    phones = fakedevice(PHONES)
    hidden = {PHONES}
    names, index = out_mod.portaudio_names, out_mod.portaudio_index

    def hide_index(name, output=True):
        if name in hidden:
            raise out_mod.NotInPortAudio(f'{name}: hidden by the test')
        return index(name, output)
    monkeypatch.setattr(out_mod, 'portaudio_names', lambda output=True: names(output) - hidden)
    monkeypatch.setattr(out_mod, 'portaudio_index', hide_index)
    refreshes = []
    refresh = ca.refresh_portaudio
    monkeypatch.setattr(ca, 'refresh_portaudio', lambda: refreshes.append(1) or refresh())
    s = config.Settings()
    s.ui.headphones = PHONES
    states = []
    ctl = Controller(overlay=None, on_state=states.append, settings=s)
    call = None
    try:
        ctl.submit(ctl._through_start).result()
        assert ctl._through is not None and ctl._through_arrival is not None, states
        call = player(cm.CALL_DEVICE, 15.0)
        assert until(lambda: ctl._through.output == SPEAKERS2), (ctl._through.output, states)
        time.sleep(2.0)  # a loop would refresh again and again meanwhile
        looped = len(refreshes)
        heard = record(SPEAKERS, 1.0)  # Test Speakers plays into BlackHole 2ch
        hidden.clear()  # the profile settled
        unplug(phones)
        phones = fakedevice(PHONES)  # a change of the saved headphones: the waiting Arrival
        assert until(lambda: ctl._through is not None and ctl._through.output == PHONES, 10), (ctl._through and ctl._through.output, states)
    finally:
        ctl.submit(ctl._through_stop).result()
        if call:
            call.terminate()
            call.wait(5)
        if before:
            set_default_output(before)
    print(f'pass-through, saved headphones unknown to PortAudio: {looped} refresh(es) in 2 s, on {SPEAKERS2} rms '
          f'{rms(heard):.3f}; then on {PHONES} after {len(refreshes)} refreshes; {states}')
    assert looped == 1
    assert rms(heard) > 0.1


@needs_blackhole
def test_pass_through_that_cannot_open_says_so():
    """N3: opening on Core Audio's event thread fails (no headphones): on_end says why, not lost in that thread."""
    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.audio import coreaudio as ca

    def pick(exclude=''):
        raise RuntimeError('no output device')
    busy = ca.Running(cm.CALL_DEVICE, lambda running: None)
    try:  # the last test's call app may still run IO on BlackHole 16ch: start() would open (and fail) at once
        assert until(lambda: not busy.running(), 10), f'{cm.CALL_DEVICE} stays busy'
    finally:
        busy.close()
    ended, call = [], None
    through = cm.PassThrough(pick, on_end=ended.append)
    try:
        through.start()
        call = player(cm.CALL_DEVICE, 4.0)
        assert until(lambda: ended), 'the failure was lost'
    finally:
        through.stop()
        if call:
            call.terminate()
            call.wait(5)
    print(f'pass-through cannot open: {ended}')
    assert 'no output device' in ended[0]


def test_a_listener_lives_while_core_audio_holds_it(monkeypatch):
    """A listener nobody closed is not freed under Core Audio (that was a SIGSEGV on its listener queue): it stays
    registered by its key, with the one C callback that is never freed; after close() a late call does nothing."""
    import gc
    import types

    from peresmeshnik.audio import coreaudio as ca
    added, removed, fired = [], [], []
    monkeypatch.setattr(ca, '_lib', lambda: types.SimpleNamespace(
        AudioObjectAddPropertyListener=lambda obj, a, cb, key: added.append((cb, key)) or 0,
        AudioObjectRemovePropertyListener=lambda obj, a, cb, key: removed.append((cb, key)) or 0))

    class Changed(ca._Listener):
        once = False

        def _check(self, obj):
            return obj

        def _fire(self, why):
            fired.append(why)
    Changed([(ca.SYSTEM, ca.DEVICES, ca.GLOBAL)], 'the devices')  # dropped at once, never closed
    gc.collect()
    (cb, key), = added
    assert cb is ca._dispatch and key in ca._listeners
    ca._dispatch(7, 1, None, key)
    assert until(lambda: fired == [7])
    ca._listeners[key].close()
    assert removed == [(ca._dispatch, key)] and key not in ca._listeners
    assert ca._dispatch(7, 1, None, key) == 0


@needs_blackhole
def test_a_stream_stopped_during_a_refresh_stays_closed():
    """N1: stop() between a refresh's suspend and resume: resume() does not open the stream again (no stream left
    open, the device not held)."""
    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.audio import coreaudio as ca
    from peresmeshnik.audio.out import SoundDeviceOut
    out = SoundDeviceOut(22050, SPEAKERS)
    cap = cm.BlackHoleIn()
    cap.start(lambda pcm: None)
    stops = []

    class Stopper:  # a holder whose suspend stops the others from another thread, as the control thread may
        def suspend(self):
            for stop in (lambda: out.stop(drain=False), cap.stop):
                stops.append(threading.Thread(target=stop))
                stops[-1].start()
            time.sleep(0.3)  # they are stopping now, the refresh still to come

        def resume(self):
            pass
    stopper = Stopper()
    ca.hold(stopper)
    try:
        r = ca.refresh_portaudio()
    finally:
        ca.release(stopper)
    for t in stops:
        t.join(5)
    print(f'stopped during a refresh: {r}; streams left {out._stream, cap._stream}')
    assert not r['failed'], r
    assert out._stream is None and cap._stream is None
    assert not {out, cap} & set(ca._holders)


@needs_blackhole
def test_the_microphone_is_kept_by_a_refresh(tmp_path):
    """M1: the wizard's voice enrollment records through mic.Mic; on macOS that is the Core Audio helper (S2),
    no PortAudio stream: not registered, and a refresh does not interrupt it."""
    import soundfile as sf

    from peresmeshnik.audio import coreaudio as ca
    from peresmeshnik.audio.mic import Mic
    from peresmeshnik.engines import SAMPLE_RATE
    t = tone(6)  # stereo: a mono file lands in the centre channel (3) of BlackHole 16ch, the helper reads the first
    sf.write(tmp_path / 'tone.wav', np.stack([t, t], axis=1), SR)
    before = set_default_output('BlackHole 16ch')  # BlackHole 2ch is the app's own: the microphone refuses it
    p = subprocess.Popen(['afplay', str(tmp_path / 'tone.wav')])
    frames = Frames()
    mic = Mic('BlackHole16ch_UID')
    try:
        time.sleep(1.0)
        mic.start(frames)
        time.sleep(0.5)
        held = mic in set(ca._holders)
        n = len(frames.pcm())
        r = ca.refresh_portaudio()
        time.sleep(1.0)
        got = frames.pcm()[n:]
    finally:
        mic.stop()
        p.wait(15)  # to its end: a killed writer may leave its last buffer looping in BlackHole
        if before:
            set_default_output(before)
    print(f'microphone: held {held}, refresh {r}, then {len(got) / SAMPLE_RATE:.2f} s rms {rms(got):.3f}')
    assert not held and not r['failed']
    assert len(got) > SAMPLE_RATE // 2 and rms(got) > 0.1, 'the microphone stopped at the refresh'


@needs_blackhole
def test_tap_restarts_at_most_once_a_second_and_gives_up(tmp_path, monkeypatch):
    """m5: a helper asking to be started again (exit 7) again and again: restarted at most once per RESTART_S; asked
    again after RESTARTS restarts within a minute, it ends with a message."""
    from peresmeshnik.audio import capture_macos as cm
    fake = tmp_path / 'tap'
    fake.write_text('#!/bin/sh\necho "peresmeshnik-tap 1 rate=48000 channels=1 format=f32le"\n'
                    'echo "peresmeshnik-tap: the tap\'s rate changed" >&2\nexit 7\n')
    fake.chmod(0o755)
    monkeypatch.setenv('PERESMESHNIK_TAP', str(fake))
    monkeypatch.setattr(cm, 'RESTART_S', 0.3)
    ended = []
    tap = cm.ProcessTap(pid=os.getpid(), on_end=ended.append)
    t0 = time.monotonic()
    tap.start(lambda pcm: None)
    try:
        assert until(lambda: ended, 10), 'no end'
        took = time.monotonic() - t0
    finally:
        tap.stop()
    gaps = np.diff(tap._restarts)
    print(f'restarts: {len(tap._restarts)} in {took:.2f} s, gaps {np.round(gaps, 2).tolist()}; {ended}')
    assert 'keeps changing' in ended[0], ended
    assert len(tap._restarts) == cm.RESTARTS and gaps.min() >= 0.3 * 0.95, gaps


MODELS = os.environ.get('PERESMESHNIK_WORK_GATE') == '1'
DATA = os.path.join(os.path.dirname(__file__), 'data')


def translate_through(tmp_path, args: list[str], speakers: str = SPEAKERS) -> dict:
    """`peresmeshnik run` on the 30 s English call with `args`, the translation played on `speakers` and recorded
    there: captions, word recall, RTF, CPU, peak memory of the run, and how much of the recording is voice."""
    import json
    import re
    import resource
    rec = recorder(speakers, 120.0, tmp_path / 'voice.npy')
    r = subprocess.run([sys.executable, '-m', 'peresmeshnik', 'run', '--in', os.path.join(DATA, 'call-en.ogg'), '--from', 'en',
                        *args, '--play', speakers, '--log', str(tmp_path / 'lat.jsonl')],
                       capture_output=True, text=True, encoding='utf-8', timeout=300)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20  # bytes on macOS (the largest child so far)
    x, sr = recorded(rec, tmp_path / 'voice.npy', stop=True)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads(r.stderr.strip().splitlines()[-1])
    original = ' '.join(ln[4:] for ln in r.stdout.splitlines() if ln.startswith('en: '))
    ref = re.findall(r"[\w']+", open(os.path.join(DATA, 'call-en.txt'), encoding='utf-8').read().lower())
    hyp = set(re.findall(r"[\w']+", original.lower()))
    s['word_recall'] = round(sum(w in hyp for w in ref) / len(ref), 2)
    s['peak_rss_mb'] = round(peak)
    s['captions_ru'] = sum(ln.startswith('ru: ') for ln in r.stdout.splitlines())
    frames = x[:len(x) // (sr // 50) * (sr // 50)].reshape(-1, sr // 50)  # 20 ms
    s['voice_heard_s'] = round(float((np.sqrt(np.mean(frames ** 2, axis=1)) > 0.01).sum() * 0.02), 1)
    clauses = [json.loads(line) for line in open(tmp_path / 'lat.jsonl', encoding='utf-8')]
    for k in ('caption_lag_s', 'voice_lag_s'):
        v = sorted(c[k] for c in clauses if k in c)
        s[k.replace('_s', '_p50_s')] = v[len(v) // 2] if v else None
        s[k.replace('_s', '_max_s')] = v[-1] if v else None
    s['stderr'] = [ln for ln in r.stderr.splitlines() if ln.startswith('tap:')]
    return s


@needs_blackhole
@pytest.mark.skipif(not MODELS, reason='needs the models: PERESMESHNIK_WORK_GATE=1')
def test_blackhole_16ch_path_gives_subtitles_and_voice(tmp_path):
    s = translate_through(tmp_path, ['--blackhole'])
    print('\nblackhole path:', s)
    assert s['word_recall'] > 0.6 and s['captions_ru'] > 0 and s['voice_heard_s'] > 5
    assert s['cpu_s_per_audio_s'] < 2.0  # rtf: a live run lasts as long as the audio (>= 1)


@needs_helper
@needs_blackhole
@pytest.mark.skipif(not MODELS, reason='needs the models: PERESMESHNIK_WORK_GATE=1')
@pytest.mark.usefixtures('granted')
def test_tap_path_gives_subtitles_and_voice(tmp_path):
    from peresmeshnik.audio.capture_macos import CALL_DEVICE
    before = set_default_output(CALL_DEVICE)  # afplay plays there: muted while tapped
    try:
        s = translate_through(tmp_path, ['--tap'])
    finally:
        if before:
            set_default_output(before)
    print('\ntap path:', s)
    assert s['word_recall'] > 0.6 and s['captions_ru'] > 0 and s['voice_heard_s'] > 5
    assert s['cpu_s_per_audio_s'] < 2.0  # rtf: a live run lasts as long as the audio (>= 1)

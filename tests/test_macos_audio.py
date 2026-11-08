"""macOS call sound, no models: the process tap helper (or its honest refusal without permission), the original muted
while tapped, BlackHole 16ch for macOS < 14.4 (capture, and the pass-through while translation is off), the
headphones through PortAudio and their reopening. Needs BlackHole 2ch and 16ch (CI: brew); BlackHole 2ch stands in
for the speakers / headphones (the runner has no real output), its input side is what they would play.

PERESMESHNIK_TCC=denied|granted: the CI step wrote that answer for "System Audio Recording" into TCC.db of the runner's
VM; unset, the tests that need one answer skip.
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


needs_blackhole = pytest.mark.skipif(not blackhole(), reason='needs BlackHole 2ch and 16ch')


def helper_built() -> bool:
    try:
        from peresmeshnik.audio.capture_macos import helper
        return bool(helper())
    except RuntimeError:
        return False


needs_helper = pytest.mark.skipif(sys.platform != 'darwin' or not helper_built(),
                                  reason='needs the tap helper: PERESMESHNIK_TAP or native/macos/tap/build (swiftc)')


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


@needs_helper
@needs_blackhole
@pytest.mark.skipif(TCC != 'denied', reason='PERESMESHNIK_TCC=denied: the runner denies System Audio Recording')
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
@pytest.mark.skipif(TCC != 'granted', reason='PERESMESHNIK_TCC=granted: the runner allows System Audio Recording')
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


@needs_blackhole
def test_blackhole_16ch_capture_and_pass_through(tmp_path):
    """macOS < 14.4: the call app plays into BlackHole 16ch; translating, the app records it; off, it passes it on."""
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

    import psutil
    before = set_default_output(cm.CALL_DEVICE)  # afplay is the call app with BlackHole 16ch as its speaker
    sf.write(tmp_path / 'tone.wav', tone(15), SR)
    call = subprocess.Popen(['afplay', str(tmp_path / 'tone.wav')])
    through = cm.PassThrough(SPEAKERS, on_end=ended.append)
    through.start()
    try:
        time.sleep(1.0)
        heard = record(SPEAKERS, 1.0)
        me = psutil.Process()
        t0, c0 = time.monotonic(), sum(me.cpu_times()[:2])
        time.sleep(10.0)
        cpu = (sum(me.cpu_times()[:2]) - c0) / (time.monotonic() - t0)
    finally:
        through.stop()
        call.kill()
        if before:
            set_default_output(before)
    print(f'pass-through: speakers rms {rms(heard):.3f}, {peak_hz(heard, SR):.0f} Hz; CPU {cpu * 100:.2f} % of a core')
    assert rms(heard) > 0.1 and abs(peak_hz(heard, SR) - 440) <= 2
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
    if not shutil.which('swiftc'):
        pytest.skip('needs swiftc')
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
@pytest.mark.skipif(TCC != 'granted', reason='PERESMESHNIK_TCC=granted: the runner allows System Audio Recording')
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
    """(Answer 19:40 (5)) the saved headphones are back while translating: PortAudio is refreshed with the streams
    suspended and resumed, the engines untouched. A tone goes BlackHole 16ch -> BlackHoleIn -> Headphones (BlackHole 2ch)."""
    import soundfile as sf

    from peresmeshnik.app import Controller, Headphones
    from peresmeshnik.audio import capture_macos as cm
    from peresmeshnik.engines import SAMPLE_RATE
    before = set_default_output(cm.CALL_DEVICE)  # afplay (another process: not touched by our refresh) is the call app
    sf.write(tmp_path / 'tone.wav', tone(8), SR)
    call = subprocess.Popen(['afplay', str(tmp_path / 'tone.wav')])
    states = []
    ctl = Controller(overlay=None, on_state=states.append)
    hp = Headphones(SPEAKERS, states.append, on_back=lambda: ctl.submit(ctl._refresh_devices))
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
        ctl.sides, ctl.on = [SimpleSide(cap, hp)], True
        phases = ctl.submit(ctl._refresh_devices).result()
        x, sr = recorded(rec, tmp_path / 'heard.npy')
    finally:
        ctl.sides, ctl.on = [], False
        cap.stop()
        q.put(None)
        t_play.join(5)
        hp.stop()
        call.kill()
        if before:
            set_default_output(before)
    gap = longest_quiet_s(x, sr)
    print(f'devices refresh: {phases} ms, longest gap heard {gap * 1000:.0f} ms; {states[-1:]}')
    assert rms(x) > 0.05, 'the tone did not come through'
    assert gap <= 0.5, f'{gap:.2f} s of silence'


class SimpleSide:
    def __init__(self, inp, out):
        self.inp, self.out, self.name = inp, out, 'them'


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
@pytest.mark.skipif(not MODELS or TCC != 'granted', reason='needs the models and System Audio Recording allowed')
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

"""macOS "me -> them" devices through the Core Audio helper (native/macos/audio): BlackHole 2ch as the virtual mic,
the microphone by UID that follows a device going away and coming back, the loop "microphone -> BlackHole 2ch" while
translation is off, and what each costs in CPU.

On the CI runner (macos-15, BlackHole 2ch and 16ch installed, the helper built) BlackHole 16ch stands in for the real
microphone: a test process plays into it. A device that goes and comes back is an aggregate device made by the helper.
Elsewhere skipped, except the checks that need no Mac.
"""
import os
import platform
import subprocess
import sys
import threading
import time
import types

import numpy as np
import psutil
import pytest

from peresmeshnik.audio import macos

MAC = sys.platform == 'darwin'
SR_IN, SR_OUT = 24_000, 16_000  # TTS rate in, call-app rate out
MIC16 = 'BlackHole16ch_UID'  # the stand-in microphone


def _ready() -> str:
    if not MAC:
        return 'macOS only'
    try:
        macos.blackhole()
    except Exception as e:  # not built, or no BlackHole: on CI that is a failure, on a dev Mac a skip
        if os.environ.get('CI'):
            raise
        return str(e)
    return ''


mac = pytest.mark.skipif(bool(_ready()), reason=_ready() if MAC else 'macOS only')


def test_blackhole_missing_says_how_to_install(monkeypatch):
    with pytest.raises(macos.BlackHoleMissing, match='brew install --cask blackhole-2ch'):
        macos.blackhole([{'uid': 'BuiltInSpeakerDevice', 'name': 'MacBook Pro Speakers', 'inputs': 0, 'outputs': 2}])
    assert macos.blackhole([{'uid': 'BlackHole2ch_UID', 'name': 'BlackHole 2ch', 'inputs': 2, 'outputs': 2}])
    with pytest.raises(ValueError, match='not your microphone'):
        macos.not_blackhole('BlackHole 2ch')
    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', 'BlackHole16ch_UID')  # as tests/conftest.py on macOS
    assert macos.not_blackhole('BlackHole16ch_UID') == 'BlackHole16ch_UID'  # the tests' stand-in


def test_the_call_device_and_virtual_inputs_are_no_microphone(monkeypatch):
    """BlackHole 16ch is the call app's speaker below 14.4: as the microphone me -> them would translate the other side
    back into the call. Virtual inputs (Zoom's, Teams') and aggregates are not offered as the microphone either."""
    from peresmeshnik.ui import selftest
    monkeypatch.delenv('PERESMESHNIK_TEST_MIC', raising=False)
    for key in macos.CALL:
        with pytest.raises(ValueError, match="carries the call's sound"):
            macos.not_blackhole(key)
    assert macos.not_blackhole('BlackHole 16ch', what='headphones') == 'BlackHole 16ch'
    dev = {'inputs': 1, 'outputs': 0, 'rate': 48000.0, 'default_input': False, 'default_output': False}
    devs = [{**dev, 'uid': 'BuiltInMicrophoneDevice', 'name': 'MacBook Pro Microphone', 'transport': 'bltn'},
            {**dev, 'uid': 'ZoomAudioDevice', 'name': 'ZoomAudioDevice', 'transport': 'virt'},
            {**dev, 'uid': 'BlackHole16ch_UID', 'name': 'BlackHole 16ch', 'transport': 'virt', 'outputs': 16},
            {**dev, 'uid': 'peresmeshnik-test-mic', 'name': 'Test Mic', 'transport': 'grup'},
            {**dev, 'uid': 'KrispMicrophone', 'name': 'krisp microphone', 'transport': 'virt'}]  # the real one, processed
    monkeypatch.setattr(selftest, 'HELPER', True)
    monkeypatch.setattr(selftest, 'WINDOWS', False)
    monkeypatch.setattr(sys, 'platform', 'darwin')  # mic.native() picks audio/macos.py (on the Windows runner too)
    monkeypatch.setattr(macos, 'devices', lambda: devs)
    # saved earlier (or by hand) as the microphone: refused when it opens, with the reason
    with pytest.raises(ValueError, match='ZoomAudioDevice is a virtual device, not your microphone'):
        macos.not_blackhole('ZoomAudioDevice')
    with pytest.raises(ValueError, match='Test Mic is an aggregate device, not your microphone'):
        macos.not_blackhole('Test Mic')
    assert macos.not_blackhole('BuiltInMicrophoneDevice') == 'BuiltInMicrophoneDevice'
    assert macos.not_blackhole('KrispMicrophone') == 'KrispMicrophone'
    assert macos.not_blackhole('') == ''  # the default input: the helper never takes a virtual one by itself
    assert selftest.devices('sources') == [('BuiltInMicrophoneDevice', 'MacBook Pro Microphone'),
                                           ('KrispMicrophone', 'krisp microphone')]
    assert ('BlackHole 16ch', 'BlackHole 16ch') not in selftest.devices('sinks')  # the call app's speaker: nobody hears it
    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', 'BlackHole16ch_UID')
    assert ('BlackHole16ch_UID', 'BlackHole 16ch') in selftest.devices('sources')
    monkeypatch.delenv('PYTEST_CURRENT_TEST')  # the app itself: the tests' hook lets nothing through
    assert ('BlackHole16ch_UID', 'BlackHole 16ch') not in selftest.devices('sources')


def cpu(p: subprocess.Popen) -> float:
    t = psutil.Process(p.pid).cpu_times()
    return t.user + t.system


def tone(seconds: float, sr: int, f: float = 440.0, level: float = 0.3) -> np.ndarray:
    return (level * np.sin(2 * np.pi * f * np.arange(int(sr * seconds)) / sr)).astype(np.float32)


class Feed:
    """A test process plays a tone into a device, as long as the test needs (the stand-in microphone)."""

    def __init__(self, dev: str = MIC16, f: float = 440.0):
        self.p = macos.spawn('play', '--to', dev, '--rate', str(SR_OUT), stdout=subprocess.DEVNULL)
        self.stop_ = threading.Event()
        chunk = tone(0.1, SR_OUT, f).tobytes()

        def run():
            t0 = time.monotonic()
            for i in range(10 ** 6):
                if self.stop_.wait(max(0.0, t0 + i * 0.1 - 0.2 - time.monotonic())):
                    break
                try:
                    self.p.stdin.write(chunk)
                    self.p.stdin.flush()
                except (BrokenPipeError, ValueError):
                    break
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def stop(self):
        self.stop_.set()
        self.t.join(5)
        self.p.terminate()  # not kill: see virtual_mic_macos.VirtualMic._stop_proc
        self.p.wait(5)


def lines(p: subprocess.Popen, out: list):
    def run():
        for b in p.stderr:
            out.append(b.decode(errors='replace').strip())
    threading.Thread(target=run, daemon=True).start()
    return out


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.05)
    return True


@mac
def test_devices():
    devs = macos.devices()
    print('\ndevices:', [(d['name'], d['transport'], d['inputs'], d['outputs'], d['rate']) for d in devs])
    bh = macos.blackhole(devs)
    assert bh['transport'] == 'virt' and bh['inputs'] == bh['outputs'] == 2
    from peresmeshnik.ui.selftest import devices
    assert not any(macos.is_blackhole(uid) for uid, _ in devices('sources') + devices('sinks'))
    print('microphone permission:', macos.permission())


@mac
def test_round_trip():
    """The translation into BlackHole 2ch, recorded from its input like a call app: tone, level, length, real time."""
    from peresmeshnik.audio.virtual_mic_macos import VirtualMic
    from peresmeshnik.ui.selftest import Recorder
    vm = VirtualMic()
    rec = Recorder(macos.BLACKHOLE_UID, SR_OUT)
    try:
        time.sleep(0.3)
        t = np.arange(int(SR_IN * 3)) / SR_IN
        x = (0.3 * np.sin(2 * np.pi * 440 * t) * (t < 1.5)).astype(np.float32)  # 1.5 s tone, 1.5 s silence
        t0 = time.perf_counter()
        for i in range(0, len(x), SR_IN // 5):  # 200 ms chunks, like TTS
            vm.play(x[i:i + SR_IN // 5], SR_IN)
        helper_cpu = cpu(vm._proc)
        vm.drain()
        wall = time.perf_counter() - t0
        time.sleep(0.3)
        y = rec.stop()
    finally:
        vm.remove()
    on = np.nonzero(np.abs(y) > 0.1)[0]
    assert len(on), 'nothing came out of BlackHole 2ch'
    seg = y[on[0]:on[0] + SR_OUT]
    print(f'\nround trip: {len(y) / SR_OUT:.2f} s recorded, tone {(on[-1] - on[0]) / SR_OUT:.3f} s, '
          f'rms {np.sqrt(np.mean(seg ** 2)):.3f}, played in {wall:.2f} s, helper CPU {helper_cpu:.3f} s')
    assert np.argmax(np.abs(np.fft.rfft(seg))) * SR_OUT / len(seg) == pytest.approx(440, abs=3)
    assert (on[-1] - on[0]) / SR_OUT == pytest.approx(1.5, abs=0.05)  # whole tone, nothing dropped
    assert np.sqrt(np.mean(seg ** 2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.1)  # level kept
    assert 2.9 < wall < 3.6  # played in real time


@mac
def test_flush_leaves_silence():
    """Barge-in and the mode switch drop queued speech (flush: the player is killed) while the call app keeps reading
    BlackHole 2ch: from then on it must hear silence, not the last of the speech again."""
    from peresmeshnik.audio.virtual_mic_macos import VirtualMic
    vm = VirtualMic()
    rec = Timed(macos.BLACKHOLE_UID)
    try:
        time.sleep(0.4)
        vm.play(tone(3.0, SR_IN), SR_IN)
        time.sleep(1.0)
        t = time.monotonic()
        vm.flush()
        time.sleep(2.5)
        y = rec.stop()
    finally:
        vm.remove()
    cut = next(n for at, n in rec.stamps if at >= t + 0.15)  # what arrived 150 ms after the flush, and later
    before, after = y[:cut - SR_OUT // 5], y[cut:]
    print(f'\nflush: rms before {np.sqrt(np.mean(before[-SR_OUT // 2:] ** 2)):.3f}, after {np.sqrt(np.mean(after ** 2)):.4f} '
          f'(max {np.abs(after).max():.4f}) over {len(after) / SR_OUT:.1f} s')
    assert np.sqrt(np.mean(before[-SR_OUT // 2:] ** 2)) > 0.15  # it was playing
    assert np.abs(after).max() < 0.01


class Timed:
    """The helper's recording of a device, each chunk stamped when it arrives: when a sample was there (at the latest)."""

    def __init__(self, dev: str, sr: int = SR_OUT):
        self.sr, self.chunks, self.stamps = sr, [], []  # stamps: (arrival, samples so far)
        self.p = macos.spawn('record', '--from', dev, '--rate', str(sr))

        def run():
            n = 0
            for b in iter(lambda: self.p.stdout.read(4 * sr // 100), b''):
                self.chunks.append(np.frombuffer(b[:len(b) // 4 * 4], '<f4'))
                n += len(self.chunks[-1])
                self.stamps.append((time.monotonic(), n))
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def at(self, i: int) -> float:
        return next(t for t, n in self.stamps if n > i)

    def stop(self) -> np.ndarray:
        self.p.stdin.close()
        self.p.wait(5)
        self.t.join(5)
        return np.concatenate(self.chunks)


def onsets(y: np.ndarray) -> list[int]:
    on = np.nonzero(np.abs(y) > 0.25)[0]
    return [i for k, i in enumerate(on) if k == 0 or i - on[k - 1] > SR_OUT // 10]


@mac
def test_latency():
    """play() -> the sound at BlackHole 2ch's input as a call app reads it, 5 clicks from silence (the first one opens
    the playback process), each once the one before has played out. Not at a fixed pace: a click is 0.52 s of sound,
    so with a click every 0.6 s a slow start of the process stayed queued ahead of the next ones (a runner: 545, 377,
    260, 160, 31 ms). Upper bound: includes the reader's own 10 ms chunks."""
    from peresmeshnik.audio.virtual_mic_macos import VirtualMic
    vm = VirtualMic()
    rec = Timed(macos.BLACKHOLE_UID)
    sent = []
    try:
        time.sleep(0.5)
        for k in range(5):
            sent.append(time.monotonic())
            vm.play(np.concatenate([np.full(SR_IN // 50, 0.5, np.float32), np.zeros(SR_IN // 2, np.float32)]), SR_IN)
            wait_for(lambda k=k: len(onsets(np.concatenate([np.zeros(0, np.float32), *rec.chunks[:]]))) > k)
            time.sleep(0.6)  # its 0.5 s of silence plays out: the next click finds the player idle
        y = rec.stop()
    finally:
        vm.remove()
    starts = onsets(y)
    lags = [rec.at(i) - t for i, t in zip(starts, sent)]
    print('\nplay -> BlackHole 2ch input, ms:', [round(x * 1000) for x in lags])
    assert len(starts) == 5
    assert max(lags[1:]) < 0.15


@mac
@pytest.mark.parametrize('what', ['play', 'loop'])
def test_blackhole_changes_its_rate_the_tone_keeps_its_pitch(what):
    """A call app or Audio MIDI Setup moves BlackHole 2ch to another rate while the translation plays into it (play) or
    the microphone is looped into it (loop): the call app still hears 440 Hz, not the old rate's samples at the new
    one. play ends with 7 and VirtualMic opens the next one at the new rate; loop restarts its output itself."""
    from test_macos_audio import peak_hz, set_rate

    from peresmeshnik.audio import coreaudio as ca
    from peresmeshnik.audio.virtual_mic_macos import VirtualMic
    from peresmeshnik.ui.selftest import Recorder
    old = ca.rate(macos.BLACKHOLE)
    new = 44100 if old != 44100 else 48000
    rec = Recorder(macos.BLACKHOLE_UID, SR_OUT)
    stop, procs, vm = threading.Event(), [], None
    if what == 'play':
        vm = VirtualMic()

        def speak():  # the translation's voice, chunk by chunk, in real time
            t0 = time.monotonic()
            for i in range(10 ** 6):
                if stop.wait(max(0.0, t0 + i * 0.1 - time.monotonic())):
                    return
                vm.play(tone(0.1, SR_IN)[:], SR_IN)
        threading.Thread(target=speak, daemon=True).start()
    else:
        procs.append(Feed(MIC16))
        procs.append(macos.spawn('loop', '--from', MIC16, '--to', macos.BLACKHOLE_UID, '--always', stdout=subprocess.DEVNULL))
    try:
        time.sleep(1.5)
        n0 = sum(len(c) for c in rec.chunks)
        set_rate(macos.BLACKHOLE, new)
        time.sleep(3.0)
        y = rec.stop()
    finally:
        stop.set()
        rec.p.poll() is None and rec.stop()
        if vm:
            vm.remove()
        for p in procs:
            p.stop() if isinstance(p, Feed) else (p.stdin.close(), p.wait(5))
        set_rate(macos.BLACKHOLE, old)
    before, after = y[SR_OUT // 2:n0], y[n0 + SR_OUT:]  # from 1 s after the change: the restart takes ~0.3 s
    print(f'\n{what}: BlackHole 2ch {old:.0f} -> {new} Hz: before {peak_hz(before, SR_OUT):.0f} Hz, after '
          f'{peak_hz(after, SR_OUT):.0f} Hz rms {np.sqrt(np.mean(after ** 2)):.3f} ({len(after) / SR_OUT:.1f} s)')
    assert abs(peak_hz(before, SR_OUT) - 440) <= 3
    assert len(after) > SR_OUT and np.sqrt(np.mean(after ** 2)) > 0.1 and abs(peak_hz(after, SR_OUT) - 440) <= 3


@mac
def test_mic_refuses_blackhole():
    from peresmeshnik.audio.mic import Mic
    with pytest.raises(ValueError, match='not your microphone'):
        Mic('BlackHole 2ch').start(lambda pcm: None)


@mac
def test_mic_follows_a_device_that_goes_and_comes_back(monkeypatch):
    """The saved microphone (an aggregate device over BlackHole 16ch) goes away: the helper takes the default input
    and comes back to the saved one when it returns; frames flow from it again."""
    from peresmeshnik.audio.mic import Mic
    uid = 'peresmeshnik-test-mic'

    def plug():
        p = macos.spawn('aggregate', '--uid', uid, '--name', 'Test Mic', '--of', MIC16)
        assert p.stdout.readline().startswith(b'created'), p.stderr.read()
        return p

    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', uid)  # an aggregate: refused as the microphone but for the tests' hook
    feed, agg = Feed(), plug()
    frames, events = [], []
    m = Mic(uid, follow=True, on_device=lambda kind, name: events.append((kind, name)), on_end=lambda why: events.append(('end', why)))
    try:
        m.start(frames.append)
        assert wait_for(lambda: ('using', 'Test Mic') in events), events
        assert wait_for(lambda: len(frames) > 25)
        assert np.abs(np.concatenate(frames[-10:])).max() > 0.2, 'no tone from the saved microphone'
        agg.stdin.close()  # unplugged
        agg.wait(5)
        assert wait_for(lambda: ('lost', 'Test Mic') in events), events
        assert wait_for(lambda: events[-1][0] in ('using', 'none') and events[-1][1] != 'Test Mic'), events
        print('\nwithout the saved microphone:', events[-1])
        agg = plug()  # plugged back
        assert wait_for(lambda: events[-1] == ('using', 'Test Mic')), events
        n = len(frames)
        assert wait_for(lambda: len(frames) > n + 25)
        assert np.abs(np.concatenate(frames[-10:])).max() > 0.2, 'no tone after it came back'
        assert m.source == 'Test Mic'
        assert not any(k == 'end' for k, _ in events)
        print('events:', events)
    finally:
        m.stop()
        agg.stdin.close()
        agg.wait(5)
        feed.stop()


def test_a_playback_that_ended_by_itself_reopens_or_says_why():
    """The helper ends by itself: 7 (BlackHole 2ch changed its rate) - the next chunk opens a new one at the new rate;
    anything else (BlackHole 2ch went away: 4) - play() says why, the translation does not go silent unseen."""
    from peresmeshnik.audio.virtual_mic_macos import VirtualMic

    def ended(code, why=''):
        p = subprocess.Popen([sys.executable, '-c', f'import sys; print({why!r}, file=sys.stderr); sys.exit({code})'],
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        return p
    opened = []

    class Stub(VirtualMic):
        def _make(self):
            return []

        def _open(self, sample_rate):
            opened.append(sample_rate)
            self._proc = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read()'],
                                          stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self._sr = sample_rate
    vm = Stub()
    x = np.zeros(2400, np.float32)
    try:
        vm.play(x, 24000)
        vm._stop_proc(vm._proc)
        vm._proc = ended(7, 'peresmeshnik-audio: the rate of BlackHole 2ch changed')
        vm.play(x, 24000)
        assert len(opened) == 2 and vm._proc.poll() is None  # opened again, nothing said
        vm._stop_proc(vm._proc)
        vm._proc = ended(4, 'peresmeshnik-audio: BlackHole 2ch went away')
        with pytest.raises(RuntimeError, match='BlackHole 2ch: the playback stopped .*BlackHole 2ch went away'):
            vm.play(x, 24000)
        vm.play(x, 24000)  # the next try opens a new one
        assert len(opened) == 3
    finally:
        vm.remove()


def test_live_voice_with_a_dead_output_says_so(monkeypatch):
    """Live voice (no translation): the output dies (BlackHole 2ch removed: VirtualMic raises). The tray hears it once,
    the frames still leave the queue, nothing plays into the dead output again."""
    from peresmeshnik import app, config
    s = config.Settings()
    s.them_to_me = 'local'
    errors, plays = [], []

    class Direction:  # its events last until it is closed, as a real one's (ended by themselves: it died)
        def __init__(self):
            self.closed = threading.Event()

        def feed(self, pcm):
            pass

        def events(self):
            self.closed.wait()
            yield from ()

        def close(self):
            self.closed.set()

    class DeadOut:
        def play(self, pcm, sr):
            plays.append(len(pcm))
            raise RuntimeError('BlackHole 2ch: the playback stopped (peresmeshnik-audio: BlackHole 2ch went away)')

        def flush(self):
            pass

        def stop(self):
            pass

    class In:
        def start(self, cb):
            self.cb = cb

        def stop(self):
            pass
    inp = In()
    monkeypatch.setattr(app.Side, '_local', lambda self, s: Direction())
    side = app.Side('them', s, inp, DeadOut(), type('O', (), {'show': lambda self, c: None})(), errors.append)
    side.start()
    side.set_live(True)
    for _ in range(5):
        inp.cb(np.zeros(320, np.float32))
    assert wait_for(lambda: side._live_q.empty() and errors)
    side.stop()
    assert errors == ['error: them → me: BlackHole 2ch: the playback stopped (peresmeshnik-audio: BlackHole 2ch went away)']
    assert plays == [320]


def test_live_voice_plays_again_when_the_output_is_back(monkeypatch):
    """The output fails (BlackHole 2ch gone), then is back: the tray heard it once; the live voice plays again by
    itself (a new stream every RETRY_S on the frames that come), without a second message for a stream that fails
    right after it opened."""
    from peresmeshnik import app, config
    s = config.Settings()
    s.them_to_me = 'local'
    errors, plays, fails = [], [], [3]  # the output fails its first three tries

    class Direction:  # its events last until it is closed, as a real one's (ended by themselves: it died)
        def __init__(self):
            self.closed = threading.Event()

        def feed(self, pcm):
            pass

        def events(self):
            self.closed.wait()
            yield from ()

        def close(self):
            self.closed.set()

    class FlakyOut:
        def play(self, pcm, sr):
            if fails[0]:
                fails[0] -= 1
                raise RuntimeError('BlackHole 2ch: the playback stopped (exit 4)')
            plays.append(len(pcm))

        def flush(self):
            pass

        def stop(self):
            pass

    class In:
        def start(self, cb):
            self.cb = cb

        def stop(self):
            pass
    inp = In()
    monkeypatch.setattr(app.Side, 'RETRY_S', 0.05)
    monkeypatch.setattr(app.Side, '_local', lambda self, s: Direction())
    side = app.Side('them', s, inp, FlakyOut(), type('O', (), {'show': lambda self, c: None})(), errors.append)
    side.start()
    side.set_live(True)
    for _ in range(40):  # 0.4 s of 10 ms frames
        inp.cb(np.zeros(160, np.float32))
        time.sleep(0.01)
    assert wait_for(lambda: side._live_q.empty())
    side.stop()
    assert errors == ['error: them → me: BlackHole 2ch: the playback stopped (exit 4)']  # once for the whole outage
    assert fails == [0] and len(plays) > 10  # it plays again


def test_followed_says_which_microphone_plays_meanwhile():
    """The saved microphone goes: the helper takes another one at once; the tray says which, then that it is back."""
    from peresmeshnik.audio.mic import followed
    states = []
    m = followed('USBMic_UID', states.append)

    def event(kind, uid, name):
        if kind == 'using':
            m.uid = uid  # as Mic's reader sets it before on_device
        m.on_device(kind, name)
    event('using', 'USBMic_UID', 'USB Mic')  # the start: nothing to say
    event('lost', 'USBMic_UID', 'USB Mic')
    event('using', 'BuiltInMicrophoneDevice', 'MacBook Pro Microphone')
    event('using', 'USBMic_UID', 'USB Mic')
    event('lost', 'USBMic_UID', 'USB Mic')
    event('none', '', '')
    event('using', 'USBMic_UID', 'USB Mic')
    assert states == ['translating (USB Mic went away: MacBook Pro Microphone meanwhile)',
                      'translating (the microphone is back: USB Mic)',
                      'error: no microphone: plug one in or choose it in the settings',
                      'translating (the microphone is back: USB Mic)']
    states.clear()
    m = followed('', states.append)  # none saved: the default input; it goes, another one plays meanwhile
    event('using', 'USBMic_UID', 'USB Mic')
    event('lost', 'USBMic_UID', 'USB Mic')
    event('using', 'BuiltInMicrophoneDevice', 'MacBook Pro Microphone')
    event('using', 'USBMic_UID', 'USB Mic')
    assert states == ['translating (USB Mic went away: MacBook Pro Microphone meanwhile)',
                      'translating (the microphone is back: USB Mic)']


@mac
def test_mic_cpu():
    """CPU of the helper recording a 48 kHz device at 16 kHz (the translation's microphone), per second of audio."""
    from peresmeshnik.audio.mic import Mic
    frames = []
    m = Mic(MIC16)
    m.start(frames.append)
    try:
        time.sleep(1)
        c0, t0, n0 = cpu(m._proc), time.monotonic(), len(frames)
        time.sleep(10)
        c, dt, n = cpu(m._proc) - c0, time.monotonic() - t0, len(frames) - n0
    finally:
        m.stop()
    print(f'\nmicrophone: {n * 0.02:.2f} s of frames in {dt:.2f} s, helper {c / dt:.4f} CPU s per s')
    assert n * 0.02 == pytest.approx(dt, rel=0.05)  # 20 ms frames in real time
    assert c / dt < 0.05


@mac
def test_loop_only_while_a_call_records():
    """Translation off: the loop "microphone -> BlackHole 2ch" opens the microphone only while another process (the
    call app) records BlackHole 2ch; idle it costs no CPU and holds no microphone."""
    from peresmeshnik.ui.selftest import Recorder, chirp, find
    loop = macos.spawn('loop', '--from', MIC16, '--to', macos.BLACKHOLE_UID, stdout=subprocess.DEVNULL)
    ev = lines(loop, [])
    try:
        assert wait_for(lambda: 'idle' in ev), ev
        c0 = cpu(loop)
        time.sleep(10)
        idle = cpu(loop) - c0
        call = Recorder(macos.BLACKHOLE_UID, SR_OUT)  # the call app takes BlackHole 2ch as its microphone
        assert wait_for(lambda: 'call' in ev and any(e.startswith('using BlackHole16ch_UID') for e in ev)), ev
        time.sleep(0.5)
        feed = macos.spawn('play', '--to', MIC16, '--rate', str(SR_OUT), stdout=subprocess.DEVNULL)
        ref = chirp(SR_OUT)
        feed.communicate(np.concatenate([np.zeros(SR_OUT // 2, np.float32), ref]).tobytes(), timeout=10)
        c1 = cpu(loop)
        time.sleep(10)
        active = cpu(loop) - c1
        y = call.stop()
        assert wait_for(lambda: ev[-1] == 'idle'), ev  # the call app let go: the microphone is closed again
    finally:
        loop.stdin.close()
        loop.wait(5)
    score, at = find(y, ref)
    print(f'\nloop: idle {idle / 10:.4f} CPU s per s, while a call records {active / 10:.4f}; chirp match {score:.2f}; events {ev}')
    assert score > 0.5
    assert idle / 10 < 0.002
    assert loop.returncode == 0


@mac
def test_loop_memory_stays_flat_over_calls():
    """S2-09: the loop lives for days with translation off and opens the microphone and BlackHole 2ch on every call;
    what a call allocates (rings, mono buffers, the pump's 1 MiB) goes with it. 30 calls (another process records
    BlackHole 2ch, then stops): the helper's memory grows by less than 3 MiB from the 5th call on."""
    loop = macos.spawn('loop', '--from', MIC16, '--to', macos.BLACKHOLE_UID, stdout=subprocess.DEVNULL)
    ev = lines(loop, [])
    rss = []
    try:
        assert wait_for(lambda: 'idle' in ev), ev
        for i in range(30):
            n = ev.count('call')
            call = macos.spawn('record', '--from', macos.BLACKHOLE_UID, '--rate', str(SR_OUT), stdout=subprocess.DEVNULL)
            assert wait_for(lambda n=n: ev.count('call') > n, 10), ev[-5:]
            time.sleep(0.3)
            n = ev.count('idle')
            call.stdin.close()
            call.wait(5)
            assert wait_for(lambda n=n: ev.count('idle') > n, 10), ev[-5:]
            rss.append(psutil.Process(loop.pid).memory_info().rss)
    finally:
        loop.stdin.close()
        loop.wait(5)
    grown = (rss[-1] - rss[4]) / 2 ** 20
    print(f'\nloop RSS over 30 calls (MiB): {[round(r / 2 ** 20, 1) for r in rss[::5]]}; from the 5th call {grown:+.2f}')
    assert grown < 3, f'the loop grew {grown:.1f} MiB over 25 calls'


def restart_coreaudiod() -> None:
    subprocess.run(['sudo', 'launchctl', 'kickstart', '-kp', 'system/com.apple.audio.coreaudiod'], check=True, timeout=60)
    for _ in range(30):  # BlackHole back for the next test
        if subprocess.run(['system_profiler', 'SPAudioDataType'], capture_output=True, text=True).stdout.count('BlackHole') >= 2:
            break
        time.sleep(2)


@mac
@pytest.mark.skipif(os.environ.get('PERESMESHNIK_COREAUDIOD_RESTART') != '1',
                    reason='restarts coreaudiod: its own CI step (PERESMESHNIK_COREAUDIOD_RESTART=1)')
@pytest.mark.parametrize('what', ['play', 'loop', 'record'])
def test_a_coreaudiod_restart_ends_the_helper(what):
    """S2-04: after `sudo killall coreaudiod` our IOProcs never run again: play would wait for an IO cycle forever
    (the translation silent, its queue growing), the loop would stay silent. play and loop exit 4 and the app starts
    them again; record (the microphone, --follow as the app runs it) opens its microphone anew and goes on."""
    args = {'play': ['play', '--to', macos.BLACKHOLE_UID, '--rate', str(SR_IN)],
            'loop': ['loop', '--from', MIC16, '--to', macos.BLACKHOLE_UID, '--always'],
            'record': ['record', '--from', MIC16, '--rate', str(SR_OUT), '--follow']}[what]
    p = macos.spawn(*args, stdout=subprocess.DEVNULL)
    ev = lines(p, [])
    try:
        time.sleep(2.0)
        assert p.poll() is None, ev
        n = sum(e.startswith('using ') for e in ev)
        restart_coreaudiod()
        if what == 'record':
            assert wait_for(lambda: sum(e.startswith('using ') for e in ev) > n, 20), f'the microphone was not opened again: {ev}'
            time.sleep(2.0)
            print(f'\nrecord after a coreaudiod restart: alive {p.poll() is None}; {ev[-3:]}')
            assert p.poll() is None, ev
            return
        try:
            code = p.wait(15)
        except subprocess.TimeoutExpired:
            pytest.fail(f'{what} still runs 15 s after coreaudiod restarted: {ev[-3:]}')
        print(f'\n{what} after a coreaudiod restart: exit {code}; {ev[-2:]}')
        assert code == 4, (code, ev)
    finally:
        if p.poll() is None:
            p.kill()


@mac
def test_self_test():
    from peresmeshnik.ui import selftest
    rs = selftest.run_all(mic=MIC16)
    print('\n' + '\n'.join(f'{r.ok} {r.name}: {r.detail}' for r in rs))
    assert all(r.ok for r in rs[:1])  # the virtual mic; the stand-in microphone is silent


class Fake:
    """"Translates" each second of input into a second of a 220 Hz tone (whole periods: seconds join cleanly)."""

    def __init__(self, speaks):
        import queue
        self.q, self.n, self.speaks = queue.Queue(), 0, speaks

    def feed(self, pcm):
        from peresmeshnik.engines import Speech
        self.n += len(pcm)
        if self.speaks and self.n >= 16000:
            self.n -= 16000
            self.q.put(Speech(tone(1.0, 24000, 220.0), 24000))

    def events(self):
        while (e := self.q.get()) is not None:
            yield e

    def close(self):
        self.q.put(None)


def after_sound() -> str:
    """What is left playing: helper processes anywhere, and what both BlackHoles carry for 0.5 s (rms, main frequency)."""
    from peresmeshnik.ui.selftest import Recorder
    procs = subprocess.run(['pgrep', '-fl', 'peresmeshnik-audio'], capture_output=True, text=True).stdout.strip().splitlines()
    out = [f'{len(procs)} helpers {procs}']
    for dev in (macos.BLACKHOLE_UID, MIC16):
        r = Recorder(dev, SR_OUT)
        time.sleep(0.6)
        y = r.stop()[SR_OUT // 10:]
        f = np.argmax(np.abs(np.fft.rfft(y))) * SR_OUT / max(len(y), 1) if len(y) else 0
        out.append(f'{dev}: rms {np.sqrt(np.mean(y ** 2)) if len(y) else 0:.3f} at {f:.0f} Hz')
    return '; '.join(out)


class Overlay:
    def side(self, name):
        return self

    def show(self, caption):
        pass

    def hide(self):
        pass


@mac
@pytest.mark.parametrize('fade_ms', [30.0, 0.0])
def test_app_me_side(monkeypatch, fade_ms):
    """The app's me -> them on macOS with a fake translation (220 Hz) and BlackHole 16ch as the microphone (300 Hz):
    what the call app hears from BlackHole 2ch: translation, live voice, translation, then translation off (the loop
    passes the microphone); no jump at the seams above the tones' own with the crossfade (fade 0: measured)."""
    from peresmeshnik import app, config
    from peresmeshnik.audio import mic as mic_mod
    from peresmeshnik.ui.selftest import Recorder
    s = config.Settings()
    s.me_to_them, s.ui.mic = 'local', MIC16
    monkeypatch.setattr(app.Side, '_local', lambda self, s: Fake(self.name == 'me'))  # no models; the rest is real
    init = mic_mod.LiveSwitch.__init__
    monkeypatch.setattr(mic_mod.LiveSwitch, '__init__', lambda self, inner: init(self, inner, fade_ms=fade_ms))
    states = []
    ctl = app.Controller(Overlay(), states.append, settings=s)
    feed = Feed(MIC16, f=300.0)
    rec = None
    try:
        ctl.startup()
        assert ctl._loop is not None and ctl._loop.poll() is None, (ctl.startup_error, states)
        ctl.set_translation(True).result()
        assert ctl._loop is None and any(x.name == 'me' for x in ctl.sides), states  # the translation speaks instead
        rec = Recorder(macos.BLACKHOLE_UID, 48_000)
        time.sleep(2.5)  # a second of input, then the translation speaks
        ctl.set_live(True)
        time.sleep(1.5)
        ctl.set_live(False)
        time.sleep(2.5)
        n_off = sum(len(c) for c in rec.chunks)  # the hotkey's part ends here: translation off is a plain stop
        ctl.set_translation(False).result()
        assert ctl._loop is not None and ctl._loop.poll() is None
        time.sleep(1.5)  # the loop sees the call app recording and passes the microphone
        x = rec.stop()
        rec = None
    finally:
        if rec:
            rec.stop()
        ctl.quit()
        feed.stop()
    left = [p.name() for p in psutil.Process().children(recursive=True) if 'peresmeshnik-audio' in ' '.join(p.cmdline())]
    print('\nstates:', states)
    print('after:', after_sound())
    assert not left, left

    sr, w = 48_000, 960  # 20 ms windows
    f = np.fft.rfftfreq(w, 1 / sr)
    wins = x[:len(x) // w * w].reshape(-1, w)
    rms = np.sqrt(np.mean(wins ** 2, axis=1))
    peak = f[np.argmax(np.abs(np.fft.rfft(wins, axis=1)), axis=1)]
    kind = np.where(rms < 0.02, '.', np.where(np.abs(peak - 220) < 30, 't', np.where(np.abs(peak - 300) < 30, 'L', '?')))
    heard = [k for k in kind if k in 'tL']
    runs = ''.join(k for i, k in enumerate(heard) if i == 0 or k != heard[i - 1])
    print('call hears:', ''.join(kind))
    assert runs == 'tLtL', runs  # translation, live, translation, then off (the loop passes the microphone)
    own = max(0.3 * 2 * np.pi * 220 / sr, 0.3 * 2 * np.pi * 300 / sr)  # the tones' own biggest step
    seams = [i * w for i in range(1, len(kind)) if kind[i - 1] != kind[i] and '?' not in (kind[i - 1], kind[i])]
    jumps = [np.abs(np.diff(x[max(0, s - 2 * w):s + 2 * w])).max() / own for s in seams]
    hotkey = [j for s, j in zip(seams, jumps) if s < n_off]
    print(f'fade {fade_ms:.0f} ms: jump at the hotkey\'s seams {", ".join(f"{j:.1f}" for j in hotkey)} x the tones\' own; '
          f'at translation off {", ".join(f"{j:.1f}" for s, j in zip(seams, jumps) if s >= n_off)}')
    if fade_ms:
        assert max(hotkey) < 3


def test_a_virtual_default_input_is_no_microphone(monkeypatch):
    """The microphone "System default" while macOS' default input is BlackHole 2ch (installing it can do that): said
    with what to do, not recorded as silence (a user's Mac, 0.3.0: the self-test heard -180 dBFS)."""
    from peresmeshnik.ui import selftest
    monkeypatch.delenv('PERESMESHNIK_TEST_MIC', raising=False)
    dev = {'inputs': 1, 'outputs': 0, 'rate': 48000.0, 'default_input': False, 'default_output': False}
    mic = {**dev, 'uid': 'BuiltInMicrophoneDevice', 'name': 'MacBook Pro Microphone', 'transport': 'bltn'}
    for default in ({**dev, 'uid': 'BlackHole2ch_UID', 'name': 'BlackHole 2ch', 'transport': 'virt', 'inputs': 2},
                    {**dev, 'uid': 'BlackHole16ch_UID', 'name': 'BlackHole 16ch', 'transport': 'virt', 'inputs': 16},
                    {**dev, 'uid': 'ZoomAudioDevice', 'name': 'ZoomAudioDevice', 'transport': 'virt'}):
        monkeypatch.setattr(macos, 'devices', lambda d=default: [mic, {**d, 'default_input': True}])
        with pytest.raises(ValueError, match=f"macOS' default input is {default['name']}, not your microphone.*Sound → Input"):
            macos.not_virtual('')
        assert macos.not_virtual('BuiltInMicrophoneDevice') == 'BuiltInMicrophoneDevice'  # chosen: no matter
    monkeypatch.setattr(macos, 'devices', lambda: [{**mic, 'default_input': True}])
    assert macos.not_virtual('') == ''
    monkeypatch.setattr(macos, 'devices', lambda: [{**mic, 'uid': 'KrispMicrophone', 'name': 'krisp microphone',
                                                    'transport': 'virt', 'default_input': True}])
    assert macos.not_virtual('') == ''  # the real microphone, processed
    monkeypatch.setattr(selftest, 'HELPER', True)  # the self-test's microphone goes through the same check
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(macos, 'devices', lambda: [mic, {**mic, 'uid': 'BlackHole2ch_UID', 'name': 'BlackHole 2ch',
                                                         'transport': 'virt', 'default_input': True}])
    with pytest.raises(ValueError, match="default input is BlackHole 2ch"):
        selftest._helper_mic('')


def test_the_self_test_says_exact_zeros_are_no_microphone(monkeypatch):
    """A microphone always hisses a little: only zeros (a silent input, no permission) are said so, not "say something"."""
    from peresmeshnik.ui import selftest

    class Zeros:
        def __init__(self, source, rate=16000):
            self.p = types.SimpleNamespace(returncode=0)

        def stop(self):
            return np.zeros(48000, np.float32)
    monkeypatch.setattr(selftest, 'Recorder', Zeros)
    monkeypatch.setattr(selftest, 'HELPER', True)
    monkeypatch.setattr(selftest, 'WINDOWS', False)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    r = selftest.check_mic('', seconds=0)
    assert not r.ok and r.detail.startswith('No sound at all came from the microphone') and 'lid closed' in r.detail
    assert 'Privacy & Security → Microphone' in r.detail and 'allow Terminal' in r.detail, r.detail
    assert 'say something' not in r.detail
    Zeros.stop = lambda self: np.full(48000, 1e-5, np.float32)  # a quiet room: still "say something"
    r = selftest.check_mic('', seconds=0)
    assert not r.ok and 'say something' in r.detail


def test_the_microphone_loop_refuses_a_virtual_default_and_follows_the_settings(monkeypatch):
    """Translation off, the helper loops the microphone into BlackHole 2ch for the call app. The microphone "System
    default" while macOS' default input is BlackHole 2ch: said, no loop (a user's Mac, 0.3.0: Zoom heard nothing). A
    microphone chosen in the settings afterwards: the loop starts with it; another one: the loop starts again with that;
    the same one: the running loop stays."""
    from peresmeshnik import app, config
    from peresmeshnik.audio import mic
    monkeypatch.delenv('PERESMESHNIK_TEST_MIC', raising=False)
    dev = {'inputs': 1, 'outputs': 0, 'rate': 48000.0, 'default_input': False, 'default_output': False}
    devs = [{**dev, 'uid': 'BuiltInMicrophoneDevice', 'name': 'MacBook Pro Microphone', 'transport': 'bltn'},
            {**dev, 'uid': 'USBHeadsetMic', 'name': 'USB Headset', 'transport': 'usb'},
            {**dev, 'uid': 'BlackHole2ch_UID', 'name': 'BlackHole 2ch', 'transport': 'virt', 'inputs': 2, 'outputs': 2,
             'default_input': True}]
    monkeypatch.setattr(macos, 'devices', lambda: devs)
    spawned, closed = [], []

    class Loop:
        def __init__(self, *args, **k):
            self.args, self.stderr = args, []
            self.stdin = types.SimpleNamespace(close=lambda: closed.append(self.args))
            spawned.append(args)

        def poll(self):
            return None if self.args not in closed else 0

        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(macos, 'spawn', Loop)
    monkeypatch.setattr(mic, 'native', lambda: macos)
    monkeypatch.setattr(app.sys, 'platform', 'darwin')
    states = []
    s = config.Settings()
    monkeypatch.setenv('PERESMESHNIK_CALL_PATH', 'tap')  # macOS 14.4+: no pass-through
    ctl = app.Controller(None, states.append, settings=s)
    ctl._vmic = types.SimpleNamespace(device={'uid': 'BlackHole2ch_UID'}, call_mic={'uid': 'BlackHole2ch_UID'})
    try:
        ctl._loop_helper(True)  # as at the start, translation off
        assert spawned == [] and len(states) == 1, states
        assert "will not hear your microphone while translation is off: macOS' default input is BlackHole 2ch" in states[0]
        assert 'System Settings → Sound → Input' in states[0]
        s.ui.mic = 'BuiltInMicrophoneDevice'  # chosen in the settings (tray.restart: mic_changed)
        ctl.mic_changed().result(10)
        assert [a[:3] for a in spawned] == [('loop', '--from', 'BuiltInMicrophoneDevice')]
        s.ui.mic = 'USBHeadsetMic'
        ctl.mic_changed().result(10)
        assert closed == [spawned[0]] and spawned[1][:3] == ('loop', '--from', 'USBHeadsetMic')
        ctl.mic_changed().result(10)  # saved again, the same microphone: the loop stays
        assert len(spawned) == 2 and len(closed) == 1
        ctl.on = True  # translating: the tray restarts the translation instead, the loop is not the call's now
        s.ui.mic = 'BuiltInMicrophoneDevice'
        ctl.mic_changed().result(10)
        assert len(spawned) == 2
    finally:
        ctl._disarm()
        ctl._pool.shutdown()


def _mac_devices():
    dev = {'inputs': 0, 'outputs': 0, 'rate': 48000.0, 'default_input': False, 'default_output': False}
    return [{**dev, 'uid': 'AirPods_UID', 'name': 'AirPods Pro', 'transport': 'blue', 'inputs': 1, 'outputs': 2},
            {**dev, 'uid': 'BuiltInMicrophoneDevice', 'name': 'MacBook Pro Microphone', 'transport': 'bltn', 'inputs': 1},
            {**dev, 'uid': 'BuiltInSpeakerDevice', 'name': 'MacBook Pro Speakers', 'transport': 'bltn', 'outputs': 2},
            {**dev, 'uid': 'BlackHole2ch_UID', 'name': 'BlackHole 2ch', 'transport': 'virt', 'inputs': 2, 'outputs': 2},
            {**dev, 'uid': 'BlackHole16ch_UID', 'name': 'BlackHole 16ch', 'transport': 'virt', 'inputs': 16, 'outputs': 16},
            {**dev, 'uid': 'ZoomAudioDevice', 'name': 'ZoomAudioDevice', 'transport': 'virt', 'inputs': 2, 'outputs': 2},
            {**dev, 'uid': 'MultiOutput', 'name': 'Multi-Output Device', 'transport': 'grup', 'outputs': 2},
            {**dev, 'uid': 'KrispSpeaker', 'name': 'krisp speaker', 'transport': 'virt', 'outputs': 2},
            {**dev, 'uid': 'eqMacDevice', 'name': 'eqMac', 'transport': 'virt', 'outputs': 2},  # plays on to the speakers
            {**dev, 'uid': 'LoopbackAudio', 'name': 'Loopback Audio', 'transport': 'virt', 'inputs': 2, 'outputs': 2},
            {**dev, 'uid': 'TeamsAudio', 'name': 'Microsoft Teams Audio', 'transport': 'virt', 'inputs': 1, 'outputs': 2}]


MIC_I, SPK_I, BH2_I = 1, 2, 3  # indexes in _mac_devices()


def _qt():
    """A QApplication for the device lists (offscreen without a display)."""
    if platform.system() == 'Linux' and not os.environ.get('DISPLAY'):  # not sys.platform: a test may fake it
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QApplication = pytest.importorskip('PySide6.QtWidgets').QApplication  # noqa: N806
    return QApplication.instance() or QApplication([])


def test_the_lists_offer_real_devices_and_system_default_only_when_it_is_real(monkeypatch):
    """Headphones: no BlackHole 2ch / 16ch, no Zoom's or Teams' speaker (nobody hears them; a user picked BlackHole and
    heard nothing); a Multi-Output Device, Krisp's speaker, eqMac and Loopback play on to real ones. "System default" in
    either list only while macOS' default is not one of those; without it the built-in microphone comes first."""
    from peresmeshnik.audio import mic
    from peresmeshnik.ui import selftest
    _qt()
    from peresmeshnik.ui.tray import device_combo
    monkeypatch.delenv('PERESMESHNIK_TEST_MIC', raising=False)
    devs = _mac_devices()
    devs[MIC_I]['default_input'] = devs[SPK_I]['default_output'] = True
    monkeypatch.setattr(macos, 'devices', lambda: devs)
    monkeypatch.setattr(mic, 'native', lambda: macos)
    monkeypatch.setattr(selftest, 'HELPER', True)
    monkeypatch.setattr(selftest, 'WINDOWS', False)
    sinks, ok = selftest.listing('sinks')
    assert ok and [n for n, _ in sinks] == ['MacBook Pro Speakers', 'AirPods Pro', 'Multi-Output Device', 'krisp speaker',
                                            'eqMac', 'Loopback Audio']
    sources, ok = selftest.listing('sources')
    assert ok and sources == [('BuiltInMicrophoneDevice', 'MacBook Pro Microphone'), ('AirPods_UID', 'AirPods Pro')]
    c = device_combo('sinks', '')
    assert c.itemData(0) == '' and c.currentData() == ''
    for d in devs:
        d['default_input'] = d['default_output'] = d['uid'] == 'BlackHole2ch_UID'
    for kind, first in (('sinks', 'MacBook Pro Speakers'), ('sources', 'BuiltInMicrophoneDevice')):
        assert selftest.listing(kind)[1] is False
        c = device_combo(kind, '')  # saved '': no "System default"; the built-in microphone first, else the first real
        assert '' not in [c.itemData(i) for i in range(c.count())] and c.currentData() == first, (kind, c.currentData())
    for d in devs:  # eqMac as the default output: it plays on, "System default" stays
        d['default_output'] = d['uid'] == 'eqMacDevice'
    assert selftest.listing('sinks')[1] is True


def test_krisp_in_either_case_is_the_microphone():
    from peresmeshnik.audio import processed
    assert processed('krisp microphone') and processed('Krisp Microphone') and processed('Microphone (nvidia broadcast)')
    assert not processed('ZoomAudioDevice')


def _fake_coreaudio(monkeypatch, devs):
    from peresmeshnik.audio import capture_macos, coreaudio
    monkeypatch.setattr(coreaudio, 'devices', lambda: [
        {'id': i, 'name': d['name'], 'uid': d['uid'], 'inputs': d['inputs'], 'outputs': d['outputs'],
         'virtual': d['transport'] in ('virt', 'grup')} for i, d in enumerate(devs)])
    monkeypatch.setattr(coreaudio, 'default', lambda output=True: next(
        (d['name'] for d in devs if d['default_output' if output else 'default_input']), ''))
    monkeypatch.setattr(capture_macos, 'path', lambda: 'tap')


def test_silent_headphones_are_refused_and_a_saved_uid_becomes_its_name(monkeypatch, tmp_path):
    """The saved headphones Zoom's or Teams' speaker, or BlackHole by its UID (0.3.0's lists saved UIDs): said, not
    played into nothing. eqMac, Loopback and a Multi-Output Device play. A saved UID is used by its name and the
    settings keep the name from then on."""
    from peresmeshnik import app, config
    devs = _mac_devices()
    devs[SPK_I]['default_output'] = True
    _fake_coreaudio(monkeypatch, devs)
    monkeypatch.setattr(app.sys, 'platform', 'darwin')
    monkeypatch.setattr(app.crash, 'private', lambda *a: None)
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    for name in ('ZoomAudioDevice', 'Microsoft Teams Audio'):
        with pytest.raises(RuntimeError, match=f'the saved output is {name}, a virtual device: nobody hears'):
            app.Headphones(name).resolve()
    with pytest.raises(RuntimeError, match='the saved output is BlackHole 2ch, a device of the call'):
        app.Headphones('BlackHole2ch_UID').resolve()  # by its UID: found by name, refused
    with monkeypatch.context() as m:  # the runner's stand-in (test_macos_audio.py), only here
        m.setattr(app, 'CALL_OUTPUTS', ('BlackHole 16ch',))
        assert app.Headphones('BlackHole 2ch').resolve() == 'BlackHole 2ch'  # BlackHole is CALL_OUTPUTS' call alone
    for name in ('Multi-Output Device', 'krisp speaker', 'eqMac', 'Loopback Audio', ''):
        assert app.Headphones(name).resolve() == (name or 'MacBook Pro Speakers')
    s = config.Settings()
    s.ui.headphones = 'AirPods_UID'
    config.save(s)
    assert app.Headphones('AirPods_UID').resolve() == 'AirPods Pro'
    assert config.load().ui.headphones == 'AirPods Pro'  # persisted once
    _qt()
    from PySide6.QtWidgets import QComboBox

    from peresmeshnik.audio import mic
    from peresmeshnik.ui import selftest
    from peresmeshnik.ui.tray import fill_combo
    monkeypatch.setattr(macos, 'devices', lambda: devs)
    monkeypatch.setattr(mic, 'native', lambda: macos)
    monkeypatch.setattr(selftest, 'HELPER', True)
    monkeypatch.setattr(selftest, 'WINDOWS', False)
    c = QComboBox()
    fill_combo(c, 'sinks', 'BuiltInSpeakerDevice')  # a UID saved by 0.3.0: shown as its device, no raw entry
    assert c.currentData() == 'MacBook Pro Speakers' and c.findData('BuiltInSpeakerDevice') < 0


def _loop_ctl(monkeypatch, devs):
    """A Controller of the app (macOS) whose helper loops are recorded, not started."""
    from peresmeshnik import app, config
    from peresmeshnik.audio import mic
    monkeypatch.delenv('PERESMESHNIK_TEST_MIC', raising=False)
    monkeypatch.setattr(macos, 'devices', lambda: devs)
    spawned = []

    class Loop:
        def __init__(self, *args, **k):
            self.args, self.stderr, self.closed = args, [], False
            self.stdin = types.SimpleNamespace(close=lambda: setattr(self, 'closed', True))
            spawned.append(args)

        def poll(self):
            return 0 if self.closed else None

        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(macos, 'spawn', Loop)
    monkeypatch.setattr(mic, 'native', lambda: macos)
    monkeypatch.setattr(app.sys, 'platform', 'darwin')
    monkeypatch.setattr(app, 'LOOP_RECHECK_S', 0.05)  # the Timer instead of Core Audio's notification: the devices
    from peresmeshnik.audio import coreaudio  # change in this fake list only, Core Audio (on a Mac) would not say so

    def no_listener(on_change):
        raise RuntimeError('no Core Audio notification for a fake device list')
    monkeypatch.setattr(coreaudio, 'InputsChanged', no_listener)
    monkeypatch.setenv('PERESMESHNIK_CALL_PATH', 'tap')  # macOS 14.4+: no pass-through
    overlay = types.SimpleNamespace(side=lambda name: types.SimpleNamespace(hide=lambda: None))
    states, s = [], config.Settings()
    return app.Controller(overlay, states.append, settings=s), states, s, spawned


VMIC = types.SimpleNamespace(device={'uid': 'BlackHole2ch_UID'}, call_mic={'uid': 'BlackHole2ch_UID'}, remove=lambda: None)


def _with_vmic(monkeypatch, ctl, missing=None):
    """ctl.virtual_mic() makes BlackHole 2ch's VirtualMic; while `missing()` is true it raises as when not installed."""
    def make():
        if missing and missing():
            raise macos.BlackHoleMissing
        ctl._vmic = VMIC
        return VMIC
    monkeypatch.setattr(ctl, 'virtual_mic', make)
    monkeypatch.setattr(ctl, '_startup_call_macos', lambda: None)


def _until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.02)
    return cond()


def test_a_refused_loop_starts_by_itself_once_the_default_input_is_real(monkeypatch):
    """Translation off, "System default" while macOS' default input is BlackHole 2ch: refused, said once at startup
    (by the tray after its "idle": startup_error, no state before it), tried again as the devices change: the default
    set back, the call hears the user. Translation on and off meanwhile: the tray keeps the refusal, not "idle"."""
    devs = _mac_devices()
    devs[BH2_I]['default_input'] = True
    ctl, states, s, spawned = _loop_ctl(monkeypatch, devs)
    _with_vmic(monkeypatch, ctl)
    try:
        ctl.submit(ctl.startup).result(10)
        assert spawned == [] and "macOS' default input is BlackHole 2ch" in ctl.startup_error, ctl.startup_error
        assert states == []  # said once: by the tray, after its "idle"
        time.sleep(0.3)  # tried again meanwhile: still refused, not said again
        assert spawned == [] and states == [] and ctl._raw_mic
        ctl.on = True  # translation on and off: the refusal stays in the tray
        ctl.submit(ctl._off).result(10)
        assert states[-1].startswith("error: the call will not hear your microphone while translation is off: macOS' "
                                     'default input is BlackHole 2ch'), states
        devs[BH2_I]['default_input'], devs[MIC_I]['default_input'] = False, True  # set back in System Settings
        assert _until(lambda: spawned), spawned
        assert spawned[0][:3] == ('loop', '--from', '') and len(spawned) == 1
        assert states[-1] == 'idle, translation off (the call hears your microphone again)' and ctl._loop_arm is None
    finally:
        ctl._disarm()
        ctl._pool.shutdown()


def test_a_listener_that_fires_in_its_constructor_still_recovers(monkeypatch):
    """macOS: Core Audio may call the listener while it is being added: that arm is spent, the retry arms a new one
    before it checks, and a later change (the default set back) still starts the loop."""
    from peresmeshnik.audio import coreaudio
    devs = _mac_devices()
    devs[BH2_I]['default_input'] = True
    made = []

    class Listener:
        def __init__(self, on_change):
            self.on_change, self.closed = on_change, False
            made.append(self)
            if len(made) == 1:
                on_change()  # from inside its own constructor

        def close(self):
            self.closed = True
    ctl, states, s, spawned = _loop_ctl(monkeypatch, devs)
    monkeypatch.setattr(coreaudio, 'InputsChanged', Listener)  # after _loop_ctl's Timer-only stand-in
    _with_vmic(monkeypatch, ctl)
    try:
        ctl.submit(ctl.startup).result(10)
        assert _until(lambda: len(made) >= 2) and made[0].closed and not made[-1].closed and spawned == []
        devs[BH2_I]['default_input'], devs[MIC_I]['default_input'] = False, True
        made[-1].on_change()  # Core Audio: the default input changed
        assert _until(lambda: spawned) and spawned[0][:3] == ('loop', '--from', '')
        assert all(x.closed for x in made)  # nothing left listening
    finally:
        ctl._disarm()
        ctl._pool.shutdown()


class _Watch:
    """winaudio.DeviceWatch's stand-in: `on_change()` is Windows saying the default input or a device changed."""
    made: list = []

    def __init__(self, on_change, flow='render'):
        self.on_change, self.flow, self.closed = on_change, flow, False
        self.made.append(self)

    def close(self):
        self.closed = True


def _win_loop(monkeypatch, devs):
    """_loop_ctl on Windows' path (the helper module stays macOS' fake): the device watch is _Watch, and every device
    list (`list`, a helper process on Windows) is counted."""
    from peresmeshnik import app
    from peresmeshnik.audio import winaudio
    ctl, states, s, spawned = _loop_ctl(monkeypatch, devs)
    _with_vmic(monkeypatch, ctl)
    monkeypatch.setattr(app.sys, 'platform', 'win32')
    monkeypatch.setattr(winaudio, 'DeviceWatch', _Watch)
    monkeypatch.setattr(_Watch, 'made', [])
    listed = []
    monkeypatch.setattr(macos, 'devices', lambda: listed.append(1) or devs)
    return ctl, states, spawned, listed


def test_a_refused_loop_on_windows_lists_no_devices_while_idle(monkeypatch):
    """Windows, translation off, the default input virtual (CABLE Output): refused, then nothing runs while idle (no
    helper `list` every few seconds: idle CPU, docs/ARCHITECTURE.md §5) until Windows says the devices changed; then
    the loop starts. A device event while still refused (no microphone yet) lists once and waits again."""
    from peresmeshnik import app
    devs = _mac_devices()
    devs[BH2_I]['default_input'] = True
    ctl, states, spawned, listed = _win_loop(monkeypatch, devs)
    monkeypatch.setattr(app, 'LOOP_RECHECK_S', 0.05)  # a Timer, if any, would list ~10 times below
    try:
        ctl.submit(ctl.startup).result(10)
        assert spawned == [] and ctl._loop_why
        n = len(listed)
        time.sleep(0.5)
        assert len(listed) == n and spawned == []  # idle: nothing listed (no helper `list` spawned), nothing started
        assert len(_Watch.made) == 1 and _Watch.made[0].flow == 'capture'  # waits on Windows' device events instead
        _Watch.made[-1].on_change()  # a device event, the default still virtual
        assert _until(lambda: len(listed) > n and len(_Watch.made) == 2)
        time.sleep(0.3)
        assert spawned == [] and _Watch.made[0].closed and not _Watch.made[1].closed and len(listed) == n + 1
        devs[BH2_I]['default_input'], devs[MIC_I]['default_input'] = False, True  # a real default
        _Watch.made[-1].on_change()
        assert _until(lambda: spawned) and spawned[0][:3] == ('loop', '--from', '')
        assert _until(lambda: ctl._loop_arm is None) and all(w.closed for w in _Watch.made)
    finally:
        ctl._disarm()
        ctl._pool.shutdown()


def test_quit_with_a_retry_armed_and_queued_starts_no_loop(monkeypatch):
    """Windows: the refused loop's device watch is armed and a retry is queued behind other work when the app quits:
    the retry finds the app quitting, nothing is spawned, the watch is closed."""
    devs = _mac_devices()
    devs[BH2_I]['default_input'] = True
    ctl, states, spawned, listed = _win_loop(monkeypatch, devs)
    ctl.submit(ctl.startup).result(10)
    arm = ctl._loop_arm
    watch = arm.handle
    assert isinstance(watch, _Watch) and not watch.closed
    gate = threading.Event()
    ctl.submit(gate.wait, 10)  # the control thread busy
    ctl.submit(ctl._loop_retry)  # queued
    devs[BH2_I]['default_input'], devs[MIC_I]['default_input'] = False, True  # a real default by now
    q = threading.Thread(target=ctl.quit)
    q.start()
    assert _until(lambda: ctl._quitting)
    gate.set()
    q.join(15)
    assert spawned == [] and ctl._loop_arm is None and arm.fired and watch.closed


def test_no_device_watch_on_windows_leaves_the_settings_to_retry(monkeypatch):
    """Windows: the device watch does not start (no loopback helper): no Timer instead, the refusal stays; saving the
    settings (mic_changed) still tries again."""
    from peresmeshnik.audio import winaudio
    devs = _mac_devices()
    devs[BH2_I]['default_input'] = True
    ctl, states, spawned, listed = _win_loop(monkeypatch, devs)

    def no_helper(on_change, flow='render'):
        raise FileNotFoundError('peresmeshnik-loopback.exe')
    monkeypatch.setattr(winaudio, 'DeviceWatch', no_helper)
    try:
        ctl.submit(ctl.startup).result(10)
        assert spawned == [] and ctl._loop_why and ctl._loop_arm.fired and ctl._loop_arm.handle is None
        devs[BH2_I]['default_input'], devs[MIC_I]['default_input'] = False, True
        ctl.mic_changed().result(10)
        assert spawned[0][:3] == ('loop', '--from', '') and ctl._loop_arm is None
    finally:
        ctl._disarm()
        ctl._pool.shutdown()


def test_a_saved_virtual_microphone_is_not_retried(monkeypatch):
    """A virtual microphone saved in the settings: refused with why, and only the settings fix it (no retry)."""
    devs = _mac_devices()
    devs[MIC_I]['default_input'] = True
    ctl, states, s, spawned = _loop_ctl(monkeypatch, devs)
    _with_vmic(monkeypatch, ctl)
    s.ui.mic = 'ZoomAudioDevice'
    try:
        ctl.submit(ctl.startup).result(10)
        assert spawned == [] and 'ZoomAudioDevice is a virtual device' in ctl.startup_error and ctl._loop_arm is None
    finally:
        ctl._disarm()
        ctl._pool.shutdown()


def test_blackhole_installed_while_the_app_runs_gets_its_loop(monkeypatch):
    """BlackHole 2ch missing at startup (said by the tray); installed later with the wizard closed: the device list
    changes, the virtual mic is made and the loop starts, with the raw microphone flag set (_loopback)."""
    devs = _mac_devices()
    devs[MIC_I]['default_input'] = True
    installed = []
    ctl, states, s, spawned = _loop_ctl(monkeypatch, devs)
    _with_vmic(monkeypatch, ctl, missing=lambda: not installed)
    try:
        ctl.submit(ctl.startup).result(10)
        assert ctl._vmic is None and 'BlackHole 2ch is not installed' in ctl.startup_error and spawned == []
        time.sleep(0.2)
        assert spawned == []
        installed.append(True)
        assert _until(lambda: spawned), states
        assert spawned[0][:5] == ('loop', '--from', '', '--to', 'BlackHole2ch_UID') and ctl._raw_mic
    finally:
        ctl._disarm()
        ctl._pool.shutdown()


def test_blackhole_installed_in_the_wizard_gets_its_loop_at_finish(monkeypatch):
    """BlackHole 2ch was missing at the start (no virtual mic, no loop); installed before the wizard's Finish
    (tray.restart -> mic_changed): the virtual mic is made and the loop starts through _loopback (the raw microphone
    flag set), without translation on once."""
    devs = _mac_devices()
    devs[MIC_I]['default_input'] = True
    ctl, states, s, spawned = _loop_ctl(monkeypatch, devs)
    _with_vmic(monkeypatch, ctl)
    assert ctl._vmic is None
    try:
        s.ui.mic = 'BuiltInMicrophoneDevice'
        ctl.mic_changed().result(10)
        assert [a[:5] for a in spawned] == [('loop', '--from', 'BuiltInMicrophoneDevice', '--to', 'BlackHole2ch_UID')]
        assert ctl._raw_mic
    finally:
        ctl._disarm()
        ctl._pool.shutdown()

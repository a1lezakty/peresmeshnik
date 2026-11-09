"""macOS "me -> them" devices through the Core Audio helper (native/macos/audio): BlackHole 2ch as the virtual mic,
the microphone by UID that follows a device going away and coming back, the loop "microphone -> BlackHole 2ch" while
translation is off, and what each costs in CPU.

On the CI runner (macos-15, BlackHole 2ch and 16ch installed, the helper built) BlackHole 16ch stands in for the real
microphone: a test process plays into it. A device that goes and comes back is an aggregate device made by the helper.
Elsewhere skipped, except the checks that need no Mac.
"""
import os
import subprocess
import sys
import threading
import time

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
    with pytest.raises(macos.BlackHoleMissing, match='brew install blackhole-2ch'):
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
            {**dev, 'uid': 'peresmeshnik-test-mic', 'name': 'Test Mic', 'transport': 'grup'}]
    monkeypatch.setattr(selftest, 'MACOS', True)
    monkeypatch.setattr(macos, 'devices', lambda: devs)
    # saved earlier (or by hand) as the microphone: refused when it opens, with the reason
    with pytest.raises(ValueError, match='ZoomAudioDevice is a virtual device, not your microphone'):
        macos.not_blackhole('ZoomAudioDevice')
    with pytest.raises(ValueError, match='Test Mic is an aggregate device, not your microphone'):
        macos.not_blackhole('Test Mic')
    assert macos.not_blackhole('BuiltInMicrophoneDevice') == 'BuiltInMicrophoneDevice'
    assert macos.not_blackhole('') == ''  # the default input: the helper never takes a virtual one by itself
    assert selftest.devices('sources') == [('BuiltInMicrophoneDevice', 'MacBook Pro Microphone')]
    assert ('BlackHole16ch_UID', 'BlackHole 16ch') in selftest.devices('sinks')  # an output: S1's (the call app's)
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


@mac
def test_latency():
    """play() -> the sound at BlackHole 2ch's input as a call app reads it, 5 clicks from silence (the first one opens
    the playback process). Upper bound: includes the reader's own 10 ms chunks."""
    from peresmeshnik.audio.virtual_mic_macos import VirtualMic
    vm = VirtualMic()
    rec = Timed(macos.BLACKHOLE_UID)
    sent = []
    try:
        time.sleep(0.5)
        for _ in range(5):
            sent.append(time.monotonic())
            vm.play(np.concatenate([np.full(SR_IN // 50, 0.5, np.float32), np.zeros(SR_IN // 2, np.float32)]), SR_IN)
            time.sleep(0.6)
        y = rec.stop()
    finally:
        vm.remove()
    on = np.nonzero(np.abs(y) > 0.25)[0]
    starts = [i for k, i in enumerate(on) if k == 0 or i - on[k - 1] > SR_OUT // 10]
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
    assert errors == ['error: them: BlackHole 2ch: the playback stopped (peresmeshnik-audio: BlackHole 2ch went away)']
    assert plays == [320]


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

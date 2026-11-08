"""Round trip through "Peresmeshnik Mic": play into the virtual mic, record its source like a call app.

Needs a running PipeWire (pipewire-pulse) or PulseAudio and pactl/pacat/parec; skipped otherwise
(CI runners have no sound server). Also checks the working-mode cost of the output path:
CPU seconds per second of audio of our process plus its pacat.
"""
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != 'linux' or not shutil.which('parec')
    or subprocess.run(['pactl', 'info'], capture_output=True).returncode != 0,
    reason='no PipeWire/PulseAudio session')

SR_IN, SR_OUT = 24_000, 16_000  # TTS rate in, call-app rate out
CPU_PER_AUDIO_S = 0.05  # our process + pacat incl. its start, per second of audio; a spinning pacat costs 0.2-1


def record(source):
    """Record the source like a call app does. Read on a thread while recording: a pipe nobody reads holds ~1 s
    (64 KB at 16 kHz), and what parec buffers past that is lost at SIGTERM (the rate-switch flake under load)."""
    from peresmeshnik.ui.selftest import Recorder
    return Recorder(source, SR_OUT)


def test_round_trip():
    from peresmeshnik.audio import virtual_mic_linux as vm
    mic = vm.VirtualMic()
    try:
        assert vm.exists()
        rec = record(vm.SOURCE)
        time.sleep(0.3)  # recorder connected
        t = np.arange(int(SR_IN * 3)) / SR_IN
        tone = (0.3 * np.sin(2 * np.pi * 440 * t) * (t < 1.5)).astype(np.float32)  # 1.5 s tone, 1.5 s silence
        me = psutil.Process()
        cpu0 = sum(me.cpu_times()[:4])  # self + waited-for children (pacat exits in drain())
        t0 = time.perf_counter()
        for i in range(0, len(tone), SR_IN // 5):  # 200 ms chunks, like TTS
            mic.play(tone[i:i + SR_IN // 5], SR_IN)
        mic.drain()
        wall = time.perf_counter() - t0
        cpu = sum(me.cpu_times()[:4]) - cpu0
        time.sleep(0.3)
        x = rec.stop()
    finally:
        mic.remove()
    assert not vm.exists(), 'device left behind'
    on = np.nonzero(np.abs(x) > 0.1)[0]
    assert len(on), 'nothing came out of the virtual mic'
    seg = x[on[0]:on[0] + SR_OUT]
    assert np.argmax(np.abs(np.fft.rfft(seg))) * SR_OUT / len(seg) == pytest.approx(440, abs=3)
    assert (on[-1] - on[0]) / SR_OUT == pytest.approx(1.5, abs=0.05)  # whole tone, nothing dropped
    assert np.sqrt(np.mean(seg ** 2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.1)  # level kept
    assert 2.9 < wall < 3.6  # played in real time
    print(f'virtual mic: {wall:.2f} s for 3 s of audio, CPU {cpu:.3f} s')
    assert cpu / 3 < CPU_PER_AUDIO_S


def test_rate_switch_plays_both_in_order():
    """A new sample rate (TTS 24 kHz -> live mic 16 kHz) must not cut the queued audio or leave a pacat behind."""
    from peresmeshnik.audio import virtual_mic_linux as vm
    mic = vm.VirtualMic()
    try:
        rec = record(vm.SOURCE)
        time.sleep(0.3)
        tone = lambda sr, hz: (0.3 * np.sin(2 * np.pi * hz * np.arange(sr) / sr)).astype(np.float32)  # noqa: E731
        mic.play(tone(24_000, 440), 24_000)
        first = mic._proc
        mic.play(tone(16_000, 880), 16_000)
        mic.drain()
        assert first.poll() is not None, 'old pacat still running'
        time.sleep(0.3)
        x = rec.stop()
    finally:
        mic.remove()
    on = np.nonzero(np.abs(x) > 0.1)[0]
    assert (on[-1] - on[0]) / SR_OUT == pytest.approx(2.0, abs=0.1)  # 1 s + 1 s, nothing dropped
    peak = lambda seg: np.argmax(np.abs(np.fft.rfft(seg))) * SR_OUT / len(seg)  # noqa: E731
    assert peak(x[on[0] + 1600:on[0] + 14400]) == pytest.approx(440, abs=5)
    assert peak(x[on[-1] - 14400:on[-1] - 1600]) == pytest.approx(880, abs=5)


def test_flush_drops_queued_audio():
    from peresmeshnik.audio import virtual_mic_linux as vm
    mic = vm.VirtualMic()
    try:
        rec = record(vm.SOURCE)
        time.sleep(0.3)
        mic.play(np.full(SR_IN * 5, 0.3, np.float32) * np.sin(np.arange(SR_IN * 5) / 10), SR_IN)  # 5 s
        time.sleep(0.5)
        mic.flush()
        time.sleep(1.0)
        x = rec.stop()
    finally:
        mic.remove()
    loud = np.nonzero(np.abs(x) > 0.1)[0]
    assert len(loud) and (loud[-1] - loud[0]) / SR_OUT < 1.5  # ~0.5 s played + buffers, not 5 s


def test_never_the_default_device():
    """Another output and input are there (here without a priority, like any module-made device): the virtual mic
    must not take over as the default sink or source (the system's sound would go into the call)."""
    from peresmeshnik.audio import virtual_mic_linux as vm
    pa = vm._pactl
    mods = [pa('load-module', 'module-null-sink', 'sink_name=test_real_out').strip(),
            pa('load-module', 'module-null-sink', 'sink_name=test_real_in_feed').strip(),
            pa('load-module', 'module-remap-source', 'master=test_real_in_feed.monitor', 'source_name=test_real_in').strip()]
    try:
        time.sleep(0.3)
        before = pa('get-default-sink').strip(), pa('get-default-source').strip()
        mic = vm.VirtualMic()
        try:
            time.sleep(0.5)  # the session manager picks defaults asynchronously
            assert (pa('get-default-sink').strip(), pa('get-default-source').strip()) == before
        finally:
            mic.remove()
    finally:
        for m in reversed(mods):
            subprocess.run(['pactl', 'unload-module', m], capture_output=True)


class Source:
    """A test microphone: a null sink fed by a tone, its monitor remapped as a source named `name`."""

    def __init__(self, name, hz):
        from peresmeshnik.audio import virtual_mic_linux as vm
        self.name = name
        self.mods = [vm._pactl('load-module', 'module-null-sink', f'sink_name={name}_feed', 'channels=1').strip(),
                     vm._pactl('load-module', 'module-remap-source', f'master={name}_feed.monitor', f'source_name={name}',
                               'channels=1').strip()]
        t = (0.2 * np.sin(2 * np.pi * hz * np.arange(1600) / 16000)).astype('<f4').tobytes()  # 0.1 s, whole periods
        # dont-reconnect: PipeWire would move it into a sink the app makes (the call sink takes over as the default)
        self.feed = subprocess.Popen(['pacat', f'--device={name}_feed', '--raw', '--format=float32le', '--rate=16000',
                                      '--channels=1', '--latency-msec=20', '--property=node.dont-reconnect=true'],
                                     stdin=subprocess.PIPE)

        def talk():  # the pipe blocks: real-time pace
            try:
                while True:
                    self.feed.stdin.write(t)
            except (BrokenPipeError, ValueError, OSError):
                pass
        import threading
        threading.Thread(target=talk, daemon=True).start()

    def remove(self):
        self.feed.kill()
        self.feed.wait()
        for m in reversed(self.mods):
            subprocess.run(['pactl', 'unload-module', m], capture_output=True)
        self.mods = []


def hz(frames):
    x = np.concatenate(frames[-25:])  # the last 0.5 s
    return np.argmax(np.abs(np.fft.rfft(x))) * 16000 / len(x)


def wait(cond, timeout=5.0, show=lambda: ''):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, f'timed out {show()}'
        time.sleep(0.02)


def test_mic_by_name_and_never_ours():
    from peresmeshnik.audio.mic import Mic
    from peresmeshnik.audio.virtual_mic_linux import SOURCE
    with pytest.raises(ValueError, match='not a microphone'):
        Mic(SOURCE).start(lambda f: None)
    with pytest.raises(RuntimeError, match='no microphone'):
        Mic('test_no_such_mic').start(lambda f: None)
    a = Source('test_mic_a', 400)
    frames, ended = [], []
    mic = Mic('test_mic_a', on_end=ended.append)
    try:
        mic.start(frames.append)
        wait(lambda: len(frames) > 40)
        assert {len(f) for f in frames} == {320} and hz(frames) == pytest.approx(400, abs=10)  # 20 ms frames of that source
        n = len(frames)
        a.remove()  # what this PipeWire / WirePlumber does to the recording when its source goes (printed, not asserted)
        time.sleep(1.0)
        print(f'source removed under parec: parec {"ended" if mic._proc.poll() is not None else "still running"}, '
              f'end reported {ended}, {len(frames) - n} frames after the removal')
    finally:
        ended.clear()
        mic.stop()
        a.remove()
    assert not ended  # stop() is no outage


@pytest.mark.parametrize('mic_dies', [False, True])
def test_microphone_survives_a_sound_server_restart(monkeypatch, mic_dies):
    """pipewire-pulse restarts mid-call: the event stream ends. The tray hears it once, the app subscribes again after
    a pause (one repeat, not polling), and the new subscription still catches an unplug. `mic_dies`: the microphone's
    stream went down with the server, which stays away until the second subscription; then the microphone reopens."""
    from peresmeshnik import app
    from peresmeshnik.app import Microphone
    from peresmeshnik.audio import mic as mic_mod
    monkeypatch.setattr(Microphone, 'RESUBSCRIBE_S', 0.3)
    start = mic_mod.Mic.start

    def silent(self, on_frames):  # only the removal event tells (WirePlumber 0.5)
        start(self, on_frames)
        self.on_end = None
    monkeypatch.setattr(mic_mod.Mic, 'start', silent)
    a, b = Source('test_mic_a', 400), Source('test_mic_b', 700)
    frames, states = [], []
    mic = Microphone('test_mic_a', states.append)
    try:
        mic.start(frames.append)
        wait(lambda: len(frames) > 40)
        first = mic._watch
        if mic_dies:
            source = app.mic_source

            def down(*a):  # the server is away while the first subscription is current
                if mic._watch is first:
                    raise app.NoMicrophone('no sound server')
                return source(*a)
            monkeypatch.setattr(app, 'mic_source', down)
            mic._mic._proc.kill()
        first.kill()
        wait(lambda: len(states) >= 2, show=lambda: states)  # the next ones may come at once
        assert states[0] == 'error: lost the sound server; reconnecting the microphone in 0.3 s'
        if mic_dies:
            assert states[1] == 'error: the microphone went away (test_mic_a ended with the sound server); reopening'
            wait(lambda: len(states) >= 3, show=lambda: states)
            assert states[2:] == ['translating (the microphone is back: test_mic_a)']
        else:
            assert states[1:] == ['translating (the sound server is back)']
        assert mic._watch is not first and mic._watch.poll() is None
        n = len(frames)
        wait(lambda: len(frames) > n + 40)
        assert hz(frames) == pytest.approx(400, abs=10)
        k = len(states)
        a.remove()  # seen by the new subscription
        wait(lambda: len(states) >= k + 2, show=lambda: states)
        assert states[k + 1:] == ['translating (the microphone is back: test_mic_b)']
    finally:
        mic.stop()
        a.remove()
        b.remove()


@pytest.mark.parametrize('end_reported', [True, False])
def test_microphone_reopens_when_the_source_goes(monkeypatch, end_reported):
    """The saved microphone is unplugged: the default one takes over, once in the tray; with none left the app waits
    for PipeWire to announce a source (an event, not polling) and takes it. `end_reported=False`: the stream outlives
    its device without an end (WirePlumber 0.5 on ubuntu-latest), only the removal event tells."""
    from peresmeshnik.app import Microphone
    from peresmeshnik.audio import mic as mic_mod
    if not end_reported:
        start = mic_mod.Mic.start

        def silent(self, on_frames):
            start(self, on_frames)
            self.on_end = None
        monkeypatch.setattr(mic_mod.Mic, 'start', silent)
    a, b = Source('test_mic_a', 400), Source('test_mic_b', 700)
    frames, states = [], []
    mic = Microphone('test_mic_a', states.append)
    c = None
    try:
        mic.start(frames.append)
        wait(lambda: len(frames) > 40)
        assert hz(frames) == pytest.approx(400, abs=10)
        a.remove()  # unplugged
        wait(lambda: len(states) == 2, show=lambda: states)
        assert states[0].startswith('error: the microphone went away (test_mic_a') and states[0].endswith('; reopening')
        assert states[1] == 'translating (the microphone is back: test_mic_b)'
        n = len(frames)
        wait(lambda: len(frames) > n + 40)
        assert hz(frames) == pytest.approx(700, abs=10)
        b.remove()  # no microphone at all now
        wait(lambda: len(states) == 3, show=lambda: states)
        time.sleep(0.5)
        assert len(states) == 3  # waiting, quietly
        c = Source('test_mic_c', 1000)  # plugged in
        wait(lambda: len(states) == 4, show=lambda: states)
        assert states[3] == 'translating (the microphone is back: test_mic_c)', states
        n = len(frames)
        wait(lambda: len(frames) > n + 40)
        assert hz(frames) == pytest.approx(1000, abs=10)
    finally:
        mic.stop()
        for s in (a, b, c):
            if s:
                s.remove()
    n = len(frames)
    time.sleep(0.2)
    assert len(frames) == n  # stopped for good

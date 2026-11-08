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
    """Record the source like a call app does."""
    return subprocess.Popen(['parec', f'--device={source}', '--format=float32le', f'--rate={SR_OUT}', '--channels=1', '--raw',
                             '--latency-msec=20', '--property=node.dont-reconnect=true'], stdout=subprocess.PIPE)


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
        rec.terminate()
        x = np.frombuffer(rec.communicate()[0], dtype=np.float32)
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
        rec.terminate()
        x = np.frombuffer(rec.communicate()[0], dtype=np.float32)
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
        rec.terminate()
        x = np.frombuffer(rec.communicate()[0], dtype=np.float32)
    finally:
        mic.remove()
    loud = np.nonzero(np.abs(x) > 0.1)[0]
    assert len(loud) and (loud[-1] - loud[0]) / SR_OUT < 1.5  # ~0.5 s played + buffers, not 5 s

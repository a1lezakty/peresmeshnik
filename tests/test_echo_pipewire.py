"""The echo gate on the app's real path on Linux, no models: our speech goes into "Peresmeshnik Mic" (VirtualMic,
through Referenced as in the app), a far end records that microphone like a call app, plays it back 150 ms later and
12 dB quieter (its loudspeaker into its microphone) between its own sentences into "Peresmeshnik Call"; the capture of
that sink goes through the gate on its own thread (Threaded). The echo is zeroed, their speech is not: the reference's
clock and the devices' latencies line up. PERESMESHNIK_REQUIRE_PIPEWIRE=1 (CI on Linux): no sound server fails."""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from peresmeshnik.audio.echo import HOP, EchoGate, Reference, Referenced, Threaded

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_echo import OURS, THEIRS  # noqa: E402

SR = 16000
DELAY, GAIN = 0.15, 10 ** (-12 / 20)


def sound_server() -> bool:
    return bool(shutil.which('pactl')) and subprocess.run(['pactl', 'info'], capture_output=True).returncode == 0


@pytest.fixture
def devices():
    if not sys.platform.startswith('linux') or not sound_server():
        (pytest.fail if os.environ.get('PERESMESHNIK_REQUIRE_PIPEWIRE') == '1' else pytest.skip)('no PipeWire/PulseAudio')
    from peresmeshnik.audio.capture_pipewire import PipeWireCapture
    from peresmeshnik.audio.virtual_mic_linux import SOURCE, VirtualMic
    mic = VirtualMic()
    cap = PipeWireCapture()
    yield mic, cap, SOURCE
    mic.stop()
    mic.remove()


def far_end(mic_source: str, call_sink: str, theirs: np.ndarray, stop: threading.Event):
    """The other side of the call on a loudspeaker: what it hears from our microphone comes back DELAY later, GAIN
    quieter, mixed with its own speech (`theirs`, from its start), 20 ms at a time."""
    rec = subprocess.Popen(['parec', f'--device={mic_source}', '--raw', '--format=float32le', f'--rate={SR}', '--channels=1',
                            '--latency-msec=20'], stdout=subprocess.PIPE)
    play = subprocess.Popen(['pacat', '--playback', f'--device={call_sink}', '--raw', '--format=float32le', f'--rate={SR}',
                             '--channels=1', '--latency-msec=20'], stdin=subprocess.PIPE)
    line, pos, n = np.zeros(int(DELAY * SR), np.float32), 0, SR // 50
    try:
        while not stop.is_set() and len(b := rec.stdout.read(n * 4)) == n * 4:
            line = np.concatenate([line, np.frombuffer(b, '<f4')])
            out, line = line[:n] * GAIN, line[n:]
            own = theirs[pos:pos + n]
            out[:len(own)] += own
            pos += n
            play.stdin.write(out.astype('<f4').tobytes())
    finally:
        rec.kill()
        play.stdin.close()
        play.wait(10)
        rec.wait()


def segments(x: np.ndarray) -> list[tuple[int, int]]:
    loud = np.convolve(np.abs(x) > 0.003, np.ones(SR * 6 // 10), 'same') > 0
    e = np.flatnonzero(np.diff(np.concatenate([[0], loud.astype(int), [0]])))
    return [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a > SR // 2]


def test_echo_through_the_devices(devices):
    mic, cap, source = devices
    ours, theirs_clips = [x[:5 * SR] for x in OURS[:3]], [x[:5 * SR] for x in THEIRS[:3]]
    plan, t = [], 0.5  # (when we speak, our clip); their speech on the far end's own timeline
    theirs = np.zeros(60 * SR, np.float32)
    for x, y in zip(ours, theirs_clips):
        plan.append((t, x))
        t += len(x) / SR + DELAY + 1.2
        theirs[int(t * SR):int(t * SR) + len(y)] = y
        t += len(y) / SR + 1.2
    ref = Reference()
    out = Referenced(mic, ref)
    gate = EchoGate(ref)
    raw, gated = [], []
    th = Threaded(gate, gated.append)

    def on_frames(pcm):  # the capture's thread: only hand on, as the app's Side does
        raw.append(pcm)
        th(pcm)
    cap.start(on_frames)
    stop = threading.Event()
    far = threading.Thread(target=far_end, args=(source, cap.sink, theirs, stop), daemon=True)
    try:
        far.start()
        t0 = time.perf_counter()
        for at, x in plan:
            time.sleep(max(0.0, t0 + at - time.perf_counter()))
            for i in range(0, len(x), 4096):  # as the TTS hands its speech out, in chunks
                out.play(x[i:i + 4096], SR)
        time.sleep(max(0.0, t0 + t + 1.0 - time.perf_counter()))
    finally:
        stop.set()
        far.join(15)
        cap.stop()
        th.close()
    x, y = np.concatenate(raw), np.concatenate(gated)
    assert len(x) == len(y)
    segs = segments(x)
    print('segments', [(round(a / SR, 2), round(b / SR, 2)) for a, b in segs], 'delay', gate.delay, 'rho', round(gate.rho, 2),
          gate.stats)
    assert len(segs) == 6, segs  # echo, them, echo, them, echo, them

    def zeroed(a, b):
        blocks = range(a // HOP, b // HOP)
        loud = [k for k in blocks if np.abs(x[k * HOP:(k + 1) * HOP]).max() > 0.003]
        return sum(not y[k * HOP:(k + 1) * HOP].any() for k in loud) / max(len(loud), 1)
    echo = [zeroed(*s) for s in segs[::2]]
    them = [zeroed(*s) for s in segs[1::2]]
    print('echo zeroed', [round(v, 3) for v in echo], 'their speech zeroed', [round(v, 3) for v in them])
    assert min(echo) >= 0.8, echo
    assert max(them) <= 0.02, them

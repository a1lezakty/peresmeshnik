"""Capture and playback through PipeWire devices, no models: a tone played into the
"Peresmeshnik Call" null sink comes back from its monitor. Skipped without a sound server."""
import shutil
import subprocess
import threading

import numpy as np
import pytest

from peresmeshnik.engines import SAMPLE_RATE

pytestmark = pytest.mark.skipif(
    not shutil.which('pactl') or subprocess.run(['pactl', 'info'], capture_output=True).returncode != 0,
    reason='no PulseAudio/PipeWire server')


def test_tone_through_call_sink():
    from peresmeshnik.audio.capture_pipewire import PipeWireCapture, pactl
    from peresmeshnik.audio.out import PulseOut
    got, lock = [], threading.Lock()

    def on_frames(x):
        with lock:
            got.append(x)

    cap = PipeWireCapture()
    cap.start(on_frames)
    try:
        assert 'Peresmeshnik Call' in pactl('list', 'sinks')
        out = PulseOut(SAMPLE_RATE, cap.sink)
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        out.play((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SAMPLE_RATE)
        out.stop()  # returns when pacat has drained
    finally:
        cap.stop()
    assert cap.sink not in pactl('list', 'short', 'sinks').split()  # removed on stop
    x = np.concatenate(got)
    loud = np.abs(x) > 0.1
    assert loud.sum() > 0.5 * SAMPLE_RATE, f'{loud.sum()} loud samples of {len(x)}'
    spectrum = np.abs(np.fft.rfft(x[loud.argmax():][:SAMPLE_RATE // 2]))
    assert abs(spectrum.argmax() * 2 - 440) <= 4  # bin = 2 Hz on a 0.5 s window

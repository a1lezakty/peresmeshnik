"""Linux capture of the call sound: a null sink "Peresmeshnik Call" and a recording of its monitor.

The call app's stream is moved to the sink (`pactl move-sink-input`, stage 2); in stage 1 a file is
played into it. Works with PipeWire (pipewire-pulse) and PulseAudio. The reader thread blocks on
the pipe from `parec` and only hands frames on; it never polls.
"""
from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable

import numpy as np

from peresmeshnik.engines import SAMPLE_RATE


def pactl(*args: str) -> str:
    return subprocess.run(['pactl', *args], check=True, capture_output=True, text=True).stdout


class PipeWireCapture:
    def __init__(self, sink: str = 'peresmeshnik_call', description: str = 'Peresmeshnik Call', frame: int = 512):
        self.sink, self.description, self.frame = sink, description, frame
        self._module = self._proc = self._thread = None

    def start(self, on_frames: Callable[[np.ndarray], None]) -> None:
        if self.sink not in pactl('list', 'short', 'sinks').split():  # reuse a sink left by a crashed run
            # priority 0: never preferred over a real output device as the default sink
            self._module = pactl('load-module', 'module-null-sink', f'sink_name={self.sink}',
                                 f'sink_properties=\'device.description="{self.description}"'
                                 ' priority.session=0 priority.driver=0\'').strip()
        # dont-reconnect: PipeWire would move the recording to the next new device (e.g. the virtual mic's sink)
        self._proc = subprocess.Popen(['parec', f'--device={self.sink}.monitor', '--raw', '--format=float32le',
                                       f'--rate={SAMPLE_RATE}', '--channels=1', '--latency-msec=20',
                                       '--property=node.dont-reconnect=true'], stdout=subprocess.PIPE)
        self._thread = threading.Thread(target=self._read, args=(on_frames,), name='capture', daemon=True)
        self._thread.start()

    def _read(self, on_frames):
        n = self.frame * 4
        while len(b := self._proc.stdout.read(n)) == n:
            on_frames(np.frombuffer(b, '<f4'))

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(5)
            self._thread.join(5)
            self._proc = None
        if self._module:
            pactl('unload-module', self._module)
            self._module = None

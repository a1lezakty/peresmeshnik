"""Translated speech to the headphones. Linux: a PulseAudio/PipeWire sink through `pacat`.

play() only enqueues; one playback thread writes to the device, blocking on its buffer.
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time

import numpy as np


def default_sink() -> str:
    """Name of the current default sink. Resolve it before creating our own sinks: on a box without a real
    output device the new null sink becomes the default."""
    return subprocess.run(['pactl', 'get-default-sink'], check=True, capture_output=True, text=True).stdout.strip()


class PulseOut:
    def __init__(self, sample_rate: int, device: str | None = None, latency_ms: int = 40):
        self.sample_rate, self.latency_s = sample_rate, latency_ms / 1000
        cmd = ['pacat', '--playback', '--raw', '--format=float32le', f'--rate={sample_rate}', '--channels=1',
               f'--latency-msec={latency_ms}', '--stream-name=Peresmeshnik translation',
               # PipeWire moves a stream to a newly created sink (ours!) despite its target unless told not to
               '--property=node.dont-reconnect=true']
        self._proc = subprocess.Popen(cmd + ([f'--device={device}'] if device else []), stdin=subprocess.PIPE)
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._end = 0.0  # monotonic time when the audio written so far has played
        self._thread = threading.Thread(target=self._run, name='playback', daemon=True)
        self._thread.start()

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        assert sample_rate == self.sample_rate, f'{sample_rate} != {self.sample_rate}'
        self._q.put(pcm)

    def flush(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        """Play what is queued, then close the device."""
        self._q.put(None)
        self._thread.join()
        self._proc.wait(5)

    def _run(self):
        while (pcm := self._q.get()) is not None:
            self._end = max(self._end, time.monotonic()) + len(pcm) / self.sample_rate
            self._proc.stdin.write(pcm.astype('<f4').tobytes())
            self._proc.stdin.flush()
        # pacat spins a core draining after EOF (0.64 s CPU for a 0.7 s tail), so close only once it has played
        time.sleep(max(0.0, self._end - time.monotonic()) + self.latency_s)  # the last write still sits in the buffer
        self._proc.stdin.close()

"""Microphone capture on every OS through PortAudio (sounddevice).

The device callback only copies the frame out (engines.py rules); `on_frames` must just enqueue.
The stream exists only between start() and stop(): when translation is off no device is open.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from peresmeshnik.engines import SAMPLE_RATE


class Mic:
    def __init__(self, device: int | str | None = None, sample_rate: int = SAMPLE_RATE, frame_ms: int = 20):
        self.device, self.sample_rate = device, sample_rate
        self.blocksize = sample_rate * frame_ms // 1000
        self._stream = None

    def start(self, on_frames: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd  # PortAudio is loaded only when a stream opens

        def callback(indata, frames, time_info, status):
            on_frames(indata[:, 0].copy())

        self._stream = sd.InputStream(device=self.device, channels=1, samplerate=self.sample_rate,
                                      blocksize=self.blocksize, dtype='float32', callback=callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def record(seconds: float, device: int | str | None = None, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Blocking recording of `seconds` from the microphone (enrollment)."""
    frames: list[np.ndarray] = []
    mic = Mic(device, sample_rate)
    mic.start(frames.append)
    try:
        time.sleep(seconds)
    finally:
        mic.stop()
    return np.concatenate(frames)[:int(seconds * sample_rate)] if frames else np.zeros(0, np.float32)

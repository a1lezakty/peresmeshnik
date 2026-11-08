"""Streaming resampler shared by the audio paths (headphones, the live switch)."""
from __future__ import annotations

import numpy as np


class Resampler:
    """Linear resampling of a stream in chunks, keeping the fractional position and the last sample between
    chunks: per-chunk rounding would add ~34 ms per minute at 16 -> 22.05 kHz.
    ponytail: linear interpolation, fine for speech and the live pass-through; soxr if quality matters."""

    def __init__(self, sr: int, to: int):
        self.step, self.pos, self.prev = sr / to, 1.0, 0.0  # pos counts from the previous chunk's last sample

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        if self.step == 1.0:
            return pcm
        x = np.concatenate([[self.prev], pcm])
        t = np.arange(self.pos, len(x) - 1, self.step)
        self.pos = (t[-1] + self.step if len(t) else self.pos) - (len(x) - 1)
        self.prev = x[-1]
        return np.interp(t, np.arange(len(x)), x).astype(np.float32)

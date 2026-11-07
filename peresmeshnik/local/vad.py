"""Silero VAD (MIT) on onnxruntime, streaming: 32 ms frames, one thread, state kept between calls."""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from peresmeshnik.engines import SAMPLE_RATE

FRAME = 512  # 32 ms at 16 kHz, the model's native hop
CONTEXT = 64  # samples of the previous frame the model expects in front of each frame


def ort_options(threads: int):
    """onnxruntime session options for every engine: fixed threads, pools sleep between calls."""
    import onnxruntime as rt
    so = rt.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    so.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return so


def silero_path() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download('istupakov/silero-vad-onnx', 'silero_vad.onnx', revision='b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6'))


class SileroVad:
    """feed() yields ('start', pcm) with a pre-roll, then ('speech', pcm) frames, then ('end', pcm)
    after `min_silence_ms` below the threshold. The trailing silence is part of the segment."""

    def __init__(self, path: Path | None = None, threshold: float = 0.5, min_silence_ms: int = 300,
                 preroll_ms: int = 192, threads: int = 1):
        import onnxruntime as rt
        self._sess = rt.InferenceSession(str(path or silero_path()), ort_options(threads), providers=['CPUExecutionProvider'])
        self.threshold = threshold
        self.min_silence_frames = max(1, round(min_silence_ms * SAMPLE_RATE / 1000 / FRAME))
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, round(preroll_ms * SAMPLE_RATE / 1000 / FRAME)))
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._ctx = np.zeros(CONTEXT, np.float32)
        self._buf = np.zeros(0, np.float32)
        self._preroll.clear()
        self.speaking, self._silent = False, 0
        self.pos, self.busy_s = 0, 0.0  # samples run through the model, seconds spent in it

    def prob(self, frame: np.ndarray) -> float:
        t = time.perf_counter()
        x = np.concatenate([self._ctx, frame])[None]
        out, self._state = self._sess.run(['output', 'stateN'], {'input': x, 'state': self._state, 'sr': np.array(SAMPLE_RATE, np.int64)})
        self._ctx = frame[-CONTEXT:]
        self.pos += len(frame)
        self.busy_s += time.perf_counter() - t
        return float(out[0, 0])

    def feed(self, pcm: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
        buf = np.concatenate([self._buf, pcm.astype(np.float32, copy=False)])
        n = len(buf) // FRAME * FRAME
        self._buf = buf[n:]
        for frame in buf[:n].reshape(-1, FRAME):
            p = self.prob(frame)
            if not self.speaking:
                if p >= self.threshold:
                    self.speaking, self._silent = True, 0
                    yield 'start', np.concatenate([*self._preroll, frame])
                    self._preroll.clear()
                else:
                    self._preroll.append(frame)
                continue
            self._silent = self._silent + 1 if p < self.threshold - 0.15 else 0
            if self._silent >= self.min_silence_frames:
                self.speaking = False
                yield 'end', frame
            else:
                yield 'speech', frame

"""Echo of our own speech, found by its reference: we know what we played, so where it comes back it is not
speech to translate. One module for both pairs:

- the call side (them -> me): our translation goes into the call through the virtual microphone; a far end on a
  loudspeaker sends it back 50-300 ms later and quieter. Reference: `VirtualMic.play`; gate: on the call capture.
- my microphone (me -> them): without headphones the translation from the speakers comes back into my microphone.
  Reference: `Headphones.play`; gate: on the microphone.

`EchoGate` compares the band envelopes of the capture with those of the reference over a range of delays (a
correlation, no adaptive filter: robust to codecs, loudspeakers and a drifting delay). A 16 ms block that is the
reference's echo and nothing more (no energy above the echo predicted from the reference) is zeroed: the VAD sees
silence. Where the other person talks over our echo (double talk) the block passes as it is. No dependencies;
while the reference is silent the gate does nothing (idle CPU ~ 0). Time is `time.perf_counter()` everywhere.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import numpy as np

from peresmeshnik.audio.resample import Resampler
from peresmeshnik.engines import SAMPLE_RATE as SR

clock = time.perf_counter
HOP = 256  # 16 ms blocks
_EDGES = np.unique(np.geomspace(3, 97, 17).round().astype(int))  # FFT bins of ~log-spaced bands, 190-6000 Hz
_BANDS = np.zeros((len(_EDGES) - 1, HOP // 2 + 1), np.float32)
for _i, (_lo, _hi) in enumerate(zip(_EDGES[:-1], _EDGES[1:])):
    _BANDS[_i, _lo:_hi] = 1 / (_hi - _lo)
_WIN = np.hanning(HOP).astype(np.float32)
FLOOR = -9.0  # log10 band power of silence (speech at -26 dBFS is ~ -4 .. -6 per band)
LOUD = FLOOR + 2.0  # a block of the reference that can make an echo
SEARCH_EVERY = 4  # blocks between delay searches (64 ms): the echo path changes slower than that
GAP_S = 0.5  # audio this much later than its samples counted: the capture lost samples (restarted), its clock resyncs


def bands(blocks: np.ndarray) -> np.ndarray:
    """(n, HOP) PCM -> (n, 16) log10 band power."""
    p = np.abs(np.fft.rfft(blocks * _WIN, axis=-1)) ** 2 / HOP
    return np.log10(p @ _BANDS.T + 10 ** FLOOR)


class Reference:
    """What we played, mono 16 kHz, the last `seconds`, on the clock: `push(pcm, sr, t_end)` from the output's
    play() (`t_end`: when its last sample plays; None: right after the audio pushed before, or now), `window(t0, t1)`
    the PCM of that time (zeros where nothing played), `features(times)` the bands of the blocks at those times.
    A TTS hands out seconds of speech at once: the ring also holds up to `ahead_s` queued before it plays."""

    def __init__(self, seconds: float = 4.0, ahead_s: float = 30.0):  # 4 s back: a gate's delays up to 2.8 s
        self.n = int((seconds + ahead_s) * SR) // HOP * HOP
        self._pcm = np.zeros(self.n, np.float32)
        self._feat = np.full((self.n // HOP, len(_BANDS)), FLOOR, np.float32)
        self.t0 = clock()  # absolute sample 0
        self._end = 0  # absolute sample after the last one written
        self.loud_until = -np.inf  # time when the last loud block ended: the gate wakes up only around it
        self._rs: dict[int, Resampler] = {}
        self._lock = threading.Lock()

    def _at(self, t: float) -> int:
        return round((t - self.t0) * SR)

    def push(self, pcm: np.ndarray, sample_rate: int, t_end: float | None = None) -> None:
        with self._lock:  # the headphones play from two threads (speech, the live pass-through)
            x = (self._rs.get(sample_rate) or self._rs.setdefault(sample_rate, Resampler(sample_rate, SR)))(
                np.asarray(pcm, np.float32))
            a1 = max(self._end, self._at(clock())) + len(x) if t_end is None else self._at(t_end)
            a0, x = a1 - len(x[-self.n:]), x[-self.n:]
            lo = max(min(a0, self._end), a1 - self.n)  # a gap since the last push is silence
            if a1 <= lo:
                return
            idx = np.arange(lo, a1) % self.n
            seg = np.zeros(a1 - lo, np.float32)
            seg[max(a0, lo) - lo:] = x[max(a0, lo) - a0:]
            self._pcm[idx] = seg
            self._end = max(self._end, a1)
            b0, b1 = lo // HOP, -(-a1 // HOP)  # blocks touched, recomputed whole
            blocks = self._pcm[(np.arange(b0 * HOP, b1 * HOP) % self.n)].reshape(-1, HOP)
            f = bands(blocks)
            self._feat[np.arange(b0, b1) % len(self._feat)] = f
            loud = np.nonzero(f.max(axis=1) > LOUD)[0]
            if len(loud):
                self.loud_until = max(self.loud_until, self.t0 + (b0 + loud[-1] + 1) * HOP / SR)

    def cut(self) -> None:
        """The output was flushed (barge-in, mode switch): what was queued after now will not play."""
        with self._lock:
            now = self._at(clock())
            if self._end > now:
                self._pcm[np.arange(max(now, self._end - self.n), self._end) % self.n] = 0
                self._feat[np.arange(now // HOP + 1, -(-self._end // HOP)) % len(self._feat)] = FLOOR
                self._end = now
                self.loud_until = min(self.loud_until, self.t0 + now / SR)

    def window(self, t0: float, t1: float) -> np.ndarray:
        a0, a1 = self._at(t0), self._at(t1)
        with self._lock:
            a = np.arange(a0, a1)
            ok = (a < self._end) & (a >= self._end - self.n)
            return np.where(ok, self._pcm[a % self.n], 0).astype(np.float32)

    def features(self, times: np.ndarray) -> np.ndarray:
        """Bands of the blocks that played at `times` (any shape), FLOOR where nothing known."""
        b = np.floor((np.asarray(times) - self.t0) * SR / HOP).astype(np.int64)
        with self._lock:
            hi = self._end // HOP  # the last block may be partly written: it counts
            ok = (b <= hi) & (b > (self._end - self.n) // HOP)
            return np.where(ok[..., None], self._feat[b % len(self._feat)], FLOOR)


class EchoGate:
    """`process(pcm, t_end)`: capture frames (16 kHz, any length; `t_end`: when the last sample was captured)
    -> the same frames, zeroed where they are only the echo of `ref`. `gated` is the last decision, `stats`
    counts blocks. The delay is searched in [min_delay_s, max_delay_s] over the last `window_s` of capture.
    One thread calls process() (a pipeline worker, never an audio callback)."""

    def __init__(self, ref: Reference, max_delay_s: float = 2.0, min_delay_s: float = -0.1, window_s: float = 1.0,
                 tail_s: float = 0.15, rho_on: float = 0.5, margin_db: float = 6.0, excess: float = 0.25):
        self.ref = ref
        self.delays = np.arange(round(min_delay_s * SR / HOP), round(max_delay_s * SR / HOP) + 1) * HOP / SR
        self.span = (min_delay_s, max_delay_s + window_s + tail_s)  # reference times that can matter now
        self.tail = np.arange(-1, round(tail_s * SR / HOP) + 1) * HOP / SR  # delay jitter and the far room's reverb
        self.rho_on, self.margin, self.excess = rho_on, margin_db / 10, excess
        self.min_blocks = round(0.2 * SR / HOP)
        self._hist: deque[tuple[float, np.ndarray]] = deque(maxlen=max(4, round(window_s * SR / HOP)))
        self._carry = np.zeros(0, np.float32)
        self._fed, self._offsets = 0, deque(maxlen=64)  # samples seen; arrival minus their duration, last frames
        self.gain: np.ndarray | None = None  # log10 echo / reference per band, learnt where the echo is clear
        self.gated, self.delay, self.rho = False, 0.0, 0.0
        self.stats = {'blocks': 0, 'gated': 0}

    def process(self, pcm: np.ndarray, t_end: float | None = None) -> np.ndarray:
        n = len(pcm)
        # captures hand audio on in bursts (parec: several frames at once): the audio's own clock is the samples
        # counted, put where they came with the least delay over the last ~2 s
        self._fed += n
        off = (clock() if t_end is None else t_end) - self._fed / SR
        if self._offsets and off - min(self._offsets) > GAP_S:  # a gap (a helper restarted): the old times are off
            self._offsets.clear()
        self._offsets.append(off)
        t_end = min(self._offsets) + self._fed / SR
        if t_end - n / SR - self.span[1] > self.ref.loud_until:  # nothing we played can be in this audio
            self._hist.clear()
            self._carry = np.zeros(0, np.float32)
            self.gated = False
            return pcm
        x = np.concatenate([self._carry, pcm.astype(np.float32, copy=False)])
        k = len(x) // HOP
        self._carry = x[k * HOP:]
        out = pcm.copy()
        t_x0 = t_end - len(x) / SR  # time of x[0]
        if k:
            feats = bands(x[:k * HOP].reshape(k, HOP))
            for i, f in enumerate(feats):
                tc = t_x0 + (i + 0.5) * HOP / SR
                self._hist.append((tc, f))
                self.gated = self._decide(tc, f)
                self.stats['blocks'] += 1
                self.stats['gated'] += self.gated
                if self.gated:  # this block's samples that are in pcm (the carry's part was put out before)
                    a = i * HOP - (len(x) - n)
                    out[max(a, 0):max(a + HOP, 0)] = 0
        if self.gated and len(self._carry):  # the start of the next block: the decision holds until it is known
            out[max(n - len(self._carry), 0):] = 0  # a frame shorter than the carry is all of it
        return out

    def _decide(self, tc: float, f: np.ndarray) -> bool:
        if len(self._hist) < 4:
            return False
        if self.stats['blocks'] % SEARCH_EVERY == 0:
            self._search()
        if self.rho < self.rho_on or self.gain is None:
            return False
        echo = self.ref.features(tc - self.delay - self.tail).max(axis=0)  # (b,) what of ours can sound now
        if echo.max() <= LOUD:
            return False
        p, e = 10 ** f, 10 ** (echo + self.gain + self.margin)
        return float(np.maximum(p - e, 0).sum() / p.sum()) < self.excess

    def _search(self) -> None:
        """The delay where the capture follows the reference best (rho), and how loud the echo comes back."""
        t = np.array([h[0] for h in self._hist])
        X0 = np.stack([h[1] for h in self._hist])  # (w, b)
        R0 = self.ref.features(t[:, None] - self.delays[None, :])  # (w, d, b)
        # only the spectral shape: syllables make every band rise and fall together, in anyone's speech;
        # only where the reference sounds: silence -> speech in both (two onsets) is no evidence of an echo
        X, R = X0 - X0.mean(axis=1, keepdims=True), R0 - R0.mean(axis=2, keepdims=True)
        m = (R0.max(axis=2) > LOUD).astype(np.float32)  # (w, d)
        cnt = np.maximum(m.sum(axis=0), 1)  # (d,)
        Rm = R * m[:, :, None]
        sx, sr = np.einsum('wd,wb->db', m, X), Rm.sum(axis=0)  # (d, b)
        cov = np.einsum('wb,wdb->d', X, Rm) - (sx * sr).sum(axis=1) / cnt
        vx = m.T @ (X ** 2).sum(axis=1) - (sx ** 2).sum(axis=1) / cnt
        vr = np.einsum('wdb,wdb->d', Rm, R) - (sr ** 2).sum(axis=1) / cnt
        rho = cov / (np.sqrt(np.maximum(vx * vr, 0)) + 1e-9)
        rho[cnt < self.min_blocks] = 0
        j = int(np.argmax(rho))
        self.rho, self.delay = float(rho[j]), float(self.delays[j])
        if self.rho >= self.rho_on + 0.1:  # a clear echo: learn how loud it comes back
            loud = m[:, j] > 0
            g = np.median(X0[loud] - R0[loud, j], axis=0)
            self.gain = g if self.gain is None else 0.8 * self.gain + 0.2 * g


class Referenced:
    """An `AudioOut` whose speech also goes to `ref` (the output's queue plays it back to back from now)."""

    def __init__(self, out, ref: Reference):
        self.out, self.ref = out, ref

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        self.ref.push(pcm, sample_rate)
        self.out.play(pcm, sample_rate)

    def flush(self) -> None:
        self.ref.cut()
        self.out.flush()

    def __getattr__(self, name):  # stop(), resolve(), name, ...
        return getattr(self.out, name)


class Threaded:
    """`EchoGate` on its own thread between an audio callback (it only enqueues: `self(pcm)`) and `on_frames`.
    If the gate fails, the audio passes ungated (logged once): an echo is better than a deaf side. If `on_frames`
    fails on a frame, that frame is lost (logged once) and the next ones go on."""

    def __init__(self, gate: EchoGate, on_frames):
        self.gate, self.on_frames, self._failed, self._lost = gate, on_frames, False, 0
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name='echo-gate', daemon=True)
        self._thread.start()

    def __call__(self, pcm: np.ndarray) -> None:
        self._q.put((clock(), pcm))

    def _run(self) -> None:
        while (item := self._q.get()) is not None:
            t, pcm = item
            if not self._failed:
                try:
                    pcm = self.gate.process(pcm, t)
                except Exception:
                    logging.getLogger('peresmeshnik').exception('echo gate failed: the audio passes ungated')
                    self._failed = True
            try:
                self.on_frames(pcm)
            except Exception:
                self._lost += 1
                if self._lost == 1:
                    logging.getLogger('peresmeshnik').exception('after the echo gate: a frame was lost')

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()

"""The proposed 0.3.2 latency fixes measured without changing the app: a variant patches the app in this process, then
runs an existing bench unchanged (research-1116 R1; quick-latency.md options 1, 2, 4, 5).

    python bench/latency_variants.py b|c|d me_them.py run ARGS...    # me -> them
    python bench/latency_variants.py c|d their_voice.py ARGS...      # them -> me (it is sped up in 0.3.1 already)

  a  0.3.1 as is: run the bench itself
  b  me -> them sped up when behind as them -> me is (from pipeline.FAST_AFTER_S 0.5 s linearly to MAX_SPEED 1.4x at
     MAX_LAG_S 4 s; the clone through voices/live.faster, WSOLA, as TheirVoice), never dropped
  c  b + local/asr.TAIL_S 0.5 (0.8) + Silero min_silence_ms 200 (300)
  d  c + each clause's synthesized trailing silence cut to KEEP_S before the queue
me_them.py rows go to bench/results/<BENCH_HOST>-<variant>.jsonl.
"""
import os
import runpy
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
sys.path[:0] = [str(BENCH.parent), str(BENCH)]
from peresmeshnik import pipeline  # noqa: E402
from peresmeshnik.local import asr, vad  # noqa: E402
from peresmeshnik.voices import live  # noqa: E402

KEEP_S, FLOOR_DB = 0.12, 30.0  # trailing silence kept; 20 ms frames this far below the clause's loudest are silence
TRIMMED = []  # seconds cut per clause (variant d), printed at exit


class NoDrop(float):
    """me -> them's max_lag_s: the speed ramp reads 4.0, but the drop test `lag > max_lag_s` is never true (Python
    tries a float subclass's reflected __lt__ first)."""

    def __lt__(self, other):
        return False


def trim(chunks, sr):
    """The chunks as they come, but the quiet after the last loud 20 ms frame held back; at the end at most KEEP_S of
    it. ponytail: 20 ms frames re-cut from each chunk's start, fine for a tail."""
    f, top, held = sr // 50, 0.0, np.zeros(0, np.float32)
    for c in chunks:
        x = np.concatenate([held, np.asarray(c, np.float32)])
        n = len(x) // f
        rms = np.sqrt(np.mean(np.square(x[:n * f].reshape(n, f)), axis=1)) if n else np.zeros(0)
        top = max(top, float(rms.max(initial=0.0)))
        loud = np.nonzero(rms > top * 10 ** (-FLOOR_DB / 20))[0] if top else []
        if len(loud):
            end = (int(loud[-1]) + 1) * f
            yield x[:end]
            held = x[end:]
        else:
            held = x
    keep = held[:int(KEEP_S * sr)]
    TRIMMED.append(round((len(held) - len(keep)) / sr, 3))
    if len(keep):
        yield keep


class SpeedTts:
    """A clone engine with the pipeline's `speed`: set by Fast below just before each synth()."""

    def __init__(self, tts):
        self.tts, self.sample_rate, self.speed = tts, tts.sample_rate, 1.0

    def __getattr__(self, name):
        return getattr(self.tts, name)

    def synth(self, text, lang, voice=None):
        return live.faster(self.tts.synth(text, lang, voice), self.speed, self.sample_rate)


def main():
    variant, script, *args = sys.argv[1:]
    assert variant in ('b', 'c', 'd'), variant
    os.environ['BENCH_HOST'] = f"{os.environ.get('BENCH_HOST', 'local')}-{variant}"
    if variant in 'cd':
        asr.TAIL_S = 0.5
        init = vad.SileroVad.__init__
        vad.SileroVad.__init__ = lambda self, *a, min_silence_ms=200, **kw: init(self, *a, min_silence_ms=min_silence_ms, **kw)
    if variant == 'd':
        faster = live.faster
        live.faster = lambda chunks, speed, sr=live.OUT_SR: trim(faster(chunks, speed, sr), sr)
        import atexit
        atexit.register(lambda: print(f'trimmed per clause, s: {TRIMMED} (total {sum(TRIMMED):.2f})', flush=True))
    sys.argv = [script, *args]
    if script != 'me_them.py':
        runpy.run_path(str(BENCH / script), run_name='__main__')
        return
    import me_them
    make_local = pipeline.make_local

    def make(src, tgt, *a, **kw):
        d = make_local(src, tgt, *a, **kw)
        if kw.get('max_lag_s') is None:  # me -> them (them -> me passes MAX_LAG_S)
            d.max_lag_s = NoDrop(pipeline.MAX_LAG_S)
        return d
    pipeline.make_local = make

    class Fast(me_them.Marked):
        def __init__(self, tts):
            super().__init__(SpeedTts(tts))

        def synth(self, text, lang, voice=None, speed=1.0):
            self.tts.speed = speed
            yield from super().synth(text, lang, voice)
    me_them.Marked = Fast
    me_them.main()


if __name__ == '__main__':
    main()

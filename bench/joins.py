"""Clause joins (audio/joins.py): a long sentence of the bench set cut in two as `max_clause_s` cuts it (at a comma near
its middle, else between words), each part said by the app's Pocket engine (int8, 2 threads), played back to back. The
join metric of a research run of 2026-11-16 (not published): silence at the join (10 ms frames under -45 dBFS,
the end of part 1 + the start of part 2), clicks at it, and the loudness step (the first 0.4 s of part 2's speech against the last 0.4 s
of part 1's: 20 ms frames within 30 dB of each part's loudest). The pitch step is not touched by the change (not here).
`first_word_s`: from the start of part 1's synthesis to its first word, were it played from its first chunk (the
synthesized pause before the first word is cut now: the voice starts that much sooner).

    python bench/joins.py <ru|en> <before|after> [--model-dir DIR] [--wav DIR]

before: as 0.3.1 (each part ends with a full stop, as its prompt() made it; no Joins); after: as the pipeline now
says them (part 1 through joins.goes_on, both through one Joins). Voices: ref_<lang>.wav of the set and its two longest
clips (FLEURS / LibriSpeech readers; never a user's voice). A row per voice to bench/results/<host>.jsonl.
"""
import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the package from this checkout
from common import SET, manifest, record  # noqa: E402

SIL_DB, LOOK_S = -45.0, 0.4


def split(t):
    w = t.split()
    mid = len(w) // 2
    commas = [i for i, x in enumerate(w[:-2]) if x.endswith(',') and i >= 2]
    i = min(commas, key=lambda i: abs(i - mid)) if commas else mid - 1
    return ' '.join(w[:i + 1]), ' '.join(w[i + 1:])


def edge(x, sr, end):
    f = sr // 100
    n = len(x) // f
    db = 20 * np.log10(np.sqrt(np.mean(x[:n * f].reshape(n, f) ** 2, 1)) + 1e-9)
    loud = np.nonzero(db > SIL_DB)[0]
    return ((n - 1 - loud[-1]) if end else loud[0]) * 0.01 if len(loud) else 0.0


def voiced(y, sr):
    f = sr // 50
    n = len(y) // f
    fr = y[:n * f].reshape(n, f)
    e = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    return fr[e > e.max() - 30].reshape(-1)


def click(p1, p2, sr):
    """The largest sample step within 5 ms of the join against the pair's 99th percentile one (> 1: a click)."""
    x, j, w = np.concatenate([p1, p2]), len(p1), int(0.005 * sr)
    d = np.abs(np.diff(x))
    return float(d[max(j - w, 0):j + w].max() / (np.percentile(d, 99) + 1e-9))


def db(y):
    return float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lang', choices=('ru', 'en'))
    ap.add_argument('mode', choices=('before', 'after'))
    ap.add_argument('--model-dir', type=Path, help='default: the registry folder of pocket-tts-<lang>')
    ap.add_argument('--wav', type=Path, help='write each pair here')
    a = ap.parse_args()
    from peresmeshnik import models, voice
    from peresmeshnik.audio.joins import Joins, goes_on
    from peresmeshnik.local.clone import make_engine
    engine = f'pocket-tts-{a.lang}'
    eng = make_engine(engine, a.model_dir or models.path(engine), threads=2)
    eng._rng = np.random.default_rng(1)  # the same draws before and after
    sr, n = eng.sample_rate, round(LOOK_S * eng.sample_rate)
    sents = [m[a.lang]['text'] for m in sorted(manifest(), key=lambda m: -len(m[a.lang]['text']))[:8]]
    clips = sorted((SET / a.lang).glob('*.wav'), key=lambda p: -sf.info(p).duration)[:2]
    root = Path(tempfile.mkdtemp(prefix='joins-'))
    for vi, path in enumerate([SET / f'ref_{a.lang}.wav', *clips]):
        x, xsr = sf.read(path, dtype='float32')
        v = voice.save(x, xsr, f'v{vi}', a.lang, root=root)
        eng.state(v)
        joins, rows = Joins(), []
        for k, t in enumerate(sents):
            c1, c2 = split(t)
            parts, heard = [], []
            for text, cut in ((c1, True), (c2, False)):
                t0, first, out = time.perf_counter(), None, []
                if a.mode == 'before':  # 0.3.1's prompt(): a comma at the end became a full stop, a word got one
                    text = text if text[-1] in '.!?…' else text.rstrip(',;:-–— ') + '.'
                elif cut:
                    text = goes_on(text)
                chunks = eng.synth(text, a.lang, v)
                if a.mode == 'after':
                    chunks = joins(chunks, sr, cut)
                for c in chunks:
                    first = first or time.perf_counter() - t0
                    out.append(np.asarray(c, np.float32))
                parts.append(np.concatenate(out))
                heard.append(first + edge(parts[-1], sr, False))  # its first word, played from its first chunk
            p1, p2 = parts
            if a.wav:
                a.wav.mkdir(parents=True, exist_ok=True)
                sf.write(a.wav / f'{a.lang}-{a.mode}-v{vi}-{k}.wav', np.concatenate(parts), sr, subtype='PCM_16')
            rows.append({'gap_s': edge(p1, sr, True) + edge(p2, sr, False),
                         'loud_step_db': db(voiced(p2, sr)[:n]) - db(voiced(p1, sr)[-n:]),
                         'first_word_s': heard[0],  # a clause that starts a sentence: from the synthesis start
                         'click': click(p1, p2, sr)})
        med = {k: round(statistics.median(r[k] for r in rows), 3) for k in rows[0]}
        record('joins', engine, mode=a.mode, voice=path.name, n=len(rows), gap_s_median=med['gap_s'],
               loud_step_db_median=med['loud_step_db'],
               loud_step_db_abs_median=round(statistics.median(abs(r['loud_step_db']) for r in rows), 2),
               rows=[{k: round(v, 3) for k, v in r.items()} for r in rows])


if __name__ == '__main__':
    main()

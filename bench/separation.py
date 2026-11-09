"""Stage 5 spike: two people at once at the far end through a separation model before the pipeline.

The call comes as one mixed channel: where two people talk at once the ASR hears a mix (bench/call.py overlap).
TIGER (Xu, Li et al., ICLR 2025; Apache-2.0, https://huggingface.co/JusperLee/TIGER-speech, ~0.8 M parameters,
16 kHz, trained on EchoSet with reverberation and noise) splits the mix into two streams; each goes through the
local pipeline as its own direction, and the clauses of both are scored as `call.py overlap`.

    # 1. in a venv with torch and the TIGER repo (github.com/JusperLee/TIGER): mix -> two streams per scenario
    python bench/separation.py separate --tiger DIR --out SEP --n 6 [--device cuda|cpu]
    # 2. in the app's venv: the mix and the two streams through the pipeline
    python bench/separation.py score SEP

Duplicated: content words recognised in both streams for one utterance (the other stream's leak would be
translated twice).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 16000


def separate(a):
    import numpy as np
    import soundfile as sf
    import torch
    sys.path.insert(0, a.tiger)
    import look2hear.models
    from call import fleurs, make
    torch.set_num_threads(a.threads)
    model = look2hear.models.TIGER.from_pretrained('JusperLee/TIGER-speech', cache_dir=str(Path(a.out) / 'cache'))
    model.to(a.device).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = fleurs('en')
    stats = []
    for seed in range(a.seed, a.seed + a.n):
        sc = make('overlap', seed, pool)
        x = sc.capture
        t = time.perf_counter()
        parts = []
        with torch.no_grad():
            for i in range(0, len(x), int(a.chunk * SR)):  # ponytail: chunks without overlap; streams may swap at a seam
                y = model(torch.from_numpy(x[i:i + int(a.chunk * SR)]).to(a.device)[None, None])
                parts.append(y[0].cpu().numpy())  # (2, n)
        dt = time.perf_counter() - t
        y = np.concatenate(parts, axis=1)
        sf.write(out / f'{seed}.mix.wav', x, SR)
        for k in range(2):
            s = y[k] * (0.05 / max(float(np.sqrt(np.mean(y[k] ** 2))), 1e-6))
            sf.write(out / f'{seed}.s{k}.wav', np.clip(s, -1, 1).astype(np.float32), SR)
        (out / f'{seed}.json').write_text(json.dumps(sc.segs, ensure_ascii=False), 'utf-8')
        from common import peak_rss_mb
        stats.append({'seed': seed, 'audio_s': round(len(x) / SR, 1), 'rtf': round(dt / (len(x) / SR), 4), 'peak_rss_mb': peak_rss_mb()})
        print(json.dumps(stats[-1]), flush=True)
    (out / 'separate.json').write_text(json.dumps({'device': a.device, 'threads': a.threads, 'chunk_s': a.chunk,
                                                   'runs': stats}), 'utf-8')


def score(a):
    from call import NoTts, clauses, overlap_score, record

    from peresmeshnik.pipeline import make_local, read_audio
    base = make_local('en', 'ru', threads=a.threads, tts=NoTts())
    sep = Path(a.sep)
    out = {'mix': [], 'separated': []}
    merged = {'mix': 0, 'separated': 0}
    dup = words = 0
    for j in sorted(sep.glob('*.json')):
        if not j.stem.isdigit():
            continue
        segs = json.loads(j.read_text('utf-8'))
        rows = clauses(base, read_audio(str(sep / f'{j.stem}.mix.wav')))
        per, _, m = overlap_score(segs, rows)
        out['mix'] += [(s['kind'], sum(p.values()) / max(len(p), 1)) for s, p in zip(segs, per)]
        merged['mix'] += m
        streams = [clauses(base, read_audio(str(sep / f'{j.stem}.s{k}.wav'))) for k in range(2)]
        per, _, m = overlap_score(segs, streams[0] + streams[1])
        out['separated'] += [(s['kind'], sum(p.values()) / max(len(p), 1)) for s, p in zip(segs, per)]
        merged['separated'] += m
        each = [overlap_score(segs, r)[0] for r in streams]  # per stream alone: a word out in both is a duplicate
        dup += sum(a_[w] and b_[w] for a_, b_ in zip(*each) for w in a_)
        words += sum(len(p) for p in per)
        print(j.stem, 'lost', sum(x < 0.5 for _, x in out['mix'][-len(segs):]), '->',
              sum(x < 0.5 for _, x in out['separated'][-len(segs):]), 'of', len(segs), flush=True)
    meta = json.loads((sep / 'separate.json').read_text('utf-8'))
    res = {}
    for name, xs in out.items():
        res[name] = {'utterances': len(xs), 'lost': sum(x < 0.5 for _, x in xs), 'merged_clauses': merged[name],
                     'recall': round(sum(x for _, x in xs) / max(len(xs), 1), 3),
                     'by_kind': {k: {'n': sum(1 for q, _ in xs if q == k), 'lost': sum(1 for q, x in xs if q == k and x < 0.5),
                                     'recall': round(sum(x for q, x in xs if q == k) / max(sum(1 for q, _ in xs if q == k), 1), 3)}
                                 for k in sorted({q for q, _ in xs})}}
    record('separation', 'tiger-speech', scenario='overlap', separate=meta, duplicated_words=round(dup / max(words, 1), 3), **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['separate', 'score'])
    ap.add_argument('sep', nargs='?', help='score: the folder `separate` wrote')
    ap.add_argument('--tiger', help='separate: the TIGER repo (its look2hear package)')
    ap.add_argument('--out', help='separate: folder for the streams')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--chunk', type=float, default=10.0, help='separate: seconds per model call')
    a = ap.parse_args()
    globals()[a.mode](a)


if __name__ == '__main__':
    main()

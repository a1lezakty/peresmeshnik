"""Stage 5: the me -> them voice as the call hears it. My turns (FLEURS ru dev sentences, other people's recordings at
-26 dBFS, pauses of 0.6-1.2 s) go in real time through the app's path: LocalDirection (VAD, GigaAM, OPUS-MT ru-en, the
clone) -> LiveSwitch -> a device that plays in real time into memory (the stream the virtual mic gets, 24 kHz). Kept:
where each clause and each TTS chunk landed in that stream.

    python bench/me_voice.py input [--n 30]                                    # $BENCH_DATA/me_voice/input.{wav,json}
    python bench/me_voice.py run --voice DIR --model-dir DIR --variant NAME    # GPU box: $BENCH_OUT/me_voice/NAME/
    python bench/me_voice.py score --variant NAME                              # tts-eval venv: one `me_voice` row
    python bench/me_voice.py tails --from NAME --voice DIR --model-dir DIR     # GPU box: its clause texts again, k times,
                                                                               # as they are / ending in ',' / '.'
    python bench/me_voice.py tails-score --from NAME                           # tts-eval venv: tails and CER per ending

`score`: per clause ECAPA similarity to the voice's recording and Whisper-small CER against its text (words after the
text: a tail), integrated loudness (BS.1770, pyloudnorm) against my own speech's, clicks at the seams (a clause's edges,
the TTS chunks inside it), holes inside a clause (its next audio was not there in time), pauses between clauses.
"""
import argparse
import bisect
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the package from this checkout
from common import ROOT, record  # noqa: E402

OUT = Path(os.environ.get('BENCH_OUT', ROOT.parent / 'out')) / 'me_voice'
IN = ROOT / 'me_voice'
SR_IN = 16_000
QUIET_DB = -45.0  # 10 ms frames below this (dBFS) are silence: pauses, a clause's own lead and tail
EDGE_DB = -40.0  # a clause edge that steps from silence by more than this (dB under the clause's peak) clicks
CLICK = 2.0  # seam_jump above this: a step the signal around does not make


# --- input

def make_input(a):
    import soundfile as sf
    from call import fleurs, level

    from peresmeshnik.pipeline import read_audio
    rng = np.random.default_rng(a.seed)
    seen, parts, segs, t = set(), [np.zeros(SR_IN, np.float32)], [], 1.0
    for i in rng.permutation(len(pool := fleurs('ru'))):
        wav, text = pool[i]
        if text in seen:  # one recording per sentence
            continue
        seen.add(text)
        x = level(read_audio(wav))
        gap = np.zeros(int(rng.uniform(0.6, 1.2) * SR_IN), np.float32)
        parts += [x, gap]
        segs.append({'start': round(t, 3), 'end': round(t + len(x) / SR_IN, 3), 'text': text})
        t += (len(x) + len(gap)) / SR_IN
        if len(segs) == a.n:
            break
    IN.mkdir(parents=True, exist_ok=True)
    sf.write(IN / 'input.wav', np.concatenate(parts + [np.zeros(2 * SR_IN, np.float32)]), SR_IN)
    (IN / 'input.json').write_text(json.dumps(segs, ensure_ascii=False, indent=0), encoding='utf-8')
    print(IN / 'input.wav', f'{t:.0f} s', len(segs), 'turns')


# --- run (the GPU box)

class Traced:
    """The clone, remembering each utterance's text, chunk lengths and timing (speech reaches LiveSwitch in this order)."""

    def __init__(self, tts):
        self.tts, self.sample_rate, self.said = tts, tts.sample_rate, []

    def __getattr__(self, name):
        return getattr(self.tts, name)

    def synth(self, text, lang, voice=None, **kw):
        u = {'text': text, 'chunks': [], 't0': time.perf_counter(), 'arrive': []}
        self.said.append(u)
        for c in self.tts.synth(text, lang, voice, **kw):
            u['chunks'].append(len(c))
            u['arrive'].append(round(time.perf_counter() - u['t0'], 3))
            yield c


class Sink:
    """A device that plays in real time into memory, back to back like a real one; play() returns where it landed."""

    def __init__(self, sr: int, seconds: float):
        self.sr, self.t0, self.end, self.x = sr, time.perf_counter(), 0.0, np.zeros(int(seconds * sr), np.float32)

    def play(self, pcm: np.ndarray) -> int:
        start = max(time.perf_counter(), self.end)
        i = int((start - self.t0) * self.sr)
        self.x[i:i + len(pcm)] = pcm[:max(0, len(self.x) - i)]
        self.end = start + len(pcm) / self.sr
        return i


def trace(sw):
    """The app's LiveSwitch, noting for each output frame its lead and the speech samples it took (`sw.trace`)."""
    step, sw.trace = sw._step, []

    def traced(pcm):
        fresh, before = time.monotonic() >= sw._out_end, len(sw._buf)
        out = step(pcm)
        if out is not None:
            sw.trace.append((sw.lead if fresh else 0, before - len(sw._buf)))
        return out
    sw._step = traced
    return sw


def play(sw, sink: Sink, events: list, captions: list | None = None) -> None:
    """The app's consumer of the me -> them side: speech into the device (`events`: where each output frame landed,
    its lead, the speech samples it took); final captions kept."""
    from peresmeshnik.engines import Caption, Speech
    k = 0
    for e in sw.events():
        if isinstance(e, Speech):
            i = sink.play(e.pcm)
            lead, took = sw.trace[k] if k < len(sw.trace) else (0, len(e.pcm))  # a stall: the rest unpaced
            k += k < len(sw.trace)
            events.append((i, lead, took))
        elif isinstance(e, Caption) and e.final and captions is not None:
            captions.append({'clause': e.clause, 'original': e.original, 'translation': e.translation})


def run(a):
    import logging

    import soundfile as sf

    from peresmeshnik import pipeline, voice
    from peresmeshnik.audio.mic import LiveSwitch
    from peresmeshnik.local.clone import XttsClone

    rows = []

    class Keep(logging.Handler):
        def emit(self, r):
            rows.append(json.loads(r.getMessage()))
    pipeline.log.addHandler(Keep())
    pipeline.log.setLevel(logging.INFO)
    if a.cuda_frac:
        import torch
        torch.cuda.set_per_process_memory_fraction(a.cuda_frac)
    tts = Traced(XttsClone(a.model_dir, a.device, a.threads))
    v = voice.load(a.voice.name, a.voice.parent)
    d = pipeline.make_local('ru', 'en', threads=a.threads, tts=tts, voice=v)  # as the app's me -> them (no queue cap)
    d.warm_up()
    tts.said.clear()
    d.realtime = True
    sw = trace(LiveSwitch(d, tts.sample_rate))
    pcm = pipeline.read_audio(str(IN / 'input.wav'))
    sink = Sink(tts.sample_rate, len(pcm) / SR_IN + 120)
    events, captions = [], []  # (where it landed, its lead, TTS samples it took); final captions

    th = threading.Thread(target=play, args=(sw, sink, events, captions))
    th.start()
    pipeline.run(sw, np.concatenate([pcm, np.zeros(a.tail_s * SR_IN, np.float32)]), realtime=True)  # the mic goes on
    d.finish()
    th.join()
    time.sleep(max(0.0, sink.end - time.perf_counter()))
    out = OUT / a.variant
    out.mkdir(parents=True, exist_ok=True)
    n = int((sink.end - sink.t0) * sink.sr) + sink.sr
    sf.write(out / 'stream.wav', sink.x[:n], sink.sr, subtype='FLOAT')
    np.save(out / 'events.npy', np.array(events, np.int64).reshape(-1, 3))
    meta = {'variant': a.variant, 'sr': sink.sr, 'device': tts.device, 'said': [{k: u[k] for k in ('text', 'chunks', 'arrive')}
                                                                               for u in tts.said],
            'rows': rows, 'captions': captions, 'summary': d.summary()}
    (out / 'run.json').write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding='utf-8')
    d.close()
    print(out, len(tts.said), 'utterances', len(events), 'output frames')


# --- score (tts-eval venv)

def db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(x, 1e-12))


def frames_db(x: np.ndarray, sr: int, ms: int = 10) -> np.ndarray:
    f = sr * ms // 1000
    n = len(x) // f
    return db((x[:n * f].reshape(n, f) ** 2).mean(axis=1))


def mapping(events: np.ndarray):
    """TTS sample index -> stream index, from the output frames (each: where it landed, its lead, samples taken)."""
    starts = np.cumsum(np.r_[0, events[:-1, 2]])  # TTS index of each frame's first sample

    def at(j: int) -> int:
        e = bisect.bisect_right(starts, j) - 1
        while events[e, 2] == 0:  # a frame of only silence (cannot hold j)
            e -= 1
        return int(events[e, 0] + events[e, 1] + j - starts[e])
    return at, starts


def holes(events: np.ndarray, starts: np.ndarray, a: int, b: int, sr: int) -> list[float]:
    """Silences inside TTS samples [a, b) of one clause: its next audio came after the stream had played out."""
    out, prev = [], None
    for e in np.nonzero((starts < b) & (starts + events[:, 2] > a) & (events[:, 2] > 0))[0]:
        s = events[e, 0] + events[e, 1]
        if prev is not None and s - prev > sr // 1000:
            out.append((s - prev) / sr)
        prev = s + events[e, 2]
    return out


def seam_jump(x: np.ndarray, p: int, sr: int) -> float:
    """A step at sample p: the biggest sample-to-sample change there (±1 sample) against the 99th percentile of those
    around it (±10 ms). A smooth signal stays <= ~1.1; a splice of speech at another phase is 2-20 (CLICK)."""
    w = sr // 100
    lo = max(0, p - w)
    d = np.abs(np.diff(x[lo:p + w]))
    c = p - lo - 1
    near, rest = d[max(0, c - 1):c + 2], np.r_[d[:max(0, c - 1)], d[c + 2:]]
    return float(near.max() / max(np.percentile(rest, 99), 1e-4)) if len(near) and len(rest) else 0.0


def edges(c: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """A clause's own silence before and after its sound (s), and the step at each edge (dB under its peak)."""
    f = frames_db(c, sr)
    loud = np.nonzero(f > QUIET_DB)[0]
    peak = max(float(np.abs(c).max()), 1e-6)
    step = lambda v: float(20 * np.log10(max(abs(v), 1e-7) / peak))  # noqa: E731
    if not len(loud):
        return 0.0, 0.0, -140.0, -140.0
    return loud[0] / 100, (len(f) - 1 - loud[-1]) / 100, step(c[0]), step(c[-1])


def loudness(x: np.ndarray, sr: int) -> float | None:
    import pyloudnorm as pyln
    if len(x) < sr // 2:
        return None
    v = pyln.Meter(sr).integrated_loudness(x.astype(np.float64))
    return round(float(v), 2) if np.isfinite(v) else None


def tail_words(ref: str, hyp: str) -> int:
    """Words the judge heard after the last word of the text (XTTS's babble after the sentence)."""
    from tts_clone import norm
    r, h = norm(ref).split(), norm(hyp).split()
    if not r:
        return len(h)
    last = max((i for i, w in enumerate(h) if w == r[-1] or (len(w) > 3 and w[:4] == r[-1][:4])), default=None)
    return len(h) - 1 - last if last is not None else 0


def judge(meta, x, sr, at, ev, starts, voice, tmp) -> list[dict]:
    """Per clause: where it is in the stream, its edges, seams, holes, loudness, ECAPA similarity, CER, tail."""
    import soundfile as sf
    from tts_clone import Judge
    j = Judge()
    ref = j.emb(str(voice / 'reference.wav'))
    per, pos = [], 0
    for u in meta['said']:
        n = sum(u['chunks'])
        if not n:
            continue
        a0, b0 = at(pos), at(pos + n - 1) + 1
        c = x[a0:b0]
        lead, trail, step0, step1 = edges(c, sr)
        inner = [at(pos + k) for k in np.cumsum(u['chunks'])[:-1]]  # where the TTS chunks meet
        sf.write(tmp, c, sr)
        de, dn, hyp = j.cer(str(tmp), u['text'], 'en')
        per.append({'text': u['text'], 'start': a0 / sr, 'end': b0 / sr, 'dur_s': round((b0 - a0) / sr, 3),
                    'lead_s': lead, 'trail_s': trail, 'edge_in_db': round(step0, 1), 'edge_out_db': round(step1, 1),
                    'inner_jump': [round(seam_jump(x, p, sr), 2) for p in inner],
                    'holes_s': holes(ev, starts, pos, pos + n, sr), 'lufs': loudness(c, sr),
                    'sim': round(float(j.emb(str(tmp)) @ ref), 3), 'cer_e': de, 'cer_n': dn, 'hyp': hyp,
                    'tail_words': tail_words(u['text'], hyp)})
        print(json.dumps(per[-1], ensure_ascii=False), flush=True)
        pos += n
    tmp.unlink(missing_ok=True)
    return per


def score(a):
    import soundfile as sf
    out = OUT / a.variant
    meta = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    x, sr = sf.read(out / 'stream.wav', dtype='float32')
    ev = np.load(out / 'events.npy')
    at, starts = mapping(ev)
    my = sf.read(IN / 'input.wav', dtype='float32')[0]
    turns = json.loads((IN / 'input.json').read_text(encoding='utf-8'))
    for s in turns:  # my live voice: each turn's loudness
        s['lufs'] = loudness(my[int(s['start'] * SR_IN):int(s['end'] * SR_IN)], SR_IN)
    my_lufs = [s['lufs'] for s in turns if s['lufs'] is not None]
    cache = out / 'clauses.json'  # the judge's work per clause: kept, so the summary can be made again
    if cache.exists():
        per = json.loads(cache.read_text(encoding='utf-8'))
    else:
        per = judge(meta, x, sr, at, ev, starts, a.voice, out / 'clause.wav')
    cache.write_text(json.dumps(per, ensure_ascii=False), encoding='utf-8')
    f = frames_db(x, sr)
    gaps, seams, b2b_sent, b2b_frag = [], [], [], []  # silence between clauses i and i+1 in the stream; the jump there
    for p, q in zip(per, per[1:]):
        lo, hi = int(p['start'] * 100), int(q['end'] * 100)
        lp = lo + np.nonzero(f[lo:int(p['end'] * 100)] > QUIET_DB)[0]  # frames before the next clause starts
        lq = int(q['start'] * 100) + np.nonzero(f[int(q['start'] * 100):hi] > QUIET_DB)[0]
        if len(lp) and len(lq):
            gaps.append((lq[0] - lp[-1] - 1) / 100)
        if abs(q['start'] - p['end']) < 1e-3:  # back to back: one seam, the pause is the clauses' own silence
            seams.append(seam_jump(x, int(q['start'] * sr), sr))
            if len(lp) and len(lq):
                (b2b_sent if not fragment(p['text']) else b2b_frag).append(gaps[-1])
    pct = lambda v, q: round(float(np.percentile(v, q)), 3) if len(v) else None  # noqa: E731
    lufs = [p['lufs'] for p in per if p['lufs'] is not None]
    inner = [v for p in per for v in p['inner_jump']]
    hole = [h for p in per for h in p['holes_s']]
    edge = [p['edge_in_db'] for p in per] + [p['edge_out_db'] for p in per]
    rows = {r['translation']: r for r in meta['rows'] if 'translation' in r}
    voice_lag = [r['voice_lag_s'] for r in rows.values() if 'voice_lag_s' in r]
    jump = []  # a clause against the turn of mine it was said in (the hotkey's crossfade between the two)
    for p in per:
        t = rows.get(p['text'], {}).get('start_s')
        turn = next((s for s in turns if t is not None and s['start'] - 0.5 <= t < s['end']), None)
        if turn and turn['lufs'] is not None and p['lufs'] is not None:
            jump.append(p['lufs'] - turn['lufs'])
    row = record('me_voice', a.variant, device=meta['device'], clauses=len(per),
                 sim_mean=round(float(np.mean([p['sim'] for p in per])), 3), sim_p10=pct([p['sim'] for p in per], 10),
                 cer=round(sum(p['cer_e'] for p in per) / max(sum(p['cer_n'] for p in per), 1), 4),
                 tails=int(sum(p['tail_words'] > 0 for p in per)), tail_words=sum(p['tail_words'] for p in per),
                 lufs_p50=pct(lufs, 50), lufs_p10=pct(lufs, 10), lufs_p90=pct(lufs, 90),
                 lufs_spread=round(pct(lufs, 90) - pct(lufs, 10), 2) if lufs else None,
                 my_lufs_p50=pct(my_lufs, 50), jump_to_live_lu=round(pct(lufs, 50) - pct(my_lufs, 50), 2) if lufs and my_lufs else None,
                 live_jump_lu_p50=pct(jump, 50), live_jump_abs_lu_p90=pct(np.abs(jump), 90), live_jumps=len(jump),
                 edge_clicks=int(sum(e > EDGE_DB for e in edge)), edges=len(edge), edge_db_p90=pct(edge, 90),
                 inner_seams=len(inner), inner_clicks=int(sum(v > CLICK for v in inner)), inner_jump_p90=pct(inner, 90),
                 back_to_back=len(seams), seam_clicks=int(sum(v > CLICK for v in seams)),
                 holes=len(hole), holes_s=round(sum(hole), 3), clauses_with_holes=int(sum(bool(p['holes_s']) for p in per)),
                 lead_s_p50=pct([p['lead_s'] for p in per], 50), lead_s_p90=pct([p['lead_s'] for p in per], 90),
                 trail_s_p50=pct([p['trail_s'] for p in per], 50), trail_s_p90=pct([p['trail_s'] for p in per], 90),
                 gap_s_p10=pct(gaps, 10), gap_s_p50=pct(gaps, 50), gap_s_p90=pct(gaps, 90),
                 gaps_under_0_1=int(sum(g < 0.1 for g in gaps)), gaps=len(gaps),
                 pause_after_sentence_b2b_p50=pct(b2b_sent, 50), pause_after_sentence_b2b_min=min(b2b_sent, default=None),
                 pause_in_sentence_b2b_p50=pct(b2b_frag, 50), b2b_after_sentence=len(b2b_sent),
                 voice_lag_p50=pct(voice_lag, 50), voice_lag_p90=pct(voice_lag, 90),
                 tts_first_ms_p50=pct([r['tts_first_ms'] for r in rows.values() if 'tts_first_ms' in r], 50),
                 stream_peak=round(float(np.abs(x).max()), 3))
    (out / 'score.json').write_text(json.dumps({'row': row, 'clauses': per, 'gaps': gaps}, ensure_ascii=False, indent=0),
                                    encoding='utf-8')
    print(json.dumps(row, ensure_ascii=False))


END = {'as-is': lambda t: t, 'comma': lambda t: t if t.endswith(',') else t.rstrip(';:') + ',',
       'period': lambda t: t.rstrip(',;:') + '.'}


def fragment(text: str) -> bool:
    """A clause that ends without a sentence's punctuation (cut at 5 s or at a comma)."""
    return not re.search(r'[.!?…]["»”)]*$', text)


def tails(a):
    """XTTS's babble after the text, by how a fragment's text ends: as it is, with a comma, with a period."""
    import soundfile as sf

    from peresmeshnik import voice
    from peresmeshnik.local.clone import XttsClone
    tts = XttsClone(a.model_dir, a.device, a.threads)
    v = voice.load(a.voice.name, a.voice.parent)
    texts = [u['text'] for u in json.loads((OUT / a.src / 'run.json').read_text(encoding='utf-8'))['said'] if u['chunks']]
    out = OUT / a.src / 'tails'
    out.mkdir(exist_ok=True)
    items = []
    for i, text in enumerate(texts):
        for name, f in END.items() if fragment(text) else [('as-is', END['as-is'])]:
            for k in range(a.k):
                wav = np.concatenate(list(tts.synth(f(text), 'en', v)))
                path = out / f'{i:03d}-{name}-{k}.wav'
                sf.write(path, wav, tts.sample_rate)
                items.append({'i': i, 'end': name, 'k': k, 'text': f(text), 'fragment': fragment(text), 'wav': path.name,
                              'dur_s': round(len(wav) / tts.sample_rate, 3)})
        print(i, text, flush=True)
    (out / 'items.json').write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding='utf-8')


def tails_score(a):
    from tts_clone import Judge
    out = OUT / a.src / 'tails'
    items, j = json.loads((out / 'items.json').read_text(encoding='utf-8')), Judge()
    for x in items:
        x['cer_e'], x['cer_n'], x['hyp'] = j.cer(str(out / x['wav']), x['text'], 'en')
        x['tail_words'] = tail_words(x['text'], x['hyp'])
    (out / 'scored.json').write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding='utf-8')
    groups = [('sentences', 'as-is', False)] + [('fragments', name, True) for name in END]
    for part, name, frag in groups:
        xs = [x for x in items if x['end'] == name and x['fragment'] == frag]
        record('me_voice_tails', name, part=part, src=a.src, n=len(xs), tails=sum(x['tail_words'] > 0 for x in xs),
               tail_words=sum(x['tail_words'] for x in xs),
               cer=round(sum(x['cer_e'] for x in xs) / max(sum(x['cer_n'] for x in xs), 1), 4),
               dur_s=round(sum(x['dur_s'] for x in xs), 1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('input')
    i.add_argument('--n', type=int, default=30, help='turns (FLEURS ru dev sentences, one recording each)')
    i.add_argument('--seed', type=int, default=1)
    r = sub.add_parser('run')
    r.add_argument('--voice', type=Path, required=True, help='voice profile folder')
    r.add_argument('--model-dir', type=Path, required=True)
    r.add_argument('--device', default='cuda')
    r.add_argument('--threads', type=int, default=2)
    r.add_argument('--cuda-frac', type=float, help='CUDA memory cap (a shared GPU)')
    r.add_argument('--tail-s', type=int, default=20, help='silence fed after the input (the mic goes on)')
    r.add_argument('--variant', required=True)
    s = sub.add_parser('score')
    s.add_argument('--variant', required=True)
    s.add_argument('--voice', type=Path, required=True, help='voice profile folder (its reference.wav)')
    t = sub.add_parser('tails')
    t.add_argument('--from', dest='src', required=True, help='the run whose clause texts are said again')
    t.add_argument('--voice', type=Path, required=True)
    t.add_argument('--model-dir', type=Path, required=True)
    t.add_argument('--device', default='cuda')
    t.add_argument('--threads', type=int, default=2)
    t.add_argument('--k', type=int, default=3, help='times each (XTTS samples: a tail comes and goes)')
    ts = sub.add_parser('tails-score')
    ts.add_argument('--from', dest='src', required=True)
    a = ap.parse_args()
    {'input': make_input, 'run': run, 'score': score, 'tails': tails, 'tails-score': tails_score}[a.cmd](a)


if __name__ == '__main__':
    main()

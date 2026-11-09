"""Stage 5 call scenarios: recordings of different people glued into the sound of a call, as the app hears it,
then through the local pipeline (them -> me). Nobody uses the app live yet, so these stand in for it.

    echo     we speak into the call (our translation), the far end is on a loudspeaker: our speech comes back
             50-300 ms later and quieter, with their room in it; they answer, a third of the time over us
    overlap  two people at the far end: turns that overlap (0.3-1.5 s) and some that start together
    long     one person, sentences back to back with no pause the VAD would take (a monologue)

Data: FLEURS dev (CC-BY-4.0, $BENCH_DATA/fleurs, bench/data.py), each recording at -26 dBFS RMS. A scenario is a
seed: it is made again from the public recordings, nothing but numbers goes to the repo.

    python bench/call.py echo --n 6 [--gate] [--delay 0.15] [--gain -12] [--no-echo-path]
    python bench/call.py overlap --n 6
    python bench/call.py long --n 4 [--realtime]

`echo` scores the clauses against the scenario's truth: an utterance of ours that comes out as a clause (its
content words recognised) is an "echo translated"; their words recognised (recall) and their speech blocks the
gate zeroed are the price.
"""
import argparse
import csv
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
from peresmeshnik.pipeline import OTHER, LocalDirection, make_local, read_audio, run  # noqa: E402  first: it pins BLAS threads
import numpy as np  # noqa: E402
# isort: on

from conversation import NoTts, Rows, norm, pct  # noqa: E402
from data import CODES  # noqa: E402

from common import ROOT, record  # noqa: E402

SR = 16000


def level(x: np.ndarray, rms: float = 0.05) -> np.ndarray:  # -26 dBFS: a call app's AGC (as tests/data)
    return (x * (rms / max(float(np.sqrt(np.mean(x ** 2))), 1e-6))).astype(np.float32)


def fleurs(lang: str = 'en') -> list[tuple[str, str]]:
    """(wav, text) of every FLEURS dev recording: recordings of one sentence are by different people."""
    d = ROOT / 'fleurs' / lang
    return [(str(d / 'dev' / r[1]), r[2]) for r in csv.reader(open(d / 'dev.tsv', encoding='utf-8'), delimiter='\t',
                                                                quoting=csv.QUOTE_NONE)]


def far_end(x: np.ndarray, delay: float, gain_db: float, rng: np.random.Generator) -> np.ndarray:
    """Our speech as it comes back from a far end on a loudspeaker: a small speaker's band (300-3400 Hz), their
    room (reverb, ~0.2 s), `delay` s later and `gain_db` quieter."""
    n = 101
    t = np.arange(n) - n // 2
    lp = lambda f: 2 * f / SR * np.sinc(2 * f / SR * t)  # noqa: E731
    band = (lp(3400) - lp(300)) * np.hamming(n)
    room = np.zeros(int(0.25 * SR))
    room[0] = 1.0
    taps = rng.integers(int(0.005 * SR), len(room), 40)
    room[taps] += rng.normal(0, 0.4, len(taps)) * np.exp(-taps / SR / 0.06)
    y = np.convolve(np.convolve(x, band), room)
    y *= 10 ** (gain_db / 20) * np.sqrt(np.mean(x ** 2) / max(np.mean(y ** 2), 1e-12))
    return np.concatenate([np.zeros(int(delay * SR), np.float32), y.astype(np.float32)])


def squeeze(x: np.ndarray, keep_s: float = 0.15, frame: int = 320) -> np.ndarray:
    """Silences (20 ms frames 40 dB under the loudest) at the ends dropped, inside cut down to `keep_s`."""
    n = len(x) // frame
    e = np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = e > e.max() * 10 ** (-40 / 20)
    keep, run = [], 0
    for i in range(int(np.argmax(loud)), n - int(np.argmax(loud[::-1]))):
        run = 0 if loud[i] else run + 1
        if run * frame <= keep_s * SR:
            keep.append(i)
    return x[:n * frame].reshape(n, frame)[keep].ravel()


class Scenario:
    """`capture`: the call sound; `ref`: what we sent into the call (our speech, when we sent it); `segs`: who
    said what when in the capture (ours: when its echo is heard)."""

    def __init__(self, seconds: float):
        self.capture = np.zeros(int(seconds * SR), np.float32)
        self.ref = np.zeros_like(self.capture)
        self.theirs = np.zeros_like(self.capture)  # their speech alone: the truth for false suppression
        self.segs: list[dict] = []

    def add(self, who: str, x: np.ndarray, at: float, text: str, echo: np.ndarray | None = None, delay: float = 0.0):
        i = int(at * SR)
        if who == 'ours':
            self.ref[i:i + len(x)] += x[:len(self.ref) - i]
            if echo is None:
                return
            x = echo
        else:
            self.theirs[i:i + len(x)] += x[:len(self.theirs) - i]
        self.capture[i:i + len(x)] += x[:len(self.capture) - i]
        start = at + (delay if who == 'ours' else 0.0)
        self.segs.append({'who': who, 'start': round(start, 2), 'end': round(start + len(x) / SR, 2), 'text': text})

    def trim(self) -> 'Scenario':
        n = int((max(s['end'] for s in self.segs) + 1.5) * SR)
        self.capture, self.ref, self.theirs = self.capture[:n], self.ref[:n], self.theirs[:n]
        return self


def make(kind: str, seed: int, pool: list[tuple[str, str]], delay: float = 0.15, gain_db: float = -12.0,
         echo_path: bool = True) -> Scenario:
    rng = np.random.default_rng(seed)
    pick = iter(rng.permutation(len(pool)))

    def clip():
        wav, text = pool[next(pick)]
        x = read_audio(wav)
        if kind == 'long':  # no pause the VAD would take: the recording's silences cut down to 0.15 s (not 'turns')
            x = squeeze(x)
        return level(x), text
    sc, t = Scenario(240), 0.5
    if kind == 'echo':
        for turn in range(6):
            x, text = clip()
            sc.add('ours', x, t, text, far_end(x, delay, gain_db, rng) if echo_path else None, delay)
            ours_end = t + len(x) / SR + delay
            y, text = clip()
            t = ours_end - rng.uniform(0.5, 1.5) if turn % 3 == 2 else ours_end + rng.uniform(0.6, 1.2)  # over us
            sc.add('them', y, t, text)
            t += len(y) / SR + rng.uniform(0.6, 1.2)
    elif kind == 'overlap':
        prev = None
        for turn in range(8):
            x, text = clip()
            kind = 'alone'
            if prev is not None and turn % 4 == 3:  # together
                t, kind = prev[0] + rng.uniform(0.0, 0.5), 'together'
            elif prev is not None and turn % 2 == 1:  # over the end of the other one
                t, kind = prev[1] - rng.uniform(0.3, 1.5), 'over'
            sc.add('them', x, t, text)
            sc.segs[-1] |= {'speaker': turn % 2, 'kind': kind}
            if kind != 'alone':
                sc.segs[-2]['kind'] = kind
            prev = (t, t + len(x) / SR)
            t = prev[1] + rng.uniform(0.4, 1.0)
    elif kind in ('long', 'turns'):
        for _ in range(5):
            x, text = clip()
            sc.add('them', x, t, text)
            t += len(x) / SR + (rng.uniform(0.05, 0.2) if kind == 'long' else rng.uniform(0.6, 1.2))  # under / over the VAD pause
    return sc.trim()


def gate(sc: Scenario, frame: int = 512, **kw) -> tuple[np.ndarray, dict]:
    """The scenario through the echo gate in its own time: the reference is pushed as it is sent. `kw`: EchoGate's."""
    from peresmeshnik.audio.echo import HOP, EchoGate, Reference
    ref = Reference()
    g = EchoGate(ref, **kw)
    out = np.empty_like(sc.capture)
    for i in range(0, len(sc.capture), frame):
        t_end = ref.t0 + min(i + frame, len(sc.capture)) / SR
        ref.push(sc.ref[i:i + frame], SR, t_end)
        out[i:i + frame] = g.process(sc.capture[i:i + frame], t_end)
    n = len(out) // HOP * HOP
    blocks = lambda x: x[:n].reshape(-1, HOP)  # noqa: E731
    theirs = np.sqrt((blocks(sc.theirs) ** 2).mean(axis=1)) > 0.005  # -46 dBFS: their speech, not its pauses
    zeroed = ~blocks(out).any(axis=1) & blocks(sc.capture).any(axis=1)
    echo = np.sqrt((blocks(sc.capture - sc.theirs) ** 2).mean(axis=1)) > 0.001
    return out, {'their_blocks_zeroed': round(float(zeroed[theirs].mean()), 4) if theirs.any() else None,
                 'echo_blocks_zeroed': round(float(zeroed[echo & ~theirs].mean()), 4) if (echo & ~theirs).any() else None,
                 'gated_blocks': g.stats['gated'], 'blocks': g.stats['blocks']}


def clauses(base: LocalDirection, pcm: np.ndarray, realtime: bool = False, summaries: list | None = None) -> list[dict]:
    d = LocalDirection(type(base.vad)(), base.asr, base.mt, base.tts, base.src, base.tgt, max_lag_s=base.max_lag_s,
                       max_clause_s=base.max_clause_s, early_s=base.early_s, partial_mt=base.partial_mt)
    d.realtime = realtime
    with Rows() as r:
        t = threading.Thread(target=lambda: list(d.events()), daemon=True)
        t.start()
        run(d, pcm, realtime=realtime)
        d.finish()
        t.join()
    if summaries is not None:
        summaries.append(d.summary())
    return r.rows


def content(text: str) -> list[str]:
    return [w for w in norm(text).split() if len(w) >= 4]


def score(sc: Scenario, rows: list[dict]) -> dict:
    """Each recognised content word goes to the utterance it belongs to (the one heard around the clause, by
    text): an utterance of ours with >= 2 and >= 25 % of its words out is an echo translated; theirs: recall."""
    got = [{w: 0 for w in content(s['text'])} for s in sc.segs]
    for r in rows:
        near = [i for i, s in enumerate(sc.segs) if s['start'] - 1.0 <= r['end_s'] and r['start_s'] <= s['end'] + 1.0]
        for w in content(r['text']):
            i = next((i for i in near if w in got[i] and got[i][w] == 0), None)
            if i is not None:
                got[i][w] = 1
    out = {}
    for who in {s['who'] for s in sc.segs}:
        idx = [i for i, s in enumerate(sc.segs) if s['who'] == who]
        words = sum(len(got[i]) for i in idx)
        out[f'{who}_utterances'] = len(idx)
        out[f'{who}_recall'] = round(sum(sum(got[i].values()) for i in idx) / max(words, 1), 3)
        out[f'{who}_out'] = sum(sum(got[i].values()) >= max(2, 0.25 * len(got[i])) for i in idx)
    return out


def echo(a):
    base = make_local('en', 'ru', threads=a.threads, tts=NoTts())
    pool = fleurs('en')
    tot = {}
    for seed in range(a.seed, a.seed + a.n):
        sc = make('echo', seed, pool, a.delay, a.gain, not a.no_echo_path)
        pcm, g = gate(sc) if a.gate else (sc.capture, {})
        s = score(sc, clauses(base, pcm)) | g
        print(seed, json.dumps(s), flush=True)
        for k, v in s.items():
            tot.setdefault(k, []).append(v)
    sums = {k: sum(v) for k, v in tot.items() if k.endswith(('_out', '_utterances', 'blocks'))}
    means = {k: round(float(np.mean(xs)), 4) if (xs := [x for x in v if x is not None]) else None  # None: JSON, not NaN
             for k, v in tot.items() if k.endswith(('recall', 'zeroed'))}
    ours = sums.get('ours_utterances', 0)
    record('call', 'echo-gate' if a.gate else 'none', scenario='echo', n=a.n, seed=a.seed, delay_s=a.delay, gain_db=a.gain,
           echo_path=not a.no_echo_path, echo_translated=round(sums.get('ours_out', 0) / ours, 3) if ours else None,
           **sums, **means)


def overlap_score(segs: list[dict], rows: list[dict]) -> tuple[list[dict], list[set], int]:
    """Content words of each utterance recognised (word -> 1), those of them voiced (not dropped), and the
    clauses with >= 2 content words of each of two people (merged)."""
    per = [{w: 0 for w in content(s['text'])} for s in segs]
    said = [set() for _ in segs]
    merged = 0
    for r in rows:
        near = [i for i, s in enumerate(segs) if s['start'] - 1.0 <= r['end_s'] and r['start_s'] <= s['end'] + 1.0]
        hit = {}
        for w in content(r['text']):
            i = next((i for i in near if w in per[i] and per[i][w] == 0), None)
            if i is not None:
                per[i][w] = 1
                hit[segs[i]['speaker']] = hit.get(segs[i]['speaker'], 0) + 1
                if 'dropped' not in r:
                    said[i].add(w)
        merged += sum(v >= 2 for v in hit.values()) >= 2
    return per, said, merged


def overlap(a):
    """Two people at the far end. Lost: an utterance with < 50 % of its content words out; merged: a clause with
    >= 2 content words of each of two people (one subtitle, one voice for both)."""
    from peresmeshnik.pipeline import MAX_LAG_S
    base = make_local('en', 'ru', threads=a.threads, **({'max_lag_s': MAX_LAG_S} if a.realtime else {'tts': NoTts()}))
    base.warm_up()
    pool = fleurs('en')
    lost = lost_voice = merged = utts = n_clauses = dropped = 0
    recall, by, sums = [], {}, []
    for seed in range(a.seed, a.seed + a.n):
        sc = make('overlap', seed, pool)
        rows = clauses(base, sc.capture, realtime=a.realtime, summaries=sums)
        per, said, m = overlap_score(sc.segs, rows)
        merged += m
        dropped += sum('dropped' in r for r in rows)
        rec = [sum(p.values()) / max(len(p), 1) for p in per]
        lost += sum(x < 0.5 for x in rec)
        lost_voice += sum(len(w) < 0.5 * max(len(p), 1) for w, p in zip(said, per))
        for sg, x in zip(sc.segs, rec):
            by.setdefault(sg['kind'], []).append(x)
        utts, n_clauses = utts + len(per), n_clauses + len(rows)
        recall += rec
        print(seed, f'lost {sum(x < 0.5 for x in rec)}/{len(per)}, clauses {len(rows)}', flush=True)
    record('call', a.variant, scenario='overlap', n=a.n, seed=a.seed, realtime=a.realtime, utterances=utts, lost=lost,
           lost_share=round(lost / utts, 3), clauses=n_clauses, merged=merged, merged_share=round(merged / max(n_clauses, 1), 3),
           recall_mean=round(float(np.mean(recall)), 3), dropped_clauses=dropped, empty_clauses=sum(x['empty'] for x in sums),
           **({'lost_in_voice': lost_voice, 'lost_in_voice_share': round(lost_voice / utts, 3)} if a.realtime else {}),
           by_kind={k: {'n': len(v), 'lost': sum(x < 0.5 for x in v), 'recall': round(float(np.mean(v)), 3)} for k, v in by.items()})
    base.close()


def target(a) -> str:
    """The language they are translated into: --to, else Russian (English for Russian speech), as the app's them -> me."""
    return a.to or OTHER.get(a.lang, 'ru')


def repeats(rows, lang: str = 'en') -> int:
    """Words (zh / ja: characters) a clause repeats from the end of the clause before it (a cut whose window timestamps
    ran ahead)."""
    from peresmeshnik.local.asr import NO_SPACES, units
    k = 4 if lang in NO_SPACES else 3
    n = 0
    for a, b in zip(rows, rows[1:]):
        x, y = units(norm(a['text']), lang)[-k:], units(norm(b['text']), lang)[:k]
        n += next((k for k in range(min(len(x), len(y)), 0, -1) if x[-k:] == y[:k]), 0)
    return n


def long(a):
    """A monologue with no pause: when each word's translation is on screen and heard (words spread evenly over
    their clause), WER of the clauses and chrF of their translations against the MT of the whole sentences."""
    import jiwer
    import sacrebleu

    from peresmeshnik.local import asr
    from peresmeshnik.pipeline import MAX_LAG_S
    base = make_local(a.lang, target(a), threads=a.threads,  # as the app's them -> me
                      **({'tts': NoTts()} if a.asap else {'max_lag_s': MAX_LAG_S}))
    base.early_s, base.partial_mt = a.early or None, not a.no_partial_mt
    base.max_clause_s = a.max_clause
    base.warm_up()
    pool = fleurs(a.lang)
    rows, refs, hyps, trs, prefs, sums, rep = [], [], [], [], [], [], 0
    for seed in range(a.seed, a.seed + a.n):
        sc = make('turns' if a.turns else 'long', seed, pool)
        r = clauses(base, sc.capture, realtime=not a.asap, summaries=sums)
        rows += r
        rep += repeats(r, a.lang)
        refs.append(norm(' '.join(s['text'] for s in sc.segs)))
        hyps.append(norm(' '.join(x['text'] for x in r)))
        trs.append(' '.join(x['translation'] for x in r))
        prefs.append(' '.join(base.mt.translate(s['text'], a.lang, target(a)) for s in sc.segs))
        print(seed, f'{len(r)} clauses', flush=True)
    nw = lambda x: max(len(asr.units(x['text'], a.lang)), 1)  # noqa: E731  words (zh / ja: characters)
    wait = lambda x, i: x['end_s'] - (x['start_s'] + (x['end_s'] - x['start_s']) * (i + 0.5) / nw(x))  # noqa: E731
    caption = [wait(x, i) + x['caption_lag_s'] for x in rows if 'caption_lag_s' in x for i in range(nw(x))]

    def subtitle(x, i):  # the first translated subtitle with word i: a translated partial (stage 5) or the final
        t = x['end_s'] + x['caption_lag_s']
        t = min([t] + [e + lag for k, e, lag in x.get('partials', ()) if k > i])
        return t - (x['end_s'] - wait(x, i))
    sub = [subtitle(x, i) for x in rows if 'caption_lag_s' in x for i in range(nw(x))]
    first_sub = [subtitle(x, 0) for x in rows if 'caption_lag_s' in x]
    voice = [wait(x, i) + x['voice_lag_s'] for x in rows if 'voice_lag_s' in x for i in range(nw(x))]
    first = [x['end_s'] - x['start_s'] + x['caption_lag_s'] for x in rows if 'caption_lag_s' in x]  # its first word
    dur = [x['end_s'] - x['start_s'] for x in rows]
    out = ROOT / 'call-long' / f'{a.variant}.json'  # the clauses as recognised, against the read text
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'refs': refs, 'hyps': hyps, 'clauses': [x['text'] for x in rows]}, ensure_ascii=False))
    from asr import lid  # bench/asr.py: letter trigrams against FLEURS test text of the 14
    classify, long_enough = lid(), [x['text'] for x in rows if len(asr.units(x['text'], a.lang)) >= (6 if a.lang in asr.NO_SPACES else 4)]
    other = [t for t in long_enough if classify(t) != a.lang]  # a clause in another language: Parakeet drifting
    record('call', a.variant, scenario='long', lang=a.lang, to=target(a), n=a.n, seed=a.seed, clauses=len(rows), clause_s_p50=pct(dur, 50),
           clause_s_p90=pct(dur, 90), caption_span_s_p50=pct(caption, 50), caption_span_s_p90=pct(caption, 90),
           subtitle_span_s_p50=pct(sub, 50), subtitle_span_s_p90=pct(sub, 90), early_s=a.early, partial_mt=not a.no_partial_mt,
           max_clause_s=a.max_clause, mt_rtf=round(float(np.mean([x['mt_rtf'] for x in sums])), 4),
           mt_partial_rtf=round(float(np.mean([x['mt_partial_rtf'] for x in sums])), 4),
           first_word_caption_s_p50=pct(first, 50), first_word_caption_s_p90=pct(first, 90),
           first_word_subtitle_s_p50=pct(first_sub, 50), first_word_subtitle_s_p90=pct(first_sub, 90),
           voice_span_s_p50=pct(voice, 50), voice_span_s_p90=pct(voice, 90),
           dropped=sum('dropped' in x for x in rows), empty=sum(x['empty'] for x in sums),
           speed_mean=round(float(np.mean([x.get('speed', 1.0) for x in rows])), 3),
           wer=round(jiwer.wer(refs, hyps), 4), repeated_words=rep,
           cer=round(jiwer.cer([r.replace(' ', '') for r in refs], [h.replace(' ', '') for h in hyps]), 4),
           other_lang=round(len(other) / max(len(long_enough), 1), 3), other_samples=other[:3],
           tail_s=asr.TAIL_S, repeat_words=asr.REPEAT_WORDS, asap=a.asap,
           chrf=round(sacrebleu.corpus_chrf(trs, [prefs]).score, 1),
           reasons={k: sum(x['reason'] == k for x in rows) for k in ('pause', 'punct', 'max', 'eof', 'early')})
    base.close()


LINKS = (('vad', 'vad_lag_s'), ('asr', 'asr_lag_s'), ('caption', 'caption_lag_s'), ('tts_first', 'synth_lag_s'),
         ('voice', 'voice_lag_s'))


def links(a):
    """Latency per link of them -> me in a call (the echo scenario through the gate, in real time, the app's queue
    policy): seconds from the end of a clause's speech to each link (pipeline.LocalDirection, Latency events)."""
    from peresmeshnik.pipeline import MAX_LAG_S
    base = make_local(a.lang, target(a), threads=a.threads, max_lag_s=MAX_LAG_S)
    base.warm_up()
    pool = fleurs(a.lang)
    rows, sums = [], []
    for seed in range(a.seed, a.seed + a.n):
        pcm, _ = gate(make('echo', seed, pool))
        rows += clauses(base, pcm, realtime=True, summaries=sums)
        print(seed, len(rows), flush=True)
    import platform

    from peresmeshnik.local import device
    out = {}
    for name, key in LINKS:
        xs = [r[key] for r in rows if key in r]
        out[name] = {'p50': pct(xs, 50), 'p90': pct(xs, 90), 'n': len(xs)}
    by_reason = {k: {'n': len(v), 'vad_p50': pct(v, 50), 'vad_p90': pct(v, 90)}
                 for k in ('pause', 'punct', 'max') if (v := [r['vad_lag_s'] for r in rows if r['reason'] == k and 'vad_lag_s' in r])}
    record('call', a.variant, scenario='links', lang=a.lang, to=target(a), n=a.n, seed=a.seed, clauses=len(rows),
           dropped=sum('dropped' in r for r in rows), asr_device=base.asr.device, mt_device=getattr(base.mt, 'device', 'cpu'),
           cuda=device.want(), cpu=platform.processor(),
           links=out, vad_by_reason=by_reason,
           rtf={k: round(float(np.mean([x[k] for x in sums])), 4) for k in sums[0] if k.endswith('_rtf')},
           reasons={k: sum(x['reason'] == k for x in rows) for k in ('pause', 'punct', 'max', 'eof', 'early')})
    base.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['echo', 'overlap', 'long', 'links'])
    ap.add_argument('--variant', default='main', help='overlap, long: name of what is measured (the result row)')
    ap.add_argument('--n', type=int, default=6, help='scenarios (seeds)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--gate', action='store_true', help='echo: through the echo gate (peresmeshnik/audio/echo.py)')
    ap.add_argument('--delay', type=float, default=0.15, help='echo: delay of the far end, s')
    ap.add_argument('--gain', type=float, default=-12.0, help='echo: level of the echo against our speech, dB')
    ap.add_argument('--early', type=float, default=0.0, help='long: cut at a comma from this clause length, s (0: off, the app)')
    ap.add_argument('--realtime', action='store_true', help='overlap: in real time with the voice (the queue policy)')
    ap.add_argument('--no-partial-mt', action='store_true', help='long: no translated partials')
    ap.add_argument('--max-clause', type=float, default=5.0)
    ap.add_argument('--no-echo-path', action='store_true', help='echo: the far end on a headset (no echo): false suppression only')
    ap.add_argument('--turns', action='store_true', help='long: the sentences with pauses between them (0.6-1.2 s), a talk not a monologue')
    ap.add_argument('--asap', action='store_true', help='long: fed as fast as it goes (the words only: no lags, same on any load)')
    ap.add_argument('--repeat', type=int, help='local/asr.py REPEAT_WORDS: words of a cut the next clause may repeat (1: before stage 5)')
    ap.add_argument('--lang', default='en', choices=sorted(CODES), help='long, links: the language they speak (FLEURS dev)')
    ap.add_argument('--to', choices=sorted(CODES), help='long, links: the language they are translated into (default: ru; en for ru)')
    ap.add_argument('--tail', type=float, help='local/asr.py TAIL_S: no cut at a word start this near the window end, s')
    a = ap.parse_args()
    from peresmeshnik.local import asr
    if a.tail is not None:
        asr.TAIL_S = a.tail
    if a.repeat is not None:
        asr.REPEAT_WORDS = a.repeat
    globals()[a.kind](a)


if __name__ == '__main__':
    main()

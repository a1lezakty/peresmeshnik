"""Them -> me in their own voice, end to end (stage 8, S2): a call of two people, the app's them side in real time.

The call is built from two LibriSpeech speakers (turns of 30, 30, 10, 10, then 5 s each, 0.7 s between turns); the
side is the app's: VAD, ASR, MT and the stock voice from the registry, their voice (voices/live.py) with the clone
engine and the speaker vector model given here (`--stock`: without it, as before). Per clause it logs whose voice
said it (stock, or person k with the seconds it was made from) and its lag; the judge is the same speaker vector model:
each clause's speech against 15 s of each speaker's other utterances. One JSON summary (BENCH stage 8, "Their voice").

    PERESMESHNIK_MODELS=<registry> python bench/their_voice.py --data <LibriSpeech dev-clean> --a 1272 --b 1462 \
        --clone <pocket-tts-ru folder> --speaker <wespeaker folder> --out <dir> [--threads 2]
"""
import argparse
import json
import logging
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from peresmeshnik.audio.resample import Resampler
from peresmeshnik.engines import SAMPLE_RATE, Speech
from peresmeshnik.local.clone import make_engine
from peresmeshnik.pipeline import MAX_LAG_S, cpu_s, make_local, read_audio, run
from peresmeshnik.voice import Speaker
from peresmeshnik.voices import live

TURNS = [30, 30, 10, 10, 5, 5, 5, 5, 5, 5]  # seconds per turn, A first
GAP_S, REF_S = 0.7, 15.0


def utterances(folder: Path) -> list[np.ndarray]:
    return [read_audio(str(f)) for f in sorted(folder.rglob('*.flac'))]


def call(a: list, b: list) -> tuple[np.ndarray, list, dict]:
    """The call's audio, its turns [(t0, t1, 'A' | 'B')], and each speaker's reference: their last REF_S s, not said."""
    refs, pools = {}, {'A': list(a), 'B': list(b)}
    for k, pool in pools.items():
        ref = []
        while sum(map(len, ref)) < REF_S * SAMPLE_RATE:
            ref.append(pool.pop())
        refs[k] = np.concatenate(ref)
    parts, turns, t = [], [], 0.0
    for i, s in enumerate(TURNS):
        k, got = 'AB'[i % 2], []
        while sum(map(len, got)) < s * SAMPLE_RATE:
            got.append(pools[k].pop(0))
        x = np.concatenate(got)
        turns.append((t, t + len(x) / SAMPLE_RATE, k))
        parts += [x, np.zeros(int(GAP_S * SAMPLE_RATE), np.float32)]
        t += len(x) / SAMPLE_RATE + GAP_S
    return np.concatenate(parts + [np.zeros(SAMPLE_RATE, np.float32)]), turns, refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--clone', type=Path, required=True, help='the clone engine folder (pocket-tts-ru)')
    ap.add_argument('--engine', default='pocket-tts-ru')
    ap.add_argument('--speaker', type=Path, required=True, help='the speaker vector model folder (wespeaker)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--stock', action='store_true', help='the stock voice only (their voice off): the baseline')
    ap.add_argument('--call', action='store_true', help='as a call app sends it: Opus 24 kbit/s there and back (ffmpeg), '
                    'white noise 20 dB below the speech')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pcm, turns, refs = call(utterances(a.data / a.a), utterances(a.data / a.b))
    if a.call:
        import subprocess
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(SAMPLE_RATE), '-ac', '1', '-i', '-',
                              '-c:a', 'libopus', '-b:a', '24k', '-f', 'ogg', '-'], input=pcm.tobytes(), capture_output=True,
                             check=True).stdout
        pcm = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-f', 'f32le', '-ar', str(SAMPLE_RATE),
                                            '-ac', '1', '-'], input=raw, capture_output=True, check=True).stdout, np.float32)
        level = np.sqrt(np.mean(pcm[np.abs(pcm) > 0.01] ** 2))
        pcm = (pcm + np.random.default_rng(0).normal(0, level / 10, len(pcm))).astype(np.float32)
    sf.write(a.out / 'call.wav', pcm, SAMPLE_RATE)

    rows = []
    h = logging.Handler()
    h.emit = lambda r: rows.append(json.loads(r.getMessage()))
    lg = logging.getLogger('peresmeshnik.latency')
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    t0 = time.perf_counter()
    d = make_local('en', 'ru', threads=a.threads, max_lag_s=MAX_LAG_S)
    d.warm_up()
    if a.stock:  # their voice off: the same wrapper (resampled to 24 kHz), never a clone
        v = live.LiveVoice(lambda: None, lambda: None, a.engine, 'en')
        v.fail('--stock')
    else:
        v = live.LiveVoice(lambda: make_engine(a.engine, a.clone, a.threads), lambda: Speaker(a.speaker), a.engine, 'en')
    said = []  # per spoken clause: (who, its speech)

    class Rec(live.TheirVoice):
        def synth(self, text, lang, voice=None, speed=1.0):
            who, chunks = ['stock', 0.0], []
            orig = self.live.voice
            def pick(clause=None):
                x = orig(clause)
                for i, p in enumerate(self.live.people):
                    if x is not None and p.voice is x:
                        who[:] = [f'p{i}', round(p.made_at, 1)]
                return x
            self.live.voice = pick
            try:
                for c in super().synth(text, lang, voice, speed):
                    chunks.append(c)
                    yield c
            finally:
                self.live.voice = orig
                said.append((who, np.concatenate(chunks) if chunks else np.zeros(1, np.float32)))
    d.tts, d.on_clause = Rec(d.tts, v), None if a.stock else v.offer
    while v.engine is None and not v.dead:  # loaded before the call: the load is not what is measured
        time.sleep(0.1)
    load_s = time.perf_counter() - t0
    out, start = [], time.perf_counter()

    def consume():
        at, rs = 0, {}
        for e in d.events():
            if isinstance(e, Speech):  # at the rate it says (FIXES-s2 r2 1), into the recording's
                x = e.pcm if e.sample_rate == live.OUT_SR else rs.setdefault(e.sample_rate, Resampler(e.sample_rate,
                                                                                                      live.OUT_SR))(e.pcm)
                gap = int((time.perf_counter() - start) * live.OUT_SR) - at
                out.extend([np.zeros(max(gap, 0), np.float32), x])
                at += max(gap, 0) + len(x)
    th = threading.Thread(target=consume)
    th.start()
    c0 = cpu_s(d)
    run(d, pcm, realtime=True)
    d.finish()
    cpu = cpu_s(d) - c0
    th.join()
    d.close()
    lg.removeHandler(h)
    sf.write(a.out / 'heard.wav', np.concatenate(out) if out else np.zeros(1), live.OUT_SR)

    judge = Speaker(a.speaker)
    ref = {k: judge.embed(x, SAMPLE_RATE) for k, x in refs.items()}
    spoken = [r for r in rows if 'dropped' not in r]
    clauses = []
    for r, (who, x) in zip(spoken, said):
        mid = (r['start_s'] + r['end_s']) / 2
        true = next((k for t0_, t1_, k in turns if t0_ <= mid <= t1_ + GAP_S), '?')
        e = judge.embed(x, live.OUT_SR) if len(x) > live.OUT_SR // 2 else None
        clauses.append({'start_s': r['start_s'], 'end_s': r['end_s'], 'true': true, 'who': who[0], 'made_s': who[1],
                        'voice_lag_s': r.get('voice_lag_s'), 'tts_first_ms': r.get('tts_first_ms'),
                        'speech_s': r.get('speech_s'), 'speed': r.get('speed', 1.0),
                        'sim_true': None if e is None else round(float(e @ ref[true]), 3) if true != '?' else None,
                        'sim_other': None if e is None or true == '?' else round(float(e @ ref['AB'[true == 'A']]), 3),
                        'text': r['text'], 'translation': r['translation']})
    with open(a.out / 'clauses.jsonl', 'w', encoding='utf-8') as f:
        for c in clauses:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    # whose voice each person is: the speaker most of their clauses were
    owner = {}
    for c in clauses:
        if c['who'] != 'stock':
            owner.setdefault(c['who'], []).append(c['true'])
    owner = {p: max(set(ts), key=ts.count) for p, ts in owner.items()}
    clone = [c for c in clauses if c['who'] != 'stock']
    first = clone[0] if clone else None
    med = lambda xs: round(statistics.median(xs), 3) if xs else None  # noqa: E731

    def turn_of(c):
        return next((i for i, (t0_, t1_, _) in enumerate(turns) if t0_ <= (c['start_s'] + c['end_s']) / 2 <= t1_ + GAP_S), -1)
    switch = []  # per turn from the third: its clauses said in the other person's voice before its own
    for i in range(2, len(turns)):
        cs = [c for c in clauses if turn_of(c) == i]
        k = next((j for j, c in enumerate(cs) if owner.get(c['who']) == c['true']), len(cs))
        switch.append(k)
    summary = {
        'pair': f'{a.a}+{a.b}', 'audio_s': round(len(pcm) / SAMPLE_RATE, 1), 'load_s': round(load_s, 1),
        'clauses': len(rows), 'spoken': len(spoken), 'dropped': len(rows) - len(spoken),
        'first_clone_at_s': first and first['start_s'], 'first_clone_made_s': first and first['made_s'],
        'stock_clauses_before': first and clauses.index(first),
        'right_person': f"{sum(owner.get(c['who']) == c['true'] for c in clone)}/{len(clone)}",
        'switch_clauses_per_turn': switch,
        'sim_true_clone': med([c['sim_true'] for c in clone if c['sim_true'] is not None]),
        'sim_other_clone': med([c['sim_other'] for c in clone if c['sim_other'] is not None]),
        'sim_true_stock': med([c['sim_true'] for c in clauses if c['who'] == 'stock' and c['sim_true'] is not None]),
        'voice_lag_clone': med([c['voice_lag_s'] for c in clone if c['voice_lag_s'] is not None]),
        'voice_lag_clone_max': max((c['voice_lag_s'] for c in clone if c['voice_lag_s'] is not None), default=None),
        'tts_first_ms_clone': med([c['tts_first_ms'] for c in clone if c['tts_first_ms'] is not None]),
        'tts_first_ms_stock': med([c['tts_first_ms'] for c in clauses if c['who'] == 'stock' and c['tts_first_ms']]),
        # the voice's pace: seconds of speech per character of the translation, at speed 1
        'pace_clone': med([c['speech_s'] * c['speed'] / len(c['translation']) for c in clone if c['translation']]),
        'pace_stock': med([c['speech_s'] * c['speed'] / len(c['translation']) for c in clauses
                           if c['who'] == 'stock' and c['translation']]),
        'sped_up': sum(c['speed'] > 1 for c in clauses),
        'people': len(v.people), 'stats': v.stats, 'dead': v.dead,
        'cpu_s_per_audio_s': round(cpu / (len(pcm) / SAMPLE_RATE), 3),
    }
    if not a.stock and (got := v.them()):  # S3's "keep their voice", then a next call knows it at once
        import tempfile

        from peresmeshnik import voice as profiles
        with tempfile.TemporaryDirectory() as tmp:
            profiles.save_them('bench', got[1], got[2], got[0], root=Path(tmp))
            back = live.kept(a.engine, Path(tmp))
            summary['kept_roundtrip'] = len(back) == 1 and np.array_equal(back[0][1], got[0][a.engine][0])
    (a.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()

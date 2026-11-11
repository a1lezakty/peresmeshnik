"""Does a short piece of another voice in the clone's prompt recolour the clone's output? (stream S3, tails, review 2 p. 3)

LibriSpeech dev-clean, pairs "me + other" of the same gender. A 10 s prompt of my speech with N = 2 / 3 / 4 s of the
other's speech at its start or its end, plus the bases (all mine, all theirs); each engine says the same phrases from
it as the app does (Pocket: local/pocket.py, int8, first REF_S s ending on a pause; XTTS: local/xtts.py Engine:
GPT latents from the first 6 s, speaker embedding from all of it). Each output against my voice and the other's
(other chapters) with the app's speaker vector (voice.Speaker, wespeaker ResNet34): "me - other" per output.

Stages, each in its own environment (PYTHONPATH: a copy of the repo):
  prompts  --data D --work W            pairs.json, prompts/*.wav, refs/*.wav (numpy + soundfile)
  pocket   --work W --model-dir M       out/pocket/*.wav (the app's env: onnxruntime, sentencepiece)
  xtts     --work W --model-dir M       out/xtts/*.wav (the voice pack's env: torch, coqui-tts)
  score    --work W --speaker S         score.json + the table (the app's env: onnx-asr)
  gate     --data D --work W --speaker S  gate.json: which recordings pass the app's live check (voice.py _start,
                                        _windows), the table again over the outputs of those that pass
Measure 7 (a quiet other in the first 10 s, before / after voice.save() writes what is under its floor as silence):
  prompts7 --data D --work W            prompts/*_q62.wav (before), *_q62s.wav (after: voice.save()'s reference.wav)
  qwen3    --work W --model-dir M       out/qwen3/*.wav (the voice pack's env: faster-qwen3-tts, CUDA)
  pocket / xtts / qwen3 --variants me,other,q62,q62s: only those prompts
  score7   --work W --speaker S [--asr MODELS]  score7.json + the table: me - other, loudness, words heard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
PROMPT_S, CHUNKS, PLACES = 10.0, (2, 3, 4), ('start', 'end')
REF_S = 30.0  # my / their reference (another chapter) for the vectors
XFADE = int(0.02 * SR)
PHRASES = ('The weather was cold this morning, so I stayed at home and read a good book.',
           'Please call me back when you get this message, I will be in the office until six.',
           'We should meet next week to talk about the new project and the budget for it.')


def variants() -> list[str]:
    return ['me', 'other'] + [f'{p}{n}' for p in PLACES for n in CHUNKS] + ['quiet']


QUIET_DB, TAIL_S = -57.0, 20.0  # 'quiet': the other's 10 s under the check's floor (-55), then my TAIL_S s at -20


def quiet(theirs: np.ndarray, mine: np.ndarray, quiet_db: float = QUIET_DB) -> np.ndarray:
    """The other's PROMPT_S s scaled so the 95th percentile of its 30 ms frames is QUIET_DB dBFS, the pauses (frames 30 dB
    under that) digital silence; then TAIL_S s of mine at the app's level: what a check dropping frames under its floor
    cannot see. Pocket takes the first 10 s of it (all quiet); XTTS its GPT latents from the first 6 s, the speaker
    vector from all of it."""
    f = 480
    x = theirs[:int(PROMPT_S * SR) // f * f].copy()  # 9.99 s: whole frames
    fr = x.reshape(-1, f)
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12)
    p95 = np.percentile(db, 95)
    fr[db < p95 - 30] = 0
    x = fr.ravel() * 10 ** ((quiet_db - p95) / 20)
    tail = mine[:int(TAIL_S * SR)]
    return np.concatenate([x, tail * (10 ** (-20 / 20) / level(tail))])


# ---------- prompts
def chapters(libri: Path) -> dict[str, dict[str, list[Path]]]:
    out: dict = {}
    for f in sorted(libri.glob('*/*/*.flac')):
        out.setdefault(f.parent.parent.name, {}).setdefault(f.parent.name, []).append(f)
    return out


def genders(libri: Path) -> dict[str, str]:
    g = {}
    for line in (libri.parent / 'SPEAKERS.TXT').read_text().splitlines():
        p = [c.strip() for c in line.split('|')]
        if not line.startswith(';') and len(p) >= 3 and p[2] == 'dev-clean':
            g[p[0]] = p[1]
    return g


def trim(x: np.ndarray, floor_db: float = 35.0) -> np.ndarray:
    """Silence at the edges cut (20 ms frames `floor_db` under the loudest): an utterance as speech end to end."""
    f = int(0.02 * SR)
    n = len(x) // f
    db = 20 * np.log10(np.sqrt(np.mean(np.square(x[:n * f].reshape(n, f)), axis=1)) + 1e-12)
    on = np.nonzero(db > db.max() - floor_db)[0]
    return x[on[0] * f:(on[-1] + 1) * f]


def speech(utts: list[Path], sec: float) -> np.ndarray:
    """`sec` s of these utterances, each trimmed at its edges, 0.15 s between them (as a pause in a recording)."""
    buf, gap = [], np.zeros(int(0.15 * SR), np.float32)
    for u in utts:
        buf += [trim(sf.read(u, dtype='float32')[0]), gap]
        if sum(map(len, buf)) >= sec * SR:
            break
    x = np.concatenate(buf)[:int(sec * SR)]
    if len(x) < int(sec * SR):
        raise ValueError(f'{utts[0].parent}: under {sec} s')
    return x


def level(x: np.ndarray) -> float:
    """RMS of the loud frames (speech, not pauses)."""
    f = x[:len(x) // 320 * 320].reshape(-1, 320)
    r = np.sqrt(np.mean(f ** 2, axis=1))
    return float(np.sqrt(np.mean(r[r > r.max() * 0.1] ** 2)))


def join(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a then b with a 20 ms crossfade; the total stays len(a) + len(b) - XFADE, so a is given XFADE more."""
    ramp = np.linspace(0, 1, XFADE, dtype=np.float32)
    return np.concatenate([a[:-XFADE], a[-XFADE:] * (1 - ramp) + b[:XFADE] * ramp, b[XFADE:]])


def splice(me: np.ndarray, other: np.ndarray, place: str, n: int) -> np.ndarray:
    """PROMPT_S s: mine with the other's first n s (their level matched to mine) at the start or the end."""
    piece = other[:n * SR] * (level(me) / level(other))
    rest = me[:int((PROMPT_S - n) * SR) + XFADE]
    return join(piece, rest) if place == 'start' else join(rest, piece)


def stage_prompts(a) -> None:
    libri, work = Path(a.data), Path(a.work)
    ch, g = chapters(libri), genders(libri)
    pairs = []
    for sex in ('F', 'M'):
        spk = [s for s in sorted(ch, key=int) if g.get(s) == sex and len(ch[s]) >= 2][:2 * a.pairs_per_gender]
        pairs += [(spk[i], spk[i + 1]) for i in range(0, len(spk) - 1, 2)]
    (work / 'prompts').mkdir(parents=True, exist_ok=True)
    (work / 'refs').mkdir(exist_ok=True)
    meta = []
    for me, other in pairs:
        cm, co = sorted(ch[me]), sorted(ch[other])
        mine, theirs = speech(ch[me][cm[0]], PROMPT_S + TAIL_S + 1), speech(ch[other][co[0]], PROMPT_S + 1)
        pid = f'{me}-{other}'
        for v in variants():
            if v == 'quiet':
                sf.write(work / 'prompts' / f'{pid}_{v}.wav', quiet(theirs, mine[int(PROMPT_S * SR):]), SR)
                continue
            x = (mine[:int(PROMPT_S * SR)] if v == 'me' else theirs[:int(PROMPT_S * SR)] if v == 'other'
                 else splice(mine, theirs, v.rstrip('0123456789'), int(v[-1])))
            x = x * (10 ** (-20 / 20) / level(x))  # the app's saved level (voice.save: TARGET_DB -20)
            sf.write(work / 'prompts' / f'{pid}_{v}.wav', np.clip(x, -1, 1), SR)
        for who, s, c in (('me', me, cm[1]), ('other', other, co[1])):
            sf.write(work / 'refs' / f'{pid}_{who}.wav', speech(ch[s][c], REF_S), SR)
        meta.append({'id': pid, 'me': me, 'other': other, 'gender': g[me], 'prompt_ch': [cm[0], co[0]],
                     'ref_ch': [cm[1], co[1]]})
    (work / 'pairs.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(meta)} pairs, {len(meta) * len(variants())} prompts -> {work}')


# ---------- engines
def todo(work: Path, engine: str, vs: list[str] | None = None):
    out = work / 'out' / engine
    out.mkdir(parents=True, exist_ok=True)
    for p in json.loads((work / 'pairs.json').read_text()):
        for v in vs or variants():
            yield p['id'], v, work / 'prompts' / f'{p["id"]}_{v}.wav', out


def stage_pocket(a) -> None:
    from types import SimpleNamespace

    from peresmeshnik.local.pocket import PocketTts
    eng = PocketTts(a.model_dir, threads=a.threads, model_id='pocket-tts-en', int8=True)  # as clone.make_engine
    for pid, v, wav, out in todo(Path(a.work), 'pocket', a.variants):
        if all((out / f'{pid}_{v}_{k}.wav').exists() for k in range(len(PHRASES))):
            continue
        t0 = time.perf_counter()
        voice = SimpleNamespace(state=eng.condition(*sf.read(wav, dtype='float32')))
        for k, text in enumerate(PHRASES):
            eng._rng = np.random.default_rng(1000 * k + int(pid.split('-')[0]))  # the same noise for every variant
            sf.write(out / f'{pid}_{v}_{k}.wav', np.concatenate(list(eng.synth(text, 'en', voice))), eng.sample_rate)
        print(f'pocket {pid} {v} {time.perf_counter() - t0:.1f} s', flush=True)


def stage_xtts(a) -> None:
    import importlib.util
    from types import SimpleNamespace

    import torch
    spec = importlib.util.spec_from_file_location('xtts', Path(a.repo) / 'peresmeshnik/local/xtts.py')
    xtts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xtts)  # as the voice pack's runner loads it
    eng = xtts.Engine(a.model_dir, device=a.device, threads=a.threads)
    for pid, v, wav, out in todo(Path(a.work), 'xtts', a.variants):
        if all((out / f'{pid}_{v}_{k}.wav').exists() for k in range(len(PHRASES))):
            continue
        t0 = time.perf_counter()
        voice = SimpleNamespace(state=eng.condition(*sf.read(wav, dtype='float32')))
        for k, text in enumerate(PHRASES):
            torch.manual_seed(1000 * k + int(pid.split('-')[0]))
            with torch.inference_mode():
                sf.write(out / f'{pid}_{v}_{k}.wav', np.concatenate(list(eng.synth(text, 'en', voice))), xtts.SR)
        print(f'xtts {pid} {v} {time.perf_counter() - t0:.1f} s', flush=True)


# ---------- score
def stage_score(a) -> None:
    from peresmeshnik.voice import Speaker
    work, spk = Path(a.work), Speaker(Path(a.speaker), threads=a.threads)
    pairs = json.loads((work / 'pairs.json').read_text())
    emb = {}

    def vec(f: Path) -> np.ndarray:
        if f not in emb:
            emb[f] = spk.embed(*sf.read(f, dtype='float32'))
        return emb[f]
    rows, res = [], {}
    for eng in sorted(p.name for p in (work / 'out').iterdir()):
        for p in pairs:
            me, other = vec(work / 'refs' / f'{p["id"]}_me.wav'), vec(work / 'refs' / f'{p["id"]}_other.wav')
            for v in variants():
                for f in sorted((work / 'out' / eng).glob(f'{p["id"]}_{v}_*.wav')):
                    e = vec(f)
                    rows.append({'engine': eng, 'pair': p['id'], 'gender': p['gender'], 'variant': v,
                                 'k': int(f.stem[-1]), 'me': float(e @ me), 'other': float(e @ other)})
        for v in variants():
            d = np.array([r['me'] - r['other'] for r in rows if r['engine'] == eng and r['variant'] == v])
            if len(d):
                res[f'{eng}/{v}'] = {'n': len(d), 'median': float(np.median(d)), 'p10': float(np.percentile(d, 10)),
                                     'p90': float(np.percentile(d, 90)), 'closer_other': float(np.mean(d < 0)),
                                     'me': float(np.median([r['me'] for r in rows if r['engine'] == eng and r['variant'] == v])),
                                     'other': float(np.median([r['other'] for r in rows if r['engine'] == eng
                                                               and r['variant'] == v]))}
    # the prompts themselves against the refs: how much of the other the prompt carries, before any engine
    for p in pairs:
        me, other = vec(work / 'refs' / f'{p["id"]}_me.wav'), vec(work / 'refs' / f'{p["id"]}_other.wav')
        for v in variants():
            e = vec(work / 'prompts' / f'{p["id"]}_{v}.wav')
            rows.append({'engine': 'prompt', 'pair': p['id'], 'gender': p['gender'], 'variant': v, 'k': 0,
                         'me': float(e @ me), 'other': float(e @ other)})
    for v in variants():
        d = np.array([r['me'] - r['other'] for r in rows if r['engine'] == 'prompt' and r['variant'] == v])
        res[f'prompt/{v}'] = {'n': len(d), 'median': float(np.median(d)), 'p10': float(np.percentile(d, 10)),
                              'p90': float(np.percentile(d, 90)), 'closer_other': float(np.mean(d < 0))}
    (work / 'score.json').write_text(json.dumps({'summary': res, 'rows': rows}, indent=1))
    print('| engine | variant | n | median me-other | p10 | p90 | closer to other |')
    print('|---|---|---|---|---|---|---|')
    for k, r in res.items():
        e, v = k.split('/')
        print(f'| {e} | {v} | {r["n"]} | {r["median"]:+.3f} | {r["p10"]:+.3f} | {r["p90"]:+.3f} | '
              f'{r["closer_other"]:.0%} |')


# ---------- gate: the outputs of the splices the app's live check lets through
RECORD_S, LIVE_S = 40.0, 8.0


def stage_gate(a) -> None:
    """Each prompt as the app would hold the recording: the prompt (10 s; 'quiet': its quiet 10 s) then my speech of the
    same chapter up to RECORD_S, at -20; checked as confirm_live does it against a live reading of ~LIVE_S of my other
    chapter (edges trimmed): every window's cosine at or above its threshold."""
    from peresmeshnik.voice import Speaker, _speech, _start, _windows
    libri, work = Path(a.data), Path(a.work)
    spk, ch = Speaker(Path(a.speaker), threads=a.threads), chapters(libri)
    pairs, gate = json.loads((work / 'pairs.json').read_text()), {}
    for p in pairs:
        cm = p['prompt_ch'][0], p['ref_ch'][0]
        mine = speech(ch[p['me']][cm[0]], RECORD_S + 1)[int(PROMPT_S * SR):int(RECORD_S * SR)]
        mine = mine * (10 ** (-20 / 20) / level(mine))
        live = spk.embed(_speech(speech(ch[p['me']][cm[1]][::-1], LIVE_S), SR, edges=True), SR)  # its last utterances
        for v in variants():
            x = sf.read(work / 'prompts' / f'{p["id"]}_{v}.wav', dtype='float32')[0][:int(PROMPT_S * SR)]
            sims = [(float(spk.embed(w, SR) @ live), m) for w, m in _windows(_start(np.concatenate([x, mine]), SR), SR)]
            margin = min(s - m for s, m in sims)
            gate[f'{p["id"]}/{v}'] = {'pass': bool(margin >= 0), 'margin': margin, 'windows': len(sims)}
    rows = json.loads((work / 'score.json').read_text())['rows']
    (work / 'gate.json').write_text(json.dumps(gate, indent=1))
    print('| variant | passed (of 15) | median margin |')
    print('|---|---|---|')
    for v in variants():
        g = [gate[f'{p["id"]}/{v}'] for p in pairs]
        print(f'| {v} | {sum(x["pass"] for x in g)} | {np.median([x["margin"] for x in g]):+.3f} |')
    print('\n| engine | variant | passed outputs | median me-other | p10 | p90 | closer to other |')
    print('|---|---|---|---|---|---|---|')
    for eng in ('pocket', 'xtts'):
        for v in variants():
            d = np.array([r['me'] - r['other'] for r in rows if r['engine'] == eng and r['variant'] == v
                          and gate[f'{r["pair"]}/{v}']['pass']])
            print(f'| {eng} | {v} | {len(d)} | ' + (f'{np.median(d):+.3f} | {np.percentile(d, 10):+.3f} | '
                  f'{np.percentile(d, 90):+.3f} | {np.mean(d < 0):.0%} |' if len(d) else '- | - | - | - |'))


# ---------- measure 7: a quiet other (-62) in the first 10 s, before / after voice.save()
Q7_DB, V7 = -62.0, ('me', 'other', 'q62', 'q62s')


def p95_db(x: np.ndarray, sr: int) -> float:
    """95th percentile of the 30 ms frames' level, dBFS (digital silence counted as -120)."""
    f = x[:len(x) // (sr * 3 // 100) * (sr * 3 // 100)].reshape(-1, sr * 3 // 100)
    return float(np.percentile(20 * np.log10(np.sqrt(np.mean(f.astype(np.float64) ** 2, axis=1)) + 1e-6), 95))


def stage_prompts7(a) -> None:
    """'q62': the other's 10 s at p95 -62 dBFS, pauses digital silence, then my 20 s at -20, as the old save() wrote it
    (its gain, PCM_16); 'q62s': the same through voice.save() itself (its reference.wav)."""
    from peresmeshnik import voice
    libri, work = Path(a.data), Path(a.work)
    ch = chapters(libri)
    for p in json.loads((work / 'pairs.json').read_text()):
        mine = speech(ch[p['me']][p['prompt_ch'][0]], PROMPT_S + TAIL_S + 1)[int(PROMPT_S * SR):]
        theirs = speech(ch[p['other']][p['prompt_ch'][1]], PROMPT_S + 1)
        x = quiet(theirs, mine, Q7_DB)
        report = voice.check(x, SR)
        gain = min(10 ** ((voice.TARGET_DB - report.level_db) / 20), 10 ** (-1 / 20) / float(np.max(np.abs(x))))
        sf.write(work / 'prompts' / f'{p["id"]}_q62.wav', np.clip(x * gain, -1, 1), SR, subtype='PCM_16')
        v = voice.save(x, SR, p['id'], 'en', root=work / 'profiles7')
        y = sf.read(v.path / 'reference.wav', dtype='float32')[0]
        sf.write(work / 'prompts' / f'{p["id"]}_q62s.wav', y, SR, subtype='PCM_16')
        b = sf.read(work / 'prompts' / f'{p["id"]}_q62.wav', dtype='float32')[0]
        n = int(PROMPT_S * SR)
        nz = np.flatnonzero(y[:n])
        print(f'{p["id"]}: before first 10 s p95 {p95_db(b[:n], SR):.1f} dBFS, rest {p95_db(b[n:], SR):.1f}; after: '
              f'{len(nz)} non-zero samples in the first 10 s' + (f' (from {nz[0] / SR:.2f} s)' if len(nz) else ''),
              flush=True)


def stage_qwen3(a) -> None:
    """Qwen3 as the voice pack's runner loads local/qwen3.py: its vector from the first REF_S s (condition()), no
    Chatterbox (a runaway is cut at the guard and counted)."""
    import importlib.util
    from types import SimpleNamespace

    import torch
    spec = importlib.util.spec_from_file_location('qwen3', Path(a.repo) / 'peresmeshnik/local/qwen3.py')
    qwen3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen3)
    eng = qwen3.Engine(a.model_dir, device=a.device or 'cuda', threads=a.threads)
    for pid, v, wav, out in todo(Path(a.work), 'qwen3', a.variants):
        if all((out / f'{pid}_{v}_{k}.wav').exists() for k in range(len(PHRASES))):
            continue
        t0, r0 = time.perf_counter(), eng.runaways
        voice = SimpleNamespace(state=eng.condition(*sf.read(wav, dtype='float32')))
        for k, text in enumerate(PHRASES):
            torch.manual_seed(1000 * k + int(pid.split('-')[0]))
            y = np.concatenate([np.zeros(0, np.float32), *eng.synth(text, 'en', voice)])
            sf.write(out / f'{pid}_{v}_{k}.wav', y, qwen3.SR)
        print(f'qwen3 {pid} {v} {time.perf_counter() - t0:.1f} s runaways {eng.runaways - r0}', flush=True)


def stage_score7(a) -> None:
    from peresmeshnik.voice import Speaker, _to, wer
    work, spk = Path(a.work), Speaker(Path(a.speaker), threads=a.threads)
    asr = None
    if a.asr:
        import os
        os.environ['PERESMESHNIK_MODELS'] = a.asr
        from peresmeshnik.local.asr import OnnxAsr
        asr = OnnxAsr('parakeet-tdt-0.6b-v3', threads=a.threads, live=False)
    pairs, rows = json.loads((work / 'pairs.json').read_text()), []
    for p in pairs:
        me, other = (spk.embed(*sf.read(work / 'refs' / f'{p["id"]}_{w}.wav', dtype='float32')) for w in ('me', 'other'))
        for eng in ('qwen3', 'pocket', 'xtts'):
            for v in V7:
                for k, text in enumerate(PHRASES):
                    f = work / 'out' / eng / f'{p["id"]}_{v}_{k}.wav'
                    if not f.exists():
                        continue
                    x, sr = sf.read(f, dtype='float32')
                    e = spk.embed(x, sr)
                    rows.append({'engine': eng, 'pair': p['id'], 'variant': v, 'k': k, 'me': float(e @ me),
                                 'other': float(e @ other), 'p95_db': p95_db(x, sr), 'sec': len(x) / sr,
                                 'wer': wer(text, asr.recognize(_to(x, sr, 16_000)).text) if asr else None})
    (work / 'score7.json').write_text(json.dumps(rows, indent=1))
    print('| engine | variant | n | median me-other | p10 | p90 | closer to other | p95 dBFS | sec | WER |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for eng in ('qwen3', 'pocket', 'xtts'):
        for v in V7:
            r = [x for x in rows if x['engine'] == eng and x['variant'] == v]
            if not r:
                continue
            d = np.array([x['me'] - x['other'] for x in r])
            w = f'{np.median([x["wer"] for x in r]):.2f}' if asr else '-'
            print(f'| {eng} | {v} | {len(d)} | {np.median(d):+.3f} | {np.percentile(d, 10):+.3f} | '
                  f'{np.percentile(d, 90):+.3f} | {np.mean(d < 0):.0%} | {np.median([x["p95_db"] for x in r]):.1f} | '
                  f'{np.median([x["sec"] for x in r]):.1f} | {w} |')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stage', choices=('prompts', 'pocket', 'xtts', 'score', 'gate', 'prompts7', 'qwen3', 'score7'))
    ap.add_argument('--data', help='LibriSpeech/dev-clean')
    ap.add_argument('--work', required=True)
    ap.add_argument('--model-dir')
    ap.add_argument('--speaker', help='wespeaker ResNet34 folder (*.onnx)')
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--device', default=None)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--pairs-per-gender', type=int, default=8)
    ap.add_argument('--variants', type=lambda s: s.split(','), help='only these prompts (engine stages)')
    ap.add_argument('--asr', help='models root with parakeet-tdt-0.6b-v3 (score7: words heard)')
    a = ap.parse_args(argv)
    {'prompts': stage_prompts, 'pocket': stage_pocket, 'xtts': stage_xtts, 'score': stage_score,
     'gate': stage_gate, 'prompts7': stage_prompts7, 'qwen3': stage_qwen3, 'score7': stage_score7}[a.stage](a)


if __name__ == '__main__':
    sys.exit(main())

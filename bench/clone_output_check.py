"""The clone's own output against the live reading: does it close the splices the recording's check lets through?
(stream S3, before 0.3.0: BRIEF-s3-clone-output-check)

LibriSpeech dev-clean. Two sets of recordings, each written by `voice.save()` itself (its level, what is under its floor
as silence) and checked by the app's live check (`voice._start`, `_windows`) against live readings of ~8 s from another
chapter of the owner (edges trimmed, `voice._speech(edges=True)`), as `confirm_live` does:
  clean   the owner's own reading as it is (a chapter's first 40 s, utterances untrimmed), every speaker with two or more
          chapters, each chapter as the recording, up to 3 live readings from the other chapters: the owner's refusals
  splice  clone_prompt_splice.py's 15 pairs of the same sex: a 10 s prompt (mine / theirs / 2-4 s of theirs at the start
          or the end / their voice under the floor) then 30 s more of mine: what passes, and what the clone then says
Each engine conditions on the profile as the app does (Pocket: its first 10 s ending on a pause, int8; XTTS: GPT latents
from the first 6 s, its speaker vector from 30 s; Qwen3: an x-vector from the first 10 s; Chatterbox: its
prepare_conditionals of the file) and says the app's TEST_PHRASES (en; clean set also ru, de: the clone speaks their
language). Also each live reading cloned by itself (`live` outputs): the output "clone of the recording" against
"clone of the live reading" is a second feature measured.

Stages (PYTHONPATH: a copy of the repo; each engine in its own environment):
  recs    --data D --work W --speaker S   profiles/, lives/, refs/, recs.json with the live check per (recording, live)
          [--mid: the splice set only, the other person's K s at 12-30 s of the file (MID_V) where XTTS takes its speaker
          vector from and the live check has only the rest's 8 s windows: FIXES-s3-40 12]
  recs_ru --data D --work W --speaker S   the same of Russian speakers (FIXES-s3-40 10: a Russian live reading, the
          clone speaking English): mTEDx ru-ru train (D: its data/train; one speaker a talk), 40 s of a talk from its
          RU_FROM_S as the recording, 3 x 8 s of it at RU_AT of the talk as the live readings (one session: the clone's
          check is what is measured); the engines then with --no-lives
  pocket  --work W --model-dir M [--shard i/n]
  xtts    --work W --model-dir M [--device cuda:0]
  qwen3   --work W --model-dir M [--device cuda:0]          (no Chatterbox fallback: a runaway is counted, cut)
  cbox    --work W --model-dir M [--device cuda:0]          Chatterbox Multilingual on every recording
  score   --work W --speaker S              score.json: every output's vector against the live readings, refs, clones
  report  --work W                          the app's rule step by step: the owner's refusals, the splices passing
  timing  --work W --engine E --model-dir M --speaker S   what the clone check adds to confirming a voice
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clone_prompt_splice import PROMPT_S, SR, chapters, genders, level, quiet, speech, splice  # noqa: E402

RECORD_S, LIVE_S, LIVES = 40.0, 8.0, 3
SPLICE_V = ['me', 'other', 'start2', 'start3', 'start4', 'end2', 'end3', 'end4', 'quiet']
MID_V = {'mid3': (3, 14.0), 'mid4': (4, 14.0), 'mid4l': (4, 22.0), 'mid8': (8, 14.0)}  # --mid: their K s at T s of the file
LANGS = {'clean': ('en', 'ru', 'de'), 'splice': ('en',), 'ru': ('en', 'ru')}  # Pocket: en only (its weights here)
RU_FROM_S, RU_AT = 120.0, (0.4, 0.6, 0.8)  # mTEDx: the recording from the talk's 3rd minute, the live readings there


def at(x: np.ndarray, db: float = -20.0) -> np.ndarray:
    return x * (10 ** (db / 20) / level(x))


# ---------- recordings, live readings, the app's live check
def stage_recs(a) -> None:
    from peresmeshnik import voice
    from peresmeshnik.voice import Speaker, _speech, _start, _windows
    libri, work = Path(a.data), Path(a.work)
    ch, g = chapters(libri), genders(libri)
    spk = Speaker(Path(a.speaker), threads=a.threads)
    for d in ('lives', 'refs', 'profiles'):
        (work / d).mkdir(parents=True, exist_ok=True)
    recs, lives = [], {}

    def live(s: str, c: str, end: bool) -> str:
        lid = f'{s}-{c}-{"e" if end else "s"}'
        if lid not in lives:
            utts = sorted(ch[s][c])
            sf.write(work / 'lives' / f'{lid}.wav', speech(utts[::-1] if end else utts, LIVE_S), SR)
            lives[lid] = spk.embed(_speech(sf.read(work / 'lives' / f'{lid}.wav', dtype='float32')[0], SR, edges=True), SR)
        return lid

    def keep(rid: str, x: np.ndarray, meta: dict, lids: list[str]) -> None:
        v = voice.save(x, SR, rid, 'en', root=work / 'profiles')
        y, rsr = sf.read(v.path / 'reference.wav', dtype='float32')  # as confirm_live reads it
        wins = _windows(_start(y, rsr), rsr)
        vecs = [(spk.embed(w, rsr), m) for w, m in wins]
        gate = {lid: float(min(e @ lives[lid] - m for e, m in vecs)) for lid in lids}
        recs.append(meta | {'id': rid, 'lives': lids, 'margin': gate})

    # clean: the owner's own reading as it is
    for s in [] if a.mid else sorted(ch, key=int):
        cs = sorted(ch[s])
        if len(cs) < 2:
            continue
        for c in cs:
            x = np.concatenate([sf.read(u, dtype='float32')[0] for u in sorted(ch[s][c])])[:int(RECORD_S * SR)]
            if len(x) < RECORD_S * SR:
                continue
            others = [o for o in cs if o != c]
            lids = [live(s, o, e) for o, e in [(o, e) for e in (False, True) for o in others][:LIVES]]
            keep(f'c{s}-{c}', x, {'set': 'clean', 'me': s, 'gender': g[s], 'chapter': c}, lids)
        print(f'clean {s}: {len(recs)} recordings', flush=True)
    # splice: the pairs and prompts of clone_prompt_splice.py, followed by 30 s more of mine
    pairs = []
    for sex in ('F', 'M'):
        sp = [s for s in sorted(ch, key=int) if g.get(s) == sex and len(ch[s]) >= 2][:16]
        pairs += [(sp[i], sp[i + 1]) for i in range(0, len(sp) - 1, 2)]
    for me, other in pairs:
        cm, co = sorted(ch[me]), sorted(ch[other])
        mine = speech(ch[me][cm[0]], RECORD_S + 1)
        theirs = speech(ch[other][co[0]], PROMPT_S + 1)
        rest = at(mine[int(PROMPT_S * SR):int(RECORD_S * SR)])
        pid = f'{me}-{other}'
        lid = f'{me}-{cm[1]}-e'
        if lid not in lives:
            sf.write(work / 'lives' / f'{lid}.wav', speech(ch[me][cm[1]][::-1], LIVE_S), SR)  # as clone_prompt_splice gate
            lives[lid] = spk.embed(_speech(sf.read(work / 'lives' / f'{lid}.wav', dtype='float32')[0], SR, edges=True), SR)
        for who, s, c in (('me', me, cm[1]), ('other', other, co[1])):
            sf.write(work / 'refs' / f'{pid}_{who}.wav', speech(ch[s][c], 30.0), SR)
        for v in ['me', *MID_V] if a.mid else SPLICE_V:
            if v in MID_V:
                k, t = MID_V[v]
                o, piece = int((t - PROMPT_S) * SR), theirs[:k * SR] * (level(rest) / level(theirs[:k * SR]))
                keep(f's{pid}_{v}', np.concatenate([at(mine[:int(PROMPT_S * SR)]), rest[:o], piece, rest[o + len(piece):]]),
                     {'set': 'splice', 'pair': pid, 'me': me, 'other': other, 'gender': g[me], 'variant': v}, [lid])
                continue
            if v == 'quiet':
                x = quiet(theirs, mine[int(PROMPT_S * SR):])[:int(PROMPT_S * SR)]
            else:
                x = (mine[:int(PROMPT_S * SR)] if v == 'me' else theirs[:int(PROMPT_S * SR)] if v == 'other'
                     else splice(mine, theirs, v.rstrip('0123456789'), int(v[-1])))
                x = at(x)
            keep(f's{pid}_{v}', np.concatenate([x, rest]),
                 {'set': 'splice', 'pair': pid, 'me': me, 'other': other, 'gender': g[me], 'variant': v}, [lid])
        print(f'splice {pid}: {len(recs)} recordings', flush=True)
    (work / 'recs.json').write_text(json.dumps(recs, indent=1))
    np.savez(work / 'lives.npz', **lives)
    for st in ('clean', 'splice'):
        r = [x for x in recs if x['set'] == st]
        m = [v for x in r for v in x['margin'].values()]
        print(f'{st}: {len(r)} recordings, {len(m)} (recording, live) checks, refused {np.mean(np.array(m) < 0):.1%}')


def at_sr(x: np.ndarray, sr: int) -> np.ndarray:
    """x (mono) at SR, band-limited (FFT)."""
    n = int(len(x) * SR / sr)
    return x.astype(np.float32) if sr == SR else (np.fft.irfft(np.fft.rfft(x)[:n // 2 + 1], n) * SR / sr).astype(np.float32)


def stage_recs_ru(a) -> None:
    from peresmeshnik import voice
    from peresmeshnik.voice import Speaker, _prompt, _speech, _start, _windows
    d, work = Path(a.data), Path(a.work)
    spk = Speaker(Path(a.speaker), threads=a.threads)
    for sub in ('lives', 'profiles'):
        (work / sub).mkdir(parents=True, exist_ok=True)
    seg: dict[str, list[tuple[float, float]]] = {}
    for line in open(d / 'txt/segments'):
        _, talk, s0, s1 = line.split()
        seg.setdefault(talk, []).append((float(s0), float(s1)))
    recs, lives = [], {}
    for talk in sorted(seg):
        f = d / 'wav' / f'{talk}.flac'
        try:
            info = sf.info(f)
            ss = sorted(seg[talk])
            start = next(s0 for s0, _ in ss if s0 >= RU_FROM_S)
            x, sr = sf.read(f, dtype='float64', start=int(start * info.samplerate), frames=int(RECORD_S * info.samplerate),
                            always_2d=True)
            lids = []
            for k, frac in enumerate(RU_AT):
                s0 = next(s0 for s0, _ in ss if s0 >= max(frac * ss[-1][1], start + RECORD_S))
                y, _ = sf.read(f, dtype='float64', start=int(s0 * sr), frames=int(LIVE_S * sr), always_2d=True)
                if len(y) < LIVE_S * sr:
                    continue
                lid = f'{talk}-{k}'
                sf.write(work / 'lives' / f'{lid}.wav', at_sr(y.mean(axis=1), sr), SR)
                lives[lid] = spk.embed(_speech(sf.read(work / 'lives' / f'{lid}.wav', dtype='float32')[0], SR, edges=True), SR)
                lids.append(lid)
        except (StopIteration, RuntimeError, OSError, ValueError) as e:  # too short, not downloaded, cut off
            print(f'{talk}: skipped ({type(e).__name__})', flush=True)
            continue
        if len(x) < RECORD_S * sr or not lids:
            continue
        v = voice.save(at_sr(x.mean(axis=1), sr), SR, f'r{talk}', 'ru', root=work / 'profiles')
        y, rsr = sf.read(v.path / 'reference.wav', dtype='float32')
        vecs = [(spk.embed(w, rsr), m) for w, m in _windows(_start(y, rsr), rsr)]
        recs.append({'set': 'ru', 'me': talk, 'id': f'r{talk}', 'lives': lids, 'prompt_s': len(_prompt(y, rsr)) / rsr,
                     'margin': {lid: float(min(e @ lives[lid] - m for e, m in vecs)) for lid in lids}})
        print(f'ru {talk}: {len(recs)} recordings', flush=True)
    (work / 'recs.json').write_text(json.dumps(recs, indent=1))
    np.savez(work / 'lives.npz', **lives)
    m = [v for x in recs for v in x['margin'].values()]
    print(f'ru: {len(recs)} recordings, {len(m)} (recording, live) checks, refused by the windows {np.mean(np.array(m) < 0):.1%}')


# ---------- engines
def todo(work: Path, engine: str, shard: str = '0/1', reverse: bool = False, lives: bool = True):
    i, n = map(int, shard.split('/'))
    recs = json.loads((work / 'recs.json').read_text())
    out = work / 'out' / engine
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(r['id'], work / 'profiles' / r['id'], LANGS[r['set']]) for r in recs]
    seen = {}
    for r in recs:
        for lid in r['lives']:
            seen[lid] = sorted(set(seen.get(lid, ())) | set(LANGS[r['set']]))
    jobs += [(f'live_{lid}', work / 'lives' / f'{lid}.wav', tuple(ls)) for lid, ls in sorted(seen.items())] if lives else []
    jobs = [j for k, j in enumerate(jobs) if k % n == i]
    return jobs[::-1] if reverse else jobs, out


def phrases(langs, engine):
    from peresmeshnik.voice import TEST_PHRASES
    return [(lang, k, TEST_PHRASES[lang][k]) for lang in langs if engine != 'pocket' or lang == 'en' for k in (0, 1)]


def run(a, engine, cond_folder, cond_pcm, say, seed=lambda s: None) -> None:
    from peresmeshnik.engines import Voice
    jobs, out = todo(Path(a.work), engine, a.shard, a.reverse, not a.no_lives)
    times = []
    for jid, src, langs in jobs:
        todo_ph = [(lg, k, t) for lg, k, t in phrases(langs, engine) if not (out / f'{jid}_{lg}{k}.wav').exists()]
        if not todo_ph:
            continue
        t0 = time.perf_counter()
        v = (cond_folder(Voice(name=src.name, engine='reference', path=src, meta={})) if src.is_dir()
             else cond_pcm(*sf.read(src, dtype='float32')))
        tc = time.perf_counter() - t0
        for lg, k, text in todo_ph:
            seed(1000 * k + sum(map(ord, jid)))
            t1 = time.perf_counter()
            y = np.concatenate([np.zeros(0, np.float32), *say(text, lg, v)])
            ts = time.perf_counter() - t1
            sf.write(out / f'{jid}_{lg}{k}.wav', y, a.sr)
            times.append({'job': jid, 'lang': lg, 'k': k, 'cond_s': tc, 'synth_s': ts, 'sec': len(y) / a.sr})
            tc = 0.0
        print(f'{engine} {jid} {time.perf_counter() - t0:.1f} s', flush=True)
    with open(out / f'times-{a.shard.replace("/", "of")}{"r" if a.reverse else ""}.jsonl', 'a') as f:
        for t in times:
            f.write(json.dumps(t) + '\n')


def stage_pocket(a) -> None:
    from types import SimpleNamespace

    from peresmeshnik.local.clone import make_engine
    t0 = time.perf_counter()
    eng = make_engine('pocket-tts-en', a.model_dir, threads=a.threads)  # as the app makes it (int8)
    print(f'pocket start {time.perf_counter() - t0:.2f} s', flush=True)
    a.sr = eng.sample_rate

    def seed(s):
        eng._rng = np.random.default_rng(s)
    run(a, 'pocket', lambda v: SimpleNamespace(state=eng.state(v)), lambda x, sr: SimpleNamespace(state=eng.condition(x, sr)),
        lambda text, lang, v: eng.synth(text, lang, v), seed)


def _load(repo: str, name: str):
    import importlib.util
    sys.path.insert(0, str(Path(repo) / 'peresmeshnik/local'))  # qwen3.py imports xtts.py as the runner does
    spec = importlib.util.spec_from_file_location(name, Path(repo) / f'peresmeshnik/local/{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stage_xtts(a) -> None:
    from types import SimpleNamespace

    import torch
    xtts = _load(a.repo, 'xtts')
    t0 = time.perf_counter()
    eng = xtts.Engine(a.model_dir, device=a.device, threads=a.threads)
    print(f'xtts start {time.perf_counter() - t0:.2f} s', flush=True)
    a.sr = xtts.SR

    def say(text, lang, v):
        with torch.inference_mode():
            yield from eng.synth(text, lang, v)
    run(a, 'xtts', lambda v: SimpleNamespace(state=eng.latents(v)), lambda x, sr: SimpleNamespace(state=eng.condition(x, sr)),
        say, torch.manual_seed)


def stage_qwen3(a) -> None:
    from types import SimpleNamespace

    import torch
    qwen3 = _load(a.repo, 'qwen3')
    t0 = time.perf_counter()
    eng = qwen3.Engine(a.model_dir, device=a.device or 'cuda', threads=a.threads)
    print(f'qwen3 start {time.perf_counter() - t0:.2f} s', flush=True)
    a.sr = qwen3.SR
    run(a, 'qwen3', lambda v: v, lambda x, sr: SimpleNamespace(state=eng.condition(x, sr)),
        lambda text, lang, v: eng.synth(text, lang, v), torch.manual_seed)
    print(f'qwen3 runaways {eng.runaways}', flush=True)


def stage_cbox(a) -> None:
    import io

    import torch
    from chatterbox.models.tokenizers import tokenizer
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    qwen3 = _load(a.repo, 'qwen3')  # its CFG, as Qwen3's fallback speaks
    d = Path(a.model_dir)
    tokenizer.hf_hub_download = lambda repo_id, filename, **kw: str(d / filename)
    t0 = time.perf_counter()
    m = ChatterboxMultilingualTTS.from_local(d, a.device or 'cuda')
    print(f'cbox start {time.perf_counter() - t0:.2f} s', flush=True)
    a.sr = m.sr

    def folder(v):
        m.prepare_conditionals(str(Path(v.path) / 'reference.wav'))
        return m.conds

    def pcm(x, sr):
        wav = io.BytesIO()
        sf.write(wav, x, sr, format='WAV', subtype='FLOAT')
        wav.seek(0)
        m.prepare_conditionals(wav)
        return m.conds

    def say(text, lang, conds):
        m.conds = conds
        yield m.generate(text, language_id=lang, cfg_weight=qwen3.CFG).squeeze(0).cpu().numpy().astype(np.float32)
    run(a, 'cbox', folder, pcm, say, torch.manual_seed)


# ---------- score
def stage_score(a) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from peresmeshnik.voice import Speaker
    work = Path(a.work)
    recs = json.loads((work / 'recs.json').read_text())
    lives = dict(np.load(work / 'lives.npz'))
    spks = [Speaker(Path(a.speaker), threads=1) for _ in range(a.threads)]
    engines = sorted(p.name for p in (work / 'out').iterdir() if p.is_dir())
    files = sorted(f for e in engines for f in (work / 'out' / e).glob('*.wav'))
    files += sorted((work / 'refs').glob('*.wav'))

    def emb(i_f):
        i, f = i_f
        try:
            x, sr = sf.read(f, dtype='float32')
        except Exception:  # cut short by a stopped process: as not made
            print(f'unreadable: {f}', flush=True)
            return str(f.relative_to(work)), None
        return str(f.relative_to(work)), spks[i % len(spks)].embed(x, sr) if len(x) > sr // 10 else None

    def cat(i_pair):
        i, (key, fs) = i_pair
        xs = [sf.read(f, dtype='float32') for f in fs]
        sr = xs[0][1]
        gap = np.zeros(int(0.3 * sr), np.float32)
        x = np.concatenate([xs[0][0], gap, xs[1][0]])
        return key, spks[i % len(spks)].embed(x, sr)
    with ThreadPoolExecutor(len(spks)) as ex:
        vec = dict(ex.map(emb, enumerate(files)))
        pairs = {}
        for f in files:
            if (f.parent.parent.name == 'out' and f.stem.endswith('0') and (f.with_name(f.stem[:-1] + '1.wav')).exists()
                    and vec.get(str(f.relative_to(work))) is not None
                    and vec.get(str(f.with_name(f.stem[:-1] + '1.wav').relative_to(work))) is not None):
                pairs[str(f.relative_to(work))[:-5] + 'cat'] = [f, f.with_name(f.stem[:-1] + '1.wav')]
        vec |= dict(ex.map(cat, enumerate(pairs.items())))
    rows = []
    for e in engines:
        for r in recs:
            for lg in LANGS[r['set']]:
                for k in ('0', '1', 'cat'):
                    o = vec.get(f'out/{e}/{r["id"]}_{lg}{k}' + ('.wav' if k != 'cat' else ''))
                    if o is None:
                        continue
                    row = {'engine': e, 'rec': r['id'], 'set': r['set'], 'lang': lg, 'k': k,
                           'variant': r.get('variant', 'own'), 'pair': r.get('pair')}
                    for lid in r['lives']:
                        lc = vec.get(f'out/{e}/live_{lid}_{lg}{k}' + ('.wav' if k != 'cat' else ''))
                        rows.append(row | {'live': lid, 'margin': r['margin'][lid], 'cos_live': float(o @ lives[lid]),
                                           'cos_lc': None if lc is None else float(o @ lc),
                                           **({'cos_me': float(o @ vec[f'refs/{r["pair"]}_me.wav']),
                                               'cos_other': float(o @ vec[f'refs/{r["pair"]}_other.wav'])}
                                              if r['set'] == 'splice' else {})})
    (work / 'score.json').write_text(json.dumps(rows))
    # every vector, and each recording's windows as the live check cuts them (the first are its head windows)
    from peresmeshnik.voice import _prompt, _start, _windows

    def wins(i_r):
        i, r = i_r
        y, rsr = sf.read(work / 'profiles' / r['id'] / 'reference.wav', dtype='float32')
        out = {f'win/{r["id"]}/{j}/{m}': spks[i % len(spks)].embed(w, rsr) for j, (w, m) in enumerate(_windows(_start(y, rsr), rsr))}
        if len(p := _prompt(y, rsr)) >= rsr // 2:
            out[f'prompt/{r["id"]}'] = spks[i % len(spks)].embed(p, rsr)
        return out
    with ThreadPoolExecutor(len(spks)) as ex:
        for d in ex.map(wins, enumerate(recs)):
            vec |= d
    np.savez(work / 'vectors.npz', **{k: v for k, v in vec.items() if v is not None}, **{f'live/{k}': v for k, v in lives.items()})
    print(f'{len(rows)} rows, {len(vec)} vectors -> {work / "score.json"}, vectors.npz')


def stage_report(a) -> None:
    """The app's rule step by step: the live check's windows (the margins of recs.json), + the engines' prompt as a window
    and the head like the rest (voice.head_rest, HEAD_REST), + the clone's two test phrases (cos_live of 'cat',
    CLONE_MATCH): the owner's refusals, the splices passing of 15 and their outputs closer to the other person."""
    from peresmeshnik import voice
    work = Path(a.work)
    rows, R = json.loads((work / 'score.json').read_text()), json.loads((work / 'recs.json').read_text())
    V, lives = np.load(work / 'vectors.npz'), dict(np.load(work / 'lives.npz'))
    E = [e for e in ('pocket', 'xtts', 'qwen3', 'cbox') if any(x['engine'] == e for x in rows)]
    ok_rec, hr = {}, {}
    for r in R:
        vecs = [(V[k], float(k.split('/')[-1])) for k in V.files if k.startswith(f'win/{r["id"]}/')]
        p = [V[f'prompt/{r["id"]}']] if f'prompt/{r["id"]}' in V.files else []
        rest = [e for e, m in vecs if m != voice.HEAD_MATCH]
        hr[r['id']] = voice.head_rest([e for e, m in vecs if m == voice.HEAD_MATCH] + p, rest) if rest else -1.0
        if r.get('prompt_s', voice.PROMPT_S) < voice.PROMPT_MIN_S:  # a silent prompt: refused as head-rest
            hr[r['id']] = -1.0
        for lid in r['lives']:
            m = r['margin'][lid] >= 0 and all(float(e @ lives[lid]) >= voice.HEAD_MATCH for e in p)
            ok_rec[(r['id'], lid)] = (r['margin'][lid] >= 0, m, m and hr[r['id']] >= voice.HEAD_REST)
    out = {(x['engine'], x['rec'], x['live'], x['lang'], x['k']): x['cos_live'] for x in rows}
    T = sorted({0.10, 0.12, 0.14, 0.16, voice.CLONE_MATCH, 0.20})
    for own in [x for x in ('clean', 'ru') if any(r['set'] == x for r in R)]:
        cl = [(r['id'], lid) for r in R if r['set'] == own for lid in r['lives']]
        print(f'## the owner, set {own} ({len({c[0] for c in cl})} recordings, {len(cl)} checks): refused by the windows / '
              f'+ the prompt / + head-rest {voice.HEAD_REST}; then + the clone at {voice.CLONE_MATCH} (two phrases), per '
              f'engine and language; added by the clone at each threshold')
        print(' / '.join(f'{np.mean([not ok_rec[c][j] for c in cl]):.1%}' for j in range(3))
              + f'; head-rest of the owner: min {min(hr[c[0]] for c in cl):.3f}')
        print('| engine | lang | refused | added by the clone | cosine p1 | min | ' + ' | '.join(f'+ at {t:.2f}' for t in T) + ' |')
        print('|---|---|---|---|---|---|' + '---|' * len(T))
        for e in E:
            for lg in LANGS[own]:
                c = [(ok_rec[k][2], out.get((e, *k, lg, 'cat'))) for k in cl]
                if all(x is None for _, x in c):
                    continue
                cos = np.array([x for _, x in c if x is not None])
                print(f'| {e} | {lg} | {np.mean([not (g and x is not None and x >= voice.CLONE_MATCH) for g, x in c]):.1%} | '
                      f'{np.mean([g and (x is None or x < voice.CLONE_MATCH) for g, x in c]):.1%} | '
                      f'{np.percentile(cos, 1):.3f} | {cos.min():.3f} | '
                      + ' | '.join(f'{np.mean([g and (x is None or x < t) for g, x in c]):.1%}' for t in T) + ' |')
    if not any(r['set'] == 'splice' for r in R):
        return
    print('\n## splices (15 pairs, the owner\'s live reading of another chapter): passing of 15 at each step; '
          'the two outputs per engine closer to the other person among the passing')
    print('| piece | windows | + prompt, head-rest | ' + ' | '.join(f'+ {e}' for e in E) + ' |')
    print('|---|---|---|' + '---|' * len(E))
    for v in dict.fromkeys(r['variant'] for r in R if r['set'] == 'splice'):
        rs = [(r['id'], r['lives'][0]) for r in R if r['set'] == 'splice' and r['variant'] == v]
        cells = []
        for e in E:
            g = [k for k in rs if ok_rec[k][2] and (out.get((e, *k, 'en', 'cat')) or 0) >= voice.CLONE_MATCH]
            near = [x for x in rows if x['engine'] == e and (x['rec'], x['live']) in g and x['k'] in ('0', '1')]
            cells.append(f'{len(g)} ({sum(x["cos_other"] > x["cos_me"] for x in near)}/{len(near)})')
        before = [x for x in rows if (x['rec'], x['live']) in rs and ok_rec[(x['rec'], x['live'])][0] and x['k'] in ('0', '1')]
        print(f'| {v} | {sum(ok_rec[k][0] for k in rs)} ({sum(x["cos_other"] > x["cos_me"] for x in before)}/{len(before)}) | '
              f'{sum(ok_rec[k][2] for k in rs)} | ' + ' | '.join(cells) + ' |')


def stage_timing(a) -> None:
    """What the clone check adds to confirming a voice, nothing else running: the engine made (as the app makes it; the
    voice pack's runner adds its own process start), the recording conditioned, voice.clone_match (both test phrases and
    one speaker vector), for --engine on the first 3 own recordings."""
    import types

    from peresmeshnik import voice
    from peresmeshnik.engines import Voice
    work = Path(a.work)
    t0 = time.perf_counter()
    if a.engine == 'pocket':
        from peresmeshnik.local.clone import make_engine
        eng = make_engine('pocket-tts-en', a.model_dir, threads=a.threads)
        cond, say = eng.state, eng.synth
    else:
        import torch
        m = _load(a.repo, 'xtts' if a.engine == 'xtts' else 'qwen3')
        eng = m.Engine(a.model_dir, device=a.device or 'cuda', threads=a.threads)
        cond = eng.latents

        def say(text, lang, v):
            with torch.inference_mode():
                yield from eng.synth(text, lang, v)
    start = time.perf_counter() - t0
    t1 = time.perf_counter()
    try:
        spk = voice.Speaker(Path(a.speaker), threads=a.threads)
    except ImportError:  # the voice pack's environment has no onnxruntime: the app makes the vector, timed with Pocket
        spk = types.SimpleNamespace(embed=lambda pcm, sr: np.ones(256, np.float32) / 16)
    load_spk = time.perf_counter() - t1
    lives, aside = dict(np.load(work / 'lives.npz')), work / 'timing-cache'  # the engines' caches set aside: made anew
    aside.mkdir(exist_ok=True)
    for r in [r for r in json.loads((work / 'recs.json').read_text()) if r['set'] == 'clean'][:3]:
        folder = work / 'profiles' / r['id']
        for f in folder.glob('*'):
            if f.name not in ('reference.wav', 'voice.json'):
                f.rename(aside / f'{a.engine}-{r["id"]}-{f.name}')
        v = Voice(name=r['id'], engine='reference', path=folder, meta={})
        t1 = time.perf_counter()
        cond(v)
        tc = time.perf_counter() - t1
        tts = types.SimpleNamespace(sample_rate=getattr(eng, 'sample_rate', 24_000), synth=say)
        t1 = time.perf_counter()
        sim = voice.clone_match(tts, v, 'en', lives[r['lives'][0]], spk)
        print(f'{a.engine} start {start:.2f} s, speaker model {load_spk:.2f} s, {r["id"]}: conditioned {tc:.2f} s, '
              f'two phrases + vector {time.perf_counter() - t1:.2f} s (cosine {sim:.3f})', flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stage', choices=('recs', 'recs_ru', 'pocket', 'xtts', 'qwen3', 'cbox', 'score', 'report', 'timing'))
    ap.add_argument('--engine', choices=('pocket', 'xtts', 'qwen3'), help='timing: which engine')
    ap.add_argument('--data', help='LibriSpeech/dev-clean')
    ap.add_argument('--work', required=True)
    ap.add_argument('--model-dir')
    ap.add_argument('--speaker', help='wespeaker ResNet34 folder (*.onnx)')
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--device', default=None)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--shard', default='0/1', help='i/n: every n-th job from the i-th (engine stages)')
    ap.add_argument('--reverse', action='store_true', help='the jobs from the last (a second process on the same ones)')
    ap.add_argument('--no-lives', action='store_true', help='only the recordings, not the live readings cloned')
    ap.add_argument('--mid', action='store_true', help='recs: the splice set only, their piece at 12-30 s (MID_V)')
    a = ap.parse_args(argv)
    {'recs': stage_recs, 'recs_ru': stage_recs_ru, 'pocket': stage_pocket, 'xtts': stage_xtts, 'qwen3': stage_qwen3, 'cbox': stage_cbox,
     'score': stage_score, 'report': stage_report, 'timing': stage_timing}[a.stage](a)


if __name__ == '__main__':
    sys.exit(main())

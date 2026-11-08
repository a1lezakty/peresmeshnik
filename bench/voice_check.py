"""Calibrate the voice checks (peresmeshnik/voice.py): which recordings make a clone that the app's ASR
understands and that sounds like the speaker, and where the thresholds go.

    python bench/voice_check.py refs OUT                  # here: references from the bench set + degraded copies
    python bench/voice_check.py clone OUT --model-dir D   # GPU box (voice pack + speechbrain): enroll, say the
                                                          # test phrases in the other language, similarity
    python bench/voice_check.py judge OUT                 # here: the app's ASR (GigaAM / Parakeet) -> WER, table

References (~40 s each): 4 LibriSpeech test-clean speakers (CC-BY-4.0) and 4 chapters of the Russian
LibriSpeech book (LibriVox, public domain), from `$BENCH_DATA/refs/*.parquet` (bench/data.py). Cases per
reference: clean; pink noise at 14 dB below speech; another speaker underneath at 14 dB (a TV, a colleague);
a reverberant room (RT60 ~0.7 s); a phone-band microphone (300-3400 Hz). Similarity: XTTS's own speaker
encoder (what the app uses, `XttsClone.similarity`) and SpeechBrain ECAPA (an independent judge), each of the
clone against its own reference and against every other reference of that language (impostors).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peresmeshnik import voice  # noqa: E402

SR = 16_000
EN_SPEAKERS = (4446, 1580, 61, 5142)
RU_BOOKS = ('poemi_01_pushkin', 'poemi_15_pushkin', 'poemi_05_pushkin', 'poemi_02_pushkin')
CASES = ('clean', 'noise14', 'babble14', 'reverb', 'phone')
OTHER = {'en': 'ru', 'ru': 'en'}


def level(x):
    return np.sqrt(np.mean(x ** 2))


def pink(n, rng):
    f = np.fft.rfftfreq(n)
    s = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    s[1:] /= np.sqrt(f[1:])
    s[0] = 0
    x = np.fft.irfft(s, n)
    return x / level(x)


def degrade(x, case, other, rng):
    speech = level(x[np.abs(x) > 0.01 * np.abs(x).max()])  # level of the speech, not of the pauses
    if case == 'noise14':
        return x + pink(len(x), rng) * speech * 10 ** (-14 / 20)
    if case == 'babble14':
        o = np.resize(other, len(x))
        return x + o / level(o[np.abs(o) > 0.01 * np.abs(o).max()]) * speech * 10 ** (-14 / 20)
    if case == 'reverb':
        t = np.arange(int(0.8 * SR)) / SR
        rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.7)  # -60 dB at 0.7 s
        rir[0] = 1.0
        y = np.convolve(x, rir)[:len(x)]
        return y / level(y) * level(x)
    if case == 'phone':
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x), 1 / SR)
        X[(f < 300) | (f > 3400)] = 0
        return np.fft.irfft(X, len(x))
    return x


def refs_main(out: Path):
    import pyarrow.parquet as pq
    root = Path(os.environ.get('BENCH_DATA', 'bench-data')) / 'refs'
    clean = {}
    t = pq.read_table(root / 'en.parquet', columns=['speaker_id', 'audio']).to_pylist()
    for spk in EN_SPEAKERS:
        parts = [sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')[0] for r in t if r['speaker_id'] == spk]
        clean[f'en{spk}'] = np.concatenate(parts)[:40 * SR]
    t = pq.read_table(root / 'ru.parquet', columns=['audio']).column('audio').to_pylist()
    for book in RU_BOOKS:
        parts = [sf.read(io.BytesIO(r['bytes']), dtype='float32')[0] for r in t if r['path'].startswith(book)]
        clean[f'ru{book.split("_")[1]}'] = np.concatenate(parts)[:40 * SR]
    rng = np.random.default_rng(0)
    (out / 'refs').mkdir(parents=True, exist_ok=True)
    rows = []
    names = list(clean)
    for i, name in enumerate(names):
        other = clean[names[(i + 1) % len(names)]]  # another voice for the babble
        for case in CASES:
            x = degrade(clean[name], case, other, rng).astype(np.float32)
            x *= min(1.0, 0.9 / np.abs(x).max())
            f = out / 'refs' / f'{name}_{case}.wav'
            sf.write(f, x, SR)
            r = voice.check(x, SR)
            rows.append({'ref': name, 'lang': name[:2], 'case': case, 'wav': f.name, 'check_ok': r.ok, 'snr_db': r.snr_db,
                         'speech_s': r.speech_s, 'verdict': voice.verdict(r)})
            print(rows[-1], flush=True)
    (out / 'refs.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')


def clone_main(out: Path, model_dir: Path):
    import torch
    import torchaudio.functional as F
    from speechbrain.inference.speaker import EncoderClassifier

    from peresmeshnik.local.clone import XttsClone
    rows = json.loads((out / 'refs.json').read_text(encoding='utf-8'))
    eng = XttsClone(model_dir, threads=2)
    ecapa = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': 'cpu'},
                                           savedir=str(Path(os.environ.get('HF_HOME', 'models')) / 'speechbrain-ecapa'))

    def ecapa_emb(x, sr):
        x = torch.from_numpy(np.asarray(x, np.float32))
        e = ecapa.encode_batch(F.resample(x, sr, SR)[None] if sr != SR else x[None]).squeeze()
        return e / e.norm()

    def xtts_emb(x, sr):
        with torch.inference_mode():
            e = eng.model.get_speaker_embedding(eng._audio(x, sr), 22_050).flatten().float()
        return e / e.norm()
    clean = {r['ref']: sf.read(out / 'refs' / r['wav'], dtype='float32')[0] for r in rows if r['case'] == 'clean'}
    ref_x = {k: xtts_emb(v, SR) for k, v in clean.items()}
    ref_e = {k: ecapa_emb(v, SR) for k, v in clean.items()}
    (out / 'clone').mkdir(exist_ok=True)
    for r in rows:
        x, _ = sf.read(out / 'refs' / r['wav'], dtype='float32')
        v = eng.enroll(x, SR, name=f'{r["ref"]}_{r["case"]}', lang=r['lang'], root=out / 'voices')
        lang = OTHER[r['lang']]
        audio = []
        for i, text in enumerate(voice.TEST_PHRASES[lang]):
            pcm = np.concatenate(list(eng.synth(text, lang, v)))
            sf.write(out / 'clone' / f'{v.name}_{i}.wav', pcm, eng.sample_rate)
            audio.append(pcm)
        pcm = np.concatenate(audio)
        ex, ee = xtts_emb(pcm, eng.sample_rate), ecapa_emb(pcm, eng.sample_rate)
        same = [k for k in clean if k[:2] == r['lang']]
        r |= {'out_lang': lang, 'sim_app': round(eng.similarity(v, pcm, eng.sample_rate), 3),  # vs its own (degraded) recording
              'sim_xtts': round(float(ex @ ref_x[r['ref']]), 3), 'sim_ecapa': round(float(ee @ ref_e[r['ref']]), 3),
              'imp_xtts': [round(float(ex @ ref_x[k]), 3) for k in same if k != r['ref']],
              'imp_ecapa': [round(float(ee @ ref_e[k]), 3) for k in same if k != r['ref']]}
        print(r, flush=True)
    (out / 'clone.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')


def judge_main(out: Path):
    from peresmeshnik import config
    from peresmeshnik.local.asr import OnnxAsr
    rows = json.loads((out / 'clone.json').read_text(encoding='utf-8'))
    loc = config.Settings().local
    for lang in ('en', 'ru'):  # one model in memory at a time
        asr = OnnxAsr(loc.asr[lang], threads=2)
        for r in (r for r in rows if r['out_lang'] == lang):
            heard = []
            for i, _ in enumerate(voice.TEST_PHRASES[lang]):
                x, sr = sf.read(out / 'clone' / f'{r["ref"]}_{r["case"]}_{i}.wav', dtype='float32')
                x = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x).astype(np.float32)
                heard.append(asr.recognize(x).text)
            r['wer'] = round(voice.wer(' '.join(voice.TEST_PHRASES[lang]), ' '.join(heard)), 3)
            r['heard'] = heard
        del asr
    (out / 'judge.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print('| ref | case | recording check | WER | sim (app: vs own recording) | sim XTTS vs clean | max impostor XTTS | '
          'sim ECAPA vs clean | max impostor ECAPA |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f'| {r["ref"]}→{r["out_lang"]} | {r["case"]} | {"ok" if r["check_ok"] else "fail"} ({r["snr_db"]:.0f} dB) | '
              f'{r["wer"]:.2f} | {r["sim_app"]:.2f} | {r["sim_xtts"]:.2f} | {max(r["imp_xtts"]):.2f} | '
              f'{r["sim_ecapa"]:.2f} | {max(r["imp_ecapa"]):.2f} |')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['refs', 'clone', 'judge'])
    ap.add_argument('out', type=Path)
    ap.add_argument('--model-dir', type=Path)
    a = ap.parse_args()
    {'refs': lambda: refs_main(a.out), 'clone': lambda: clone_main(a.out, a.model_dir), 'judge': lambda: judge_main(a.out)}[a.cmd]()

"""Pocket TTS alone as the app makes it (clone.make_engine: int8 FlowLM, 2 threads): first audio and RTF per FLEURS dev
sentence, warm (the voice state made and one sentence said before the timed ones). Research R1 (2026-11-16): the first
Apple Silicon numbers, from the GitHub macOS arm64 runner.

    python bench/pocket_speed.py profiles DIR          # ref_en / ref_ru of the bench set as voice profiles in DIR
    python bench/pocket_speed.py standin DIR           # an English stand-in (below) in DIR
    python bench/pocket_speed.py speed pocket-tts-ru --voice DIR/ref_en [--model-dir M] [--n 10] [--threads 2]

English stand-in: Kyutai's gated `pocket-tts-en` cannot be fetched without a token, so the open
`kyutai/pocket-tts-without-voice-cloning` English weights (the same model, its Mimi encoder zeroed) get the Mimi encoder
of the registry's Shtorm-PocketTTS-RU (bit for bit Kyutai's, BENCH stage 8). Same architecture and shapes: for speed.
"""
import argparse
import json
import statistics
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the package from this checkout
from common import SET, manifest, record  # noqa: E402

STANDIN = ('https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/'
           'a7056f505a4a8ca72d8b5fb1440071d6f1a35739/languages/english/')
WARM = {'en': 'Hello, this is a warm-up sentence.', 'ru': 'Здравствуйте, это фраза для разогрева.'}


def profiles(a):
    from peresmeshnik import voice
    for lang in ('en', 'ru'):
        x, sr = sf.read(SET / f'ref_{lang}.wav', dtype='float32')
        v = voice.save(x, sr, f'ref_{lang}', lang, root=Path(a.dir))
        print(v.path, round(len(x) / sr, 1), 's')


def write_safetensors(path: Path, w: dict) -> None:
    kinds = {'<f4': 'F32', '<f2': 'F16', '<f8': 'F64', '<i8': 'I64', '<i4': 'I32'}
    header, off, arrays = {}, 0, []
    for k, x in w.items():
        x = np.ascontiguousarray(x)
        header[k] = {'dtype': kinds[x.dtype.str], 'shape': list(x.shape), 'data_offsets': [off, off + x.nbytes]}
        off += x.nbytes
        arrays.append(x)
    h = json.dumps(header).encode()
    h += b' ' * (-len(h) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h)) + h)
        for x in arrays:
            f.write(x.tobytes())


def standin(a):
    from peresmeshnik import models
    from peresmeshnik.local.pocket import read_safetensors, weights_file
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('model.safetensors', 'tokenizer.model'):
        if not (out / f'open-{name}').exists():
            urllib.request.urlretrieve(STANDIN + name, out / f'open-{name}')
    (out / 'tokenizer.model').write_bytes((out / 'open-tokenizer.model').read_bytes())
    w = read_safetensors(out / 'open-model.safetensors')
    ru = read_safetensors(weights_file(models.path('pocket-tts-ru')))
    enc = [k for k in w if k.startswith('mimi.encoder')]
    print(f'open weights: {len(enc)} Mimi encoder tensors, {sum(bool(w[k].any()) for k in enc)} of them not zero')
    for k in enc:
        assert ru[k].shape == w[k].shape, k
        w[k] = ru[k]
    write_safetensors(out / 'model.safetensors', w)
    print(f'{out}: {len(enc)} Mimi encoder tensors from pocket-tts-ru, {len(w)} tensors')


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 3)


def speed(a):
    from peresmeshnik import models, voice
    from peresmeshnik.local.clone import make_engine
    lang = a.engine.rsplit('-', 1)[1]
    t = time.perf_counter()
    eng = make_engine(a.engine, a.model_dir or models.path(a.engine), a.threads)
    load_s = time.perf_counter() - t
    v = voice.load(a.voice.name, a.voice.parent)
    t = time.perf_counter()
    eng.state(v)
    state_s = time.perf_counter() - t
    for _ in eng.synth(WARM[lang], lang, v):
        pass
    rows = []
    for it in manifest()[:a.n]:
        text = it[lang]['text']
        t0, c0, first, n = time.perf_counter(), time.process_time(), None, 0
        for c in eng.synth(text, lang, v):
            first = first or time.perf_counter() - t0
            n += len(c)
        dt, cpu, audio = time.perf_counter() - t0, time.process_time() - c0, n / eng.sample_rate
        rows.append({'first_s': round(first, 3), 'rtf': round(dt / audio, 3), 'cpu_per_audio_s': round(cpu / audio, 3),
                     'audio_s': round(audio, 2), 'chars': len(text)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    fs, rt = [r['first_s'] for r in rows], [r['rtf'] for r in rows]
    audio = sum(r['audio_s'] for r in rows)
    record('pocket_speed', a.engine, model_dir=str(a.model_dir or ''), voice=a.voice.name, threads=a.threads, int8=True,
           n=len(rows), load_s=round(load_s, 2), state_s=round(state_s, 2),
           first_s_median=round(statistics.median(fs), 3), first_s_p90=pct(fs, 90), first_s_max=max(fs),
           rtf_median=round(statistics.median(rt), 3), rtf_p90=pct(rt, 90),
           rtf_total=round(sum(r['rtf'] * r['audio_s'] for r in rows) / audio, 3),
           cpu_per_audio_s=round(sum(r['cpu_per_audio_s'] * r['audio_s'] for r in rows) / audio, 3), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('profiles', 'standin'):
        sub.add_parser(name).add_argument('dir')
    s = sub.add_parser('speed')
    s.add_argument('engine')
    s.add_argument('--voice', type=Path, required=True, help='voice profile folder')
    s.add_argument('--model-dir', type=Path)
    s.add_argument('--n', type=int, default=10)
    s.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    {'profiles': profiles, 'standin': standin, 'speed': speed}[a.cmd](a)


if __name__ == '__main__':
    main()

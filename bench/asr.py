"""ASR benchmark: latency, RTF, CPU load, peak RAM and WER on the bench set.

    python bench/asr.py fw large-v3-turbo en       # faster-whisper (CTranslate2), int8
    python bench/asr.py onnx gigaam-v3-e2e-ctc ru  # onnx-asr: gigaam-*, t-tech/t-one, nemo-parakeet-tdt-0.6b-v3, ...
    python bench/asr.py onnx onnx-community/whisper-large-v3-turbo ja --fleurs --device cuda

`--fleurs`: the first `--n` sentences of FLEURS dev in any of the 14 languages (bench/data.py), with CER the
measure for zh / ja (no spaces) and *other language* = the share of results that read as another of the 14
(a model without a language setting drifting, e.g. Parakeet into Ukrainian; `lid`). One model per process, so peak
RSS belongs to that model. Threads are fixed (--threads, default 4).
"""
import argparse
import re
import time
import unicodedata

import numpy as np
import soundfile as sf
from data import CODES, fleurs

from common import SET, Timer, manifest, record

NO_SPACES = {'zh', 'ja'}


def norm(s):
    s = unicodedata.normalize('NFKC', s).lower().replace('ё', 'е')
    s = re.sub(r"[^\w\s']|_", ' ', s)
    return ' '.join(s.split())


def grams(text, n=3):
    t = f' {norm(text)} '
    return {g: t.count(g) for g in {t[i:i + n] for i in range(len(t) - n + 1)}}


def lid(profiles=None):
    """Language of a text by its letter trigrams against each language's FLEURS test text (other sentences
    than dev): {lang: profile} -> classify(text) -> lang or ''."""
    if profiles is None:
        profiles = {}
        for lang in CODES:
            g = grams(' '.join(t for _, t in fleurs(lang, split='test', audio=False).values()))
            profiles[lang] = (g, float(np.sqrt(sum(v * v for v in g.values()))))

    def classify(text):
        g = grams(text)
        if not norm(text):
            return ''
        return max(profiles, key=lambda lang: sum(v * profiles[lang][0].get(k, 0) for k, v in g.items()) / profiles[lang][1])
    return classify


def load_fw(name, lang, threads, device):
    from faster_whisper import WhisperModel
    m = WhisperModel(name, device=device, compute_type='int8' if device == 'cpu' else 'float16', cpu_threads=threads)

    def run(x):
        segs, _ = m.transcribe(x, language=lang, beam_size=1, vad_filter=False, condition_on_previous_text=False)
        return ' '.join(s.text.strip() for s in segs)
    return run, None


def load_onnx(name, lang, threads, device, path=None, quant='auto'):
    import onnx_asr

    from peresmeshnik.local.vad import ort_options
    so = ort_options(threads)  # as the app: fixed threads, no spinning (pools of 3 sessions on 2 cores fight otherwise)
    providers = {'coreml': ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
                 'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider']}.get(device, ['CPUExecutionProvider'])
    if quant == 'auto':
        quant = 'int8' if 'parakeet' in name or ('whisper' in name and device == 'cpu') else None
    if path is None:  # a registry model: its pinned files and quantization
        from peresmeshnik import models
        from peresmeshnik.local.asr import ONNX_ASR
        rid = next((i for i, (n, _) in ONNX_ASR.items() if n == name), None)
        path, quant = (models.path(rid), ONNX_ASR[rid][1]) if rid else (None, quant)
    m = onnx_asr.load_model(name, path, quantization=quant, sess_options=so, providers=providers)
    if 'whisper' in name:  # as the app
        from peresmeshnik.local.asr import cap_whisper
        cap_whisper(m)

    def run(x):
        r = m.recognize(x, language=lang) if 'canary' in name or 'whisper' in name else m.recognize(x)
        return r if isinstance(r, str) else r.text

    chunk = None
    asr = getattr(m, 'asr', None)
    if asr is not None and hasattr(asr, '_encode_chunk'):  # T-one: true streaming, fixed 300 ms chunks at 8 kHz
        n, state = asr._chunk_size, np.zeros((1, asr._state_size), np.float16)

        def chunk():
            nonlocal state
            _, state = asr._encode_chunk(np.zeros((1, n), np.float32), state)
        chunk.ms = n / 8
    return run, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('engine', choices=['fw', 'onnx'])
    ap.add_argument('model')
    ap.add_argument('lang', choices=sorted(CODES))
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--path', help='onnx: the model files (default: the registry model of that onnx-asr name)')
    ap.add_argument('--quant', help="onnx: the files' quantization suffix (int8, fp16; 'none' = plain)")
    ap.add_argument('--fleurs', action='store_true', help='FLEURS dev of the language, not the en/ru bench set')
    a = ap.parse_args()
    if fl := a.fleurs or a.lang not in ('en', 'ru'):
        rows = fleurs(a.lang)
        items = [{a.lang: {'wav': str(rows[i][0]), 'text': rows[i][1]}} for i in sorted(rows)[:a.n]]
    else:
        items = manifest()[:a.n]
    import jiwer
    classify = lid()

    t_load = time.perf_counter()
    run, chunk = load_fw(a.model, a.lang, a.threads, a.device) if a.engine == 'fw' else \
        load_onnx(a.model, a.lang, a.threads, a.device, a.path, {None: 'auto', 'none': None}.get(a.quant, a.quant))
    t_load = time.perf_counter() - t_load
    run(np.zeros(16000, np.float32))  # warm-up

    lat, refs, hyps, audio, other, short = [], [], [], 0.0, {}, {}
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for it in items:
        x, sr = sf.read(SET / it[a.lang]['wav'], dtype='float32')
        with Timer() as t:
            hyp = run(x)
        lat.append(t.s)
        audio += len(x) / sr
        refs.append(norm(it[a.lang]['text']))
        hyps.append(norm(hyp))
        if (got := classify(hyp)) != a.lang:
            other[got or 'empty'] = other.get(got or 'empty', 0) + 1
        # a stream's first partials: 1.5 s. Counted when the reference's as many first words read as the language
        # (a few words alone mislead the trigrams too) and the result does not
        head = run(x[:int(1.5 * sr)])
        ref_head = ' '.join(it[a.lang]['text'].split()[:max(len(head.split()), 1)]) if a.lang not in NO_SPACES else \
            it[a.lang]['text'][:max(len(head), 1)]
        if (got := classify(head)) not in (a.lang, '') and classify(ref_head) == a.lang:
            short[got] = short.get(got, 0) + 1
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    chars = (lambda t: t.replace(' ', '')) if a.lang in NO_SPACES else (lambda t: t)

    # cost of one short update: what a streaming UI would pay per refresh on a 1 / 2 / 3 / 5 s window
    x5 = np.concatenate([sf.read(SET / it[a.lang]['wav'], dtype='float32')[0] for it in items[:3]])[:80000]
    win = {}
    for sec in (1, 2, 3, 5):
        ts = [Timer() for _ in range(3)]
        for t in ts:
            with t:
                run(x5[:sec * 16000])
        win[sec] = round(1000 * float(np.median([t.s for t in ts])))
    extra = {}
    if chunk:
        ts = []
        for _ in range(50):
            with Timer() as t:
                chunk()
            ts.append(t.s)
        extra = {'chunk_ms': chunk.ms, 'chunk_latency_ms': round(1000 * float(np.median(ts)), 1)}
    record('asr', f'{a.engine}:{a.model}', lang=a.lang, device=a.device, threads=a.threads, n=len(items),
           audio_s=round(audio, 1), load_s=round(t_load, 2),
           utt_latency_ms_median=round(1000 * float(np.median(lat))), utt_latency_ms_p90=round(1000 * float(np.percentile(lat, 90))),
           win2s_latency_ms=win[2], win_latency_ms=win,
           rtf=round(wall / audio, 3), cpu_cores=round(cpu / wall, 2), cpu_s_per_audio_s=round(cpu / audio, 3),
           wer=round(jiwer.wer(refs, hyps), 4), cer=round(jiwer.cer([chars(r) for r in refs], [chars(h) for h in hyps]), 4),
           other_lang=round(sum(other.values()) / len(items), 3), other=other,
           other_lang_1_5s=round(sum(short.values()) / len(items), 3), other_1_5s=short,
           data='fleurs-dev' if fl else 'set', sample=hyps[0][:120], **extra)


if __name__ == '__main__':
    main()

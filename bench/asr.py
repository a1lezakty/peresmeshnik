"""ASR benchmark: latency, RTF, CPU load, peak RAM and WER on the bench set.

    python bench/asr.py fw large-v3-turbo en       # faster-whisper (CTranslate2), int8
    python bench/asr.py onnx gigaam-v3-e2e-ctc ru  # onnx-asr: gigaam-*, t-tech/t-one, nemo-parakeet-tdt-0.6b-v3, ...

One model per process, so peak RSS belongs to that model. Threads are fixed (--threads, default 4).
"""
import argparse
import re
import time
import unicodedata

import numpy as np
import soundfile as sf

from common import SET, Timer, manifest, record


def norm(s):
    s = unicodedata.normalize('NFKC', s).lower().replace('ё', 'е')
    s = re.sub(r"[^\w\s']|_", ' ', s)
    return ' '.join(s.split())


def load_fw(name, lang, threads, device):
    from faster_whisper import WhisperModel
    m = WhisperModel(name, device=device, compute_type='int8' if device == 'cpu' else 'float16', cpu_threads=threads)

    def run(x):
        segs, _ = m.transcribe(x, language=lang, beam_size=1, vad_filter=False, condition_on_previous_text=False)
        return ' '.join(s.text.strip() for s in segs)
    return run, None


def load_onnx(name, lang, threads, device):
    import onnx_asr
    import onnxruntime as rt
    so = rt.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if device == 'coreml' else ['CPUExecutionProvider']
    quant = 'int8' if 'parakeet' in name else None
    m = onnx_asr.load_model(name, quantization=quant, sess_options=so, providers=providers)

    def run(x):
        r = m.recognize(x, language=lang) if 'canary' in name else m.recognize(x)
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
    ap.add_argument('lang', choices=['en', 'ru'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    items = manifest()[:a.n]
    import jiwer

    t_load = time.perf_counter()
    run, chunk = (load_fw if a.engine == 'fw' else load_onnx)(a.model, a.lang, a.threads, a.device)
    t_load = time.perf_counter() - t_load
    run(np.zeros(16000, np.float32))  # warm-up

    lat, refs, hyps, audio = [], [], [], 0.0
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for it in items:
        x, sr = sf.read(SET / it[a.lang]['wav'], dtype='float32')
        with Timer() as t:
            hyp = run(x)
        lat.append(t.s)
        audio += len(x) / sr
        refs.append(norm(it[a.lang]['text']))
        hyps.append(norm(hyp))
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0

    # cost of one short update: what a streaming UI would pay per refresh on a 2 s window
    x2 = sf.read(SET / items[0][a.lang]['wav'], dtype='float32')[0][:32000]
    w2 = [Timer() for _ in range(3)]
    for t in w2:
        with t:
            run(x2)
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
           win2s_latency_ms=round(1000 * float(np.median([t.s for t in w2]))),
           rtf=round(wall / audio, 3), cpu_cores=round(cpu / wall, 2), cpu_s_per_audio_s=round(cpu / audio, 3),
           wer=round(jiwer.wer(refs, hyps), 4), cer=round(jiwer.cer(refs, hyps), 4),
           sample=hyps[0][:120], **extra)


if __name__ == '__main__':
    main()

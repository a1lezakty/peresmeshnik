"""TTS without cloning: time to first audio, RTF, CPU, peak RAM; intelligibility in a separate pass.

    python bench/tts.py piper en|ru     # Piper (rhasspy/piper-voices, medium)
    python bench/tts.py silero en|ru    # Silero TTS (torch)
    python bench/tts.py judge piper ru  # back-transcribe saved wavs with faster-whisper small -> CER

The 10 shortest bench sentences per language; wavs go to $BENCH_OUT/tts/<engine>-<lang>/.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ROOT, manifest, record

OUT = Path(os.environ.get('BENCH_OUT', ROOT.parent / 'out')) / 'tts'
PIPER = {'en': 'en/en_US/lessac/medium/en_US-lessac-medium', 'ru': 'ru/ru_RU/irina/medium/ru_RU-irina-medium'}
SILERO = {'en': ('v3_en', 'en_0'), 'ru': ('v5_6_ru', 'xenia')}


def sentences(lang, n=10):
    return sorted((it[lang]['text'] for it in manifest()), key=len)[:n]


def piper(lang, threads):
    import onnxruntime as rt
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice
    onnx = hf_hub_download('rhasspy/piper-voices', PIPER[lang] + '.onnx')
    hf_hub_download('rhasspy/piper-voices', PIPER[lang] + '.onnx.json')
    v = PiperVoice.load(onnx)
    so = rt.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    v.session = rt.InferenceSession(onnx, sess_options=so, providers=['CPUExecutionProvider'])

    def stream(text):  # one chunk per sentence
        for c in v.synthesize(text):
            yield c.audio_float_array, c.sample_rate
    return stream


def silero(lang, threads):
    import torch
    torch.set_num_threads(threads)
    pkg, speaker = SILERO[lang]
    model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts', language=lang, speaker=pkg, trust_repo=True)

    def stream(text):
        yield model.apply_tts(text=text, speaker=speaker, sample_rate=24000).numpy(), 24000
    return stream


def judge(engine, lang, threads):
    import jiwer
    from asr import norm
    from faster_whisper import WhisperModel
    m = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=threads)
    refs, hyps = [], []
    for i, text in enumerate(sentences(lang)):
        x, sr = sf.read(OUT / f'{engine}-{lang}' / f'{i}.wav', dtype='float32')
        x = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x).astype(np.float32)  # to 16 kHz
        segs, _ = m.transcribe(x, language=lang, beam_size=1)
        refs.append(norm(text))
        hyps.append(norm(' '.join(s.text for s in segs)))
    record('tts_cer', engine, lang=lang, judge='fw:small', cer=round(jiwer.cer(refs, hyps), 4), wer=round(jiwer.wer(refs, hyps), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('engine', choices=['piper', 'silero', 'judge'])
    ap.add_argument('args', nargs='+')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    if a.engine == 'judge':
        return judge(*a.args, a.threads)
    lang = a.args[0]
    t0 = time.perf_counter()
    stream = globals()[a.engine](lang, a.threads)
    load_s = time.perf_counter() - t0
    for _ in stream('Hello.' if lang == 'en' else 'Привет.'):  # warm-up
        pass
    out = OUT / f'{a.engine}-{lang}'
    out.mkdir(parents=True, exist_ok=True)
    ttfa, total, audio, cpu0, wall0 = [], 0.0, 0.0, time.process_time(), time.perf_counter()
    for i, text in enumerate(sentences(lang)):
        parts, t = [], time.perf_counter()
        for x, sr in stream(text):
            if not parts:
                ttfa.append(time.perf_counter() - t)
            parts.append(x)
        total += time.perf_counter() - t
        x = np.concatenate(parts)
        audio += len(x) / sr
        sf.write(out / f'{i}.wav', x, sr)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    record('tts', a.engine, lang=lang, voice=PIPER[lang].rsplit('/', 1)[1] if a.engine == 'piper' else '/'.join(SILERO[lang]),
           threads=a.threads, n=len(ttfa), load_s=round(load_s, 1), ttfa_ms_median=round(1000 * float(np.median(ttfa))),
           rtf=round(total / audio, 3), cpu_cores=round(cpu / wall, 2), audio_s=round(audio, 1))


if __name__ == '__main__':
    main()

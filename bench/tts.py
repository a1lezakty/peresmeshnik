"""TTS without cloning: time to first audio, RTF, CPU, peak RAM; intelligibility in a separate pass.

    python bench/tts.py run joe          # a voice from VOICES
    python bench/tts.py judge joe        # back-transcribe its wavs with faster-whisper small -> CER

Piper is GPL, so it is never imported: it runs as its own process, the upstream CLI from an isolated
uv tool env ($PIPER_CMD, default `uvx --from piper-tts==1.8.0 piper`), fed one line per sentence.
The 10 shortest bench sentences per language; wavs go to $BENCH_OUT/tts/<engine>-<voice>/.
"""
import argparse
import os
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ROOT, manifest, record

OUT = Path(os.environ.get('BENCH_OUT', ROOT.parent / 'out')) / 'tts'
PV = 'rhasspy/piper-voices'
VOICES = {  # name: (engine, lang, voice); chosen voices first, lessac/irina were measured before (non-free data)
    'joe': ('piper', 'en', 'en/en_US/joe/medium/en_US-joe-medium'),
    'john': ('piper', 'en', 'en/en_US/john/medium/en_US-john-medium'),
    'denis': ('piper', 'ru', 'ru/ru_RU/denis/medium/ru_RU-denis-medium'),
    'dmitri': ('piper', 'ru', 'ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium'),
    'lessac': ('piper', 'en', 'en/en_US/lessac/medium/en_US-lessac-medium'),
    'irina': ('piper', 'ru', 'ru/ru_RU/irina/medium/ru_RU-irina-medium'),
    'silero-en': ('silero', 'en', 'v3_en/en_0'),
    'silero-ru': ('silero', 'ru', 'v5_6_ru/xenia'),
}


def sentences(lang, n=10):
    return sorted((it[lang]['text'] for it in manifest()), key=len)[:n]


def piper(path, threads):
    from huggingface_hub import hf_hub_download
    onnx = hf_hub_download(PV, path + '.onnx')
    hf_hub_download(PV, path + '.onnx.json')
    tmp = OUT / 'piper-tmp' / str(os.getpid())
    cmd = shlex.split(os.environ.get('PIPER_CMD', 'uvx --from piper-tts==1.8.0 piper'))
    # The CLI has no thread option and onnxruntime pins its default pool to every physical core
    # (taskset does not bound it), so Piper here runs with one thread per physical core.
    cmd += ['-m', onnx, '--output_dir', str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    def stream(text):  # Piper synthesizes a sentence at once: its first audio is the whole wav
        proc.stdin.write(' '.join(text.split()) + '\n')
        proc.stdin.flush()
        for line in proc.stderr:
            if 'Wrote ' in line:
                wav = Path(line.rsplit('Wrote ', 1)[1].strip())
                x, sr = sf.read(wav, dtype='float32')
                yield x, sr
                return
        raise RuntimeError('piper exited')
    stream.proc = proc
    return stream


def silero(spec, threads):
    import torch
    torch.set_num_threads(threads)
    pkg, speaker = spec.split('/')
    lang = pkg.rsplit('_', 1)[1]
    model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts', language=lang, speaker=pkg, trust_repo=True)

    def stream(text):
        yield model.apply_tts(text=text, speaker=speaker, sample_rate=24000).numpy(), 24000
    return stream


def tree_cpu_s(pid):
    import psutil
    total = 0.0
    p = psutil.Process(pid)
    for q in [p, *p.children(recursive=True)]:
        t = q.cpu_times()
        total += t.user + t.system
    return total


def judge(name, threads):
    import jiwer
    from asr import norm
    from faster_whisper import WhisperModel
    engine, lang, spec = VOICES[name]
    m = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=threads)
    refs, hyps = [], []
    for i, text in enumerate(sentences(lang)):
        x, sr = sf.read(OUT / f'{engine}-{name}' / f'{i}.wav', dtype='float32')
        x = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x).astype(np.float32)  # to 16 kHz
        segs, _ = m.transcribe(x, language=lang, beam_size=1)
        refs.append(norm(text))
        hyps.append(norm(' '.join(s.text for s in segs)))
    record('tts_cer', engine, lang=lang, voice=spec.rsplit('/', 1)[-1] if engine == 'piper' else spec, judge='fw:small',
           cer=round(jiwer.cer(refs, hyps), 4), wer=round(jiwer.wer(refs, hyps), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['run', 'judge'])
    ap.add_argument('voice', choices=list(VOICES))
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    if a.step == 'judge':
        return judge(a.voice, a.threads)
    engine, lang, spec = VOICES[a.voice]
    t0 = time.perf_counter()
    stream = globals()[engine](spec, a.threads)
    for _ in stream('Hello.' if lang == 'en' else 'Привет.'):  # warm-up
        pass
    load_s = time.perf_counter() - t0  # includes one warm-up sentence
    proc = getattr(stream, 'proc', None)
    out = OUT / f'{engine}-{a.voice}'
    out.mkdir(parents=True, exist_ok=True)
    cpu_child0 = tree_cpu_s(proc.pid) if proc else 0.0
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
    extra = {}
    if proc:
        cpu += tree_cpu_s(proc.pid) - cpu_child0
        proc.stdin.close()
        proc.wait(30)
        r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # the Piper process tree, now reaped
        extra['peak_rss_mb'] = round(r / (2**20 if sys.platform == 'darwin' else 1024))
    if proc:
        import psutil
        a.threads = psutil.cpu_count(logical=False)
    record('tts', engine, lang=lang, voice=spec.rsplit('/', 1)[-1] if engine == 'piper' else spec, threads=a.threads,
           n=len(ttfa), load_s=round(load_s, 1), ttfa_ms_median=round(1000 * float(np.median(ttfa))),
           rtf=round(total / audio, 3), cpu_cores=round(cpu / wall, 2), audio_s=round(audio, 1), **extra)


if __name__ == '__main__':
    main()

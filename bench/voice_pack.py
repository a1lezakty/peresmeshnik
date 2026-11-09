"""The voice clone in its own process (the voice pack's runner) on a GPU box: time to first audio, RTF, how fast a
barge-in stops it, peak VRAM.

    python bench/voice_pack.py --voice VOICE_DIR --model-dir XTTS_DIR [--vram-gb 6] [--runs 3] [--threads 2]

The runner starts from $PERESMESHNIK_VOICE_CMD (the voice pack's own Python and voice-runner/voice_runner.py), else
from this Python (a venv with torch CUDA and coqui-tts: bench/requirements-tts-clone-xtts.txt). --vram-gb caps its CUDA
memory (a 6 GB card on a bigger one). VRAM: torch's peak inside the runner (allocated, reserved) and nvidia-smi's peak
for the runner's process, sampled every 0.2 s. The first sentence is a warm-up, not timed. One `voice_pack` row per
language in bench/results/<host>.jsonl; with --out the wavs too.
"""
import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the package from this checkout
from peresmeshnik.engines import Voice  # noqa: E402
from peresmeshnik.local.clone import RUNNER, VoicePackClone  # noqa: E402

from common import record  # noqa: E402

TEXTS = {
    'en': ['Good morning, can you hear me well?',
           'I think we should move the release to next week, the tests are not ready yet.',
           'Thank you, that was very helpful.'],
    'ru': ['Доброе утро, вы меня хорошо слышите?',
           'Я думаю, релиз стоит перенести на следующую неделю: тесты ещё не готовы.',
           'Спасибо, это очень помогло.'],
}


def smi_peak(pid: int, stop: threading.Event, peak: list) -> None:
    while not stop.wait(0.2):
        try:
            out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        for line in out.splitlines():
            p, mb = (x.strip() for x in line.split(','))
            if p == str(pid) and mb.isdigit():
                peak[0] = max(peak[0], int(mb))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--voice', type=Path, required=True, help='voice profile folder (reference.wav)')
    ap.add_argument('--model-dir', type=Path, required=True, help='XTTS-v2 folder')
    ap.add_argument('--vram-gb', type=float)
    ap.add_argument('--device', help='cpu: no CUDA needed (slower than real time)')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--runs', type=int, default=3, help='each sentence this many times')
    ap.add_argument('--langs', default='en,ru')
    ap.add_argument('--out', type=Path, help='folder for the wavs')
    a = ap.parse_args()
    cmd = shlex.split(os.environ['PERESMESHNIK_VOICE_CMD']) if os.environ.get('PERESMESHNIK_VOICE_CMD') \
        else [sys.executable, str(RUNNER)]
    cmd += [*(['--vram-gb', str(a.vram_gb)] if a.vram_gb else []), *(['--device', a.device] if a.device else [])]
    t = time.perf_counter()
    eng = VoicePackClone(a.model_dir, a.threads, cmd)
    load_s = time.perf_counter() - t
    print(f'runner up in {load_s:.1f} s: {eng.info}', flush=True)
    stop, peak = threading.Event(), [0]
    threading.Thread(target=smi_peak, args=(eng._p.pid, stop, peak), daemon=True).start()
    v = Voice(a.voice.name, 'xtts', a.voice)
    try:
        eng.latents(v)
        for _ in eng.synth(TEXTS['en'][0], 'en', v):  # warm-up
            pass
        for lang in a.langs.split(','):
            ttfa, rtf, stops = [], [], []
            for i in range(a.runs):
                for j, text in enumerate(TEXTS[lang]):
                    t, chunks = time.perf_counter(), []
                    for c in eng.synth(text, lang, v):
                        if not chunks:
                            ttfa.append(time.perf_counter() - t)
                        chunks.append(c)
                    wav = np.concatenate(chunks)
                    rtf.append((time.perf_counter() - t) * eng.sample_rate / len(wav))
                    if a.out and i == 0:
                        import soundfile as sf
                        a.out.mkdir(parents=True, exist_ok=True)
                        sf.write(a.out / f'{lang}_{j}.wav', wav, eng.sample_rate)
                gen = eng.synth(TEXTS[lang][1], lang, v)  # barge-in after the first chunk: how long close() takes
                next(gen)
                t = time.perf_counter()
                gen.close()
                stops.append(time.perf_counter() - t)
            s = eng.stats()
            record('voice_pack', 'xtts-v2', lang=lang, device=eng.info.get('device'), gpu=eng.info.get('gpu'),
                   vram_cap_gb=a.vram_gb, threads=a.threads, runs=a.runs, load_s=round(load_s, 2),
                   ttfa_s=round(float(np.median(ttfa)), 3), ttfa_max_s=round(max(ttfa), 3),
                   rtf=round(float(np.median(rtf)), 3), rtf_max=round(max(rtf), 3),
                   stop_ms=round(1000 * float(np.median(stops))), stop_max_ms=round(1000 * max(stops)),
                   vram_peak_gb=s.get('vram_peak_gb'), vram_reserved_peak_gb=s.get('vram_reserved_peak_gb'),
                   smi_peak_mb=peak[0] or None)
    finally:
        stop.set()
        eng.close()


if __name__ == '__main__':
    main()

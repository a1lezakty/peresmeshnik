"""Voice pack runner: XTTS-v2 voice clone in its own process, on the voice pack's own Python (torch CUDA, coqui-tts).

The app never imports torch for the clone where the voice pack is installed: it starts this script and talks to it
over stdin/stdout (local/clone.py VoicePackClone). Its own process keeps a second OpenMP runtime, the pack's numpy
and huggingface packages and a CUDA crash away from the app, and the CPML-licensed weights behind the pipe. It needs
no `peresmeshnik` package: the model code is one file, peresmeshnik/local/xtts.py (or an `xtts.py` beside this one).

    python voice_runner.py --model-dir XTTS_DIR [--device cuda|cpu] [--threads 2] [--vram-gb 6]

Without --device it needs CUDA (on CPU the clone is slower than real time: an explicit --device cpu only).
Protocol, all JSON in ASCII:
- start: one JSON line {"sample_rate": 24000, "device": "cuda", "cuda": true, "vram_total_gb": ..., ...} or
  {"error": "..."} (then it exits);
- requests: one JSON line each, with a client-chosen "id": {"cmd": "synth", "text", "lang", "voice": <profile folder>},
  {"cmd": "latents", "voice", "fresh": bool}, {"cmd": "similarity", "voice", "pcm": <base64 float32 LE>, "sr"},
  {"cmd": "info"}; {"cmd": "stop", "id"} at any time stops that synth after its current chunk (barge-in);
- a synth reply: chunks of uint32 LE byte count + float32 LE mono PCM; 0 ends it; 0xFFFFFFFF + a JSON line
  {"error"} ends it with an error. Any other reply is one JSON line: the result fields, or {"error": "..."}.
The process sleeps in blocking reads between requests; stdin closed ends it.
"""
import argparse
import base64
import importlib.util
import json
import os
import queue
import struct
import sys
import threading
import traceback
from pathlib import Path

ERR = 0xFFFFFFFF
HERE = Path(__file__).resolve().parent


def engine_path() -> Path:
    """The model code: beside this script (a voice pack folder), else in the checkout's package."""
    return next((p for p in (HERE / 'xtts.py', HERE.parent / 'peresmeshnik' / 'local' / 'xtts.py') if p.exists()),
                HERE / 'xtts.py')


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True, help='XTTS-v2 folder (config.json, model.pth, vocab.json)')
    ap.add_argument('--device', help='cuda or cpu (default: cuda, or an error without it)')
    ap.add_argument('--threads', type=int, default=2, help='CPU threads of torch')
    ap.add_argument('--vram-gb', type=float, help='cap of CUDA memory: emulates a smaller card')
    ap.add_argument('--engine', type=Path, default=engine_path(), help=argparse.SUPPRESS)  # tests: a stub engine
    a = ap.parse_args()
    out = os.fdopen(os.dup(1), 'wb')  # the protocol; the libraries' prints go to stderr instead
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj) -> None:
        out.write(json.dumps(obj).encode() + b'\n')
        out.flush()

    try:
        xtts = load_module(a.engine)
        info = xtts.cuda_info()
        if a.device is None and not info.get('cuda'):
            raise RuntimeError(f'no CUDA GPU for the voice clone (torch {info.get("torch")} sees none)')
        device = a.device or 'cuda'
        if a.vram_gb and device == 'cuda':
            xtts.cap_vram(a.vram_gb)
            info['vram_cap_gb'] = a.vram_gb
        engine = xtts.Engine(a.model_dir, device=device, threads=a.threads)
    except Exception as e:  # the client says why instead of "exited (1)"
        traceback.print_exc()
        say({'error': f'{type(e).__name__}: {e}'})
        return 1
    say({'sample_rate': engine.sample_rate, 'device': device, 'model': str(a.model_dir), **info})

    requests: queue.Queue = queue.Queue()
    stop = {'id': None, 'eof': False}  # the synth to stop; stdin closed: any

    def read() -> None:  # blocking reads: no CPU between requests
        for line in sys.stdin.buffer:
            req = json.loads(line)
            if req.get('cmd') == 'stop':
                stop['id'] = req.get('id')
            else:
                requests.put(req)
        stop['eof'] = True
        requests.put(None)
    threading.Thread(target=read, name='stdin', daemon=True).start()

    while (req := requests.get()) is not None:
        cmd, rid = req.get('cmd'), req.get('id')
        try:
            if cmd == 'synth':
                gen = engine.synth(req['text'], req['lang'], req['voice'])
                try:
                    for c in gen:
                        if stop['id'] == rid or stop['eof']:
                            break
                        pcm = c.astype('<f4').tobytes()
                        out.write(struct.pack('<I', len(pcm)) + pcm)
                        out.flush()
                finally:
                    gen.close()  # stops XTTS's generation at once
                out.write(struct.pack('<I', 0))
                out.flush()
            elif cmd == 'latents':
                if req.get('fresh'):
                    engine.forget(req['voice'])
                engine.latents(req['voice'])
                say({'ok': True})
            elif cmd == 'similarity':
                import numpy as np
                pcm = np.frombuffer(base64.b64decode(req['pcm']), '<f4')
                say({'similarity': engine.similarity(req['voice'], pcm, req['sr'])})
            elif cmd == 'info':
                say(xtts.cuda_info())
            else:
                say({'error': f'unknown request {cmd!r}'})
        except Exception as e:  # CUDA out of memory, a bad voice folder: the reply says so, the runner goes on
            traceback.print_exc()
            msg = {'error': f'{type(e).__name__}: {e}'}
            if cmd == 'synth':
                out.write(struct.pack('<I', ERR))
            say(msg)
    return 0


if __name__ == '__main__':
    sys.exit(main())

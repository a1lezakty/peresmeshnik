"""Voice pack runner: the voice clone (XTTS-v2 or Qwen3-TTS) in its own process, on the voice pack's own Python (torch CUDA).

The app never imports torch for the clone where the voice pack is installed: it starts this script and talks to it
over stdin/stdout (local/clone.py VoicePackClone). Its own process keeps a second OpenMP runtime, the pack's numpy
and huggingface packages and a CUDA crash away from the app, and the CPML-licensed weights behind the pipe. It needs
no `peresmeshnik` package: the model code is one file per kind, peresmeshnik/local/<kind>.py (or `<kind>.py` beside this
one): xtts.py (the default), qwen3.py (Qwen3-TTS; Chatterbox Multilingual from --fallback-dir takes over a voice it runs
away on; it uses xtts.py's guards).

    python voice_runner.py --model-dir MODEL_DIR [--kind xtts|qwen3] [--fallback-dir CHATTERBOX_DIR] [--device cuda|cpu]
                           [--threads 2] [--vram-gb 6]

Without --device it needs CUDA (on CPU the clone is slower than real time: an explicit --device cpu only).
Protocol, all JSON in ASCII:
- start: one JSON line {"sample_rate": 24000, "device": "cuda", "cuda": true, "vram_total_gb": ..., ...} or
  {"error": "..."} (then it exits);
- requests: one JSON line each, with a client-chosen "id": {"cmd": "synth", "text", "lang", "voice": <profile folder>},
  {"cmd": "latents", "voice", "fresh": bool}, {"cmd": "similarity", "voice", "pcm": <base64 float32 LE>, "sr"},
  {"cmd": "info"} (CUDA memory; qwen3 adds its runaways and the voices Chatterbox speaks); {"cmd": "stop", "id"} at any
  time stops that synth after its current chunk (barge-in);
- a voice from a sample, no profile folder (the other person in a call, voices/live.py): {"cmd": "condition", "pcm":
  <base64 float32 LE mono>, "sr"} -> {"state": "<id>"}; then {"cmd": "synth", "text", "lang", "state": "<id>"} speaks
  in it. The runner keeps the last MAX_STATES of them in memory (a synth uses one again: it stays); a state it no longer
  has, or another runner's, is that synth's error;
- a synth reply: chunks of uint32 LE byte count + float32 LE mono PCM; 0 ends it; 0xFFFFFFFF + a JSON line
  {"error"} ends it with an error. Any other reply is one JSON line: the result fields, or {"error": "..."}.
The process sleeps in blocking reads between requests; stdin closed ends it.
"""
import argparse
import base64
import contextlib
import importlib.util
import json
import os
import queue
import secrets
import struct
import sys
import threading
import traceback
import types
from collections import OrderedDict
from pathlib import Path

ERR = 0xFFFFFFFF
HERE = Path(__file__).resolve().parent
# voices from a sample kept: each is a few hundred kB (XTTS's latents on the GPU, Qwen3's vector and its 10 s sample);
# live.py makes up to 4 per person in a call. ponytail: a fixed count, a client's "forget" if calls outgrow it
MAX_STATES = 32


def engine_path(kind: str = 'xtts') -> Path:
    """The model code: beside this script (a voice pack folder), else in the checkout's package."""
    return next((p for p in (HERE / f'{kind}.py', HERE.parent / 'peresmeshnik' / 'local' / f'{kind}.py') if p.exists()),
                HERE / f'{kind}.py')


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True, help='the model folder: XTTS-v2 (config.json, model.pth, vocab.json) '
                    'or Qwen3-TTS 0.6B Base')
    ap.add_argument('--kind', choices=['xtts', 'qwen3'], default='xtts', help='the clone engine (default: xtts)')
    ap.add_argument('--fallback-dir', help='qwen3: Chatterbox Multilingual folder (ve.pt, t3_mtl23ls_v2.safetensors, '
                    's3gen.pt, ...): it speaks a voice Qwen3 ran away on')
    ap.add_argument('--device', help='cuda or cpu (default: cuda, or an error without it)')
    ap.add_argument('--threads', type=int, default=2, help='CPU threads of torch')
    ap.add_argument('--vram-gb', type=float, help='cap of CUDA memory: emulates a smaller card')
    ap.add_argument('--engine', type=Path, help=argparse.SUPPRESS)  # tests: a stub engine
    a = ap.parse_args()
    out = os.fdopen(os.dup(1), 'wb')  # the protocol; the libraries' prints go to stderr instead
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj) -> None:
        out.write(json.dumps(obj).encode() + b'\n')
        out.flush()

    try:
        code = load_module(a.engine or engine_path(a.kind))
        info = code.cuda_info()
        if a.device is None and not info.get('cuda'):
            raise RuntimeError(f'no CUDA GPU for the voice clone (torch {info.get("torch")} sees none)')
        device = a.device or 'cuda'
        if a.vram_gb and device == 'cuda':
            code.cap_vram(a.vram_gb)
            info['vram_cap_gb'] = a.vram_gb
        kw = {'fallback_dir': a.fallback_dir} if a.fallback_dir else {}
        engine = code.Engine(a.model_dir, device=device, threads=a.threads, **kw)
    except Exception as e:  # the client says why instead of "exited (1)"
        with contextlib.suppress(OSError):  # a stderr nobody reads (a pipe of a windowed app) must not hide the reason
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

    states: OrderedDict = OrderedDict()  # id -> the engine's state of a voice from a sample, oldest first

    def voice(req):
        """A synth's voice: its profile folder, or a state of `condition` (as an object with `state`, as live.py's)."""
        if 'state' not in req:
            return req['voice']
        if (sid := req['state']) not in states:
            raise LookupError(f'no voice state {sid!r}: this runner keeps the last {MAX_STATES} it made, and none '
                              'of a runner before it')
        states.move_to_end(sid)
        return types.SimpleNamespace(state=states[sid])

    def samples(req):
        import numpy as np
        return np.frombuffer(base64.b64decode(req['pcm']), '<f4')

    while (req := requests.get()) is not None:
        cmd, rid = req.get('cmd'), req.get('id')
        try:
            if cmd == 'synth':
                gen = engine.synth(req['text'], req['lang'], voice(req))
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
                say({'similarity': engine.similarity(req['voice'], samples(req), req['sr'])})
            elif cmd == 'condition':
                sid = secrets.token_hex(8)  # random: an id of a runner before this one is never another voice here
                states[sid] = engine.condition(samples(req), req['sr'])
                while len(states) > MAX_STATES:
                    states.popitem(last=False)
                say({'state': sid})
            elif cmd == 'info':
                say({**code.cuda_info(), **(engine.info() if hasattr(engine, 'info') else {})})
            else:
                say({'error': f'unknown request {cmd!r}'})
        except Exception as e:  # CUDA out of memory, a bad voice folder: the reply says so, the runner goes on
            with contextlib.suppress(OSError):
                traceback.print_exc()
            msg = {'error': f'{type(e).__name__}: {e}'}
            if cmd == 'synth':
                out.write(struct.pack('<I', ERR))
            say(msg)
    return 0


if __name__ == '__main__':
    sys.exit(main())

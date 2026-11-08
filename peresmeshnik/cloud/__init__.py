"""Cloud mode: Gemini Live behind the Direction interface.

    peresmeshnik cloud --in call.wav --out heard.wav                 # them -> me, file at real-time pace
    peresmeshnik cloud --direction me-them --subtitles --in me.wav --out sent.wav
    parec -d <sink>.monitor --format=s16le --rate=16000 --channels=1 \\
      | peresmeshnik cloud --in - --out - | pacat --format=s16le --rate=24000 --channels=1 -d <sink>

`-` = raw s16le mono on stdin (16 kHz) / stdout (24 kHz), e.g. PipeWire devices through
parec / pacat. Captions go to stdout (stderr when stdout carries audio); per-link latency goes
to stderr, with a median summary at the end.
"""
from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from peresmeshnik import config
from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech

FRAME = SAMPLE_RATE // 50  # 20 ms, like a capture callback


def api_key(cfg: config.Cloud) -> str:
    """$GEMINI_API_KEY, else `cloud.key_file`, else `gemini-key` in the config folder
    (~/.config/Peresmeshnik on Linux; the lowercase ~/.config/peresmeshnik is read too)."""
    key = os.environ.get('GEMINI_API_KEY', '').strip()
    d = config.config_dir()
    paths = [Path(cfg.key_file)] if cfg.key_file else [d / 'gemini-key', d.with_name(d.name.lower()) / 'gemini-key']
    for path in paths:
        if not key and path.is_file():
            key = path.read_text().strip()
    if not key:
        raise SystemExit(f'no Gemini API key: set GEMINI_API_KEY or put the key into {paths[0]}')
    return key


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SAMPLE_RATE):
            raise SystemExit(f'{path}: need 16 kHz mono 16-bit wav (ffmpeg -i in -ac 1 -ar 16000 out.wav)')
        return np.frombuffer(w.readframes(w.getnframes()), '<i2')


def frames(src: str, tail_max: float, out_quiet):
    """Yield 20 ms int16 frames at capture pace, then silence until the output has been quiet for
    2.5 s (at least 3 s, at most `tail_max`) so the translator can finish."""
    t0, i = time.monotonic(), 0
    if src == '-':  # the pipe sets the pace. os.read, not sys.stdin.buffer: a daemon thread blocked inside the
        rest = b''  # buffered reader holds its lock, and Python aborts at exit (a failed connect: core dump, not an error)
        while b := os.read(sys.stdin.fileno(), FRAME * 2):
            b, rest = rest + b, b''
            if len(b) % 2:
                b, rest = b[:-1], b[-1:]
            if b:  # one byte read: nothing whole yet
                yield np.frombuffer(b, '<i2')
    else:
        pcm = read_wav(src)
        for i in range(0, len(pcm), FRAME):
            time.sleep(max(0.0, t0 + (i + FRAME) / SAMPLE_RATE - time.monotonic()))  # a frame leaves when captured
            yield pcm[i:i + FRAME]
    end, z = time.monotonic(), np.zeros(FRAME, np.int16)
    while (t := time.monotonic()) - end < tail_max and not (t - end > 3 and t - out_quiet() > 2.5):
        time.sleep(FRAME / SAMPLE_RATE)
        yield z


def cli(argv: list[str]) -> int:
    s = config.load()
    ap = argparse.ArgumentParser(prog='peresmeshnik cloud', description='Gemini Live direction on a file or a raw PCM pipe.')
    ap.add_argument('--in', dest='src', required=True, help='16 kHz mono wav, or - for raw s16le on stdin')
    ap.add_argument('--out', help='translated speech: wav (24 kHz), or - for raw s16le on stdout')
    ap.add_argument('--direction', choices=['them-me', 'me-them'], default='them-me')
    ap.add_argument('--subtitles', action='store_true', default=s.cloud.subtitles, help='also run transcribe-live')
    ap.add_argument('--reconnect-s', type=float, default=s.cloud.reconnect_s)
    ap.add_argument('--tail-max', type=float, default=15.0, help='max seconds of silence after the input ends')
    a = ap.parse_args(argv)
    src, tgt = (s.their_lang, s.my_lang) if a.direction == 'them-me' else (s.my_lang, s.their_lang)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s', stream=sys.stderr)
    for noisy in ('google_genai', 'websockets', 'httpx'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    text_out = sys.stderr if a.out == '-' else sys.stdout
    cfg = config.Cloud(**{**vars(s.cloud), 'subtitles': a.subtitles, 'reconnect_s': a.reconnect_s})

    from peresmeshnik.cloud.gemini import GeminiDirection
    d = GeminiDirection(src, tgt, api_key(cfg), cfg)
    last_audio = [time.monotonic()]
    audio: list[np.ndarray] = []

    def feed():
        try:
            for f in frames(a.src, a.tail_max, lambda: last_audio[0]):
                d.feed(f)
        finally:
            d.close()
    threading.Thread(target=feed, daemon=True).start()
    try:
        for e in d.events():
            if isinstance(e, Speech):
                last_audio[0] = time.monotonic()
                pcm = (np.clip(e.pcm, -1, 1) * 32767).astype('<i2')
                if a.out == '-':
                    sys.stdout.buffer.write(pcm.tobytes())
                    sys.stdout.buffer.flush()
                else:
                    audio.append(pcm)
            elif isinstance(e, Caption):
                tag = 'translation' if e.translation else 'final' if e.final else 'partial'
                print(f'[{tag}] {e.original}' + (f' => {e.translation}' if e.translation else ''), file=text_out, flush=True)
    except KeyboardInterrupt:
        d.close()
    if a.out and a.out != '-':
        from peresmeshnik.cloud.gemini import OUT_RATE
        with wave.open(a.out, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(OUT_RATE)
            w.writeframes(b''.join(x.tobytes() for x in audio))
    by: dict[str, list[float]] = {}
    for name, v in d.latency:
        by.setdefault(name, []).append(v)
    for name, v in by.items():
        p90 = sorted(v)[-(-9 * len(v) // 10) - 1]  # nearest rank
        print(f'latency {name}: median {statistics.median(v):.2f} s, p90 {p90:.2f} s, n={len(v)}', file=sys.stderr)
    return 0

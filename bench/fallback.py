"""The cloud side's local fallback with the real local path (Parakeet / GigaAM, OPUS-MT, Piper) beside a fake Gemini Live
server (tests/fake_gemini.py, a real WebSocket over TLS): a recording plays in real time, a fault comes at `--at` s, and
the times are measured: fault -> the side decides (the tray message), fault -> its first local voice; with `back`, the
server answers again and the phrase start -> back in the cloud. One JSON line per run.

    python bench/fallback.py --fault drop --runs 3      # drop | quota | slow | back
"""
import argparse
import contextlib
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

import numpy as np  # noqa: E402
from fake_gemini import FakeGemini  # noqa: E402

from peresmeshnik import config  # noqa: E402
from peresmeshnik.cloud import fallback  # noqa: E402
from peresmeshnik.cloud.gemini import GeminiDirection  # noqa: E402
from peresmeshnik.engines import SAMPLE_RATE, Caption, Latency, Speech  # noqa: E402
from peresmeshnik.pipeline import MAX_LAG_S, make_local, read_audio  # noqa: E402


def one(fault: str, at: float, rec: str, s) -> dict:
    fake = FakeGemini(lag=1.0)
    states, got = [], []
    s.cloud.fallback_after_s = 6.0

    def tts():
        from peresmeshnik.local.tts import PiperTts
        return PiperTts(s.local.tts['ru'], threads=s.local.threads)
    t0 = time.perf_counter()
    fb = fallback.Fallback(
        'them', 'ru', False,
        make_cloud=lambda: GeminiDirection('en', 'ru', cfg=s.cloud, client=fake.client(), fail_fast=True),
        make_local=lambda t: make_local('en', 'ru', tts=t, settings=s, max_lag_s=MAX_LAG_S),
        make_tts=tts, preload=True, on_state=lambda x: states.append((time.monotonic(), x)))
    fb.cloud.ready.wait(10)
    fb.muted.load()  # Piper up, as the app's preload has it by the first phrase
    load_s = time.perf_counter() - t0
    consumer = threading.Thread(target=lambda: [got.append((time.monotonic(), e)) for e in fb.events()], daemon=True)
    consumer.start()
    pcm = read_audio(rec)
    pcm = np.concatenate([pcm, pcm]) if fault == 'back' else pcm  # back: the second pass is after the server is up
    t_fault = t_up = None
    start = time.perf_counter()
    for i in range(0, len(pcm), 512):
        time.sleep(max(0.0, start + i / SAMPLE_RATE - time.perf_counter()))
        now = i / SAMPLE_RATE
        if t_fault is None and now >= at:
            t_fault = time.monotonic()
            if fault == 'slow':
                fake.lag = 60.0
            elif fault == 'quota':
                fake.refuse = (1011, 'RESOURCE_EXHAUSTED: You exceeded your current quota')
                fake.drop()
            else:
                fake.down = True
                fake.drop()
        if fault == 'back' and t_up is None and now >= at + 8:
            fake.down, t_up = False, time.monotonic()
        fb.feed(pcm[i:i + 512])
    time.sleep(3)
    fb.close()
    fake.close()
    down = next((t for w, t in fb.switched if w == 'down'), None)
    local = [t for t, e in got if isinstance(e, Speech) and e.sample_rate == fallback.Muted.sample_rate and t > t_fault
             and not np.allclose(e.pcm[:200], 0) and down is not None and t >= down]
    r = {'fault': fault, 'engines_load_s': round(load_s, 1), 'decided_s': round(down - t_fault, 2) if down else None,
         'first_local_voice_s': round(local[0] - t_fault, 2) if local else None, 'mode': fb.mode,
         'messages': [x for _, x in states]}
    if fault == 'back':
        back = next((t for w, t in fb.switched if w == 'back'), None)
        r['back_after_up_s'] = round(back - t_up, 2) if back else None
    return r


def ends(pcm) -> list[float]:
    """The phrase ends in a recording (s from its start) by the client's own energy gate ('pause' 300 ms after one): the
    client's clock for the first voice after a clause, without the server's word for it."""
    from peresmeshnik.cloud.gemini import BLOCK, _Gate
    g, x, out = _Gate(), (np.clip(pcm, -1, 1) * 32767).astype(np.int16), []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        if g.update(x[i:i + BLOCK], i / SAMPLE_RATE) == 'pause':
            out.append((i + BLOCK) / SAMPLE_RATE - 0.3)
    return out


def q(v: list[float]) -> dict:
    v = sorted(v)
    return {'p50': round(statistics.median(v), 2), 'p95': round(v[min(len(v) - 1, int(0.95 * len(v)))], 2),
            'max': round(v[-1], 2), 'n': len(v)} if v else {}


def remote(fault: str, at: float, rec: str, s, side: str, paired: dict, tunnel: str) -> dict:
    import shlex
    import signal
    import subprocess
    proc = subprocess.Popen(shlex.split(tunnel)) if tunnel else None
    try:
        time.sleep(2 if proc else 0)  # the tunnel up
        states, got = [], []
        t0 = time.monotonic()
        fb = fallback.make(side, s, '', on_state=lambda x: states.append(x), remote=paired)
        fb.cloud.ready.wait(120)  # client.READY_S: a cold server loads its models
        ready_s = time.monotonic() - t0
        consumer = threading.Thread(target=lambda: [got.append((time.monotonic(), e)) for e in fb.events()], daemon=True)
        consumer.start()
        pcm = read_audio(rec)
        pcm = np.concatenate([pcm, pcm]) if fault == 'back' else pcm
        t_fault = t_up = None
        start = time.monotonic()
        for i in range(0, len(pcm), 512):
            time.sleep(max(0.0, start + i / SAMPLE_RATE - time.monotonic()))
            now = i / SAMPLE_RATE
            if fault != 'none' and t_fault is None and now >= at:
                t_fault = time.monotonic()
                proc.kill() if fault == 'drop' else proc.send_signal(signal.SIGSTOP)
            if fault == 'back' and t_up is None and now >= at + 8:
                proc.send_signal(signal.SIGCONT)
                t_up = time.monotonic()
            fb.feed(pcm[i:i + 512])
        time.sleep(5)
        fb.close()
    finally:
        if proc:
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGCONT)
            proc.kill()
            proc.wait()
    cut = t_fault or float('inf')
    before = [(t, e) for t, e in got if t < cut]
    voice = [t for t, e in before if isinstance(e, Speech)]
    starts = [t for i, t in enumerate(voice) if i == 0 or t - voice[i - 1] > 0.15]  # a voice after a quiet gap
    ends_m = [start + x for x in ends(pcm)]
    after_final, heard = [], []
    for t, e in before:
        if isinstance(e, Caption) and e.final:
            if (nxt := next((v for v in voice if v > t), None)) is not None:
                after_final.append(nxt - t)  # the client's "no voice slow_s after a subtitle" (any voice clears it)
            end = max((x for x in ends_m if t - 3 < x < t), default=None)
            if end is not None and (first := next((v for v in starts if v > t), None)) is not None:
                heard.append(first - end)  # ponytail: a voice queued right after another has no gap: counted late
    down = next((t for w, t in fb.switched if w == 'down'), None)
    back = next((t for w, t in fb.switched if w == 'back'), None)
    return {'fault': fault, 'side': side, 'ready_s': round(ready_s, 1), 'rtt_s': fb.cloud.rtt if fb.mode == 'cloud' else None,
            'caption_s': q([e.s for _, e in before if isinstance(e, Latency) and e.what == 'caption']),
            'voice_s': q([e.s for _, e in before if isinstance(e, Latency) and e.what == 'voice']),
            'heard_s': q(heard), 'voice_after_subtitle_s': q(after_final),
            'decided_s': round(down - t_fault, 2) if down and t_fault else None,
            'back_after_up_s': round(back - t_up, 2) if back and t_up else None, 'mode': fb.mode, 'messages': states}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fault', choices=['drop', 'quota', 'slow', 'back', 'stall', 'none'], default='drop')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--at', type=float, default=8.0, help='seconds into the recording')
    ap.add_argument('--in', dest='rec', default=str(ROOT / 'tests' / 'data' / 'call-en.ogg'))
    ap.add_argument('--remote', nargs='?', const='', help="your own server: the pairing file (default: this computer's)")
    ap.add_argument('--tunnel', default='', help='the command of the tunnel to the server, run and faulted here')
    ap.add_argument('--side', choices=['them', 'me'], default='them')
    ap.add_argument('--voice', default='', help='me: the voice profile that goes up (confirmed live)')
    a = ap.parse_args()
    s = config.Settings()
    s.their_lang, s.my_lang = 'en', 'ru'
    if a.remote is not None:
        from peresmeshnik.remote import pairing
        if a.fault in ('quota', 'slow') or (a.fault != 'none' and not a.tunnel):
            ap.error('--remote: --fault none, or drop | stall | back with --tunnel')
        if (paired := pairing.load(Path(a.remote) if a.remote else None)) is None:
            ap.error('not paired with a server: tray menu, Server…')
        s.them_to_me = s.me_to_them = 'remote'
        s.local.voice = a.voice
        rec = a.rec if a.side == 'them' or a.rec != ap.get_default('rec') else str(ROOT / 'tests' / 'data' / 'call-ru.ogg')
        for _ in range(a.runs):
            print(json.dumps(remote(a.fault, a.at, rec, s, a.side, paired, a.tunnel), ensure_ascii=False), flush=True)
        return 0
    if a.fault == 'back':
        fallback.RETRY_S = 0.0
    rows = [one(a.fault, a.at, a.rec, s) for _ in range(a.runs)]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    for k in ('decided_s', 'first_local_voice_s', 'back_after_up_s'):
        if v := [r[k] for r in rows if r.get(k) is not None]:
            print(f'{a.fault} {k}: median {statistics.median(v):.2f} s, max {max(v):.2f} s, n={len(v)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

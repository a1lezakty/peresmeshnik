"""The cloud side's local fallback with the real local path (Parakeet / GigaAM, OPUS-MT, Piper) beside a fake Gemini Live
server (tests/fake_gemini.py, a real WebSocket over TLS): a recording plays in real time, a fault comes at `--at` s, and
the times are measured: fault -> the side decides (the tray message), fault -> its first local voice; with `back`, the
server answers again and the phrase start -> back in the cloud. One JSON line per run.

    python bench/fallback.py --fault drop --runs 3      # drop | quota | slow | back
"""
import argparse
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
from peresmeshnik.engines import SAMPLE_RATE, Speech  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fault', choices=['drop', 'quota', 'slow', 'back'], default='drop')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--at', type=float, default=8.0, help='seconds into the recording')
    ap.add_argument('--in', dest='rec', default=str(ROOT / 'tests' / 'data' / 'call-en.ogg'))
    a = ap.parse_args()
    s = config.Settings()
    s.their_lang, s.my_lang = 'en', 'ru'
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

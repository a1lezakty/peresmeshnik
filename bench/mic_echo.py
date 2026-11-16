"""Stage 5: echo of the translation in my microphone (no headphones). Them -> me plays its translation (Piper, my
language) from the speakers; the microphone hears it back a little later through the room, as loud as my own voice or
quieter, and I answer, a third of the time over its end. The me -> them side would translate it back to them, and its
partial subtitle would cut the translation I hear (barge-in at 3 words). The gate is S1's (audio/echo.py, the app's
`EchoGate(mic_echo)` on the 'me' side, with its threshold [local] mic_echo_rho_on); its reference is what the
headphones / speakers play (`Referenced`).

    python bench/mic_echo.py --n 6 [--delay 0.08] [--gain 0] [--rt60 0.4] [--no-echo-path]

Data: my turns are FLEURS ru dev recordings (CC-BY-4.0, other people) at -26 dBFS; what the speakers play is Piper
`dmitri` reading other FLEURS ru sentences. Scored as bench/call.py `echo` (content words per utterance), one row
without the gate and one with it: *echo translated* = played utterances that come out as a clause; *my recall*, *my
blocks zeroed* (false suppression); *false barge-ins* = clauses of >= 3 words made of the echo alone (a partial that
long cuts the translation I hear).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
from peresmeshnik.pipeline import make_local, read_audio  # noqa: E402  first: it pins BLAS threads
import numpy as np  # noqa: E402
# isort: on

from call import SR, NoTts, Scenario, clauses, content, fleurs, gate, level, score  # noqa: E402

from common import record  # noqa: E402

BARGE_IN_WORDS = 3  # app.py: my partial subtitle this long cuts the other side's voice


def room(x: np.ndarray, delay: float, gain_db: float, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """What the microphone hears of `x` played on the speakers: their band (laptop speakers, 200-8000 Hz), the room
    (a few reflections, then a tail decaying 60 dB in `rt60` s), `delay` s later (output and input buffers) and
    `gain_db` against my own speech at the microphone."""
    n = 101
    t = np.arange(n) - n // 2
    lp = lambda f: 2 * f / SR * np.sinc(2 * f / SR * t)  # noqa: E731
    band = (lp(7900) - lp(200)) * np.hamming(n)
    k = int(rt60 * SR)
    h = rng.normal(0, 1, k) * 10 ** (-3 * np.arange(k) / k) * 0.08
    h[0] = 1.0
    for d in rng.integers(int(0.002 * SR), int(0.02 * SR), 6):  # early reflections: walls, the desk
        h[d] += rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
    y = np.convolve(np.convolve(x, band), h)
    y *= 10 ** (gain_db / 20) * 0.05 / max(float(np.sqrt(np.mean(y ** 2))), 1e-12)  # my speech is at 0.05 (-26 dBFS)
    return np.concatenate([np.zeros(int(delay * SR), np.float32), y.astype(np.float32)])


def make(seed: int, pool: list, played: list, a) -> Scenario:
    """6 rounds: the speakers say a translation, I answer 0.6-1.2 s after it (every third round 0.5-1.5 s before its
    end: over it)."""
    rng = np.random.default_rng(seed)
    pick, said = iter(rng.permutation(len(pool))), iter(rng.permutation(len(played)))
    sc, t = Scenario(240), 0.5
    for turn in range(6):
        x, text = played[next(said)]
        sc.add('ours', x, t, text, None if a.no_echo_path else room(x, a.delay, a.gain, a.rt60, rng), a.delay)
        end = t + len(x) / SR + a.delay
        wav, mine = pool[next(pick)]
        y = level(read_audio(wav))
        t = end - rng.uniform(0.5, 1.5) if turn % 3 == 2 else end + rng.uniform(0.6, 1.2)
        sc.add('them', y, t, mine)  # 'them' in Scenario's terms: the speech that is not our echo (mine here)
        t += len(y) / SR + rng.uniform(0.6, 1.2)
    return sc.trim()


def barge_ins(sc: Scenario, rows: list[dict]) -> int:
    """Clauses of >= BARGE_IN_WORDS words whose content words are the echo's and none mine."""
    n = 0
    for r in rows:
        near = [s for s in sc.segs if s['start'] - 1.0 <= r['end_s'] and r['start_s'] <= s['end'] + 1.0]
        words = content(r['text'])
        echo = sum(any(w in content(s['text']) for s in near if s['who'] == 'ours') for w in words)
        mine = sum(any(w in content(s['text']) for s in near if s['who'] == 'them') for w in words)
        n += len(r['text'].split()) >= BARGE_IN_WORDS and echo >= 2 and not mine
    return n


def mic_gate(sc: Scenario, rho_on: float | None = None) -> tuple[np.ndarray, dict]:
    """bench/call.py `gate` with the microphone side's threshold: [local] mic_echo_rho_on, as the app's 'me' side."""
    from peresmeshnik import config
    from peresmeshnik.audio import echo
    rho, init = config.Local().mic_echo_rho_on if rho_on is None else rho_on, echo.EchoGate.__init__
    echo.EchoGate.__init__ = lambda self, ref, **k: init(self, ref, **{'rho_on': rho, **k})
    try:
        return gate(sc)
    finally:
        echo.EchoGate.__init__ = init


def synth_played(n: int, seed: int) -> list:
    """What them -> me says from the speakers: Piper `dmitri` (as the app's ru voice) reading FLEURS ru sentences."""
    from peresmeshnik.audio.resample import Resampler
    from peresmeshnik.local.tts import PiperTts
    texts = sorted({t for _, t in fleurs('ru')})
    rng = np.random.default_rng(1000 + seed)
    tts, out = PiperTts('piper-ru_RU-dmitri-medium', threads=2), []
    for i in rng.permutation(len(texts))[:n]:
        x = np.concatenate(list(tts.synth(texts[i], 'ru')))
        out.append((Resampler(tts.sample_rate, SR)(x.astype(np.float32)), texts[i]))
    tts.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6, help='scenarios (seeds)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--delay', type=float, default=0.08, help='speakers -> microphone, after play() is called, s')
    ap.add_argument('--gain', type=float, default=0.0, help='the echo at the microphone against my own voice, dB')
    ap.add_argument('--rt60', type=float, default=0.4, help='the room: reverb time, s')
    ap.add_argument('--no-echo-path', action='store_true', help='headphones: the reference plays, nothing comes back')
    ap.add_argument('--rho-on', type=float, help="the gate's threshold (default: the app's [local] mic_echo_rho_on)")
    a = ap.parse_args()
    if a.rho_on is None:
        from peresmeshnik import config
        a.rho_on = config.Local().mic_echo_rho_on
    base = make_local('ru', 'en', threads=a.threads, tts=NoTts())  # the me -> them side, its subtitles
    pool, played = fleurs('ru'), synth_played(40, a.seed)
    tot: dict = {'none': {}, 'echo-gate': {}}
    for seed in range(a.seed, a.seed + a.n):
        sc = make(seed, pool, played, a)
        pcm, g = mic_gate(sc, a.rho_on)
        for name, x, extra in (('none', sc.capture, {}), ('echo-gate', pcm, g)):
            rows = clauses(base, x)
            s = score(sc, rows) | extra | {'barge_ins': barge_ins(sc, rows)}
            print(seed, name, json.dumps(s), flush=True)
            for k, v in s.items():
                tot[name].setdefault(k, []).append(v)
    for name, t in tot.items():
        sums = {k: sum(v) for k, v in t.items() if k.endswith(('_out', '_utterances', 'blocks', 'barge_ins'))}
        means = {k: round(float(np.mean(xs)), 4) if (xs := [x for x in v if x is not None]) else None
                 for k, v in t.items() if k.endswith(('recall', 'zeroed'))}
        echo_n = sums.get('ours_utterances', 0)
        record('mic_echo', name, n=a.n, seed=a.seed, rho_on=a.rho_on, delay_s=a.delay, gain_db=a.gain, rt60_s=a.rt60,
               echo_path=not a.no_echo_path,
               echo_translated=round(sums.get('ours_out', 0) / echo_n, 3) if echo_n else None, echo_out=sums.get('ours_out'),
               echo_utterances=echo_n, my_out=sums.get('them_out'), my_utterances=sums.get('them_utterances'),
               my_recall=means.get('them_recall'), echo_recall=means.get('ours_recall'),
               my_blocks_zeroed=means.get('their_blocks_zeroed'), echo_blocks_zeroed=means.get('echo_blocks_zeroed'),
               false_barge_ins=sums.get('barge_ins'))
    base.close()


if __name__ == '__main__':
    main()

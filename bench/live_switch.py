"""Crossfade length of "translation <-> live voice" (audio/mic.py LiveSwitch) on real speech: how big the jump at
the seam is against the two signals' own biggest jumps around it, for several fade lengths.

    python bench/live_switch.py TRANSLATION.wav [...] --live LIVE.wav [--n 300]

TRANSLATION: clone output (24 kHz), LIVE: a microphone recording; seams at random moments where both are speech.
The fade is LiveSwitch's: equal power (cos / sin), `fade_ms` long, starting at the seam.
"""
import argparse

import numpy as np
import soundfile as sf

SR = 24_000


def load(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    return np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x).astype(np.float32) if sr != SR else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('translation', nargs='+')
    ap.add_argument('--live', required=True)
    ap.add_argument('--n', type=int, default=300)
    a = ap.parse_args()
    tts, live = np.concatenate([load(p) for p in a.translation]), load(a.live)
    tts, live = tts / np.abs(tts).max() * 0.5, live / np.abs(live).max() * 0.5  # both at the same peak
    rng = np.random.default_rng(0)
    w = SR // 10  # +-100 ms around the seam for the signals' own jumps
    rms = lambda x: np.sqrt(np.mean(x ** 2))  # noqa: E731
    seams = []
    while len(seams) < a.n:
        i, j = rng.integers(w, len(tts) - w), rng.integers(w, len(live) - w)
        if rms(tts[i - 480:i + 480]) > 0.03 and rms(live[j - 480:j + 480]) > 0.03:  # both talking at the seam
            seams.append((i, j))
    def hf(x, at, n=240):  # energy above 4 kHz in a 10 ms Hann window centred at `at`
        seg = x[at - n // 2:at + n // 2] * np.hanning(n)
        p = np.abs(np.fft.rfft(seg)) ** 2
        return p[np.fft.rfftfreq(n, 1 / SR) >= 4000].sum() + 1e-12

    print('| fade | jump at the seam / the signals\' own: median | p95 | max | seams above 2x | '
          'energy > 4 kHz at the seam / the signals\' own: p95 | seams above +6 dB |')
    print('|---|---|---|---|---|---|---|')
    for fade_ms in (0, 2, 5, 10, 20, 30, 50):
        f = max(1, round(SR * fade_ms / 1000))
        g = np.clip(np.arange(1, 2 * w + 1) - w, 0, f) / f  # 0 before the seam, ramp, 1 after
        ratios, hfs = [], []
        for i, j in seams:
            a_, b_ = tts[i - w:i + w], live[j - w:j + w]
            out = a_ * np.cos(g * np.pi / 2) + b_ * np.sin(g * np.pi / 2)
            own = max(np.abs(np.diff(a_)).max(), np.abs(np.diff(b_)).max())
            ratios.append(np.abs(np.diff(out[w - 120:w + f + 120])).max() / own)
            seam_hf = max(hf(out, w + d) for d in range(0, f + 1, 60))  # anywhere along the fade
            own_hf = max(hf(x, w + d) for x in (a_, b_) for d in range(-120, f + 121, 60))  # the signals there
            hfs.append(10 * np.log10(seam_hf / own_hf))
        r, h = np.array(ratios), np.array(hfs)
        print(f'| {fade_ms} ms | {np.median(r):.2f} | {np.percentile(r, 95):.2f} | {r.max():.2f} | {np.mean(r > 2):.0%} | '
              f'{np.percentile(h, 95):+.1f} dB | {np.mean(h > 6):.0%} |')


if __name__ == '__main__':
    main()

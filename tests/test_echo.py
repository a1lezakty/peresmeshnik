"""The echo gate (peresmeshnik/audio/echo.py) on real speech, no models: our speech comes back from a far end on a
loudspeaker (bench/call.py far_end: a small speaker's band, a room, 50-300 ms, -6..-20 dB) and is zeroed; their own
speech is not, also over our echo (double talk) and while we speak to a far end on a headset (no echo path).
Ours: the Russian recordings of tests/data, theirs: the English ones (other people)."""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from peresmeshnik.audio.echo import HOP, EchoGate, Reference, Referenced, Threaded
from peresmeshnik.pipeline import read_audio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from call import SR, Scenario, far_end, gate, level  # noqa: E402

DATA = Path(__file__).parent / 'data'


def utterances(name):
    """The three sentences of a test recording (split at its 0.8 s pauses)."""
    x = read_audio(str(DATA / name))
    loud = np.convolve(np.abs(x) > 0.01, np.ones(SR * 7 // 10), 'same') > 0  # speech, with its pauses under 0.7 s
    edges = np.flatnonzero(np.diff(np.concatenate([[0], loud.astype(int), [0]])))
    return [level(x[a:b]) for a, b in zip(edges[::2], edges[1::2]) if b - a > SR]


OURS, THEIRS = utterances('call-ru.ogg'), utterances('call-en.ogg')


def scenario(delay, gain_db, echo_path=True, double_talk=False):
    rng = np.random.default_rng(0)
    sc, t = Scenario(80), 0.5
    for x, y in zip(OURS, THEIRS):
        sc.add('ours', x, t, '', far_end(x, delay, gain_db, rng) if echo_path else None, delay)
        t += len(x) / SR + delay
        t = t - 1.0 if double_talk else t + 0.8
        sc.add('them', y, t, '')
        t += len(y) / SR + 0.8
    return sc.trim()


def test_the_recordings_split():
    assert len(OURS) == len(THEIRS) == 3


@pytest.mark.parametrize('delay', [0.05, 0.15, 0.3])
@pytest.mark.parametrize('gain_db', [-6, -20])
def test_echo_is_zeroed_their_speech_is_not(delay, gain_db):
    _, s = gate(scenario(delay, gain_db))
    print(delay, gain_db, s)
    assert s['echo_blocks_zeroed'] >= 0.85, s
    assert s['their_blocks_zeroed'] <= 0.02, s


def test_double_talk_keeps_their_speech():
    _, s = gate(scenario(0.15, -12, double_talk=True))
    print(s)
    assert s['their_blocks_zeroed'] <= 0.05, s


def test_no_echo_path_no_suppression():
    """The far end wears a headset: we speak, they speak over us, nothing comes back. Their speech stays."""
    _, s = gate(scenario(0.15, -12, echo_path=False, double_talk=True))
    print(s)
    assert s['their_blocks_zeroed'] <= 0.01, s


def test_silent_reference_costs_nothing():
    ref = Reference()
    g = EchoGate(ref)
    x = THEIRS[0]
    out = np.concatenate([g.process(x[i:i + 512], ref.t0 + (i + 512) / SR) for i in range(0, len(x), 512)])
    assert np.array_equal(out, x) and g.stats['blocks'] == 0


def test_reference_timeline():
    ref = Reference(seconds=1.0)
    t = ref.t0 + 10.0
    ref.push(np.ones(SR // 2, np.float32), SR, t_end=t)  # played in [t - 0.5, t]
    w = ref.window(t - 1.0, t + 0.25)
    assert not w[:SR // 2 - 16].any() and w[SR // 2 + 16:SR - 16].all() and not w[SR + 16:].any()
    ref.push(np.ones(441, np.float32), 44100, t_end=t + 0.5)  # 10 ms at 44.1 kHz, after a gap of silence
    w = ref.window(t, t + 0.5)
    assert not w[16:SR * 49 // 100].any() and w[-SR // 200:].all()
    assert ref.features(np.array([t - 2.0, t - 0.25])).max(axis=-1)[0] == pytest.approx(-9.0)


def test_cut_after_a_flush():
    """Barge-in flushes the output: the audio queued after now never plays, the timeline goes back to now."""
    class Out:
        def play(self, pcm, sr):
            pass

        def flush(self):
            pass
    ref = Reference()
    out = Referenced(Out(), ref)
    out.play(np.ones(SR * 2, np.float32), SR)  # 2 s queued from now
    out.flush()
    now = time.perf_counter()
    assert not ref.window(now + 0.05, now + 1.0).any()
    assert ref.loud_until <= now + 0.05
    out.play(np.ones(SR // 10, np.float32), SR)  # the next speech plays from now, not after the flushed audio
    assert ref.window(time.perf_counter(), time.perf_counter() + 0.05).all()


def test_threaded_gate_passes_audio_if_it_fails():
    class Broken:
        def process(self, pcm, t):
            raise ValueError('boom')
    got, done = [], threading.Event()
    th = Threaded(Broken(), lambda pcm: (got.append(pcm), len(got) == 3 and done.set()))
    for _ in range(3):
        th(np.ones(HOP, np.float32))
    assert done.wait(5)
    th.close()
    assert all(g.all() for g in got)

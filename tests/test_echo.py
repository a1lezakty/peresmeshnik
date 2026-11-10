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


def scenario(delay, gain_db, echo_path=True, double_talk=False, overlap_s=1.0, their_db=0.0):
    """`double_talk`: they begin `overlap_s` before our echo ends; `their_db`: their level against ours."""
    rng = np.random.default_rng(0)
    sc, t = Scenario(80), 0.5
    for x, y in zip(OURS, THEIRS):
        sc.add('ours', x, t, '', far_end(x, delay, gain_db, rng) if echo_path else None, delay)
        t += len(x) / SR + delay
        t = t - overlap_s if double_talk else t + 0.8
        sc.add('them', y * 10 ** (their_db / 20), t, '')
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


@pytest.mark.parametrize('frame', [100, 160, 320, 512])
def test_any_frame_size(frame):
    """Captures hand on frames of any size (BlackHole at 96 kHz: ~170 samples at 16 kHz): a frame shorter than a block
    is zeroed whole when the gate says echo."""
    _, s = gate(scenario(0.15, -12), frame=frame)
    print(frame, s)
    assert s['echo_blocks_zeroed'] >= 0.85, s
    assert s['their_blocks_zeroed'] <= 0.02, s


@pytest.mark.parametrize('delay', [0.6, 0.9, 1.4])
def test_a_long_round_trip(delay):
    """A call's round trip (two networks, two jitter buffers, device buffers) can be over 1 s."""
    _, s = gate(scenario(delay, -12))
    print(delay, s)
    assert s['echo_blocks_zeroed'] >= 0.85, s
    assert s['their_blocks_zeroed'] <= 0.02, s


def test_beyond_the_limit_the_echo_passes():
    """An echo later than `echo_max_delay_s` (2 s) is not found: it is translated as speech, nothing else is zeroed."""
    _, s = gate(scenario(2.5, -12))
    print(s)
    assert s['echo_blocks_zeroed'] <= 0.05, s
    assert s['their_blocks_zeroed'] == 0, s


@pytest.mark.parametrize('setting, top, delay', [(-1.0, 0.0, 0.0), (10.0, 2.8, 2.5)])
def test_the_delay_limit_is_kept_in_range(setting, top, delay):
    """A setting below 0 would leave no delay to search (the side without a gate all session), above 2.8 s it would
    look past the 4 s the reference keeps: both are taken to the nearest edge, and an echo at that edge is zeroed."""
    g = EchoGate(Reference(), max_delay_s=setting)
    assert len(g.delays) and g.delays.max() == pytest.approx(top, abs=0.02)
    _, s = gate(scenario(delay, -12), max_delay_s=setting)
    assert s['echo_blocks_zeroed'] >= 0.85 and s['their_blocks_zeroed'] <= 0.02, s


@pytest.mark.parametrize('gain_db', [-6, -12])
def test_double_talk_keeps_their_speech(gain_db):
    _, s = gate(scenario(0.15, gain_db, double_talk=True))
    print(gain_db, s)
    assert s['their_blocks_zeroed'] <= 0.01, s



@pytest.mark.parametrize('below_db, most', [(6, 0.07), (10, 0.11)])
def test_double_talk_keeps_their_quiet_speech(below_db, most):
    """They answer quietly, 6-10 dB under our echo, over its last 2 s: where the echo drowns them the gate zeroes the
    block, the rest of their speech passes (bounds: measured ~0.05 and ~0.09, a gate three times as eager ~0.10, ~0.13)."""
    _, s = gate(scenario(0.15, -6, double_talk=True, overlap_s=2.0, their_db=-6 - below_db))
    print(below_db, s)
    assert s['their_blocks_zeroed'] <= most, s

def test_the_clock_resyncs_after_a_gap_in_the_capture():
    """A capture that stops for a while (its helper restarted) and goes on: the gate's time of the audio follows,
    else for ~2 s it would look for the echo in the wrong place of the reference."""
    ref = Reference()
    ref.loud_until = np.inf  # the gate works on every frame
    g = EchoGate(ref)
    t = ref.t0 + 1.0
    for burst in range(20):  # bursts of 4 frames, up to 0.1 s late: no gap
        for k in range(4):
            g.process(np.zeros(512, np.float32), t + (4 * burst + k + 1) * 512 / SR + (0.1 if k < 3 else 0.0))
    t += 80 * 512 / SR
    assert abs(g._hist[-1][0] - (t - HOP / 2 / SR)) < 0.02
    t += 1.5  # 1.5 s that never come; then the audio at its own pace again
    for _ in range(12):
        t += 512 / SR
        g.process(np.zeros(512, np.float32), t)
    assert abs(g._hist[-1][0] - (t - HOP / 2 / SR)) < 0.02, g._hist[-1][0] - t


def test_a_stall_without_a_loss_keeps_the_times():
    """The capture stops for 0.8 s and then hands on all it held at once: no sample is lost, so every frame of that
    backlog keeps its own time (it is not taken for a gap)."""
    ref = Reference()
    ref.loud_until = np.inf
    g = EchoGate(ref)
    t0 = ref.t0 + 1.0
    fed, worst = 0, 0.0
    for k in range(80):
        fed += 512
        arrive = t0 + fed / SR if not 40 <= k < 65 else t0 + 65 * 512 / SR  # frames 40-64 come together, 0.8 s late
        g.process(np.zeros(512, np.float32), arrive)
        worst = max(worst, abs(g._hist[-1][0] - (t0 + fed / SR - HOP / 2 / SR)))
    assert worst < 0.02, worst


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


def test_threaded_gate_goes_on_after_on_frames_fails():
    got, done = [], threading.Event()

    def on_frames(pcm):
        got.append(pcm)
        if len(got) == 1:
            raise RuntimeError('the side choked once')
        if len(got) == 3:
            done.set()
    th = Threaded(EchoGate(Reference()), on_frames)
    for _ in range(3):
        th(np.ones(HOP, np.float32))
    assert done.wait(5)
    th.close()
    assert th._lost == 1


class _Out:
    def __init__(self, name='out'):
        self.name, self.flushed = name, 0

    def play(self, pcm, sr):
        pass

    def flush(self):
        self.flushed += 1

    def resolve(self):
        pass

    def stop(self):
        pass


def _controller(monkeypatch, **kw):
    from peresmeshnik import app
    monkeypatch.setattr(app, 'Headphones', lambda *a, **k: _Out('headphones'))
    ctl = app.Controller(type('O', (), {'side': lambda self, name: None})(), lambda state: None, **kw)
    monkeypatch.setattr(ctl, 'virtual_mic', lambda: _Out('virtual mic'))
    monkeypatch.setattr(ctl, '_input', lambda name, s: object())
    made = {}
    monkeypatch.setattr(app, 'Side', lambda name, s, inp, out, overlay, on_state, echo=None:
                        made.setdefault(name, type('S', (), {'name': name, 'out': out, 'echo': echo})()))
    return ctl, made


def test_each_side_gates_with_the_other_sides_output(monkeypatch):
    """'them' plays into the headphones (the microphone's echo), 'me' into the call (the far end's echo); each side's
    gate takes the reference of the other side's output."""
    from peresmeshnik import config
    s = config.Settings()
    s.local.echo_max_delay_s = 1.5
    ctl, made = _controller(monkeypatch)
    assert ctl._output('them', s).ref is ctl.mic_echo and ctl._output('me', s).ref is ctl.call_echo
    them, me = ctl._side('them', s), ctl._side('me', s)
    assert them.echo.ref is ctl.call_echo and me.echo.ref is ctl.mic_echo
    assert them.out.ref is ctl.mic_echo and me.out.ref is ctl.call_echo
    assert them.echo.delays.max() == pytest.approx(1.5, abs=0.02)


def test_no_gate_for_a_file_or_the_wizards_test(monkeypatch, tmp_path):
    from peresmeshnik import config
    s = config.Settings()
    ctl, made = _controller(monkeypatch, test=True)
    assert ctl._side('them', s).echo is None
    import soundfile as sf
    sf.write(tmp_path / 'call.wav', np.zeros(SR, np.float32), SR)
    ctl, made = _controller(monkeypatch, them_file=str(tmp_path / 'call.wav'))
    assert ctl._side('them', s).echo is None and ctl._side('me', s).echo is not None


def test_barge_in_and_live_cut_the_reference():
    """Side.interrupt() and set_live() flush the output: what was queued will not play, so the reference forgets it."""
    from peresmeshnik import app
    for act in (lambda side: side.interrupt(), lambda side: side.set_live(True)):
        ref = Reference()
        side = app.Side.__new__(app.Side)
        side.name, side.live, side.directions, side.out = 'them', False, [(object(), True, True)], Referenced(_Out(), ref)
        side.out.play(np.ones(SR * 2, np.float32), SR)  # 2 s of speech queued from now
        act(side)
        now = time.perf_counter()
        assert side.out.out.flushed == 1 and not ref.window(now + 0.05, now + 1.0).any()

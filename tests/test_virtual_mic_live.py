""""Translation <-> live voice" of the me -> them direction (audio/mic.py LiveSwitch): the crossfade leaves no click,
the switch drops the queued translation, the output is clocked by the input. No devices needed."""
import queue
import threading
import time

import numpy as np
import pytest

from peresmeshnik.audio.mic import LiveSwitch, click_ratio
from peresmeshnik.engines import Caption, Speech

SR = 24_000
FRAME = 320  # 20 ms at 16 kHz


class FakeDirection:
    """Says whatever the test puts in `say`; records what it is fed."""

    def __init__(self):
        self.say, self.fed = queue.Queue(), []

    def feed(self, pcm):
        self.fed.append(len(pcm))

    def events(self):
        while (e := self.say.get()) is not None:
            if isinstance(e, Exception):
                raise e
            yield e

    def close(self):
        self.say.put(None)


def tone(sr, hz, seconds, offset):
    """A low tone on a DC offset: a hard cut between two of them always jumps by ~0.8, the tones' own steps are tiny."""
    t = np.arange(int(sr * seconds)) / sr
    return (offset + 0.1 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def run(fade_ms, frames=100, live_at=50, speech_s=10.0):
    inner = FakeDirection()
    sw = LiveSwitch(inner, SR, fade_ms=fade_ms)
    out, done = [], threading.Event()

    def consume():
        for e in sw.events():
            out.append(e)
        done.set()
    threading.Thread(target=consume, daemon=True).start()
    inner.say.put(Speech(tone(SR, 200, speech_s, 0.4), SR))  # the translation, queued ahead
    inner.say.put(Caption('hello', 'привет', True, 0))
    while sw._quiet:  # the speech has reached the switch
        time.sleep(0.001)
    mic = tone(16_000, 300, frames * FRAME / 16_000, -0.4)
    seam, lead = None, round(SR * 0.06)

    def put_out(n):  # wait until the events thread has put out n samples (a loaded CI runner is slow)
        end = time.monotonic() + 5
        while sum(len(e.pcm) for e in out if isinstance(e, Speech)) < n and time.monotonic() < end:
            time.sleep(0.002)
        return sum(len(e.pcm) for e in out if isinstance(e, Speech))
    for i in range(frames):
        if i == live_at:
            seam = put_out(lead + i * FRAME * 3 // 2 - 2)  # everything before the switch (the resampler may be 1 short)
            sw.set_live(True)
        sw.feed(mic[i * FRAME:(i + 1) * FRAME])
    put_out(lead + frames * FRAME * 3 // 2 - 2)
    sw.close()
    assert done.wait(2)
    return inner, out, seam, mic


@pytest.mark.parametrize('fade_ms,click', [(0, True), (30, False)])
def test_crossfade_leaves_no_click(fade_ms, click):
    inner, out, seam, mic = run(fade_ms)
    x = np.concatenate([e.pcm for e in out if isinstance(e, Speech)])
    refs = [tone(SR, 200, 0.1, 0.4), tone(SR, 300, 0.1, -0.4)]  # the two signals' own steps
    r = max(click_ratio(x, seam + d, SR, refs, half_ms=10) for d in range(0, 2400, 240))
    print(f'fade {fade_ms} ms: biggest jump at the seam {r:.1f} x the signals\' own')
    assert (r > 5) == click
    assert any(isinstance(e, Caption) for e in out)  # captions pass through
    lead = round(SR * 0.06)
    assert len(x) == pytest.approx(lead + 100 * FRAME * 1.5, abs=2)  # clocked by the input: 2 s in, 2 s (+ lead) out
    assert inner.fed[:50] == [FRAME] * 50 and inner.fed[50] == 8000 and len(inner.fed) == 51  # a pause, then not fed
    assert np.all(x[-SR // 2:] < -0.25)  # live at the end: only the microphone, the 8 s of queued translation dropped


def test_back_to_translation_and_speech_while_live():
    inner = FakeDirection()
    sw = LiveSwitch(inner, SR)
    out = []
    threading.Thread(target=lambda: out.extend(sw.events()), daemon=True).start()
    mic = np.full(FRAME, -0.4, np.float32)
    sw.set_live(True)
    for _ in range(10):
        sw.feed(mic)
    inner.say.put(Speech(np.full(SR, 0.4, np.float32), SR))  # a clause from before the switch: not said while live
    while not inner.say.empty():  # the pump has it (and queues it before the switch back)
        time.sleep(0.001)
    time.sleep(0.02)
    sw.set_live(False)
    for _ in range(10):
        sw.feed(mic)
    time.sleep(0.05)
    inner.say.put(Speech(np.full(SR // 10, 0.4, np.float32), SR))  # new translation after the switch back
    while sw._quiet:
        time.sleep(0.001)
    for _ in range(10):
        sw.feed(mic)
    time.sleep(0.05)
    sw.close()
    time.sleep(0.05)
    x = np.concatenate([e.pcm for e in out])
    assert np.sum(x > 0.3) == SR // 10  # only the new clause, in full
    first = np.argmax(x > 0.3)
    assert np.max(np.abs(np.diff(x[:first - 1]))) < 0.01  # live faded in from silence and out again
    assert np.all(np.abs(x[first - 100:first]) < 1e-6)  # the live voice had faded out to silence


def test_stalled_input_says_the_rest():
    """The input ends (a file) or stalls (a microphone being reopened): the queued translation still goes out."""
    inner = FakeDirection()
    sw = LiveSwitch(inner, SR)
    out = []
    threading.Thread(target=lambda: out.extend(sw.events()), daemon=True).start()
    inner.say.put(Speech(np.full(2 * SR, 0.4, np.float32), SR))
    while sw._quiet:
        time.sleep(0.001)
    for _ in range(5):
        sw.feed(np.zeros(FRAME, np.float32))
    end = time.monotonic() + LiveSwitch.STALL_S + 5
    while sum(len(e.pcm) for e in out) < 2 * SR and time.monotonic() < end:
        time.sleep(0.01)
    assert sum(len(e.pcm) for e in out) == pytest.approx(round(SR * 0.06) + 2 * SR, abs=2)  # lead + all of it
    sw.close()


def test_live_voice_feeds_only_the_switch(monkeypatch):
    """me -> them in the cloud with local subtitles: while live, the subtitles' direction gets no input (nothing is
    translated, no CPU), only a pause that closes its open clause; back to translation, both are fed again."""
    from peresmeshnik import config, pipeline
    from peresmeshnik.app import Side
    s = config.Settings()
    s.me_to_them, s.cloud.subtitles = 'cloud', False
    voice_dir, captions = FakeDirection(), FakeDirection()
    captions.warm_up = lambda: None
    monkeypatch.setattr(Side, '_cloud', lambda self, s: voice_dir)
    monkeypatch.setattr(pipeline, 'make_local', lambda *a, **k: captions)
    out = type('O', (), {'play': lambda *a: None, 'flush': lambda *a: None, 'stop': lambda *a: None})()
    inp = type('I', (), {'start': lambda self, cb: setattr(self, 'cb', cb), 'stop': lambda self: None})()
    side = Side('me', s, inp, out, type('V', (), {'show': lambda self, c: None})())
    assert isinstance(side.directions[0][0], LiveSwitch) and side.directions[1][0] is captions
    side.start()
    frame = np.zeros(512, np.float32)
    inp.cb(frame)
    assert voice_dir.fed == [512] and captions.fed == [512]
    side.set_live(True)
    inp.cb(frame)
    assert voice_dir.fed == [512, 8000] and captions.fed == [512, 8000]  # the pause each, then no input
    side.set_live(False)
    inp.cb(frame)
    assert voice_dir.fed == [512, 8000, 512] and captions.fed == [512, 8000, 512]
    side.stop()


def test_dead_inner_direction_raises():
    inner = FakeDirection()
    sw = LiveSwitch(inner, SR)
    inner.say.put(RuntimeError('quota'))
    with pytest.raises(RuntimeError, match='quota'):
        list(sw.events())


# --- end to end: the app under X, the hotkey by xdotool, what the call hears recorded from "Peresmeshnik Mic" ---

APP = """if True:
    import queue, sys
    import numpy as np
    from peresmeshnik import app
    from peresmeshnik.audio import mic
    from peresmeshnik.engines import Speech

    class Fake:  # "translates" each second of input into a second of a 220 Hz tone (whole periods: seconds join cleanly)
        def __init__(self, speaks):
            self.q, self.n, self.speaks = queue.Queue(), 0, speaks
        def feed(self, pcm):
            self.n += len(pcm)
            if self.speaks and self.n >= 16000:
                self.n -= 16000
                self.q.put(Speech((0.3 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)).astype(np.float32), 24000))
        def events(self):
            while (e := self.q.get()) is not None:
                yield e
        def close(self):
            self.q.put(None)
    app.Side._local = lambda self, s: Fake(self.name == 'me')  # no models: the wiring, the switch and the devices are real
    init, fade = mic.LiveSwitch.__init__, float(sys.argv[1])
    mic.LiveSwitch.__init__ = lambda self, inner: init(self, inner, fade_ms=fade)
    sys.exit(app.cli(sys.argv[2:]))
"""


def _e2e_ready():
    import os
    import shutil
    import subprocess
    import sys
    return (sys.platform.startswith('linux') and os.environ.get('DISPLAY') and shutil.which('xdotool')
            and shutil.which('pactl') and subprocess.run(['pactl', 'info'], capture_output=True).returncode == 0)


@pytest.mark.skipif(not _e2e_ready(), reason='needs PipeWire / PulseAudio, an X display and xdotool')
@pytest.mark.parametrize('fade_ms', [30.0, 0.0])
def test_hotkey_switches_live_voice_without_a_click(tmp_path, fade_ms):
    """Translation on (a fake translation: a 220 Hz tone), the hotkey switches to the live microphone (300 Hz) and
    back; the call's side of "Peresmeshnik Mic" is recorded: translation, live, translation again, and no jump at the
    seams above the tones' own (fade 0: measured, printed for BENCH)."""
    import os
    import subprocess
    import sys

    from test_app_linux import ours, stop_app, wait_line
    from test_virtual_mic_linux import Source

    from peresmeshnik.ui.selftest import Recorder
    cfg = tmp_path / 'settings.toml'
    cfg.write_text('them_to_me = "local"\nme_to_them = "local"\n'
                   '[ui]\nhotkey = "Ctrl+Shift+H"\nmic = "test_live_mic"\nheadphones = "test_live_phones"\n')
    src = Source('test_live_mic', 300)
    phones = subprocess.run(['pactl', 'load-module', 'module-null-sink', 'sink_name=test_live_phones'], check=True,
                            capture_output=True, text=True).stdout.strip()
    env = {**os.environ, 'PERESMESHNIK_CONFIG': str(cfg)}
    p = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-c', APP, str(fade_ms), '--no-wizard', '-v', '--on'],
                         env=env, stdout=subprocess.PIPE, stderr=(tmp_path / 'app.err').open('w'), text=True)
    p.log = tmp_path / 'app.err'
    import queue as q_
    import threading
    p.lines = q_.Queue()
    threading.Thread(target=lambda: [p.lines.put(x) for x in p.stdout] + [p.lines.put(None)], daemon=True).start()
    rec = None
    try:
        wait_line(p, 'translating', 30)
        rec = Recorder('peresmeshnik_mic', 48_000)
        time.sleep(2.5)  # a second of input, then the translation speaks
        subprocess.run(['xdotool', 'key', 'ctrl+shift+h'], check=True)
        assert wait_line(p, 'mode:', 10)[-1] == 'mode: live voice'
        time.sleep(1.5)
        subprocess.run(['xdotool', 'key', 'ctrl+shift+h'], check=True)
        assert wait_line(p, 'mode:', 10)[-1] == 'mode: translation'
        time.sleep(2.5)
        x = rec.stop()
        rec = None
    finally:
        try:
            if rec:
                rec.stop()
            stop_app(p)
        finally:  # the test devices go even when the app did not stop cleanly
            src.remove()
            subprocess.run(['pactl', 'unload-module', phones], capture_output=True)
    assert not ours()

    sr, w = 48_000, 960  # 20 ms windows
    f = np.fft.rfftfreq(w, 1 / sr)
    wins = x[:len(x) // w * w].reshape(-1, w)
    rms = np.sqrt(np.mean(wins ** 2, axis=1))
    kind = np.where(rms < 0.02, '.', np.where(np.abs(f[np.argmax(np.abs(np.fft.rfft(wins, axis=1)), axis=1)] - 220) < 30, 't',
                                              np.where(rms > 0.08, 'L', '?')))
    runs = ''.join(k for i, k in enumerate(kind) if i == 0 or k != kind[i - 1])
    print('call hears:', ''.join(kind))
    assert runs.strip('.?').replace('?', '') in ('tLt', 'tL.t'), runs  # translation, live voice, translation again
    own = max(0.3 * 2 * np.pi * 220 / sr, 0.2 * 2 * np.pi * 300 / sr)  # the tones' own biggest step
    seams = [i * w for i in range(1, len(kind)) if {kind[i - 1], kind[i]} in ({'t', 'L'}, {'L', '.'}, {'t', '?'}, {'L', '?'})]
    jumps = [np.abs(np.diff(x[max(0, s - 2 * w):s + 2 * w])).max() / own for s in seams]
    print(f'fade {fade_ms:.0f} ms: jump at the seams {", ".join(f"{j:.1f}" for j in jumps)} x the tones\' own')
    if fade_ms:
        assert max(jumps) < 3

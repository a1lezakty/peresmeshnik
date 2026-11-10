""""Translation <-> live voice" of the me -> them direction (audio/mic.py LiveSwitch): the crossfade leaves no click,
the switch drops the queued translation, the output is clocked by the input. No devices needed."""
import queue
import threading
import time

import numpy as np
import pytest

from peresmeshnik.audio.mic import MATCH_DB, PEAK, RISE_DB, LiveSwitch, click_ratio
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
    assert np.sum(x > 0.01) == SR // 10  # only the new clause, in full (at LEVEL_DB: a flat microphone is no speech)
    first = np.argmax(x > 0.01)
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


def db(x):
    return 10 * np.log10(np.mean(x ** 2))


def talk(rng, level_db, seconds, noise_db=-70.0):
    """My speech as the microphone's level sees it: 0.6 s syllables at `level_db`, 0.16 s pauses down to the room's
    noise at `noise_db` (whole 32 ms frames at 16 kHz)."""
    n = int(16_000 * seconds)
    on = np.arange(n) // 512 % 24 < 19
    return ((np.where(on, 10 ** (level_db / 20), 0) + 10 ** (noise_db / 20)) * rng.standard_normal(n)).astype(np.float32)


def feed(sw, mic):
    for i in range(0, len(mic), FRAME):
        sw.feed(mic[i:i + FRAME])


@pytest.mark.parametrize('mic_db,tts_db', [(-36, -14), (-26, -17), (-10, -30), (-44, -10)])
def test_translation_comes_at_the_live_voice_level(mic_db, tts_db):
    """The clone spoke ~9 dB louder than a microphone at a call's level, peaks near full scale (BENCH stage 5): the
    switch brings the translation to my live voice's level (its pauses do not count), never above PEAK."""
    rng = np.random.default_rng(0)
    inner = FakeDirection()
    sw = LiveSwitch(inner, SR)
    out = []
    threading.Thread(target=lambda: out.extend(sw.events()), daemon=True).start()
    mic = talk(rng, mic_db, 4)
    mic[FRAME * 50:FRAME * 100] = 0  # a pause in my speech
    feed(sw, mic)
    tts = (10 ** (tts_db / 20) * rng.standard_normal(2 * SR)).astype(np.float32)
    tts[SR // 2] = 0.99  # a peak
    inner.say.put(Speech(tts[:SR], SR))
    inner.say.put(Speech(np.zeros(SR // 4, np.float32), SR))  # the clone's pause after a sentence: exact zeros
    inner.say.put(Speech(tts[SR:], SR))
    while sw._quiet:
        time.sleep(0.001)
    end = time.monotonic() + 5
    while sum(len(e.pcm) for e in out) < 2 * SR + SR // 4 and time.monotonic() < end:
        sw.feed(np.zeros(FRAME, np.float32))  # the input clocks the output (my side is quiet now)
        time.sleep(0.005)
    sw.close()
    x = np.concatenate([e.pcm for e in out if isinstance(e, Speech)])
    assert len(x) >= 2 * SR + SR // 4  # all of it went out, the pause too
    said = np.r_[x[round(SR * 0.06):][SR // 4:SR], x[round(SR * 0.06):][SR + SR // 4:2 * SR + SR // 4]]  # not the pause
    said = said[said != 0]  # nor a lead of silence: a slow runner feeds slower than real time, each frame a new stretch
    want = tts_db + float(np.clip(mic_db - tts_db, *MATCH_DB))
    print(f'mic {mic_db} dB, TTS {tts_db} dB -> {db(said):.1f} dB, peak {np.abs(x).max():.2f}')
    assert np.abs(x).max() <= PEAK + 1e-6
    if 10 ** (want / 20) * 0.99 / 10 ** (tts_db / 20) <= PEAK:  # the peak does not hold it back
        assert db(said) == pytest.approx(want, abs=1.0)


@pytest.mark.parametrize('noise_db', [-55, -50])
@pytest.mark.parametrize('mic_db', [-26, -18])
def test_the_first_clause_after_a_minute_of_listening_is_not_quiet(noise_db, mic_db):
    """Translation on, the other side speaks first: a minute of my room's noise, then 2 s of my speech, then its first
    clause. The noise is not my level (it was learnt as such: the first clauses went out 13-18 dB quiet)."""
    rng = np.random.default_rng(0)
    sw = LiveSwitch(FakeDirection(), SR)
    feed(sw, (10 ** (noise_db / 20) * rng.standard_normal(16_000 * 60)).astype(np.float32))
    feed(sw, talk(rng, mic_db, 2, noise_db))
    y = sw._match(np.zeros(0, np.float32), (10 ** (-17 / 20) * rng.standard_normal(SR)).astype(np.float32))
    sw.close()
    print(f'noise {noise_db} dB, me {mic_db} dB -> my level {sw._mic.db:.1f} dB, the first clause {db(y):.1f} dB')
    assert db(y) == pytest.approx(mic_db, abs=3)


def test_a_peak_eases_the_gain_down_and_it_comes_back_slowly():
    """The cloud says 20-40 ms chunks, each with its own peak. A peak that holds the gain down eases it over up to 5 ms
    of the speech queued before it (the gain stepped down at the seam, up to ~9 dB), and the gain comes back at RISE_DB,
    so the chunks' peaks do not pump the level from one chunk to the next."""
    rng = np.random.default_rng(1)
    sw = LiveSwitch(FakeDirection(), SR)
    feed(sw, talk(rng, -16, 2))  # 14 dB above the speech: the gain is at MATCH_DB's top, +10 dB
    sizes = rng.integers(SR // 50, SR // 25, 150)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    x = (10 ** (-30 / 20) * rng.standard_normal(int(sizes.sum()))).astype(np.float32)
    x[starts[1::3] + 2] = 0.6  # 2 samples into every third chunk: PEAK holds the gain at +4 dB
    y = np.zeros(0, np.float32)
    for a, n in zip(starts, sizes, strict=True):
        y = sw._match(y, x[a:a + n])
    sw.close()
    g = 20 * np.log10(np.abs(y / x)[np.abs(x) > 1e-4])
    step, swing = np.abs(np.diff(g)).max(), np.ptp(g[len(g) // 4:])
    print(f'gain {g.min():.1f}..{g.max():.1f} dB, biggest step {step:.3f} dB, swing after the start {swing:.1f} dB')
    assert np.abs(y).max() <= PEAK + 1e-6
    assert step < 0.2  # dB from one sample to the next
    assert swing < 2 * RISE_DB * 0.1  # comes back up ~1.8 dB between two peaks ~90 ms apart; it pumped 6 dB


@pytest.mark.parametrize('to_db', [-8, -36])
def test_a_new_voice_mid_session_comes_to_my_level(to_db):
    """The cloud fails and a local voice goes on in the same LiveSwitch at another level (cloud/fallback.py): how soon
    the translation is back within 2 dB of my level. Louder: the level follows it (~2.6 s for 12 dB); quieter by more
    than RANGE_DB: 1 s of it starts the level over, then the gain comes up at RISE_DB."""
    rng = np.random.default_rng(2)
    sw = LiveSwitch(FakeDirection(), SR)
    feed(sw, talk(rng, -26, 2))
    cloud = (10 ** (-20 / 20) * rng.standard_normal(3 * SR)).astype(np.float32)
    local = (10 ** (to_db / 20) * rng.standard_normal(4 * SR)).astype(np.float32)
    y = np.zeros(0, np.float32)
    for c in [*np.split(cloud, 75), *np.split(local, 8)]:  # the cloud's 40 ms chunks, then Piper's clauses
        y = sw._match(y, c)
    sw.close()
    w = 10 * np.log10(np.mean(y[len(cloud):].reshape(-1, SR // 10) ** 2, axis=1))  # 100 ms windows
    back = (np.nonzero(np.abs(w + 26) > 2)[0].max() + 1) / 10
    print(f'{to_db} dB after -20: back at my level after {back:.1f} s; the cloud {db(y[2 * SR:len(cloud)]):.1f} dB')
    assert db(y[2 * SR:len(cloud)]) == pytest.approx(-26, abs=1)
    assert back <= 3.0


def test_live_voice_feeds_only_the_switch(monkeypatch):
    """me -> them in the cloud with local subtitles (one cloud/fallback.Fallback holding both): while live, neither the
    voice's nor the subtitles' direction gets input (nothing is translated, no CPU), only a pause that closes their open
    clause; back to translation, both are fed again."""
    from peresmeshnik import config
    from peresmeshnik.app import Side
    from peresmeshnik.cloud.fallback import Fallback
    s = config.Settings()
    s.me_to_them, s.cloud.subtitles = 'cloud', False
    voice_dir, captions = FakeDirection(), FakeDirection()
    captions.warm_up, captions.mt = (lambda: None), None
    both = Fallback('me', 'en', False, make_cloud=lambda: voice_dir, make_local=lambda tts: captions, make_tts=lambda: None)
    monkeypatch.setattr(Side, '_cloud', lambda self, s: both)
    out = type('O', (), {'play': lambda *a: None, 'flush': lambda *a: None, 'stop': lambda *a: None})()
    inp = type('I', (), {'start': lambda self, cb: setattr(self, 'cb', cb), 'stop': lambda self: None})()
    side = Side('me', s, inp, out, type('V', (), {'show': lambda self, c: None})())
    assert isinstance(side.directions[0][0], LiveSwitch) and side.directions[0][0].inner is both
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


def test_record_counts_sound_not_wall_time(monkeypatch):
    """The voice recording asks for 3 s of sound: a microphone that starts late (the helper, ~0.5 s) still gives 3 s;
    one that goes away gives what came, START_S later at the most (it was 2.5 s of 3 on the macOS runner)."""
    import threading
    import time

    import numpy as np

    from peresmeshnik.audio import mic as mic_mod

    class LateMic:
        """Its sound begins `late` s after start (the helper starting), then `give` s of it in 20 ms pieces as fast as it
        can, and it goes silent: counted pieces, no clock per piece (no runner is too slow for it)."""

        def __init__(self, sample_rate, late, give):
            self.sr, self.late, self.give, self.stopped = sample_rate, late, give, threading.Event()

        def start(self, on_frames):
            def run():
                time.sleep(self.late)
                for _ in range(round(self.give * 50)):
                    if self.stopped.is_set():
                        return
                    on_frames(np.full(self.sr // 50, 0.1, np.float32))
            threading.Thread(target=run, daemon=True).start()

        def stop(self):
            self.stopped.set()

    progress = []  # its sound begins after the 1.5 s asked: by wall time none came yet, counted in sound all of it
    monkeypatch.setattr(mic_mod, 'Mic', lambda device, sample_rate, **k: LateMic(sample_rate, late=1.7, give=2.0))
    x = mic_mod.record(1.5, sample_rate=8000, on_progress=progress.append)
    assert len(x) == 12000 and progress == [1, 1.5]
    monkeypatch.setattr(mic_mod, 'START_S', 0.5)
    monkeypatch.setattr(mic_mod, 'Mic', lambda device, sample_rate, **k: LateMic(sample_rate, late=0.5, give=0.4))
    t = time.monotonic()
    x = mic_mod.record(1.5, sample_rate=8000)
    assert len(x) == 3200 and time.monotonic() - t < 1.5 + 0.5 + 0.5

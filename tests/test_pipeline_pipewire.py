"""Capture, playback and the call app's routing through PipeWire devices, no models: a tone played into
the "Peresmeshnik Call" null sink comes back from its monitor; the call app's streams (fake Telegram and
browser players) move into the call sink and back; the headphones' end-of-stream callback and flush.
Skipped without a sound server."""
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import psutil
import pytest

from peresmeshnik.engines import SAMPLE_RATE

pytestmark = pytest.mark.skipif(
    not shutil.which('pactl') or subprocess.run(['pactl', 'info'], capture_output=True).returncode != 0,
    reason='no PulseAudio/PipeWire server')


def test_tone_through_call_sink():
    from peresmeshnik.audio.capture_pipewire import PipeWireCapture, pactl
    from peresmeshnik.audio.out import PulseOut
    got, lock = [], threading.Lock()

    def on_frames(x):
        with lock:
            got.append(x)

    cap = PipeWireCapture()
    cap.start(on_frames)
    try:
        assert 'Peresmeshnik Call' in pactl('list', 'sinks')
        out = PulseOut(SAMPLE_RATE, cap.sink)
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        out.play((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SAMPLE_RATE)
        out.stop()  # returns when pacat has drained
    finally:
        cap.stop()
    assert cap.sink not in pactl('list', 'short', 'sinks').split()  # removed on stop
    x = np.concatenate(got)
    loud = np.abs(x) > 0.1
    assert loud.sum() > 0.5 * SAMPLE_RATE, f'{loud.sum()} loud samples of {len(x)}'
    spectrum = np.abs(np.fft.rfft(x[loud.argmax():][:SAMPLE_RATE // 2]))
    assert abs(spectrum.argmax() * 2 - 440) <= 4  # bin = 2 Hz on a 0.5 s window


@pytest.fixture
def speakers():
    """A null sink as the user's speakers, made the default output; the call sink next to it."""
    from peresmeshnik.audio.capture_pipewire import CALL_SINK, pactl
    for line in reversed(pactl('list', 'short', 'modules').splitlines()):  # left by a killed run
        if 'sink_name=test_speakers' in line or f'sink_name={CALL_SINK}' in line:
            pactl('unload-module', line.split()[0])
    before = pactl('get-default-sink').strip()
    mods = [pactl('load-module', 'module-null-sink', 'sink_name=test_speakers').strip()]
    pactl('set-default-sink', 'test_speakers')
    yield 'test_speakers', mods
    for m in reversed(mods):
        subprocess.run(['pactl', 'unload-module', m], capture_output=True)
    if before and before != 'test_speakers':
        subprocess.run(['pactl', 'set-default-sink', before], capture_output=True)


def call_sink(mods):
    from peresmeshnik.audio.capture_pipewire import CALL_SINK, pactl
    mods.append(pactl('load-module', 'module-null-sink', f'sink_name={CALL_SINK}', 'sink_properties=priority.session=-10').strip())
    return CALL_SINK


RUN = f'{os.getpid()}'  # in every app name: WirePlumber remembers moves per name, an earlier failed run must not count


def player(name: str, binary: str, procs: list) -> subprocess.Popen:
    """A call app or a browser playing: a pacat stream with that application name (+ RUN) and program."""
    p = subprocess.Popen(['pacat', '--playback', '--raw', '--format=s16le', '--rate=16000', '--channels=1', '--stream-name=Playback',
                          f'--property=application.name={name} {RUN}', f'--property=application.process.binary={binary}'],
                         stdin=subprocess.PIPE)
    procs.append(p)
    return p


def sinks_of(key: str, n: int = 1, timeout: float = 5.0) -> list[str]:
    """Sinks of the `key` app's streams once there are `n` of them (and nothing moving)."""
    from peresmeshnik.audio.capture_pipewire import players
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = sorted(p['sink'] for p in players() if p['key'] == key)
        if len(got) >= n:
            time.sleep(0.2)  # a move in flight lands
            return sorted(p['sink'] for p in players() if p['key'] == key)
        time.sleep(0.05)
    raise AssertionError(f'{n} streams of {key} never appeared')


def wait_sinks(key: str, want: list[str], timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while (got := sinks_of(key, len(want))) != want and time.monotonic() < end:
        time.sleep(0.05)
    assert got == want, got


@pytest.fixture
def procs():
    ps = []
    yield ps
    for p in ps:
        p.kill()
        p.wait()


def test_call_router_moves_the_call_app_and_gives_it_back(speakers, procs):
    from peresmeshnik.audio.capture_pipewire import CallRouter, apps, guess_call_app, pactl, players, return_strays
    spk, mods = speakers
    call = call_sink(mods)
    player('TelegramDesktop', 'Telegram', procs)
    player('Google Chrome', 'chrome', procs)
    assert sinks_of('Telegram') == sinks_of('chrome') == [spk]
    assert dict(apps())['chrome'] == f'Google Chrome {RUN} (all its sound, every tab)' and guess_call_app() == 'Telegram'
    moved = []
    r = CallRouter('Telegram', call, moved.append)
    r.start()
    try:
        wait_sinks('Telegram', [call])  # a stream this young may be moved twice (WirePlumber relinks it once)
        assert sinks_of('chrome') == [spk]  # the browser stays
        # the call reconnects: a new stream of the same program, under a name WirePlumber has not seen moved
        player('TelegramDesktop (call 2)', 'Telegram', procs)
        wait_sinks('Telegram', [call, call])
        assert len(moved) == 2
    finally:
        r.stop()
    wait_sinks('Telegram', [spk, spk])
    player('TelegramDesktop', 'Telegram', procs)  # the move is forgotten: the next stream follows the default
    assert sinks_of('Telegram', 3) == [spk] * 3
    # left in our sink by a killed run: back to the default output
    web = next(p['index'] for p in players() if p['key'] == 'chrome')
    pactl('move-sink-input', web, call)
    wait_sinks('chrome', [call])
    assert return_strays(call) == [f'Google Chrome {RUN}']
    wait_sinks('chrome', [spk])


def test_call_router_back_to_the_default_when_its_device_is_gone(speakers, procs):
    from peresmeshnik.audio.capture_pipewire import CallRouter, pactl, players
    spk, mods = speakers
    call = call_sink(mods)
    headset = pactl('load-module', 'module-null-sink', 'sink_name=test_headset').strip()
    try:
        player('zoom', 'zoom', procs)
        sinks_of('zoom')
        pactl('move-sink-input', next(p['index'] for p in players() if p['key'] == 'zoom'), 'test_headset')
        wait_sinks('zoom', ['test_headset'])
        r = CallRouter('zoom', call)
        r.start()
        wait_sinks('zoom', [call])
        pactl('unload-module', headset)  # unplugged while translating
        headset = None
        r.stop()
        wait_sinks('zoom', [spk])
    finally:
        if headset:
            pactl('unload-module', headset)


def test_app_moves_the_call_on_and_back_off_and_on_exit(speakers, procs, monkeypatch, tmp_path):
    """The app's controller with stand-in sides (no models): the known call app is found and remembered,
    moved while translating, its reconnect too, given back when translation goes off and at exit."""
    from peresmeshnik import app, config
    spk, mods = speakers
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))

    class Side:
        def __init__(self, name, s, inp, out, overlay, on_error):
            self.name, self.engine = name, 'local'

        def start(self):
            pass

        def set_live(self, on):
            pass

        def interrupt(self):
            pass

        def stop(self):
            pass
    monkeypatch.setattr(app, 'Side', Side)
    monkeypatch.setattr(app.Controller, '_input', lambda self, name, s=None: None)
    monkeypatch.setattr(app.Controller, '_output', lambda self, name, s: None)
    states = []
    hide = type('H', (), {'hide': lambda self: None})()
    ctl = app.Controller(type('O', (), {'side': lambda self, name: hide})(), states.append)
    ctl.startup()
    try:
        call = ctl._call.name
        player('Telegram', 'telegram-desktop', procs)
        player('Firefox', 'firefox', procs)
        sinks_of('telegram-desktop')
        ctl.set_translation(True).result()
        assert 'call sound: telegram-desktop;' in states[-1], states
        assert config.load().ui.call_app == 'telegram-desktop'  # remembered
        wait_sinks('telegram-desktop', [call])
        assert sinks_of('firefox') == [spk]
        player('Telegram (reconnect)', 'telegram-desktop', procs)
        wait_sinks('telegram-desktop', [call, call])
        ctl.set_translation(False).result()
        wait_sinks('telegram-desktop', [spk, spk])
        assert ctl._router is None and not [p for p in psutil.Process().children(recursive=True)
                                            if p.cmdline()[:2] == ['pactl', 'subscribe']]  # idle: no event reader left
        ctl.set_call_app('firefox').result()  # chosen by hand in the tray while off: nothing moves yet
        assert sinks_of('firefox') == [spk]
        ctl.set_translation(True).result()
        wait_sinks('firefox', [call])
        assert sinks_of('telegram-desktop', 2) == [spk, spk]
        ctl.set_call_app('telegram-desktop').result()  # changed while translating: swapped at once
        wait_sinks('telegram-desktop', [call, call])
        wait_sinks('firefox', [spk])
    finally:
        ctl.quit()
    wait_sinks('telegram-desktop', [spk, spk])
    assert call not in subprocess.run(['pactl', 'list', 'short', 'sinks'], capture_output=True, text=True).stdout


def test_headphones_end_callback_and_flush(speakers):
    """pacat killed (the device went away): on_end once, not on stop(). flush() cuts the speech being played."""
    from peresmeshnik.audio.out import PulseOut
    spk, _ = speakers
    ended = []
    out = PulseOut(SAMPLE_RATE, spk, on_end=ended.append)
    out._proc.kill()
    out._proc.wait()
    for _ in range(100):
        if ended:
            break
        time.sleep(0.01)
    assert len(ended) == 1 and ended[0].startswith('pacat ended') and not out.alive
    out.play(np.zeros(1600, np.float32), SAMPLE_RATE)  # nothing to write to: no error
    out.stop()
    quiet = PulseOut(SAMPLE_RATE, spk, on_end=ended.append)
    quiet.stop()
    assert len(ended) == 1  # stop() is not an end

    rec = subprocess.Popen(['parec', f'--device={spk}.monitor', '--raw', '--format=float32le', f'--rate={SAMPLE_RATE}', '--channels=1',
                            '--latency-msec=20'], stdout=subprocess.PIPE)
    got = []
    threading.Thread(target=lambda: got.extend(iter(lambda: rec.stdout.read(3200), b'')), daemon=True).start()
    time.sleep(0.3)
    out = PulseOut(22050, spk)
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(3 * 22050) / 22050)).astype(np.float32)
    out.play(tone, 22050)  # 3 s in one chunk, as Piper hands a clause
    out.play(tone, 22050)
    time.sleep(1.0)
    out.flush()
    time.sleep(1.0)
    out.stop()
    rec.terminate()
    rec.wait()
    x = np.frombuffer(b''.join(got), '<f4')
    loud = np.flatnonzero(np.abs(x) > 0.1)
    played = (loud[-1] - loud[0]) / SAMPLE_RATE
    assert 0.8 < played < 1.0 + 0.3, f"{played:.2f} s played"  # flushed after 1 s: ≤ 0.3 s more, nothing stale later


def test_headphones_end_when_the_sink_goes_but_pacat_stays(speakers):
    """WirePlumber 0.5: a `node.dont-reconnect` stream may outlive its device. pacat is frozen here so it cannot end
    by itself: the sink's removal event ends the output, on_end once; stop() leaves no pactl subscribe behind."""
    import signal

    from peresmeshnik.audio.capture_pipewire import pactl
    from peresmeshnik.audio.out import PulseOut
    mod = pactl('load-module', 'module-null-sink', 'sink_name=test_headset').strip()
    ended = []
    out = PulseOut(SAMPLE_RATE, 'test_headset', on_end=ended.append)
    try:
        os.kill(out._proc.pid, signal.SIGSTOP)
        pactl('unload-module', mod)
        for _ in range(200):
            if ended:
                break
            time.sleep(0.01)
        time.sleep(0.3)  # a second on_end, from pacat's exit, would come by now
        assert ended == ['test_headset was removed'] and not out.alive
    finally:
        out.stop()
    assert not [p.name() for p in psutil.Process().children(recursive=True) if p.name() in ('pactl', 'pacat')]


def test_call_sink_never_the_default():
    """Another output without a priority is there (as any module-made device; WirePlumber 0.4 counts it as -1): the
    call sink, the app's or the capture's own, must not take over as the default (the system's sound would go silent
    into it)."""
    from peresmeshnik.app import CallSink
    from peresmeshnik.audio.capture_pipewire import PipeWireCapture, pactl
    real = pactl('load-module', 'module-null-sink', 'sink_name=test_real_out').strip()
    try:
        time.sleep(0.3)
        before = pactl('get-default-sink').strip()
        sink = CallSink()
        try:
            time.sleep(0.5)  # the session manager picks the default asynchronously
            assert pactl('get-default-sink').strip() == before
        finally:
            sink.remove()
        cap = PipeWireCapture()
        cap.start(lambda x: None)
        try:
            time.sleep(0.5)
            assert pactl('get-default-sink').strip() == before
        finally:
            cap.stop()
    finally:
        pactl('unload-module', real)

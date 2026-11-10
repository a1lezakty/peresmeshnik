"""The app on Linux with real devices: the loopback self-test on virtual PipeWire devices, the setup
wizard through every Linux page, the app's subtitles from file-fed directions, the hotkey via xdotool,
and the idle watchdog with the tray and a visible overlay.

Needs a PipeWire/PulseAudio server (pactl); the app tests also need an X display (Xvfb). The
end-to-end ones download and run models and the cloud: PERESMESHNIK_E2E=1 (local, not in CI).
"""
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

LINUX = sys.platform.startswith('linux')
SOUND = LINUX and shutil.which('pactl') and subprocess.run(['pactl', 'info'], capture_output=True).returncode == 0
DISPLAY = bool(os.environ.get('DISPLAY'))
E2E = os.environ.get('PERESMESHNIK_E2E') == '1'
PAIR_MODELS = os.environ.get('PERESMESHNIK_PAIR_MODELS', '')  # CI: the models folder (cached) of the pair test, which must run
if (PAIR_MODELS or os.environ.get('PERESMESHNIK_REQUIRE_PIPEWIRE') == '1') and not (SOUND and DISPLAY):
    pytest.fail(f'these tests must run here: PipeWire {bool(SOUND)}, display {DISPLAY}', pytrace=False)  # not a skip
pytestmark = pytest.mark.skipif(not SOUND, reason='needs PipeWire / PulseAudio')
DATA = Path(__file__).parent / 'data'


@pytest.fixture
def devices():
    """A null sink as the headphones and a virtual source as the microphone, playing a tone."""
    from peresmeshnik.ui.selftest import pactl
    for line in reversed(pactl('list', 'short', 'modules').splitlines()):  # left by a killed run
        if 'test_headphones' in line or 'test_mic' in line:
            pactl('unload-module', line.split()[0])
    mods = [pactl('load-module', 'module-null-sink', 'sink_name=test_headphones').strip(),
            pactl('load-module', 'module-null-sink', 'sink_name=test_mic_feed', 'channels=1').strip(),
            pactl('load-module', 'module-remap-source', 'master=test_mic_feed.monitor', 'source_name=test_mic', 'channels=1').strip()]
    tone = (0.1 * np.sin(2 * np.pi * 300 * np.arange(8000) / 16000)).astype('<f4').tobytes()  # 0.5 s, whole periods
    feed = subprocess.Popen(['pacat', '--device=test_mic_feed', '--raw', '--format=float32le', '--rate=16000', '--channels=1',
                             '--latency-msec=20'], stdin=subprocess.PIPE)

    def talk():  # the pipe blocks: real-time pace
        try:
            while True:
                feed.stdin.write(tone)
        except (BrokenPipeError, ValueError, OSError):
            pass
    threading.Thread(target=talk, daemon=True).start()
    yield {'headphones': 'test_headphones', 'mic': 'test_mic'}
    feed.kill()
    feed.wait()
    for m in reversed(mods):
        subprocess.run(['pactl', 'unload-module', m], capture_output=True)


def test_selftest_green_on_virtual_devices(devices):
    from peresmeshnik.ui import selftest
    rs = selftest.run_all(devices['headphones'], devices['mic'])
    for r in rs:
        print(('OK  ' if r.ok else 'FAIL'), r.name, r.detail)
    assert [r.ok for r in rs] == [True] * 4
    from peresmeshnik.audio.virtual_mic_linux import exists
    assert not exists()  # the self-test's own virtual mic is gone again


def start_app(tmp_path, *args, settings: str = '', env=None):
    cfg = tmp_path / 'settings.toml'
    cfg.write_text(settings)
    e = {**os.environ, 'PERESMESHNIK_CONFIG': str(cfg), **(env or {})}
    p = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-m', 'peresmeshnik', 'app', '--no-wizard', '-v', *args], env=e,
                         stdout=subprocess.PIPE, stderr=(tmp_path / 'app.err').open('w'), text=True)
    p.log = tmp_path / 'app.err'
    p.lines = queue.Queue()  # read by a thread: select() on a buffered pipe misses lines that arrived together

    def pump():
        for line in p.stdout:
            p.lines.put(line)
        p.lines.put(None)
    threading.Thread(target=pump, daemon=True).start()
    wait_line(p, 'idle', 30)
    return p


def wait_line(p, what: str, timeout: float = 60) -> list[str]:
    """Lines up to the first with `what`; on a timeout the app's threads are dumped (faulthandler) into the error."""
    seen, end = [], time.monotonic() + timeout
    while (left := end - time.monotonic()) > 0:
        try:
            line = p.lines.get(timeout=left)
        except queue.Empty:
            break
        if line is None:
            break
        seen.append(line.strip())
        if what in line:
            return seen
    if p.poll() is None:
        p.send_signal(signal.SIGABRT)
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:  # still writing its core dump
            p.kill()
            p.wait(10)
    raise AssertionError(f'no {what!r} in {seen}; the app said:\n{p.log.read_text()[-4000:]}')


def stop_app(p):
    p.terminate()
    assert p.wait(20) == 0, p.log.read_text()[-4000:]


def ours() -> list[str]:
    """Our PipeWire modules: "module name args" lines."""
    from peresmeshnik.ui.selftest import pactl
    return [' '.join(m.split('\t')[1:3]) for m in pactl('list', 'short', 'modules').splitlines() if 'peresmeshnik_' in m]


@pytest.mark.skipif(not (DISPLAY and shutil.which('xdotool')), reason='needs an X display and xdotool')
def test_hotkey_switches_live_voice(tmp_path, devices):
    """Off: the hotkey only says so (the call hears the microphone through the loopback). On (E2E: the models
    are there, both sides fed from recordings): it switches live voice and back."""
    on = ['--on', '--them-file', str(DATA / 'call-en.ogg'), '--me-file', str(DATA / 'call-ru.ogg')] if E2E else []
    p = start_app(tmp_path, *on, settings='me_to_them = "local"\n[ui]\nhotkey = "Ctrl+Shift+H"\nheadphones = "test_headphones"\n'
                  if E2E else '[ui]\nhotkey = "Ctrl+Shift+H"\n')
    try:
        if E2E:
            wait_line(p, 'translating', 120)
        time.sleep(0.5)
        subprocess.run(['xdotool', 'key', 'ctrl+shift+h'], check=True)
        if not E2E:
            assert wait_line(p, 'mode:', 10)[-1] == 'mode: translation is off'
            return
        assert wait_line(p, 'mode:', 10)[-1] == 'mode: live voice'
        subprocess.run(['xdotool', 'key', 'ctrl+shift+h'], check=True)
        assert wait_line(p, 'mode:', 10)[-1] == 'mode: translation'
    finally:
        stop_app(p)
        assert not ours()  # every device removed on exit (SIGTERM)


@pytest.mark.skipif(not (DISPLAY and shutil.which('xdotool')), reason='needs an X display and xdotool')
def test_devices_live_with_the_app_and_the_mic_is_looped_while_off(tmp_path, devices):
    """Call sink and virtual mic exist from the start (stale ones from a killed run replaced), the microphone loopback
    while a call app records Peresmeshnik Mic; a second instance leaves them alone, SIGHUP removes them."""
    from peresmeshnik.ui.selftest import pactl
    pactl('load-module', 'module-null-sink', 'sink_name=peresmeshnik_call')  # as left by a SIGKILL
    p = start_app(tmp_path, settings='[ui]\nmic = "test_mic"\n')
    call_app = None
    try:
        wait_for(lambda: any('source_name=peresmeshnik_mic ' in m for m in ours()), 20)
        call_app = record_peresmeshnik_mic()
        assert wait_for(lambda: any(m.startswith('module-loopback') for m in ours()), 5), ours()
        mods = ours()
        calls = [m for m in mods if 'sink_name=peresmeshnik_call' in m]
        assert len(calls) == 1 and 'priority.session=-10' in calls[0]  # the app's own sink (module ids get reused)
        assert any(m.startswith('module-loopback') and 'source=test_mic sink=peresmeshnik_mic_sink' in m for m in mods), mods
        assert not any('source=peresmeshnik_' in m for m in mods)  # never our own device looped back
        second = subprocess.run([sys.executable, '-m', 'peresmeshnik', 'app', '--no-wizard'], capture_output=True, text=True,
                                timeout=30, env={**os.environ, 'PERESMESHNIK_CONFIG': str(tmp_path / 'settings.toml')})
        assert second.returncode == 3 and 'already running' in second.stderr
        assert ours() == mods  # the first one's devices untouched
        p.send_signal(signal.SIGHUP)  # the terminal closed
        assert p.wait(20) == 0
    finally:
        if call_app:
            call_app.kill()
        if p.poll() is None:
            stop_app(p)
    assert not ours()


def record_peresmeshnik_mic() -> subprocess.Popen:
    """A call app recording "Peresmeshnik Mic"."""
    return subprocess.Popen(['parec', '--device=peresmeshnik_mic', '--raw', '--format=float32le', '--rate=16000',
                             '--channels=1'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for(cond, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.1)
    return cond()


def mic_state(name: str) -> str:
    from peresmeshnik.ui.selftest import pactl
    return next((ln.split('\t')[-1] for ln in pactl('list', 'short', 'sources').splitlines() if ln.split('\t')[1] == name), '')


def test_the_microphone_is_open_only_while_a_call_app_records(devices):
    """M-16: translation off and no call: the real microphone is not held open (no indicator, a Bluetooth headset keeps
    its music profile), as on macOS and Windows; a call app recording Peresmeshnik Mic gets it looped in at once."""
    from peresmeshnik import app, config
    s = config.Settings()
    s.ui.mic = devices['mic']
    side = type('S', (), {'hide': lambda self: None})()
    ctl = app.Controller(type('O', (), {'side': lambda self, name: side})(), lambda state: None, settings=s)
    ctl.startup()
    call_app = None
    try:
        loops = lambda: [m for m in ours() if m.startswith('module-loopback') and 'source=test_mic' in m]  # noqa: E731
        assert ctl.startup_error == '' and not loops(), ours()
        call_app = record_peresmeshnik_mic()
        assert wait_for(lambda: loops(), 3), ours()
        call_app.kill()
        call_app.wait()
        assert wait_for(lambda: not loops(), 3), ours()
    finally:
        if call_app:
            call_app.kill()
        ctl.quit()
    assert not ours()


def test_our_devices_come_back_when_the_sound_server_does(devices):
    """M-01: the sound server restarted (its events end, our modules are gone): Peresmeshnik Mic and Call are made
    again, without polling while it runs; a side starting finds them there."""
    from peresmeshnik import app, config
    from peresmeshnik.ui.selftest import pactl
    s = config.Settings()
    s.ui.mic = devices['mic']
    side, states = type('S', (), {'hide': lambda self: None})(), []
    ctl = app.Controller(type('O', (), {'side': lambda self, name: side})(), states.append, settings=s)
    ctl.startup()
    try:
        names = lambda: sorted(m.split()[1] for m in ours())  # noqa: E731
        before = names()
        assert any('peresmeshnik_call' in m for m in before) and any('peresmeshnik_mic ' in m for m in ours())
        for m in reversed(pactl('list', 'short', 'modules').splitlines()):  # what a restart leaves: none of ours
            if 'peresmeshnik_' in m:
                pactl('unload-module', m.split('\t')[0])
        ctl._events.p.kill()  # ... and the end of the event stream
        assert wait_for(lambda: names() == before, 10), (names(), before)
        assert wait_for(lambda: any('sound server restarted' in x for x in states), 5), states
        assert any(x == 'error: the sound server is gone, waiting for it' for x in states), states  # said meanwhile
        for m in reversed(pactl('list', 'short', 'modules').splitlines()):  # gone again while the app idles
            if 'peresmeshnik_' in m:
                pactl('unload-module', m.split('\t')[0])
        assert ctl._ensure_devices() and names() == before  # what _on and a restarting side ask first
    finally:
        ctl.quit()
    assert not ours()


class FakeSide:
    """A side as _on makes it, without models: where it plays (me -> them: Peresmeshnik Mic)."""
    engine = 'fake'

    def __init__(self, name: str, vmic):
        self.name, self.vmic = name, vmic

    def set_live(self, on: bool) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def test_every_side_moves_to_the_new_devices_when_one_ends_first(devices, monkeypatch):
    """M-01, the review: the sound server is back before its event ends and a side's input ended first (_restart before
    _sound_back): every side plays into the devices made again, not only that one (me -> them would play into the old
    Peresmeshnik Mic: no sound for the call, its queue growing)."""
    from peresmeshnik import app, config
    from peresmeshnik.ui.selftest import pactl
    monkeypatch.setattr(app.crash, 'report', lambda *a, **k: None)
    s = config.Settings()
    s.ui.mic = devices['mic']
    overlay_side = type('S', (), {'hide': lambda self: None})()
    ctl = app.Controller(type('O', (), {'side': lambda self, name: overlay_side})(), lambda state: None, settings=s)
    ctl.startup()
    ctl._route = lambda s: ''
    ctl._side = lambda name, s: FakeSide(name, ctl._vmic)
    try:
        ctl._on()
        old = ctl._vmic
        assert ctl.on and [x.vmic for x in ctl.sides] == [old, old]
        for m in reversed(pactl('list', 'short', 'modules').splitlines()):  # a restart: none of ours, the events go on
            if 'peresmeshnik_' in m:
                pactl('unload-module', m.split('\t')[0])
        ctl._restart(ctl.sides[0], 'the call sound ended', None)
        assert ctl.on and ctl._vmic is not old and [x.vmic for x in ctl.sides] == [ctl._vmic, ctl._vmic]
    finally:
        ctl.quit()
    assert not ours()


def test_restarts_on_devices_made_again_have_a_limit(devices, monkeypatch):
    """M-01 (review of PR #22): a side that keeps failing on devices made again each time is restarted RESTARTS times
    within a minute, each with a report, then translation stops and says why (not an endless off/on)."""
    from peresmeshnik import app, config
    from peresmeshnik.ui.selftest import pactl
    reports = []
    monkeypatch.setattr(app.crash, 'report', lambda *a, **k: reports.append(a[0]))
    s = config.Settings()
    s.ui.mic = devices['mic']
    overlay_side, states = type('S', (), {'hide': lambda self: None})(), []
    ctl = app.Controller(type('O', (), {'side': lambda self, name: overlay_side})(), states.append, settings=s)
    ctl.startup()
    ctl._route = lambda s: ''
    ctl._side = lambda name, s: FakeSide(name, ctl._vmic)
    try:
        ctl.set_translation(True).result(10)
        for _ in range(app.RESTARTS + 1):
            assert ctl.on, states
            for m in reversed(pactl('list', 'short', 'modules').splitlines()):
                if 'peresmeshnik_' in m:
                    pactl('unload-module', m.split('\t')[0])
            ctl.submit(ctl._restart, ctl.sides[0], 'the call sound ended', None).result(10)
        assert not ctl.on and len(reports) == app.RESTARTS + 1 and 'translation stopped' in states[-1], states
    finally:
        ctl.quit()
    assert not ours()


def test_translation_turned_off_while_the_sound_server_is_gone_stays_off(devices, monkeypatch):
    """M-01 (review of PR #22): the sound server goes, the user turns translation off meanwhile; when the server is
    back translation stays off (it went on again: a cloud side sending the microphone, the tray's box unticked)."""
    from peresmeshnik import app, config
    from peresmeshnik.ui.selftest import pactl
    s = config.Settings()
    s.ui.mic = devices['mic']
    overlay_side, states = type('S', (), {'hide': lambda self: None})(), []
    ctl = app.Controller(type('O', (), {'side': lambda self, name: overlay_side})(), states.append, settings=s)
    ctl.startup()
    ctl._route = lambda s: ''
    ctl._side = lambda name, s: FakeSide(name, ctl._vmic)
    try:
        ctl.set_translation(True).result(10)
        assert ctl.on
        for m in reversed(pactl('list', 'short', 'modules').splitlines()):  # the server went: none of ours
            if 'peresmeshnik_' in m:
                pactl('unload-module', m.split('\t')[0])
        assert ctl._down.acquire(blocking=False)  # as _sound_gone does
        ctl.set_translation(False).result(10)  # the user, meanwhile
        ctl.submit(ctl._sound_back).result(10)  # the server is back
        assert not ctl.on and not ctl.sides and 'peresmeshnik_mic' in pactl('list', 'short', 'sources'), states
        ctl.set_translation(True).result(10)  # and on again by the user: on the devices made again
        assert ctl.on and [x.vmic for x in ctl.sides] == [ctl._vmic, ctl._vmic]
    finally:
        ctl.quit()
    assert not ours()


def test_translation_turned_on_while_the_sound_server_is_gone_goes_on_when_it_is_back(devices):
    """M-01 (review of PR #22, round 2): translation off, the sound server goes, the user turns translation on (it
    cannot start: no server); when the server is back, translation goes on."""
    from peresmeshnik import app, config
    s = config.Settings()
    s.ui.mic = devices['mic']
    overlay_side, states = type('S', (), {'hide': lambda self: None})(), []
    ctl = app.Controller(type('O', (), {'side': lambda self, name: overlay_side})(), states.append, settings=s)
    ctl.startup()
    ctl._route = lambda s: ''

    def no_server(name, s):
        raise RuntimeError('Connection refused')
    ctl._side = no_server
    try:
        ctl.set_translation(True).result(10)  # the user, while the server is gone: nothing starts
        assert not ctl.on, states
        ctl._side = lambda name, s: FakeSide(name, ctl._vmic)
        ctl._events.p.kill()  # the server's event (it went and is back): translation off at that moment
        assert wait_for(lambda: ctl.on, 10), states
    finally:
        ctl.quit()
    assert not ours()


def test_the_loop_follows_a_recording_moved_onto_peresmeshnik_mic(devices):
    """M-16, the review: the call app records another microphone, then the user picks Peresmeshnik Mic for it
    (pavucontrol, the default changed): the stream moves (a 'change' event only) and the loop comes."""
    from peresmeshnik import app, config
    from peresmeshnik.ui.selftest import pactl
    s = config.Settings()
    s.ui.mic = devices['mic']
    side = type('S', (), {'hide': lambda self: None})()
    ctl = app.Controller(type('O', (), {'side': lambda self, name: side})(), lambda state: None, settings=s)
    ctl.startup()
    call_app = subprocess.Popen(['parec', '--device=test_mic_feed.monitor', '--raw', '--format=float32le',
                                 '--rate=16000', '--channels=1'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        loops = lambda: [m for m in ours() if m.startswith('module-loopback') and 'source=test_mic' in m]  # noqa: E731
        feed = next(ln.split('\t')[0] for ln in pactl('list', 'short', 'sources').splitlines()
                    if ln.split('\t')[1] == 'test_mic_feed.monitor')
        mine = lambda: [ln.split('\t')[0] for ln in pactl('list', 'short', 'source-outputs').splitlines()  # noqa: E731
                        if ln.split('\t')[1] == feed]  # the call app's stream, by the source it records
        assert wait_for(lambda: mine(), 3) and not loops(), ours()
        time.sleep(0.5)  # its 'new' event handled: what follows is the move only
        pactl('move-source-output', mine()[0], 'peresmeshnik_mic')
        assert wait_for(lambda: loops(), 3), ours()
    finally:
        call_app.kill()
        call_app.wait()
        ctl.quit()
    assert not ours()


def test_one_device_failing_leaves_the_others(devices, monkeypatch):
    from peresmeshnik import app, config

    def broken():
        raise RuntimeError('no call sink here')
    monkeypatch.setattr(app, 'CallSink', broken)
    s = config.Settings()
    s.ui.mic = devices['mic']
    side = type('S', (), {'hide': lambda self: None})()
    ctl = app.Controller(type('O', (), {'side': lambda self, name: side})(), lambda state: None, settings=s)
    ctl.startup()
    try:
        assert ctl.startup_error == 'call sink: no call sink here'
        mods = ours()
        assert any('source_name=peresmeshnik_mic ' in m for m in mods), mods  # the virtual mic is still made
        call_app = record_peresmeshnik_mic()
        try:  # and fed while a call app records it
            assert wait_for(lambda: any(m.startswith('module-loopback') and 'source=test_mic' in m for m in ours()), 3), ours()
        finally:
            call_app.kill()
    finally:
        ctl.quit()
    assert not ours()


def test_selftest_cleans_up_when_playing_fails(devices, monkeypatch):
    from peresmeshnik.ui import selftest
    recorders = []
    real = selftest.Recorder

    def track(*a, **k):
        recorders.append(real(*a, **k))
        return recorders[-1]
    monkeypatch.setattr(selftest, 'Recorder', track)

    def broken(*a):
        raise OSError('pacat: connection refused')
    monkeypatch.setattr(selftest, 'play', broken)
    rs = selftest.run_all(devices['headphones'], devices['mic'])
    assert [r.ok for r in rs][:3] == [False, True, False]  # the virtual mic plays through its own class
    assert recorders and all(r.p.poll() is not None for r in recorders)  # no parec left recording
    assert not ours()  # the temporary call sink and virtual mic are gone


def test_a_signal_quits_after_the_running_slot():
    """SIGTERM's Python handler runs between any two bytecodes of the GUI thread: a direct quit() there ran
    aboutToQuit (hotkey.close: XCloseDisplay) inside the hotkey's X read, which then crashed (SIGSEGV on CI)."""
    code = """if True:
        import os, signal
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from peresmeshnik.app import _quit_on_signals
        qapp = QApplication([])
        seen = []
        qapp.aboutToQuit.connect(lambda: seen.append('quit'))
        _quit_on_signals(qapp)
        def slot():
            os.kill(os.getpid(), signal.SIGTERM)
            seen.append('slot end')
        QTimer.singleShot(0, slot)
        qapp.exec()
        print(seen)"""
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30,
                       env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})
    assert r.stdout.strip() == "['slot end', 'quit']", r.stdout + r.stderr


def test_headphones_reopen_when_the_device_goes(devices):
    from peresmeshnik.app import Headphones
    errors = []
    out = Headphones(devices['headphones'], errors.append)
    tone = np.zeros(2400, np.float32)
    out.play(tone, 24000)
    first = out._out
    first._proc.kill()  # what a Bluetooth profile switch does to pacat
    first._proc.wait()
    time.sleep(0.1)
    out.play(tone, 24000)
    out.play(tone, 24000)
    assert out._out is not first and out._out._proc.poll() is None  # a new pacat on the same sink
    first._thread.join(2)
    assert not first._thread.is_alive()  # the dead one's writer thread ended too
    assert errors == ['error: the headphones went away; reopening',  # once, in the tray
                      'translating (the headphones are back: test_headphones)']  # and the tray says it is over
    out.stop()


@pytest.mark.skipif(not (E2E and DISPLAY), reason='PERESMESHNIK_E2E=1 and a display: models, cloud key')
@pytest.mark.parametrize('them,me,subtitles', [('local', 'cloud', False), ('cloud', 'cloud', True)])
def test_subtitles_from_file_directions(tmp_path, devices, them, me, subtitles):
    """Both sides fed from recordings; their captions reach the overlay (its log), speech plays."""
    p = start_app(tmp_path, '--on', '--them-file', str(DATA / 'call-en.ogg'), '--me-file', str(DATA / 'call-ru.ogg'),
                  settings=f'them_to_me = "{them}"\nme_to_them = "{me}"\n[cloud]\nsubtitles = {str(subtitles).lower()}\n'
                           '[ui]\nheadphones = "test_headphones"\n')
    try:
        wait_line(p, 'translating', 120)
        time.sleep(40)  # 30 s recordings + the cloud's tail
    finally:
        stop_app(p)
    log = (tmp_path / 'app.err').read_text()
    rows = [line.split('overlay ', 1)[1] for line in log.splitlines() if ' overlay ' in line]
    print('\n'.join(rows[-12:]))
    for side, words in (('them', ('Mart', 'Март')), ('me', ('Mart', 'Март'))):  # Martelly / Мартейи, as the ASR hears it
        mine = [r for r in rows if r.startswith(side + ':')]
        assert mine, f'no captions from {side}'
        assert any('|' in r and r.split('|')[1].strip() for r in mine), f'no translation from {side}'
        assert any(w in r for r in mine for w in words), f'{side}: {mine[-3:]}'


def pump(cond, timeout: float, what: str):
    from PySide6.QtCore import QCoreApplication
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise AssertionError(f'timed out waiting for {what}')
        QCoreApplication.processEvents()
        time.sleep(0.02)


@pytest.mark.skipif(not (E2E and DISPLAY), reason='PERESMESHNIK_E2E=1 and a display: downloads models, uses the cloud key')
def test_wizard_linux_pages(tmp_path, devices, monkeypatch):
    """Every Linux page: languages and engines, devices, model download with licenses, the self-test,
    the hotkey and a test phrase each way; settings are saved at the end."""
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from peresmeshnik import config, models
    from peresmeshnik.app import Controller
    from peresmeshnik.ui.overlay import SubtitleOverlay
    from peresmeshnik.ui.tray import Tray
    from peresmeshnik.ui.wizard import DevicesPage, HotkeyPage, LanguagesPage, ModelsPage, SelfTestPage, SetupWizard
    overlay, states = SubtitleOverlay(), []
    ctl = Controller(overlay, states.append)
    w = SetupWizard(Tray(ctl))
    w.show()
    try:
        assert isinstance(w.currentPage(), LanguagesPage)  # defaults: ru/en, them -> me local, me -> them cloud (no GPU)
        w.next()
        page = w.currentPage()
        assert isinstance(page, DevicesPage)
        page.headphones.setCurrentIndex(page.headphones.findData(devices['headphones']))
        page.mic.setCurrentIndex(page.mic.findData(devices['mic']))
        w.next()
        page = w.currentPage()
        assert isinstance(page, ModelsPage)
        print('models:', [m.id for m in page.ms])
        t = time.monotonic()
        for _ in range(3):  # a failed download keeps its .part: "Download" again resumes, as a user would
            page.button.click()
            pump(lambda: not page.busy, 900, 'the downloads')
            if page.isComplete():
                break
            print('retry:', page.state.text())
        print(f'models ready in {time.monotonic() - t:.0f} s: {page.state.text()}')
        assert all(models.is_fetched(m) for m in page.ms)
        w.next()
        page = w.currentPage()
        assert isinstance(page, SelfTestPage)
        page.button.click()
        pump(lambda: page.ran, 60, 'the self-test')
        print(page.out.text().replace('<br>', '\n'))
        assert page.ok
        w.next()
        page = w.currentPage()
        assert isinstance(page, HotkeyPage)
        for side in ('them', 'me'):
            n = overlay.updates
            page.tests[side].click()
            pump(lambda: page.out.text() == 'test done' or page.out.text().startswith('error'), 180, f'the {side} test phrase')
            assert page.out.text() == 'test done'
            print(side, [r for r in overlay._shown])
            assert overlay.updates > n and any(r[2] for r in overlay._shown), f'no translated subtitle from {side}'
        w.next()
        w.accept()
        s = config.load()
        assert (s.ui.headphones, s.ui.mic, s.them_to_me, s.me_to_them) == ('test_headphones', 'test_mic', 'local', 'cloud')
    finally:
        w.close()
        ctl.quit()


def pair_models(monkeypatch, s, my: str, their: str) -> str:
    """The pair's own models when the registry has them (stage 6, S1/S2); until then stand-ins: the en/ru models' files
    under the pair's languages, so the whole real path runs (its words are wrong: this checks the path, not the
    translation). Returns which."""
    from dataclasses import replace

    from peresmeshnik import languages, models
    if all(languages.direction(s, k).local for k in languages.SIDES):
        return 'real models'
    # only the work-gate's models (its CI cache): nothing to download for the stand-ins
    asr, voices = 'parakeet-tdt-0.6b-v3', {my: 'piper-ru_RU-dmitri-medium', their: 'piper-ru_RU-dmitri-medium'}
    mts = {f'{their}-{my}': 'opus-mt-en-ru', f'{my}-{their}': 'opus-mt-en-ru'}
    reg = dict(models.registry())
    for mid, langs in [(asr, (my, their)), *((v, (k,)) for k, v in voices.items()), *((v, (k,)) for k, v in mts.items())]:
        reg[mid] = replace(reg[mid], langs=reg[mid].langs + langs)  # the same files: the downloaded folders stay valid
    monkeypatch.setattr(models, 'registry', lambda: reg)
    s.local.asr.update({my: asr, their: asr})
    s.local.mt.update(mts)
    s.local.tts.update(voices)
    from peresmeshnik import pipeline  # its warm-up phrase is en/ru until S1's stage 6 (a KeyError for other sources)
    monkeypatch.setattr(pipeline, 'HELLO', {my: 'Hallo.', their: 'こんにちは。', **pipeline.HELLO})
    return 'stand-ins'


@pytest.mark.skipif(not (PAIR_MODELS or E2E), reason='PERESMESHNIK_PAIR_MODELS (CI) or PERESMESHNIK_E2E=1: runs the models')
def test_a_new_pair_through_the_wizard_and_the_app(tmp_path, devices, monkeypatch):
    """de -> ja, both sides local: the languages page offers local for both, the models page lists only the pair's
    models (no GigaAM, no clone: the stock voice), the test phrase in each language goes through the app's real path
    (Silero VAD, ASR, MT, Piper, the headphones) and shows translated subtitles; the pair is saved at the end."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from peresmeshnik import config, languages
    from peresmeshnik.app import Controller
    from peresmeshnik.ui.overlay import SubtitleOverlay
    from peresmeshnik.ui.tray import Tray
    from peresmeshnik.ui.wizard import DevicesPage, HotkeyPage, LanguagesPage, ModelsPage, SelfTestPage, SetupWizard, VoicePage
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    if PAIR_MODELS:
        monkeypatch.setenv('PERESMESHNIK_MODELS', PAIR_MODELS)
    s = config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local')
    s.local.clone = 'piper'  # the stock voice: no recording, no voice pack
    print('de -> ja on', pair_models(monkeypatch, s, 'de', 'ja'))
    config.save(s)
    overlay, states = SubtitleOverlay(), []
    ctl = Controller(overlay, states.append)
    w = SetupWizard(Tray(ctl))
    w.show()
    try:
        page = w.currentPage()
        assert isinstance(page, LanguagesPage) and (page.my_lang.currentData(), page.their_lang.currentData()) == ('de', 'ja')
        for key in languages.SIDES:
            assert page.engine[key].currentText() == 'local' and page.why[key].text().startswith('✔'), page.why[key].text()
        print(page.download.text())
        w.next()
        page = w.currentPage()
        assert isinstance(page, DevicesPage)
        page.headphones.setCurrentIndex(page.headphones.findData(devices['headphones']))
        page.mic.setCurrentIndex(page.mic.findData(devices['mic']))
        w.next()
        page = w.currentPage()
        assert isinstance(page, ModelsPage)
        ids = [m.id for m in page.ms]
        print('models:', ids, 'to download:', [m.id for m in page.missing()])
        assert ids == languages.needed_models(w.s) and not {'gigaam-v3-e2e-ctc', 'xtts-v2', 't-one'} & set(ids)
        for _ in range(3):  # a cut download resumes on the next click, as a user would
            if page.isComplete():
                break
            page.button.click()
            pump(lambda: not page.busy, 900, 'the downloads')
        assert page.isComplete(), page.state.text()
        w.next()
        page = w.currentPage()
        assert isinstance(page, SelfTestPage)
        page.button.click()
        pump(lambda: page.ran, 60, 'the self-test')
        w.next()
        page = w.currentPage()
        assert isinstance(page, VoicePage) and page.isComplete()  # the stock voice: nothing to record
        w.next()
        page = w.currentPage()
        assert isinstance(page, HotkeyPage)
        for side in ('them', 'me'):  # them: ja.ogg -> German; me: de.ogg -> Japanese
            n = overlay.updates
            page.tests[side].click()
            pump(lambda: page.out.text() == 'test done' or page.out.text().startswith('test done with'), 180,
                 f'the {side} test phrase')
            print(side, overlay._shown)
            assert page.out.text() == 'test done', page.out.text()
            assert overlay.updates > n and any(r[2] for r in overlay._shown), f'no translated subtitle from {side}'
        w.next()
        w.accept()
        got = config.load()
        assert (got.my_lang, got.their_lang, got.them_to_me, got.me_to_them) == ('de', 'ja', 'local', 'local')
    finally:
        w.close()
        ctl.quit()

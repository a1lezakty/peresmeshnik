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
pytestmark = pytest.mark.skipif(not SOUND, reason='needs PipeWire / PulseAudio')
DISPLAY = bool(os.environ.get('DISPLAY'))
E2E = os.environ.get('PERESMESHNIK_E2E') == '1'
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
    assert p.wait(20) == 0


@pytest.mark.skipif(not (DISPLAY and shutil.which('xdotool')), reason='needs an X display and xdotool')
def ours() -> list[str]:
    """Our PipeWire modules: "module name args" lines."""
    from peresmeshnik.ui.selftest import pactl
    return [' '.join(m.split('\t')[1:3]) for m in pactl('list', 'short', 'modules').splitlines() if 'peresmeshnik_' in m]


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


def test_devices_live_with_the_app_and_the_mic_is_looped_while_off(tmp_path, devices):
    """Call sink, virtual mic and the microphone loopback exist from the start (stale ones from a killed run
    replaced), a second instance leaves them alone, SIGHUP removes them."""
    from peresmeshnik.ui.selftest import pactl
    pactl('load-module', 'module-null-sink', 'sink_name=peresmeshnik_call')  # as left by a SIGKILL
    p = start_app(tmp_path, settings='[ui]\nmic = "test_mic"\n')
    try:
        mods = ours()
        calls = [m for m in mods if 'sink_name=peresmeshnik_call' in m]
        assert len(calls) == 1 and 'priority.session=0' in calls[0]  # the app's own sink (module ids get reused)
        assert any(m.startswith('module-loopback') and 'source=test_mic sink=peresmeshnik_mic_sink' in m for m in mods), mods
        assert not any('source=peresmeshnik_' in m for m in mods)  # never our own device looped back
        second = subprocess.run([sys.executable, '-m', 'peresmeshnik', 'app', '--no-wizard'], capture_output=True, text=True,
                                timeout=30, env={**os.environ, 'PERESMESHNIK_CONFIG': str(tmp_path / 'settings.toml')})
        assert second.returncode == 3 and 'already running' in second.stderr
        assert ours() == mods  # the first one's devices untouched
        p.send_signal(signal.SIGHUP)  # the terminal closed
        assert p.wait(20) == 0
    finally:
        if p.poll() is None:
            stop_app(p)
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


@pytest.mark.filterwarnings('ignore::pytest.PytestUnhandledThreadExceptionWarning')  # out.py's writer sees the pipe go
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
    assert errors == ['error: the headphones went away; reopening']  # once, in the tray
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

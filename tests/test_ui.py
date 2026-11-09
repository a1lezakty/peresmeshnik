"""UI and app wiring without devices or models: the overlay repaints only on new text, captions land
in the right rows, the live switch routes audio around a direction, the self-test finds its chirp,
the wizard asks for the right models, hotkey combos parse. Qt runs offscreen unless a display is set."""
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import pytest

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from peresmeshnik import config  # noqa: E402
from peresmeshnik.app import Side  # noqa: E402
from peresmeshnik.engines import Caption, Speech  # noqa: E402
from peresmeshnik.ui import selftest  # noqa: E402
from peresmeshnik.ui.hotkey import HotkeyError, parse  # noqa: E402
from peresmeshnik.ui.overlay import SubtitleOverlay  # noqa: E402

app = QApplication.instance() or QApplication([])


def settle(seconds=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def test_overlay_repaints_only_on_new_text():
    o = SubtitleOverlay()
    flags = o.windowFlags()
    for f in (Qt.WindowTransparentForInput, Qt.WindowStaysOnTopHint, Qt.FramelessWindowHint):
        assert flags & f, f  # click-through, on top, no frame
    them = o.side('them')
    them.show(Caption('hello', clause=0))
    settle()
    assert o.isVisible() and o.updates == 1
    paints = o.paints
    for _ in range(5):  # the same text again: no repaint
        them.show(Caption('hello', clause=0))
    settle(0.2)
    assert (o.updates, o.paints) == (1, paints)
    them.show(Caption('hello world', clause=0))
    them.show(Caption('Hello world.', 'Привет, мир.', final=True, clause=0))
    settle()
    assert o.updates == 3
    o.paint_ms.clear()
    for i in range(20):  # each a new text, painted at once
        them.show(Caption(f'Сказали {i} раз, and more words to fill a line', f'Said {i} times', final=True, clause=i))
        settle(0.01)
        o.repaint()
    ms = sorted(o.paint_ms)
    print(f'overlay paint: median {ms[len(ms) // 2]:.2f} ms, max {ms[-1]:.2f} ms, n={len(ms)}')
    assert ms[int(len(ms) * 0.9)] < 16  # the budget per repaint (ARCHITECTURE §4)
    them.hide()
    settle()
    assert not o.isVisible()


def test_overlay_rows():
    o = SubtitleOverlay(rows=2)
    me, them = o.side('me'), o.side('them')
    them.show(Caption('one', clause=0))
    them.show(Caption('two', clause=1))  # the partial of clause 1 comes before the final of 0
    them.show(Caption('One.', 'Один.', final=True, clause=0))
    settle()
    assert [r[1:] for r in o._shown] == [('One.', 'Один.'), ('two', '')]
    # cloud captions have no clause number: partials grow one row, a final closes it,
    # a translation goes to the oldest row of that side still without one
    o._on_clear()
    me.show(Caption('Привет'))
    me.show(Caption('Привет всем', final=True))
    me.show(Caption('Как'))
    me.show(Caption('Привет всем.', 'Hello everyone.', final=True))
    settle()
    assert [r for r in o._shown] == [('» ', 'Привет всем.', 'Hello everyone.'), ('» ', 'Как', '')]


class FakeDirection:
    def __init__(self):
        self.fed, self.events_q = [], []
        self.closed = threading.Event()

    def feed(self, pcm):
        self.fed.append(pcm)

    def events(self):
        yield Caption('x', clause=0)
        yield Speech(np.ones(10, np.float32), 24000)
        self.closed.wait()

    def close(self):
        self.closed.set()


class FakeOut:
    def __init__(self):
        self.played, self.flushed = [], 0

    def play(self, pcm, sr):
        self.played.append((len(pcm), sr))

    def flush(self):
        self.flushed += 1

    def stop(self):
        pass


class FakeIn:
    def start(self, cb):
        self.cb = cb

    def stop(self):
        pass


def test_live_switch_routes_around_the_direction(monkeypatch):
    s = config.Settings()
    s.them_to_me = 'local'
    d, out, inp, shown = FakeDirection(), FakeOut(), FakeIn(), []
    monkeypatch.setattr(Side, '_local', lambda self, s: d)
    side = Side('them', s, inp, out, type('O', (), {'show': lambda self, c: shown.append(c)})())
    side.start()
    time.sleep(0.1)
    assert out.played == [(10, 24000)] and len(shown) == 1  # translated speech and its caption
    frame = np.zeros(512, np.float32)
    inp.cb(frame)
    assert len(d.fed) == 1
    side.set_live(True)
    assert out.flushed == 1  # queued translation dropped
    inp.cb(frame)  # the capture callback only queues; the pass-through thread plays
    time.sleep(0.1)
    assert len(d.fed) == 1 and out.played[-1] == (512, 16000)  # the original goes straight out
    side.set_live(False)
    inp.cb(frame)
    assert len(d.fed) == 2
    side.stop()


def test_selftest_finds_the_chirp_only_when_it_is_there():
    ref = selftest.chirp()
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(selftest.SR * 2) * 0.01).astype(np.float32)
    rec = noise.copy()
    rec[30_000:30_000 + len(ref)] += 0.3 * ref  # quieter and delayed, like a device path
    score, at = selftest.find(rec, ref)
    assert score > 0.9 and at == 30_000
    assert selftest.find(noise, ref)[0] < 0.1
    assert selftest.find(np.zeros(selftest.SR * 2, np.float32), ref)[0] < 0.1  # silence is no match
    sine = np.sin(2 * np.pi * 1000 * np.arange(selftest.SR * 2) / selftest.SR).astype(np.float32) * 0.3
    assert selftest.find(sine, ref)[0] < selftest.MIN_SCORE  # a steady tone is no chirp


def test_wizard_asks_for_the_right_models():
    from peresmeshnik.ui.wizard import needed_models
    s = config.Settings()  # them -> me local, me -> them cloud with local subtitles
    assert needed_models(s) == ['silero-vad', 'parakeet-tdt-0.6b-v3', 'opus-mt-en-ru', 'piper-ru_RU-dmitri-medium',
                                'gigaam-v3-e2e-ctc', 'opus-mt-ru-en']
    s.cloud.subtitles, s.them_to_me = True, 'cloud'
    assert needed_models(s) == ['silero-vad']
    s.me_to_them = 'local'
    assert needed_models(s) == ['silero-vad', 'gigaam-v3-e2e-ctc', 'opus-mt-ru-en', 'xtts-v2']


def test_hotkey_combos():
    assert parse('Ctrl+Alt+Space') == (['ctrl', 'alt'], 'space')
    assert parse('Meta+F9') == (['meta'], 'F9')
    assert parse('Ctrl+Shift+L') == (['ctrl', 'shift'], 'l')
    for bad in ('', 'Hyper+K', 'Ctrl+'):
        with pytest.raises(HotkeyError):
            parse(bad)


@pytest.mark.skipif(not (sys.platform.startswith('linux') and os.environ.get('DISPLAY') and shutil.which('xdotool')),
                    reason='needs an X display and xdotool')
def test_x11_hotkey_fires_once_per_press():
    from peresmeshnik.ui.hotkey import X11Hotkey
    hk, hits = X11Hotkey(), []
    hk.register('Ctrl+Shift+J', lambda: hits.append(1))
    try:
        with pytest.raises(HotkeyError, match='taken'):
            X11Hotkey().register('Ctrl+Shift+J', lambda: None)  # a second grab of the same combo
        for keys in ('ctrl+shift+j', 'ctrl+shift+j', 'ctrl+j'):
            subprocess.run(['xdotool', 'key', keys], check=True)
        settle(0.5)
        assert len(hits) == 2, f'grabbed {hk._grabs}; the X server sends: {_root_key_events(hk)}'
    finally:
        hk.close()


def _root_key_events(hk) -> list:
    """Diagnosis: (type, keycode, state) of the key events the root window sees for the same keys."""
    import ctypes

    from peresmeshnik.ui.hotkey import _Event
    hk.unregister('Ctrl+Shift+J')
    hk.x.XSelectInput.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_long]
    hk.x.XSelectInput(hk.dpy, hk.root, 3)  # KeyPressMask | KeyReleaseMask
    hk.x.XSync(hk.dpy, 0)
    subprocess.run(['xdotool', 'key', 'ctrl+shift+j'], check=True)
    time.sleep(0.3)
    out, ev = [], _Event()
    while hk.x.XPending(hk.dpy):
        hk.x.XNextEvent(hk.dpy, ctypes.byref(ev))
        out.append((ev.type, ev.key.keycode, hex(ev.key.state)))
    return out


def test_idle_cpu_with_tray_and_visible_overlay(tmp_path):
    """tests/test_idle_cpu.py with the overlay on screen too (hidden while idle otherwise) and the first-run
    wizard open: no repaint loop anywhere. CPU per quarter of the window shows a one-off from a steady cost."""
    window = float(os.environ.get('IDLE_WINDOW_S', 60))
    p = subprocess.Popen([sys.executable, '-m', 'peresmeshnik', 'app', '--show-overlay'], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, 'PERESMESHNIK_CONFIG': str(tmp_path / 'settings.toml')})  # no file: first run
    try:
        assert 'idle' in p.stdout.readline()
        proc = psutil.Process(p.pid)

        def cpu():
            return sum(sum(q.cpu_times()[:2]) for q in [proc, *proc.children(recursive=True)])
        if sys.platform == 'darwin':  # AppKit's one-off work after the first show, as in tests/test_idle_cpu.py: printed
            first = cpu()
            time.sleep(15)
            print(f'idle CPU with tray, overlay and wizard in the first 15 s (not gated): {cpu() - first:.3f} s')
        marks = [cpu()]
        for _ in range(4):
            time.sleep(window / 4)
            marks.append(cpu())
        assert p.poll() is None, 'the app exited while idle'
    finally:
        p.terminate()
        p.wait(20)
    used = marks[-1] - marks[0]
    quarters = ' '.join(f'{b - a:.3f}' for a, b in zip(marks, marks[1:]))
    print(f'idle CPU with tray, overlay and wizard: {used:.3f} s in {window:.0f} s, per quarter {quarters} '
          f'({os.environ.get("QT_QPA_PLATFORM") or "native"})')
    assert used < 0.13 * window / 60


@pytest.mark.skipif(not (sys.platform.startswith('linux') and shutil.which('dbus-daemon')), reason='needs dbus-daemon')
def test_portal_hotkey_against_a_fake_portal():
    """The XDG GlobalShortcuts flow on a private bus: CreateSession and BindShortcuts answer through
    Request.Response signals, then Activated for our session fires the callback on the GUI thread."""
    from jeepney import DBusAddress, HeaderFields, MessageType, message_bus, new_method_return, new_signal
    from jeepney.io.blocking import open_dbus_connection

    from peresmeshnik.ui.hotkey import PORTAL, PORTAL_PATH, SHORTCUTS, PortalHotkey
    bus = subprocess.Popen(['dbus-daemon', '--session', '--nofork', '--print-address=1'], stdout=subprocess.PIPE, text=True)
    address = bus.stdout.readline().strip()
    service = open_dbus_connection(address)
    service.send_and_get_reply(message_bus.RequestName(PORTAL))
    calls, session, refuse = [], '/org/freedesktop/portal/desktop/session/1_2/peresmeshnik', set()
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                m = service.receive(timeout=0.5)
            except TimeoutError:
                continue
            if m.header.message_type != MessageType.method_call:
                continue
            f = m.header.fields
            member, sender = f[HeaderFields.member], f[HeaderFields.sender][1:].replace('.', '_')
            calls.append(member)
            if member == 'Get':
                service.send(new_method_return(m, 'v', (('u', 1),)))
                continue
            if member == 'Close':
                service.send(new_method_return(m))
                continue
            options = m.body[-1]
            request = f'{PORTAL_PATH}/request/{sender}/{options["handle_token"][1]}'
            service.send(new_method_return(m, 'o', (request,)))
            results = ({'session_handle': ('s', session)} if member == 'CreateSession' else
                       {'shortcuts': ('a(sa{sv})', [(m.body[1][0][0], {'trigger_description': ('s', 'Ctrl+Alt+Space')})])})
            code = 2 if member in refuse else 0  # 2: the user cancelled
            response = DBusAddress(request, interface='org.freedesktop.portal.Request')
            service.send(new_signal(response, 'Response', 'ua{sv}', (code, results)))
    threading.Thread(target=serve, daemon=True).start()
    hk = None
    try:
        hk, hits = PortalHotkey(bus=address, timeout=5), []
        hk.register('Ctrl+Alt+Space', lambda: hits.append(threading.current_thread() is threading.main_thread()))
        assert hk.session == session and hk.trigger == {'toggle-live': 'Ctrl+Alt+Space'}
        assert calls == ['Get', 'CreateSession', 'BindShortcuts']
        for sess in (session, '/other/session', session):
            service.send(new_signal(DBusAddress(PORTAL_PATH, interface=SHORTCUTS), 'Activated', 'osta{sv}', (sess, 'toggle-live', 1, {})))
        settle(0.5)
        assert hits == [True, True]  # ours only, on the GUI thread
        refuse.add('BindShortcuts')  # a new combo the desktop refuses: the old session and callback stay
        with pytest.raises(HotkeyError, match='refused'):
            hk.register('Ctrl+Alt+K', lambda: hits.append('new'))
        assert hk.session == session and calls[-3:] == ['CreateSession', 'BindShortcuts', 'Close']  # the new session closed
        service.send(new_signal(DBusAddress(PORTAL_PATH, interface=SHORTCUTS), 'Activated', 'osta{sv}', (session, 'toggle-live', 2, {})))
        settle(0.3)
        assert hits[-1] is True
        refuse.clear()  # a new combo: a new session (one bind per session, the spec), then the old one closes
        calls.clear()
        hk.register('Ctrl+Alt+K', lambda: hits.append('new'))
        assert calls == ['CreateSession', 'BindShortcuts', 'Close'] and hk.session == session  # the fake reuses the path
    finally:
        stop.set()
        if hk:
            hk.close()
        service.close()
        bus.terminate()
        bus.wait(5)


def test_wizard_models_page_downloads_with_consent(tmp_path, monkeypatch, server):
    """The models page: restricted licenses need the box ticked; a download cut mid-file resumes; then complete."""
    import hashlib

    from peresmeshnik import models
    from peresmeshnik.ui import wizard
    httpd, url = server
    files = {'a.bin': b'a' * 300_000, 'b.bin': b'b' * 200_000}
    for n, b in files.items():
        (httpd.root / n).write_bytes(b)
    httpd.cut.add('/a.bin')
    pins = {n: {'sha256': hashlib.sha256(b).hexdigest(), 'size': len(b)} for n, b in files.items()}
    reg = {'free': models.Model('free', 'mt', ('en-ru',), 'MIT', 'https://x', url, {'b.bin': pins['b.bin']}),
           'nc': models.Model('nc', 'clone', ('en',), 'CPML', 'https://x', url, {'a.bin': pins['a.bin']}, consent=True)}
    monkeypatch.setattr(models, 'registry', lambda: reg)
    monkeypatch.setattr(wizard, 'needed_models', lambda s: ['free', 'nc'])
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    page = wizard.ModelsPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    assert not page.isComplete() and not page.consent.isHidden() and 'nc (CPML)' in page.consent.text()
    page.download()
    assert 'Accept' in page.state.text() and not page.busy  # nothing starts without consent
    page.consent.setChecked(True)
    page.download()
    end = time.monotonic() + 30
    while page.busy and time.monotonic() < end:
        settle(0.05)
    assert page.isComplete(), page.state.text()
    assert page.state.text() == 'All models are here.'
    assert ('/a.bin', None) in httpd.ranges and any(p == '/a.bin' and r for p, r in httpd.ranges)  # resumed with Range


def test_tray_menu_and_settings_window(tmp_path, monkeypatch):
    """The menu reflects the settings; choosing an engine saves it; the settings window saves its fields
    (the Gemini key into its own file, never into settings.toml) and asks the app to rebind the hotkey."""
    from peresmeshnik.ui import tray as tray_mod
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    calls = []

    class Ctl:
        live, on = False, False

        def set_translation(self, on):
            calls.append(('translation', on))
    t = tray_mod.Tray(Ctl())
    texts = [a.text() for a in t.menu.actions()]
    assert {'Translation', 'Live voice', 'Settings…', 'Setup wizard…', 'Quit'} <= set(texts)
    cloud = next(a for a in t.engines['them_to_me'].actions() if a.data()[1] == 'cloud')
    cloud.trigger()
    assert config.load().them_to_me == 'cloud'
    t.set_state('translating (them: local)')
    assert t.translation.isChecked() and calls == []  # the state shows, it does not switch anything
    t.reload_hotkey = lambda combo: calls.append(('hotkey', combo)) or ''
    d = tray_mod.SettingsDialog(t)
    d.engine['me_to_them'].setCurrentText('local')
    d.key.setText('secret-key')
    from PySide6.QtGui import QKeySequence
    d.hotkey.setKeySequence(QKeySequence('Ctrl+Shift+K'))
    d.save()
    s = config.load()
    assert (s.them_to_me, s.me_to_them, s.ui.hotkey) == ('cloud', 'local', 'Ctrl+Shift+K')
    assert ('hotkey', 'Ctrl+Shift+K') in calls
    assert (tmp_path / 'gemini-key').read_text().strip() == 'secret-key'
    assert 'secret-key' not in (tmp_path / 'settings.toml').read_text()
    if sys.platform != 'win32':
        assert (tmp_path / 'gemini-key').stat().st_mode & 0o077 == 0  # only the user reads it



@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the Call app menu is Linux only')
def test_tray_call_menu_lists_the_apps_off_the_gui_thread(tmp_path, monkeypatch):
    """Opening "Call app" lists the playing apps on a thread (pactl may take seconds): the GUI thread does not wait."""
    from peresmeshnik.audio import capture_pipewire
    from peresmeshnik.ui import tray as tray_mod
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    release, threads = threading.Event(), []

    def slow_apps():
        threads.append(threading.current_thread())
        release.wait(10)
        return [('Telegram', 'Telegram')]
    monkeypatch.setattr(capture_pipewire, 'apps', slow_apps)

    class Ctl:
        live, on = False, False
    t = tray_mod.Tray(Ctl())
    t._fill_call_menu()  # returns while apps() still waits
    assert [a.text() for a in t.call_menu.actions()] == ['Looking for the apps playing now…']
    release.set()
    end = time.monotonic() + 5
    while 'Telegram' not in [a.text() for a in t.call_menu.actions()] and time.monotonic() < end:
        settle()
    assert 'Telegram' in [a.text() for a in t.call_menu.actions()]
    assert threads and threads[0] is not threading.main_thread()

def test_hotkey_state_ignores_locks_buttons_and_layout_group():
    from peresmeshnik.ui.hotkey import ALT, CTRL, SHIFT, X11Hotkey
    mask = CTRL | SHIFT
    for extra in (0, 0x2, 0x10, 0x100, 0x2000, 0x2000 | 0x10):  # Lock, NumLock, button 1, the second layout (ru)
        assert X11Hotkey.matches(mask | extra, mask), hex(extra)
    assert not X11Hotkey.matches(mask | ALT, mask) and not X11Hotkey.matches(CTRL, mask)


def test_side_reports_a_dead_direction_and_closes_the_cloud_on_a_failed_start(monkeypatch):
    s = config.Settings()
    s.them_to_me = 'local'

    class Dead(FakeDirection):
        def events(self):
            yield Caption('x', clause=0)
            raise ConnectionError('quota exceeded')
    errors = []
    monkeypatch.setattr(Side, '_local', lambda self, s: Dead())
    side = Side('them', s, FakeIn(), FakeOut(), type('O', (), {'show': lambda self, c: None})(), errors.append)
    side.start()
    side.stop()
    assert errors == ['error: them: quota exceeded']

    s.them_to_me, s.cloud.subtitles = 'cloud', False  # cloud voice + local subtitles that fail to load
    cloud = FakeDirection()
    monkeypatch.setattr(Side, '_cloud', lambda self, s: cloud)
    from peresmeshnik import pipeline

    def broken(*a, **k):
        raise RuntimeError('onnxruntime: bad model file')
    monkeypatch.setattr(pipeline, 'make_local', broken)
    with pytest.raises(RuntimeError, match='bad model'):
        Side('them', s, FakeIn(), FakeOut(), None)
    assert cloud.closed.is_set()  # its session does not stay open


def test_supports_live_direction_gets_the_switch_and_keeps_its_input(monkeypatch):
    s = config.Settings()
    s.me_to_them = 'local'

    class Crossfading(FakeDirection):
        supports_live = True

        def __init__(self):
            super().__init__()
            self.live_calls = []

        def set_live(self, on):
            self.live_calls.append(on)
    d, out, inp = Crossfading(), FakeOut(), FakeIn()
    monkeypatch.setattr(Side, '_local', lambda self, s: d)
    side = Side('me', s, inp, out, type('O', (), {'show': lambda self, c: None})())
    side.start()
    side.set_live(True)
    inp.cb(np.zeros(512, np.float32))
    time.sleep(0.1)
    assert d.live_calls == [True] and len(d.fed) == 1  # the direction does the switch (crossfade) and still gets the input
    assert out.played == [(10, 24000)] and out.flushed == 0  # its Speech (the live voice) plays in live mode
    side.stop()


def test_resampler_keeps_the_phase_between_chunks():
    from peresmeshnik.app import Resampler
    rs, n_in, out = Resampler(16000, 22050), 0, []
    x = np.sin(2 * np.pi * 440 * np.arange(16000 * 60) / 16000).astype(np.float32)
    for i in range(0, len(x), 320):  # 20 ms chunks for a minute
        out.append(rs(x[i:i + 320]))
        n_in += len(x[i:i + 320])
    y = np.concatenate(out)
    assert abs(len(y) - n_in * 22050 / 16000) <= 1  # no drift (per-chunk rounding gave +34 ms per minute)
    ref = np.sin(2 * np.pi * 440 * np.arange(len(y)) / 22050)
    assert np.max(np.abs(y[100:] - ref[100:])) < 0.02  # continuous: no click at chunk joins


def test_default_devices_that_are_ours_are_refused(monkeypatch):
    from peresmeshnik import app as app_mod
    with pytest.raises(RuntimeError, match='our own device'):
        app_mod.not_ours('peresmeshnik_mic_sink.monitor', 'default input')
    if sys.platform == 'darwin':
        pytest.skip('the rest is PipeWire: on macOS BlackHole 2ch is refused in tests/test_virtual_mic_macos.py')
    monkeypatch.setattr(app_mod, 'pactl', lambda *a: 'peresmeshnik_mic_sink.monitor\n' if a == ('get-default-source',) else '')
    ctl = app_mod.Controller(None, lambda st: None)
    with pytest.raises(RuntimeError, match='our own device'):
        ctl._input('me')  # me -> them would translate its own translation in a loop


@pytest.mark.skipif(not (sys.platform.startswith('linux') and os.environ.get('DISPLAY') and shutil.which('xdotool')),
                    reason='needs an X display and xdotool')
def test_hotkey_rebinds_after_a_failed_first_grab():
    from peresmeshnik.ui.hotkey import Binding, X11Hotkey
    other = X11Hotkey()  # another app holds the combo
    other.register('Ctrl+Shift+J', lambda: None)
    hits, done = [], []
    b = Binding(X11Hotkey(), lambda: hits.append(1))
    b.done.connect(lambda combo, why: done.append((combo, bool(why))))
    try:
        assert 'taken' in b.bind('Ctrl+Shift+J') and b.combo == ''
        assert b.bind('Ctrl+Shift+U') == '' and b.combo == 'Ctrl+Shift+U'  # works after the failure
        subprocess.run(['xdotool', 'key', 'ctrl+shift+u'], check=True)
        settle(0.5)
        assert hits == [1] and done == [('', True), ('Ctrl+Shift+U', False)]
        assert 'taken' in b.bind('Ctrl+Shift+J') and b.combo == 'Ctrl+Shift+U'  # a refused new combo keeps the old one
        subprocess.run(['xdotool', 'key', 'ctrl+shift+u'], check=True)
        settle(0.5)
        assert hits == [1, 1]
    finally:
        b.close()
        other.close()


@pytest.mark.skipif(not (sys.platform.startswith('linux') and os.environ.get('DISPLAY') and shutil.which('xdotool')),
                    reason='needs an X display and xdotool')
def test_overlay_is_click_through_for_real():
    """A click on the overlay lands in the window under it (X input shape), not in the overlay."""
    from PySide6.QtWidgets import QWidget
    clicks = []

    class Under(QWidget):
        def mousePressEvent(self, e):  # noqa: N802
            clicks.append('under')
    under = Under()
    o = SubtitleOverlay()
    under.setGeometry(o.geometry().adjusted(-50, -50, 50, 50))
    under.show()
    settle(0.3)
    o.side('them').show(Caption('click me', 'нажми меня', True, 0))
    settle(0.5)
    o.mousePressEvent = lambda e: clicks.append('overlay')
    c = o.geometry().center()
    subprocess.run(['xdotool', 'mousemove', str(c.x()), str(c.y()), 'click', '1'], check=True)
    settle(0.5)
    o.close()
    under.close()
    assert clicks == ['under']

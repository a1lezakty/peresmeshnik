"""macOS shell on the runner's own desktop: the Carbon hotkey (registration, refusals, a synthetic press), the
menu-bar icon as a template image, the overlay's window (level, Spaces, click-through) and an accessory app whose
windows still get the focus. Synthetic input goes through CGEventPost; when the runner drops it (no permission to
post events), the press tests skip and say so."""
import ctypes
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from ctypes import c_bool, c_double, c_uint16, c_uint32, c_uint64, c_void_p

import pytest

if sys.platform != 'darwin':
    pytest.skip('macOS only', allow_module_level=True)
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget  # noqa: E402

from peresmeshnik import config  # noqa: E402
from peresmeshnik.engines import Caption  # noqa: E402
from peresmeshnik.ui import macos  # noqa: E402
from peresmeshnik.ui.hotkey import HotkeyError  # noqa: E402

app = QApplication.instance() or QApplication([])
macos.accessory()  # as the app does: only an accessory app's windows may join another app's full-screen Space
cg = macos.cg
cg.CGEventCreateKeyboardEvent.restype, cg.CGEventCreateKeyboardEvent.argtypes = c_void_p, [c_void_p, c_uint16, c_bool]
cg.CGEventSetFlags.argtypes = [c_void_p, c_uint64]
cg.CGEventPost.argtypes = [c_uint32, c_void_p]


class _Point(ctypes.Structure):
    _fields_ = [('x', c_double), ('y', c_double)]


cg.CGEventCreateMouseEvent.restype, cg.CGEventCreateMouseEvent.argtypes = c_void_p, [c_void_p, c_uint32, _Point, c_uint32]
CG_FLAGS = {macos.CMD: 1 << 20, macos.SHIFT: 1 << 17, macos.OPTION: 1 << 19, macos.CONTROL: 1 << 18}
COMBO = 'Ctrl+Alt+Shift+K'  # ⌥⇧⌘K: no macOS shortcut


def settle(seconds=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def press(code: int, mods: int = 0) -> None:
    """A key press and release from the HID level, as a keyboard would send it."""
    for down in (True, False):
        e = cg.CGEventCreateKeyboardEvent(None, code, down)
        cg.CGEventSetFlags(e, sum(f for m, f in CG_FLAGS.items() if mods & m))
        cg.CGEventPost(0, e)  # kCGHIDEventTap
        macos.cf.CFRelease(e)
        time.sleep(0.03)


def click(x: float, y: float) -> None:
    for kind in (1, 2):  # left mouse down, up
        e = cg.CGEventCreateMouseEvent(None, kind, _Point(x, y), 0)
        cg.CGEventPost(0, e)
        macos.cf.CFRelease(e)
        time.sleep(0.05)


def synthetic_keys_arrive() -> bool:
    """The control for the press tests: a typed 'a' reaches a focused text field of ours."""
    edit = QLineEdit()
    edit.show()
    macos.activate()
    edit.activateWindow()
    edit.setFocus()
    settle(0.5)
    press(macos.KEYS['a'])
    settle(0.5)
    text = edit.text()
    edit.close()
    return text == 'a'


def test_system_shortcuts_are_read_and_the_default_combo_is_free():
    taken = macos.system_hotkeys()
    print('macOS shortcuts enabled on this machine:', ' '.join(sorted(macos.native(*k) for k in taken)))
    assert taken, 'CopySymbolicHotKeys gave nothing'
    assert (macos.KEYS['space'], macos.CMD) in taken  # ⌘Space, Spotlight
    assert macos.key_code(config.Ui().hotkey) not in taken, f'the default hotkey {config.Ui().hotkey} is a macOS shortcut'


def test_key_codes_follow_qt_on_macos():
    assert macos.key_code('Ctrl+Shift+Space') == (0x31, macos.CMD | macos.SHIFT)  # Qt's Ctrl is ⌘
    assert macos.key_code('Meta+Alt+K') == (0x28, macos.CONTROL | macos.OPTION)  # Qt's Meta is ⌃
    assert macos.key_code('Ctrl+F9') == (0x65, macos.CMD)
    assert macos.native(*macos.key_code('Ctrl+Shift+Space')) == '⇧⌘Space'
    assert macos.native(*macos.key_code('Meta+Alt+Shift+Ctrl+L')) == '⌃⌥⇧⌘L'
    with pytest.raises(HotkeyError):
        macos.key_code('Ctrl+Hyper')


def test_hotkey_refuses_a_macos_shortcut_a_bare_key_and_a_taken_combo():
    hk = macos.MacHotkey()
    try:
        with pytest.raises(HotkeyError, match='macOS shortcut'):
            hk.register('Ctrl+Space', lambda: None)  # ⌘Space
        with pytest.raises(HotkeyError, match='alone'):
            hk.register('F13', lambda: None)
        other = subprocess.Popen([sys.executable, '-c', f'''
from peresmeshnik.ui.macos import MacHotkey
import sys
h = MacHotkey(); h.register({COMBO!r}, lambda: None); print('held', flush=True); sys.stdin.read()'''],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert other.stdout.readline().strip() == 'held'
            with pytest.raises(HotkeyError, match='taken by another application'):
                hk.register(COMBO, lambda: None)
        finally:
            other.stdin.close()
            other.wait(10)
        hk.register(COMBO, lambda: None)  # free again once the other app is gone
        hk.unregister(COMBO)
    finally:
        hk.close()


def test_binding_says_why_and_keeps_the_old_combo():
    """The tray and the wizard's hotkey page show `why`; a refused new combo keeps the bound one."""
    from peresmeshnik.ui.hotkey import Binding
    done = []
    b = Binding(macos.MacHotkey(), lambda: None)
    b.done.connect(lambda combo, why: done.append((combo, why)))
    try:
        assert b.bind(COMBO) == '' and b.combo == COMBO
        why = b.bind('Ctrl+Alt+Space')  # ⌥⌘Space: Finder search
        assert 'macOS shortcut' in why and 'Live voice' in why and b.combo == COMBO
        assert done == [(COMBO, ''), (COMBO, why)]
    finally:
        b.close()


def test_hotkey_fires_once_per_press():
    if not synthetic_keys_arrive():
        pytest.skip('the runner does not deliver synthetic key events (CGEventPost); the press is in USER-CHECKS')
    hk, hits = macos.MacHotkey(), []
    try:
        hk.register(COMBO, lambda: hits.append(1))
        code, mods = macos.key_code(COMBO)
        press(code, mods)
        settle(0.5)
        press(code, mods)
        press(code, mods & ~macos.SHIFT)  # another combo: not ours
        settle(0.5)
        assert hits == [1, 1]
    finally:
        hk.close()


def test_menu_bar_icon_is_a_template_image():
    from peresmeshnik.ui.tray import tray_icon
    icon = tray_icon()
    assert icon.isMask() and not icon.isNull()
    assert {s.height() for s in icon.availableSizes()} >= {22, 44}  # the menu bar at 1x and 2x


def test_overlay_window_floats_over_everything_and_lets_clicks_through():
    from peresmeshnik.ui.overlay import SubtitleOverlay
    clicks = []

    class Under(QWidget):
        def mousePressEvent(self, e):  # noqa: N802
            clicks.append('under')
    o = SubtitleOverlay()
    under = Under()
    under.setGeometry(o.geometry().adjusted(-50, -50, 50, 50))
    under.show()
    o.side('them').show(Caption('click me', 'нажми меня', True, 0))
    settle(0.5)
    st = macos.window_state(int(o.winId()))
    print('overlay window:', st)
    assert st['level'] == macos.STATUS_LEVEL and st['behavior'] & macos.ALL_SPACES == macos.ALL_SPACES
    assert st['ignores_mouse'] and not st['hides'] and st['visible']
    g = o.geometry()
    mine = [w for w in macos.windows_of(os.getpid()) if w['layer'] == macos.STATUS_LEVEL and w['bounds'][2] == g.width()]
    assert mine, f'the window server shows no overlay: {macos.windows_of(os.getpid())}'
    o.mousePressEvent = lambda e: clicks.append('overlay')
    macos.activate()
    under.activateWindow()
    settle(0.3)
    click(g.center().x(), g.center().y())
    settle(0.5)
    o.close()
    under.close()
    if not clicks:
        pytest.skip('the runner does not deliver synthetic clicks; click-through is in USER-CHECKS')
    assert clicks == ['under']


def test_overlay_shows_over_another_apps_full_screen_window():
    """Another app goes full screen (macOS gives it a Space of its own): a plain window of ours stays behind on
    the desktop's Space, the overlay is on screen above the full-screen window."""
    from peresmeshnik.ui.overlay import SubtitleOverlay
    plain = QWidget()
    plain.setGeometry(100, 100, 300, 200)
    plain.show()
    o = SubtitleOverlay()
    o.side('them').show(Caption('over full screen', 'поверх полного экрана', True, 0))
    settle(1.0)
    mine = macos.windows_of(os.getpid())
    assert len(mine) >= 2, mine
    other = subprocess.Popen([sys.executable, '-c', '''
from PySide6.QtWidgets import QApplication, QWidget
from peresmeshnik.ui import macos
a = QApplication([])
w = QWidget()
macos.activate()
w.showFullScreen()
w.raise_()
print('full', flush=True)
a.exec()'''], stdout=subprocess.PIPE, text=True)
    try:
        assert other.stdout.readline().strip() == 'full'
        time.sleep(4)  # the Space animation
        settle(0.5)
        full = [w for w in macos.windows_of(other.pid) if w['layer'] == 0]
        now = macos.windows_of(os.getpid())
        print('full-screen app:', full, 'ours now:', now)
        scr = app.primaryScreen().geometry()
        assert any(w['bounds'][2:] == (scr.width(), scr.height()) for w in full), 'the other app is not full screen'
        assert not any(w['layer'] == 0 for w in now), 'still on the desktop Space: the test shows nothing'
        assert any(w['layer'] == macos.STATUS_LEVEL and w['bounds'][2] == o.width() for w in now), 'no overlay over it'
    finally:
        other.terminate()
        other.wait(10)
        o.close()
        plain.close()
        time.sleep(2)  # back to the desktop's Space: activating an app during that animation is refused
        settle(0.2)


def test_accessory_app_window_gets_the_focus(tmp_path, monkeypatch):
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    from peresmeshnik.ui import tray as tray_mod

    class Ctl:
        live, on = False, False
    macos.accessory()
    assert macos.activation_policy() == 1
    t = tray_mod.Tray(Ctl())
    t.open_settings()
    w = t.windows[-1]
    end = time.monotonic() + 3
    while not (macos.is_active() and macos.window_state(int(w.winId()))['key']) and time.monotonic() < end:
        settle(0.1)
    st, active = macos.window_state(int(w.winId())), macos.is_active()
    front = next((x for x in macos.windows_of(os.getpid()) if x['layer'] == 0), None)  # the list is front to back
    w.close()
    t.icon.hide()
    assert active and st['key'], f'app active {active}, settings window {st}'
    assert front and front['bounds'][2] == w.width(), front  # with its title bar: the width only


def _lines(p) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [q.put(line) for line in p.stdout] and None, daemon=True).start()
    return q


def _wait(q: queue.Queue, text: str, timeout: float) -> str:
    end = time.monotonic() + timeout
    seen = []
    while time.monotonic() < end:
        try:
            line = q.get(timeout=max(0.0, end - time.monotonic()))
        except queue.Empty:
            break
        seen.append(line)
        if text in line:
            return line
    raise AssertionError(f'no {text!r} in {timeout} s; got {seen}')


def test_app_menu_bar_icon_hotkey_and_sigterm(tmp_path):
    """The app as the user starts it: its menu-bar item is on screen, the hotkey reaches it, SIGTERM ends it cleanly."""
    settings = tmp_path / 'settings.toml'
    settings.write_text(f'[ui]\nhotkey = "{COMBO}"\n')
    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, '-m', 'peresmeshnik', 'app', '--no-wizard'], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, 'PERESMESHNIK_CONFIG': str(settings)})
    try:
        q = _lines(p)
        _wait(q, 'idle', 60)
        import psutil
        print(f'start from python: {time.perf_counter() - t0:.2f} s to idle, RSS {psutil.Process(p.pid).memory_info().rss / 1e6:.0f} MB')
        wins = macos.windows_of(p.pid)
        print('app windows:', wins)
        assert any(w['layer'] == macos.STATUS_LEVEL and w['bounds'][1] == 0 for w in wins), 'no menu-bar item on screen'
        if synthetic_keys_arrive():
            press(*macos.key_code(COMBO))
            _wait(q, 'mode: translation is off', 5)  # the hotkey reached the app (translation is off: a tray message)
        else:
            print('synthetic key events are not delivered here: the hotkey press is not checked')
        p.send_signal(signal.SIGTERM)
        assert p.wait(15) == 0
    finally:
        if p.poll() is None:
            p.kill()

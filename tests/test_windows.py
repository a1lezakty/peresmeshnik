"""Windows shell on the runner's own desktop: the RegisterHotKey hotkey (registration, refusals, a synthetic press),
the tray icon in the notification area and its light / dark glyph, the overlay's window (styles, click-through, no
focus taken, above full-screen windows) and the app closed by Ctrl+Break and by WM_CLOSE (taskkill). Synthetic input
goes through SendInput; when the runner drops it, the press tests skip and say so
(PERESMESHNIK_REQUIRE_SYNTHETIC=1: they fail)."""
import ctypes
import gc
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from ctypes import wintypes

import pytest

if sys.platform != 'win32':
    pytest.skip('Windows only', allow_module_level=True)
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget  # noqa: E402

from peresmeshnik import config  # noqa: E402
from peresmeshnik.engines import Caption  # noqa: E402
from peresmeshnik.ui import windows  # noqa: E402
from peresmeshnik.ui.hotkey import HotkeyError  # noqa: E402

app = QApplication.instance() or QApplication([])
user32, shell32 = windows.user32, ctypes.WinDLL('shell32')
COMBO = 'Ctrl+Alt+Shift+K'  # nothing on Windows
VK_SHIFT, VK_CONTROL, VK_MENU, VK_LWIN = 0x10, 0x11, 0x12, 0x5B
MOD_KEYS = {windows.MOD_CONTROL: VK_CONTROL, windows.MOD_ALT: VK_MENU, windows.MOD_SHIFT: VK_SHIFT, windows.MOD_WIN: VK_LWIN}


class _Key(ctypes.Structure):
    _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD), ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
                ('dwExtraInfo', ctypes.c_size_t)]


class _Mouse(ctypes.Structure):
    _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG), ('mouseData', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD), ('dwExtraInfo', ctypes.c_size_t)]


class _Input(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [('ki', _Key), ('mi', _Mouse), ('pad', ctypes.c_byte * 32)]
    _anonymous_ = ('u',)
    _fields_ = [('type', wintypes.DWORD), ('u', _U)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.FindWindowExW.restype, user32.FindWindowExW.argtypes = wintypes.HWND, [wintypes.HWND, wintypes.HWND, wintypes.LPCWSTR,
                                                                             wintypes.LPCWSTR]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM, wintypes.UINT,
                                       wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
_ENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [_ENUM, wintypes.LPARAM]


def settle(seconds=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def send(*events) -> None:
    arr = (_Input * len(events))(*events)
    assert user32.SendInput(len(events), arr, ctypes.sizeof(_Input)) == len(events), ctypes.get_last_error()


def key(vk: int, up: bool = False) -> _Input:
    i = _Input(type=1)  # INPUT_KEYBOARD
    i.ki = _Key(wVk=vk, dwFlags=2 if up else 0)  # KEYEVENTF_KEYUP
    return i


def press(mods: int, vk: int, downs: int = 1) -> None:
    """The modifiers down, the key down (`downs` times: a held key repeats) and up, the modifiers up."""
    held = [v for m, v in MOD_KEYS.items() if mods & m]
    send(*[key(v) for v in held], *[key(vk)] * downs, key(vk, True), *[key(v, True) for v in reversed(held)])
    time.sleep(0.05)


def click(x: int, y: int) -> None:
    """A left click at a physical screen point."""
    user32.SetCursorPos(x, y)
    send(*[_Input(type=0, mi=_Mouse(dwFlags=f)) for f in (2, 4)])  # INPUT_MOUSE: left down, left up
    time.sleep(0.05)


def front(hwnd: int) -> None:
    """Bring a window of ours to the front: a tapped Alt first lifts Windows' foreground lock for this process."""
    send(key(VK_MENU), key(VK_MENU, True))
    user32.SetForegroundWindow(hwnd)
    settle(0.3)


def no_synthetic(what: str) -> None:
    why = f'the runner does not deliver synthetic {what} (SendInput)'
    if os.environ.get('PERESMESHNIK_REQUIRE_SYNTHETIC') == '1':
        pytest.fail(why)
    pytest.skip(f'{why}; it is in USER-CHECKS')


def synthetic_keys_arrive() -> bool:
    """The control for the press tests: a typed 'a' reaches a focused text field of ours."""
    edit = QLineEdit()
    edit.show()
    settle(0.2)
    front(int(edit.winId()))
    edit.setFocus()
    settle(0.2)
    press(0, ord('A'))
    settle(0.5)
    text = edit.text()
    edit.close()
    return text == 'a'


def rect(hwnd) -> tuple[int, int, int, int]:
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def pid_of(hwnd) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def tree(pid: int) -> set[int]:
    """The process and its children: a venv's python.exe is a launcher that runs the real one as its child."""
    import psutil
    return {pid, *(c.pid for c in psutil.Process(pid).children(recursive=True))}


def toplevel() -> list[int]:
    """Every top-level window, front to back (the z-order)."""
    out: list[int] = []
    cb = _ENUM(lambda h, _: out.append(h) or True)
    user32.EnumWindows(cb, 0)
    return out


def describe(hwnd) -> str:
    cls, title = ctypes.create_unicode_buffer(128), ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(hwnd, cls, 128)
    user32.GetWindowTextW(hwnd, title, 128)
    return f'{cls.value!r} {title.value!r} visible={bool(user32.IsWindowVisible(hwnd))} ex=0x{windows.ex_style(hwnd):x}'


class _NotifyIconId(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('hWnd', wintypes.HWND), ('uID', wintypes.UINT), ('guid', ctypes.c_byte * 16)]


shell32.Shell_NotifyIconGetRect.argtypes = [ctypes.POINTER(_NotifyIconId), ctypes.POINTER(wintypes.RECT)]


def tray_icon_rect(pid: int):
    """Where Explorer shows a tray icon of that process (its notification area or the overflow), or None: asks the shell
    for each window of the process (Qt's tray window is a hidden top-level one) and icon id."""
    pids = tree(pid)
    hwnds = [h for h in toplevel() if pid_of(h) in pids]
    h = None
    while h := user32.FindWindowExW(wintypes.HWND(-3), h, None, None):  # message-only windows (HWND_MESSAGE)
        if pid_of(h) in pids:
            hwnds.append(h)
    for h in hwnds:
        for uid in range(4):
            nid, r = _NotifyIconId(cbSize=ctypes.sizeof(_NotifyIconId), hWnd=h, uID=uid), wintypes.RECT()
            if shell32.Shell_NotifyIconGetRect(ctypes.byref(nid), ctypes.byref(r)) == 0:
                return describe(h), uid, (r.left, r.top, r.right, r.bottom)
    return None


def test_key_codes():
    assert windows.key_code('Ctrl+Alt+Space') == (windows.MOD_CONTROL | windows.MOD_ALT, 0x20)
    assert windows.key_code('Meta+F9') == (windows.MOD_WIN, 0x78)  # Qt's Meta is the Windows key
    assert windows.key_code('Ctrl+Shift+K') == (windows.MOD_CONTROL | windows.MOD_SHIFT, ord('K'))
    assert windows.key_code('Ctrl+Alt+5') == (windows.MOD_CONTROL | windows.MOD_ALT, ord('5'))
    assert windows.key_code('Ctrl+!') == (windows.MOD_CONTROL | windows.MOD_SHIFT, ord('1'))  # US layout: Shift+1
    with pytest.raises(HotkeyError):
        windows.key_code('Ctrl+Hyper')


def test_default_combo_registers_and_what_windows_holds():
    hk = windows.WinHotkey()
    try:
        held = {}
        for combo in ('Ctrl+Alt+Space', 'Ctrl+Shift+Space', 'Meta+E', 'Meta+D', 'Meta+Alt+Space', 'Ctrl+Alt+P'):
            try:
                hk.register(combo, lambda: None)
                held[combo] = 'free'
                hk.unregister(combo)
            except HotkeyError as e:
                held[combo] = str(e)
        print('RegisterHotKey on this machine:', held)
        hk.register(config.Ui().hotkey, lambda: None)  # the default must be free on a clean Windows
    finally:
        hk.close()


def test_hotkey_refuses_windows_keys_a_bare_key_and_a_taken_combo():
    hk = windows.WinHotkey()
    try:
        for combo, why in (('Alt+Tab', 'switches windows'), ('Ctrl+Space', 'input method'), ('Meta+Space', 'input language'),
                           ('F9', 'alone'), ('K', 'typing'), ('Shift+K', 'typing'), ('Shift+Left', 'typing'),
                           ('Shift+Home', 'typing'), ('Shift+Space', 'input method')):
            with pytest.raises(HotkeyError, match=why):
                hk.register(combo, lambda: None)
        mods, vk = windows.key_code(COMBO)
        other = subprocess.Popen([sys.executable, '-c', f'''
import ctypes, sys
assert ctypes.windll.user32.RegisterHotKey(None, 1, {mods}, {vk})
print('held', flush=True)
sys.stdin.read()'''], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert other.stdout.readline().strip() == 'held'
            with pytest.raises(HotkeyError, match='taken by another application'):
                hk.register(COMBO, lambda: None)
        finally:
            other.stdin.close()
            other.wait(10)
        hk.register(COMBO, lambda: None)  # free again once the other app is gone
        hk.register('Shift+F9', lambda: None)  # Shift alone: with a function key it types nothing
    finally:
        hk.close()


def test_binding_says_why_and_keeps_the_old_combo():
    from peresmeshnik.ui.hotkey import Binding
    done = []
    b = Binding(windows.WinHotkey(), lambda: None)
    b.done.connect(lambda combo, why: done.append((combo, why)))
    try:
        assert b.bind(COMBO) == '' and b.combo == COMBO
        why = b.bind('Alt+Tab')
        assert 'switches windows' in why and 'Live voice' in why and b.combo == COMBO
        assert done == [(COMBO, ''), (COMBO, why)]
    finally:
        b.close()


def test_hotkey_fires_once_per_press():
    hk, hits = windows.WinHotkey(), []
    try:
        hk.register(COMBO, lambda: hits.append(1))
        hk.register(COMBO, lambda: hits.append(1))  # again (a settings save): replaces, does not add a second one
        mods, vk = windows.key_code(COMBO)
        press(mods, vk, downs=3)  # held: the key repeats, the hotkey does not (MOD_NOREPEAT)
        settle(0.5)
        press(mods, vk)
        press(mods & ~windows.MOD_SHIFT, vk)  # another combo: not ours
        settle(0.5)
        got = list(hits)
    finally:
        hk.close()
    if got != [1, 1] and not synthetic_keys_arrive():
        no_synthetic('key events')
    assert got == [1, 1]


def test_two_hotkey_objects_on_the_one_window():
    """RegisterHotKey ids are per window, and the window is the app's one: each object's combo calls its own callback."""
    a, b, hits = windows.WinHotkey(), windows.WinHotkey(), []
    try:
        a.register(COMBO, lambda: hits.append('a'))
        b.register('Ctrl+Alt+Shift+J', lambda: hits.append('b'))
        press(*windows.key_code(COMBO))
        settle(0.5)
        got = list(hits)
    finally:
        a.close()
        b.close()
    if got != ['a'] and not synthetic_keys_arrive():
        no_synthetic('key events')
    assert got == ['a']


def test_tray_icon_follows_the_taskbar_theme(tmp_path, monkeypatch):
    """Explorer has the icon; the glyph is white on a dark taskbar and black on a light one, switched when Windows
    says the theme changed (the registry value flipped and WM_SETTINGCHANGE broadcast, as Settings does)."""
    import winreg

    from peresmeshnik.ui import tray as tray_mod
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))

    class Ctl:
        live, on = False, False

    def lightness() -> float:
        img = t.icon.icon().pixmap(32, 32).toImage()
        px = [img.pixelColor(x, y) for x in range(img.width()) for y in range(img.height())]
        opaque = [c.lightnessF() for c in px if c.alphaF() > 0.5]
        return sum(opaque) / len(opaque)

    def set_light(on: int) -> None:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, windows.PERSONALIZE) as k:
            winreg.SetValueEx(k, 'SystemUsesLightTheme', 0, winreg.REG_DWORD, on)
        res, text = ctypes.c_size_t(), ctypes.create_unicode_buffer('ImmersiveColorSet')
        user32.SendMessageTimeoutW(wintypes.HWND(0xFFFF), windows.WM_SETTINGCHANGE, 0, ctypes.addressof(text), 2, 2000,
                                   ctypes.byref(res))  # HWND_BROADCAST, SMTO_ABORTIFHUNG
        settle(0.3)

    was = windows.light_taskbar()
    t = tray_mod.Tray(Ctl())
    try:
        settle(1.0)
        where = tray_icon_rect(os.getpid())
        print('tray icon:', where, '| taskbar light:', was)
        assert where, 'Explorer has no tray icon of ours'
        set_light(0)
        dark_bar = lightness()
        set_light(1)
        light_bar = lightness()
        print(f'glyph lightness: dark taskbar {dark_bar:.2f}, light taskbar {light_bar:.2f}')
        assert dark_bar > 0.8 and light_bar < 0.2
    finally:
        set_light(int(was))
        t.icon.hide()


def test_overlay_styles_no_focus_and_clicks_through():
    from peresmeshnik.ui.overlay import SubtitleOverlay
    clicks = []

    class Under(QWidget):
        def mousePressEvent(self, e):  # noqa: N802
            clicks.append('under')
    o = SubtitleOverlay()
    o.mousePressEvent = lambda e: clicks.append('overlay')
    under = Under()
    under.setGeometry(o.geometry().adjusted(-50, -50, 50, 50))
    under.show()
    settle(0.3)
    front(int(under.winId()))
    fg = user32.GetForegroundWindow()
    o.side('them').show(Caption('click me', 'нажми меня', True, 0))
    settle(0.5)
    hwnd = int(o.winId())
    style = windows.ex_style(hwnd)
    print('overlay:', describe(hwnd), '| foreground before', describe(fg) if fg else None, 'after', user32.GetForegroundWindow() == fg)
    try:
        assert style & windows.OVERLAY_STYLE == windows.OVERLAY_STYLE, f'ex style 0x{style:x}'
        assert user32.GetForegroundWindow() == fg, 'the overlay took the focus'
        assert toplevel().index(hwnd) < toplevel().index(int(under.winId()))
        paints = o.paints
        o.side('them').show(Caption('click me', 'нажми меня', True, 0))  # the same text again: no repaint
        settle(0.3)
        assert o.paints == paints
        left, top, right, bottom = rect(hwnd)
        click((left + right) // 2, (top + bottom) // 2)
        settle(0.5)
    finally:
        o.clear.emit()
        settle(0.1)
        o.close()
        under.close()
    if not clicks:
        no_synthetic('clicks')
    assert clicks == ['under']


FULL = '''
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget
a = QApplication([])
w = QWidget(None, Qt.WindowStaysOnTopHint if sys.argv[1] == 'topmost' else Qt.Widget)
w.setStyleSheet('background: #204080')
w.showFullScreen()
w.raise_()
w.activateWindow()
print(int(w.winId()), flush=True)
a.exec()'''


@pytest.mark.parametrize('kind', ['plain', 'topmost'])
def test_overlay_stays_above_a_full_screen_window(kind):
    """Another app's full-screen window, also always on top in the second case, comes to the front: the subtitles stay
    above it."""
    from peresmeshnik.ui.overlay import SubtitleOverlay
    o = SubtitleOverlay()
    o.side('them').show(Caption('over full screen', 'поверх полного экрана', True, 0))
    settle(0.5)
    other = subprocess.Popen([sys.executable, '-c', FULL, kind], stdout=subprocess.PIPE, text=True)
    try:
        full = int(other.stdout.readline())
        send(key(VK_MENU), key(VK_MENU, True))
        user32.SetForegroundWindow(full)  # as a click on it would
        settle(1.5)
        order, hwnd = toplevel(), int(o.winId())
        scr = rect(full)
        print(f'{kind} full screen {describe(full)} {scr}; overlay at {order.index(hwnd)}, it at {order.index(full)}; '
              f'foreground is it: {user32.GetForegroundWindow() == full}; re-raised {o.on_top.raises} times')
        assert order.index(hwnd) < order.index(full), 'the full-screen window covers the subtitles'
        if kind == 'topmost':  # it came to the front above ours: the hook put ours back
            assert o.on_top.raises > 1
    finally:
        other.terminate()
        other.wait(10)
        o.clear.emit()
        settle(0.1)
        o.close()
    gc.collect()  # overlays of earlier tests, gone
    assert windows._hook is None, f'the hook stays while no subtitles are shown ({len(windows._shown)} shown)'


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


def _start(tmp_path, wizard=False, **kw):
    settings = tmp_path / 'settings.toml'
    if not wizard:  # no settings file: a first run, the wizard opens
        settings.write_text(f'[ui]\nhotkey = "{COMBO}"\n')
    p = subprocess.Popen([sys.executable, '-m', 'peresmeshnik', 'app', '-v'], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, 'PERESMESHNIK_CONFIG': str(settings)}, **kw)
    q = _lines(p)
    _wait(q, 'idle', 60)
    return p, q


def test_app_tray_icon_hotkey_and_wm_close(tmp_path):
    """The app as the user starts it: its tray icon is in the notification area, the hotkey reaches it, and taskkill
    (WM_CLOSE to its windows, as installers do too) ends it cleanly."""
    p, q = _start(tmp_path)
    try:
        wins = [describe(h) for h in toplevel() if pid_of(h) in tree(p.pid)]
        print('app windows:', wins)
        where = tray_icon_rect(p.pid)
        print('app tray icon:', where)
        assert where, 'no tray icon of the app'
        mods, vk = windows.key_code(COMBO)
        press(mods, vk)
        try:
            _wait(q, 'mode: translation is off', 5)  # the hotkey reached the app (translation is off: a tray message)
            pressed = True
        except AssertionError:
            pressed = False
        real = {pid_of(h) for h in toplevel() if pid_of(h) in tree(p.pid)}  # the venv launcher has no windows: its child
        r = subprocess.run(['taskkill', *[a for pid in real for a in ('/PID', str(pid))]], capture_output=True, text=True)
        print('taskkill:', r.returncode, r.stdout.strip(), r.stderr.strip())
        assert r.returncode == 0 and p.wait(15) == 0
        if not pressed and not synthetic_keys_arrive():
            no_synthetic('key events')
        assert pressed, 'the hotkey did not reach the app'
    finally:
        if p.poll() is None:
            p.kill()


def test_app_quits_on_ctrl_break(tmp_path):
    p, _ = _start(tmp_path, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        os.kill(p.pid, signal.CTRL_BREAK_EVENT)
        assert p.wait(15) == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_app_quits_cleanly_when_windows_signs_out(tmp_path):
    """Sign-out or shutdown: WM_QUERYENDSESSION, then WM_ENDSESSION to the app's windows (here to its hidden one, as
    Windows sends them to every top-level window): Qt runs aboutToQuit (the call app gets its sound back) and quits."""
    p, _ = _start(tmp_path)
    try:
        hidden = [h for h in toplevel() if pid_of(h) in tree(p.pid) and describe(h).split("'")[3] == windows.HOST_TITLE]
        assert hidden, [describe(h) for h in toplevel() if pid_of(h) in tree(p.pid)]
        res = ctypes.c_size_t()
        logoff = 0x80000000  # ENDSESSION_LOGOFF
        assert user32.SendMessageTimeoutW(hidden[0], 0x0011, 0, logoff, 2, 5000, ctypes.byref(res))  # WM_QUERYENDSESSION
        assert res.value == 1, 'the app refused to end the session'
        user32.SendMessageTimeoutW(hidden[0], 0x0016, 1, logoff, 2, 5000, ctypes.byref(res))  # WM_ENDSESSION
        assert p.wait(15) == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_closing_the_wizard_by_its_x_keeps_the_app(tmp_path):
    """The user's close (X, Alt+F4: SC_CLOSE, then WM_CLOSE) closes that window only; a WM_CLOSE from outside quits."""
    p, _ = _start(tmp_path, wizard=True)
    try:
        wizard = [h for h in toplevel() if pid_of(h) in tree(p.pid) and 'Peresmeshnik setup' in describe(h)]
        assert wizard and user32.IsWindowVisible(wizard[0])
        print('wizard window:', wizard[0], describe(wizard[0]))
        res = ctypes.c_size_t()
        user32.SendMessageTimeoutW(wizard[0], 0x0112, 0xF060, 0, 2, 5000, ctypes.byref(res))  # WM_SYSCOMMAND, SC_CLOSE
        time.sleep(2)
        assert p.poll() is None, 'the app quit with its wizard'
        assert not user32.IsWindowVisible(wizard[0]), 'the wizard is still open'
        real = {pid_of(h) for h in toplevel() if pid_of(h) in tree(p.pid)}
        r = subprocess.run(['taskkill', *[a for pid in real for a in ('/PID', str(pid))]], capture_output=True, text=True)
        assert r.returncode == 0 and p.wait(15) == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_wm_close_to_a_hidden_helper_of_qt_quits(tmp_path):
    """taskkill with no window on screen (right after the start, the wizard not shown yet) posts WM_CLOSE to a hidden
    window, maybe one of Qt's helpers ('_q_titlebar' of the wizard's title bar metrics): the app quits (it stayed, with
    the wizard, in 10 of 40 runs from the Start menu shortcut, BENCH stage 5)."""
    p, _ = _start(tmp_path, wizard=True)
    try:
        helper = [h for h in toplevel() if pid_of(h) in tree(p.pid) and describe(h).startswith("'_q_titlebar'")]
        assert helper, [describe(h) for h in toplevel() if pid_of(h) in tree(p.pid)]
        assert user32.PostMessageW(helper[0], 0x0010, 0, 0)  # WM_CLOSE, posted as taskkill does
        assert p.wait(15) == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_wizard_pages_show_all_their_text(tmp_path, monkeypatch):
    """QWizard's Windows style sized the pages without their wrapped text: the last lines were cut off. Every page."""
    from PySide6.QtWidgets import QLabel

    from peresmeshnik.ui.wizard import SetupWizard
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))

    class Ctl:
        live, on = False, False

    class Tray:
        ctl, hotkey_why = Ctl(), ''
        reload_hotkey = staticmethod(lambda combo: '')
    monkeypatch.setattr('peresmeshnik.ui.wizard.has_key', lambda s: True)  # the default cloud direction asks for a key
    w = SetupWizard(Tray())
    try:
        w.show()
        seen = []
        for _ in w.pageIds():
            settle(0.5)
            page = w.currentPage()
            seen.append(type(page).__name__)
            cut = [(lab.text()[:40], lab.height(), lab.heightForWidth(lab.width())) for lab in page.findChildren(QLabel)
                   if lab.wordWrap() and lab.isVisible() and lab.height() < lab.heightForWidth(lab.width())]
            print(f'wizard {seen[-1]} {w.width()}x{w.height()}, page {page.width()}x{page.height()}, style {w.wizardStyle()}')
            assert not cut, (seen[-1], cut)
            w.next()  # as the Next button, without waiting for a download or a recording
        assert len(set(seen)) == len(w.pageIds()), f'stopped on a page: {seen}'
    finally:
        w.close()


def test_a_single_wm_close_to_the_hidden_window_quits(tmp_path):
    """The hidden window has no X: a WM_CLOSE to it is from outside even when the filter sees it once (sent, or posted
    while a native modal loop such as the file dialog runs)."""
    p, _ = _start(tmp_path)
    try:
        hidden = [h for h in toplevel() if pid_of(h) in tree(p.pid) and describe(h).split("'")[3] == windows.HOST_TITLE]
        assert hidden
        res = ctypes.c_size_t()
        user32.SendMessageTimeoutW(hidden[0], 0x0010, 0, 0, 2, 5000, ctypes.byref(res))  # WM_CLOSE, sent
        assert p.wait(15) == 0
    finally:
        if p.poll() is None:
            p.kill()

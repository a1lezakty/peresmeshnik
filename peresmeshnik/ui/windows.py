"""Windows: the global hotkey (`RegisterHotKey`), the overlay kept above other windows (full-screen ones too)
and the taskbar's light / dark theme for the tray icon. Through ctypes on user32, no pywin32.

Windows tells the app things through a hidden top-level window of ours (a QWindow, never shown): WM_HOTKEY
of our combos, WM_SETTINGCHANGE "ImmersiveColorSet" (the theme changed). A native event filter reads them in Qt's
own message loop: nothing polls. It also sees WM_CLOSE to any window of the app. One posted from outside (taskkill,
an installer) quits the app, whichever window it went to; one a window sends itself (its X button, Alt+F4) closes
that window only. With no window on screen (a start, before the wizard shows) taskkill picks a hidden one, maybe one of
Qt's helpers (the title bar metrics', the tray's): they have no X, so a WM_CLOSE to them is from outside too.
WM_ENDSESSION (sign-out, shutdown, an installer's Restart Manager) Qt turns into aboutToQuit.

Qt's combo text on Windows: 'Meta' is the Windows key.
"""
from __future__ import annotations

import ctypes
import logging
import weakref
import winreg
from collections.abc import Callable
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication, QObject, Signal
from PySide6.QtGui import QWindow

from peresmeshnik.ui.hotkey import HotkeyError, parse

log = logging.getLogger('peresmeshnik.ui')
user32 = ctypes.WinDLL('user32', use_last_error=True)
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.VkKeyScanW.restype, user32.VkKeyScanW.argtypes = ctypes.c_short, [wintypes.WCHAR]
user32.GetWindowLongPtrW.restype, user32.GetWindowLongPtrW.argtypes = ctypes.c_ssize_t, [wintypes.HWND, ctypes.c_int]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
WINEVENTPROC = ctypes.WINFUNCTYPE(None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND, wintypes.LONG, wintypes.LONG,
                                  wintypes.DWORD, wintypes.DWORD)
user32.SetWinEventHook.restype = wintypes.HANDLE
user32.SetWinEventHook.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.HMODULE, WINEVENTPROC, wintypes.DWORD,
                                   wintypes.DWORD, wintypes.DWORD]
user32.UnhookWinEvent.argtypes = [wintypes.HANDLE]

WM_CLOSE, WM_SETTINGCHANGE, WM_HOTKEY = 0x0010, 0x001A, 0x0312
GWLP_WNDPROC = -4
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 1, 2, 4, 8, 0x4000
MODS = {'alt': MOD_ALT, 'ctrl': MOD_CONTROL, 'shift': MOD_SHIFT, 'meta': MOD_WIN}
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
VK = {'space': 0x20, 'Return': 0x0D, 'Tab': 0x09, 'Escape': 0x1B, 'BackSpace': 0x08, 'Insert': 0x2D, 'Delete': 0x2E,
      'Home': 0x24, 'End': 0x23, 'Prior': 0x21, 'Next': 0x22, 'Left': 0x25, 'Up': 0x26, 'Right': 0x27, 'Down': 0x28,
      'Print': 0x2C, 'Pause': 0x13} | {f'F{i}': 0x6F + i for i in range(1, 25)}
TAB, ESC, SPACE, DEL, F4, PAUSE = 0x09, 0x1B, 0x20, 0x2E, 0x73, 0x13
F_KEYS = range(0x70, 0x88)  # F1..F24
# what Windows itself does with these: RegisterHotKey takes some of them and they never arrive, or it takes them from everyone
WINDOWS_KEYS = {
    (MOD_ALT, TAB): 'switches windows', (MOD_ALT | MOD_SHIFT, TAB): 'switches windows', (MOD_CONTROL | MOD_ALT, TAB): 'switches windows',
    (MOD_ALT, ESC): 'switches windows', (MOD_CONTROL, ESC): 'opens Start', (MOD_CONTROL | MOD_SHIFT, ESC): 'opens Task Manager',
    (MOD_CONTROL | MOD_ALT, DEL): 'opens the security screen', (MOD_ALT, F4): 'closes the window',
    (MOD_ALT, SPACE): 'opens the window menu', (MOD_WIN, ord('L')): 'locks the computer',
    (MOD_WIN, SPACE): 'switches the input language', (MOD_WIN | MOD_SHIFT, SPACE): 'switches the input language',
    (MOD_WIN | MOD_CONTROL, SPACE): 'switches the input language',
    (MOD_CONTROL, SPACE): 'turns a Chinese, Japanese or Korean input method on and off',
    (MOD_SHIFT, SPACE): 'switches an input method between full and half width',
}


def key_code(combo: str) -> tuple[int, int]:
    """'Ctrl+Alt+Space' -> (RegisterHotKey modifiers, virtual key)."""
    mods, sym = parse(combo)
    if sym in VK:
        vk = VK[sym]
    elif len(sym) == 1 and sym.isascii() and sym.isalnum():
        vk = ord(sym.upper())  # VK_A..VK_Z, VK_0..VK_9: the same in every layout
    elif len(sym) == 1 and (scan := user32.VkKeyScanW(sym)) != -1:
        vk = scan & 0xFF  # punctuation: where the current layout has it, with the Shift / Ctrl / Alt it needs there
        return sum(MODS[m] for m in mods) | sum(m for bit, m in ((1, MOD_SHIFT), (2, MOD_CONTROL), (4, MOD_ALT)) if scan >> 8 & bit), vk
    else:
        raise HotkeyError(f'cannot use the key {sym!r} of {combo!r} for the hotkey on Windows')
    return sum(MODS[m] for m in mods), vk


HOST_TITLE = 'Peresmeshnik F900FAE6'  # the hidden window, found by the installer (Peresmeshnik.iss); not a folder's name


class _Filter(QAbstractNativeEventFilter):
    def __init__(self, host: Host):
        super().__init__()
        self.host = host
        self.last_close = None  # the window of the WM_CLOSE seen last, when it was the last message

    def nativeEventFilter(self, kind, message):  # noqa: N802 (Qt name)
        try:
            msg = wintypes.MSG.from_address(int(message))
            # A posted WM_CLOSE comes here twice in a row: from Qt's dispatcher (PeekMessage), then from the window
            # procedure (DispatchMessage); a sent one, from the window's own X / Alt+F4, once (Qt 6 qeventdispatcher_win,
            # qwindowscontext windowsProc). Any other message in between: a new count.
            twice, self.last_close = msg.message == WM_CLOSE and msg.hWnd == self.last_close, None
            if msg.message == WM_CLOSE:
                # the hidden window has no X: any WM_CLOSE to it is from outside (once only when it came inside a native
                # modal loop, e.g. the file dialog, or was sent); neither has a window without Qt's window procedure, and
                # the filter sees one posted to it once: from the dispatcher only
                if twice or msg.hWnd == self.host.hwnd or not self.host.is_qt(msg.hWnd):
                    log.info('WM_CLOSE from outside: quitting')
                    self.host.closing.emit()
                    return True, 0
                self.last_close = msg.hWnd
            elif msg.hWnd == self.host.hwnd:
                return self.host.handle(msg.message, msg.wParam, msg.lParam), 0
        except Exception:  # an exception must not cross into Qt's message loop
            log.exception('native event')
        return False, 0


class Host(QObject):
    """The app's hidden window and the filter that reads what Windows sends it. One per process: `host()`."""
    closing = Signal()  # WM_CLOSE from outside: taskkill, an installer
    theme = Signal()  # the taskbar's light / dark mode changed

    def __init__(self):
        super().__init__()
        self.window = QWindow()
        self.window.setTitle(HOST_TITLE)
        self.window.create()  # a native window, never shown
        self.hwnd = int(self.window.winId())
        self.proc = user32.GetWindowLongPtrW(self.hwnd, GWLP_WNDPROC)  # Qt's window procedure, of every QWindow
        self.hotkeys: dict[int, Callable[[], None]] = {}
        self.last_id = 0  # RegisterHotKey ids: unique per window, so per process (any number of WinHotkey)
        self._filter = _Filter(self)
        QCoreApplication.instance().installNativeEventFilter(self._filter)
        QCoreApplication.instance().aboutToQuit.connect(unhook)  # quitting with subtitles on screen

    def is_qt(self, hwnd: int) -> bool:
        """A QWindow (the wizard, the overlay): a message posted to it reaches the filter twice."""
        return user32.GetWindowLongPtrW(hwnd, GWLP_WNDPROC) == self.proc

    def handle(self, message: int, wparam: int, lparam: int) -> bool:
        """True: the message is ours, Qt does not see it. WM_HOTKEY comes once per press: posted, it reaches the
        filter from Qt's dispatcher and (inside a modal loop) from the window procedure, never both."""
        if message == WM_HOTKEY:
            if cb := self.hotkeys.get(wparam):
                try:
                    cb()
                except Exception:
                    log.exception('hotkey')
            return True
        if message == WM_SETTINGCHANGE and lparam and ctypes.wstring_at(lparam) == 'ImmersiveColorSet':
            self.theme.emit()
        return False


_host: Host | None = None


def host() -> Host:
    global _host
    if _host is None:
        _host = Host()
    return _host


class WinHotkey:
    """RegisterHotKey on the app's hidden window; callbacks run on the GUI thread (Qt's message loop)."""

    def __init__(self):
        self.host = host()
        self._ids: dict[str, int] = {}

    def register(self, combo: str, callback: Callable[[], None]) -> None:
        mods, vk = key_code(combo)
        if what := WINDOWS_KEYS.get((mods, vk)):
            raise HotkeyError(f'{combo} {what} in Windows')
        if mods == MOD_SHIFT and vk not in F_KEYS and vk != PAUSE or not mods and (vk == SPACE or 0x30 <= vk <= 0x5A or vk >= 0xBA):
            # Shift with a letter, an arrow, Home, Tab...: typing and selecting text everywhere
            raise HotkeyError(f'{combo} would take the key from typing in every app: add Ctrl, Alt or Win')
        if not mods:
            raise HotkeyError(f'{combo} alone would take the key from every app: add Ctrl, Alt or Win')
        self.unregister(combo)  # the same combo again (a settings save): its old registration goes
        self.host.last_id += 1
        i = self.host.last_id
        if not user32.RegisterHotKey(self.host.hwnd, i, mods | MOD_NOREPEAT, vk):  # no repeats while held
            err = ctypes.get_last_error()
            if err == ERROR_HOTKEY_ALREADY_REGISTERED:
                raise HotkeyError(f'{combo} is already taken by another application or by Windows')
            raise HotkeyError(f'cannot register {combo} (Windows error {err})')
        self._ids[combo] = i
        self.host.hotkeys[i] = callback

    def unregister(self, combo: str) -> None:
        if (i := self._ids.pop(combo, None)) is not None:
            user32.UnregisterHotKey(self.host.hwnd, i)
            self.host.hotkeys.pop(i, None)

    def close(self) -> None:
        for combo in list(self._ids):
            self.unregister(combo)


# overlay ---------------------------------------------------------------------------------------------------
GWL_EXSTYLE = -20
WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x8, 0x20, 0x80, 0x80000, 0x08000000
OVERLAY_STYLE = WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_NOACTIVATE
HWND_TOPMOST = wintypes.HWND(-1)
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE, SWP_NOOWNERZORDER = 0x1, 0x2, 0x10, 0x200
EVENT_SYSTEM_FOREGROUND, WINEVENT_SKIPOWNPROCESS = 3, 2  # WINEVENT_OUTOFCONTEXT = 0


def ex_style(hwnd: int) -> int:
    return user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)


class OnTop:
    """While the subtitles are on screen: back on top each time another app's window comes to the front. Without it
    a window that is itself always on top (a slide show, a player's full screen) covers them once it is activated.
    An out-of-context WinEvent hook: Windows calls it from our message loop, on that event only; no hook while hidden."""

    def __init__(self):
        self.hwnd, self.raises = 0, 0

    def start(self, hwnd: int) -> None:
        global _hook
        host()  # unhooks when the app quits
        self.hwnd = hwnd
        _shown.add(self)
        if _hook is None:
            _hook = user32.SetWinEventHook(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, None, _PROC, 0, 0,
                                           WINEVENT_SKIPOWNPROCESS)
        self.raise_()

    def stop(self) -> None:
        _shown.discard(self)
        if not _shown:
            unhook()

    def raise_(self) -> None:
        user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER)
        self.raises += 1


_shown: weakref.WeakSet[OnTop] = weakref.WeakSet()
_hook = None
# one callback for the process, alive as long as the module: Windows calls it as long as the hook lives (a callback
# owned by a collected overlay was freed memory to Windows: an access violation at the next foreground change)
_PROC = WINEVENTPROC(lambda *_: [t.raise_() for t in list(_shown)] and None)


def unhook() -> None:
    global _hook
    if _hook:
        user32.UnhookWinEvent(_hook)
        _hook = None


# tray icon theme --------------------------------------------------------------------------------------------
PERSONALIZE = r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize'


def light_taskbar() -> bool:
    """Settings → Personalization → Colors → "Choose your default Windows mode" (the taskbar's, not the apps')."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, PERSONALIZE) as k:
            return winreg.QueryValueEx(k, 'SystemUsesLightTheme')[0] == 1
    except OSError:  # Windows 10 before 1903: always dark
        return False

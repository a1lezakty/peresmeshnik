"""Global hotkey on Linux: an X11 key grab, or the XDG GlobalShortcuts portal on Wayland.

Both wait on a socket (Qt socket notifier / a thread blocked in recv) and never poll. Callbacks
run on the Qt GUI thread. `make()` picks one, or returns (None, reason) when neither works: then
the tray item is the only switch, and the reason is shown to the user.

Combos are Qt key-sequence text: 'Ctrl+Alt+Space', 'Meta+F9', 'Ctrl+Shift+L'.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QSocketNotifier, Signal

log = logging.getLogger('peresmeshnik.ui')
SHIFT, LOCK, CTRL, ALT, NUMLOCK, SUPER = 1, 2, 4, 8, 16, 64  # X11 modifier masks (Mod1 = Alt, Mod2 = NumLock, Mod4 = Super)
MODS = {'ctrl': CTRL, 'alt': ALT, 'shift': SHIFT, 'meta': SUPER}
KEYSYM = {'esc': 'Escape', 'ins': 'Insert', 'del': 'Delete', 'pgup': 'Prior', 'pgdown': 'Next', 'backspace': 'BackSpace',
          'return': 'Return', 'enter': 'Return', 'space': 'space', 'tab': 'Tab', 'print': 'Print', 'pause': 'Pause'}
PORTAL_MODS = {'ctrl': 'CTRL', 'alt': 'ALT', 'shift': 'SHIFT', 'meta': 'LOGO'}


class HotkeyError(RuntimeError):
    pass


def parse(combo: str) -> tuple[list[str], str]:
    """'Ctrl+Alt+Space' -> (['ctrl', 'alt'], 'space'): modifiers and the X keysym name of the key."""
    *mods, key = [p.strip() for p in combo.replace(' ', '').split('+')] if combo else ['']
    if not key or any(m.lower() not in MODS for m in mods):
        raise HotkeyError(f'cannot use {combo!r} as a hotkey: modifiers are Ctrl, Alt, Shift, Meta, then one key')
    sym = KEYSYM.get(key.lower(), key.lower() if len(key) == 1 else key)  # letters are lowercase keysyms; F1, Home ... as is
    return [m.lower() for m in mods], sym


class _KeyEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int), ('display', ctypes.c_void_p),
                ('window', ctypes.c_ulong), ('root', ctypes.c_ulong), ('subwindow', ctypes.c_ulong), ('time', ctypes.c_ulong),
                ('x', ctypes.c_int), ('y', ctypes.c_int), ('x_root', ctypes.c_int), ('y_root', ctypes.c_int),
                ('state', ctypes.c_uint), ('keycode', ctypes.c_uint), ('same_screen', ctypes.c_int)]


class _Event(ctypes.Union):
    _fields_ = [('type', ctypes.c_int), ('key', _KeyEvent), ('pad', ctypes.c_long * 24)]


class _ErrorEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('display', ctypes.c_void_p), ('resourceid', ctypes.c_ulong), ('serial', ctypes.c_ulong),
                ('error_code', ctypes.c_ubyte), ('request_code', ctypes.c_ubyte), ('minor_code', ctypes.c_ubyte)]


_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_ErrorEvent))
_x_errors: list[int] = []  # Xlib has one error handler per process: one list for every connection
_x_handler = _ERROR_HANDLER(lambda d, e: _x_errors.append(e.contents.error_code) or 0)  # the default handler exits
KEY_PRESS, KEY_RELEASE, BAD_ACCESS = 2, 3, 10


class X11Hotkey(QObject):
    """XGrabKey on the root window over our own Xlib connection, read when its socket has data."""

    def __init__(self, display: str | None = None):
        super().__init__()
        name = ctypes.util.find_library('X11')
        if not name:
            raise HotkeyError('libX11 not found')
        x = self.x = ctypes.CDLL(name)
        x.XOpenDisplay.restype, x.XOpenDisplay.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
        x.XDefaultRootWindow.restype, x.XDefaultRootWindow.argtypes = ctypes.c_ulong, [ctypes.c_void_p]
        x.XStringToKeysym.restype, x.XStringToKeysym.argtypes = ctypes.c_ulong, [ctypes.c_char_p]
        x.XKeysymToKeycode.restype, x.XKeysymToKeycode.argtypes = ctypes.c_ubyte, [ctypes.c_void_p, ctypes.c_ulong]
        x.XGrabKey.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        x.XUngrabKey.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_ulong]
        for f in ('XPending', 'XConnectionNumber', 'XFlush', 'XCloseDisplay'):
            getattr(x, f).argtypes = [ctypes.c_void_p]
        x.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Event)]
        x.XkbSetDetectableAutoRepeat.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.dpy = x.XOpenDisplay(display.encode() if display else None)
        if not self.dpy:
            raise HotkeyError(f'cannot open X display {display or os.environ.get("DISPLAY")!r}')
        self.root = x.XDefaultRootWindow(self.dpy)
        x.XSetErrorHandler(_x_handler)
        x.XkbSetDetectableAutoRepeat(self.dpy, 1, None)  # a held key repeats KeyPress only: one toggle per press
        self._grabs: dict[str, tuple[int, int, Callable[[], None]]] = {}
        self._down: set[int] = set()
        self._ev = _Event()
        self._notifier = QSocketNotifier(x.XConnectionNumber(self.dpy), QSocketNotifier.Read, self)
        self._notifier.activated.connect(self._read)

    def register(self, combo: str, callback: Callable[[], None]) -> None:
        mods, sym = parse(combo)
        keycode = self.x.XKeysymToKeycode(self.dpy, self.x.XStringToKeysym(sym.encode()))
        if not keycode:
            raise HotkeyError(f'no key {sym!r} on this keyboard')
        mask = sum(MODS[m] for m in mods)
        x_errors = _x_errors
        x_errors.clear()
        for extra in (0, LOCK, NUMLOCK, LOCK | NUMLOCK):  # the grab must not depend on Caps Lock / Num Lock
            self.x.XGrabKey(self.dpy, keycode, mask | extra, self.root, 1, 1, 1)  # GrabModeAsync = 1
        self.x.XSync(self.dpy, 0)
        if x_errors:
            self._ungrab(keycode, mask)
            raise HotkeyError(f'{combo} is already taken by another application' if BAD_ACCESS in x_errors
                              else f'X error {x_errors} grabbing {combo}')
        self._grabs[combo] = keycode, mask, callback
        self._read()  # XSync may have queued events that the socket will not announce again

    @staticmethod
    def matches(state: int, mask: int) -> bool:
        """The modifiers of a key event against a grab. Only Shift/Ctrl/Alt/Super count: the state also
        carries Lock, NumLock, mouse buttons and the XKB layout group (bits 13-14: the second layout is on)."""
        return state & (SHIFT | CTRL | ALT | SUPER) == mask

    def _ungrab(self, keycode: int, mask: int) -> None:
        for extra in (0, LOCK, NUMLOCK, LOCK | NUMLOCK):
            self.x.XUngrabKey(self.dpy, keycode, mask | extra, self.root)
        self.x.XSync(self.dpy, 0)

    def unregister(self, combo: str) -> None:
        if (g := self._grabs.pop(combo, None)) is not None:
            self._ungrab(g[0], g[1])

    def _read(self, *_) -> None:
        while self.x.XPending(self.dpy):
            self.x.XNextEvent(self.dpy, ctypes.byref(self._ev))
            k = self._ev.key
            if self._ev.type == KEY_RELEASE:
                self._down.discard(k.keycode)
            elif self._ev.type == KEY_PRESS and k.keycode not in self._down:
                self._down.add(k.keycode)
                for keycode, mask, cb in list(self._grabs.values()):
                    if k.keycode == keycode and self.matches(k.state, mask):
                        cb()

    def close(self) -> None:
        for combo in list(self._grabs):
            self.unregister(combo)
        self._notifier.setEnabled(False)
        self.x.XCloseDisplay(self.dpy)


PORTAL = 'org.freedesktop.portal.Desktop'
PORTAL_PATH = '/org/freedesktop/portal/desktop'
SHORTCUTS = 'org.freedesktop.portal.GlobalShortcuts'


class PortalHotkey(QObject):
    """XDG GlobalShortcuts portal (Wayland). The desktop shows its own dialog to confirm or change the
    key; `trigger` holds what it reports. A session binds its shortcuts once (the spec), so a new combo
    gets a new session; the old one closes only after the new one is bound. `register` waits for the
    user's answer (up to `timeout`): call it off the GUI thread (`Binding` does). jeepney's router
    thread blocks on the D-Bus socket; ours blocks on the queue of Activated signals."""
    _fired = Signal(str)
    sid = 'toggle-live'  # ponytail: one shortcut id; give each combo its own id if the app ever needs two

    def __init__(self, bus: str = 'SESSION', timeout: float = 60.0):
        super().__init__()
        import queue

        from jeepney import DBusAddress, MatchRule, Properties, message_bus
        from jeepney.io.threading import DBusRouter, open_dbus_connection
        self._MatchRule, self._bus, self._Address, self.timeout = MatchRule, message_bus, DBusAddress, timeout
        try:
            self._conn = open_dbus_connection(bus)
        except Exception as e:  # no session bus
            raise HotkeyError(f'no D-Bus session bus ({e})') from e
        self.router = DBusRouter(self._conn)
        self.portal = DBusAddress(PORTAL_PATH, PORTAL, SHORTCUTS)
        try:
            reply = self.router.send_and_get_reply(Properties(self.portal).get('version'), timeout=5)
            if reply.header.message_type.name == 'error':
                raise HotkeyError(reply.body[0] if reply.body else reply.header.fields)
        except Exception as e:
            self.close()
            raise HotkeyError(f'no GlobalShortcuts portal on this desktop ({str(e)[:120]})') from e
        self._sender = self._conn.unique_name[1:].replace('.', '_')
        self._tokens = 0
        self._lock = threading.Lock()  # one bind at a time
        self.session = ''
        self._callbacks: dict[str, Callable[[], None]] = {}
        self.trigger: dict[str, str] = {}
        rule = MatchRule(type='signal', interface=SHORTCUTS, member='Activated', path=PORTAL_PATH)
        self.router.send_and_get_reply(message_bus.AddMatch(rule), timeout=5)
        self._activated = self.router.filter(rule, queue=queue.Queue())
        self._fired.connect(lambda sid: (cb := self._callbacks.get(sid)) and cb())
        self._thread = threading.Thread(target=self._listen, name='hotkey-portal', daemon=True)
        self._thread.start()

    def _request(self, method: str, sig: str, args: Callable[[str], tuple]) -> dict:
        """Call a portal method that answers through a Request object's Response signal."""
        from jeepney import new_method_call
        self._tokens += 1
        tok = f'peresmeshnik{os.getpid()}_{self._tokens}'
        path = f'{PORTAL_PATH}/request/{self._sender}/{tok}'
        rule = self._MatchRule(type='signal', interface='org.freedesktop.portal.Request', member='Response', path=path)
        self.router.send_and_get_reply(self._bus.AddMatch(rule), timeout=5)  # before the call: the answer can be immediate
        with self.router.filter(rule) as q:
            reply = self.router.send_and_get_reply(new_method_call(self.portal, method, sig, args(tok)), timeout=10)
            if reply.header.message_type.name == 'error':
                raise HotkeyError(f'{method}: {reply.body[0] if reply.body else "D-Bus error"}')
            try:
                code, results = q.get(timeout=self.timeout).body
            except Exception as e:
                raise HotkeyError(f'{method}: no answer from the portal in {self.timeout:.0f} s') from e
        if code != 0:
            raise HotkeyError(f'{method}: the desktop refused the shortcut (response {code})')
        return results

    def _close_session(self, session: str) -> None:
        from jeepney import new_method_call
        try:
            self.router.send_and_get_reply(new_method_call(self._Address(session, PORTAL, 'org.freedesktop.portal.Session'), 'Close'),
                                           timeout=5)
        except Exception as e:  # the session ends with our connection anyway
            log.warning('portal: closing %s: %s', session, e)

    def register(self, combo: str, callback: Callable[[], None]) -> None:
        mods, sym = parse(combo)
        trigger = '+'.join([PORTAL_MODS[m] for m in mods] + [sym])
        with self._lock:
            session = self._request('CreateSession', 'a{sv}', lambda tok: (
                {'handle_token': ('s', tok), 'session_handle_token': ('s', tok)},))['session_handle'][1]
            try:
                res = self._request('BindShortcuts', 'oa(sa{sv})sa{sv}', lambda tok: (
                    session, [(self.sid, {'description': ('s', 'Peresmeshnik: translation / live voice'),
                                          'preferred_trigger': ('s', trigger)})], '', {'handle_token': ('s', tok)}))
            except Exception:
                self._close_session(session)  # the old session and its callback stay
                raise
            old, self.session = self.session, session
            self._callbacks[self.sid] = callback
            for got_id, props in res.get('shortcuts', ('', []))[1]:
                self.trigger[got_id] = props.get('trigger_description', ('s', ''))[1]
            if old:
                self._close_session(old)

    def unregister(self, combo: str) -> None:
        pass  # a new session replaces the old one in register(); the last one ends with the connection

    def _listen(self) -> None:
        while (msg := self._activated.queue.get()) is not None:
            if msg.body[0] == self.session:
                self._fired.emit(msg.body[1])  # queued to the GUI thread

    def close(self) -> None:
        if hasattr(self, '_activated'):
            self._activated.queue.put(None)
        self.router.close()
        self._conn.close()


class Binding(QObject):
    """The app's one hotkey on a backend (or none). `bind()` keeps the old combo when a new one fails and
    works again after a failed first grab. X11 binds at once; the portal waits for the desktop's dialog,
    so it binds on a thread and reports through `done(combo, why)` on the GUI thread."""
    done = Signal(str, str)  # combo, '' or why it is not bound

    def __init__(self, backend, callback: Callable[[], None], why: str = ''):
        super().__init__()
        self.backend, self.callback, self.combo, self.why = backend, callback, '', why

    def bind(self, combo: str) -> str:
        """'' when bound (or, for the portal, asked), else why not."""
        if self.backend is None:
            return ''  # nothing to bind on this desktop (`why` says so); the combo is kept for where it works
        if isinstance(self.backend, PortalHotkey):
            threading.Thread(target=self._bind_portal, args=(combo,), name='hotkey-bind', daemon=True).start()
            return ''
        old = self.combo
        if old:
            self.backend.unregister(old)
        try:
            self.backend.register(combo, self.callback)
        except HotkeyError as e:
            try:
                if old:
                    self.backend.register(old, self.callback)
            except HotkeyError:
                self.combo = ''
            self.why = f'{e}. Use the tray menu item "Live voice"'
            self.done.emit(self.combo, self.why)
            return self.why
        self.combo, self.why = combo, ''
        self.done.emit(combo, '')
        return ''

    def _bind_portal(self, combo: str) -> None:
        try:
            self.backend.register(combo, self.callback)
            self.combo, self.why = combo, ''
        except Exception as e:  # refused, no answer: the previous session and combo stay
            self.why = f'{e}. Use the tray menu item "Live voice"'
        self.done.emit(self.combo, self.why)

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()


def make() -> tuple[object | None, str]:
    """(hotkey, '') or (None, why there is no global hotkey)."""
    if not sys.platform.startswith('linux'):
        return None, 'the global hotkey on this OS comes in a later version: use the tray menu'
    wayland = os.environ.get('XDG_SESSION_TYPE') == 'wayland' or (os.environ.get('WAYLAND_DISPLAY') and not os.environ.get('DISPLAY'))
    try:
        if wayland:  # an X11 grab under XWayland only sees keys while an X11 window has focus
            return PortalHotkey(), ''
        if os.environ.get('DISPLAY'):
            return X11Hotkey(), ''
        return None, 'no X11 display and no Wayland session: use the tray menu'
    except HotkeyError as e:
        return None, f'no global hotkey: {e}. Use the tray menu item "Live voice"'

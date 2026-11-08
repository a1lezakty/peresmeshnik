"""macOS: a menu-bar app without a Dock icon, the overlay above other apps (full-screen ones too) and the
global hotkey. Through ctypes on the Objective-C runtime and Carbon, no PyObjC.

The hotkey is Carbon's `RegisterEventHotKey`: no Accessibility permission, the key event goes to the app
through its own run loop (nothing polls). It is registered exclusively, so a combo another app holds is
refused, and a combo of an enabled macOS shortcut (Spotlight, input sources, ...) is refused before that.
Qt's macOS convention holds for the combo text: 'Ctrl' is ⌘ Command, 'Meta' is ⌃ Control.
"""
from __future__ import annotations

import ctypes
import logging
from collections.abc import Callable
from ctypes import POINTER, byref, c_bool, c_double, c_int32, c_int64, c_long, c_uint32, c_ulong, c_void_p

from peresmeshnik.ui.hotkey import HotkeyError, parse

log = logging.getLogger('peresmeshnik.ui')
objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
objc.objc_getClass.restype, objc.objc_getClass.argtypes = c_void_p, [ctypes.c_char_p]
objc.sel_registerName.restype, objc.sel_registerName.argtypes = c_void_p, [ctypes.c_char_p]
objc.object_getClassName.restype, objc.object_getClassName.argtypes = ctypes.c_char_p, [c_void_p]
_SEND = ctypes.cast(objc.objc_msgSend, c_void_p).value
_protos: dict = {}


def send(obj, sel: str, *args, restype=c_void_p, argtypes=()):
    """[obj sel:args...]. arm64 needs objc_msgSend called with the method's exact prototype."""
    f = _protos.get((restype, argtypes))
    if f is None:
        f = _protos[restype, argtypes] = ctypes.CFUNCTYPE(restype, c_void_p, c_void_p, *argtypes)(_SEND)
    return f(obj, objc.sel_registerName(sel.encode()), *args)


def nsapp():
    return send(objc.objc_getClass(b'NSApplication'), 'sharedApplication')


def accessory() -> None:
    """No Dock icon, no app menu (what LSUIElement does for the .app; Qt makes a plain executable a regular app)."""
    send(nsapp(), 'setActivationPolicy:', 1, restype=c_bool, argtypes=(c_long,))  # NSApplicationActivationPolicyAccessory


def activate() -> None:
    """Bring the app forward: an accessory app's windows get no focus otherwise."""
    send(nsapp(), 'activateIgnoringOtherApps:', True, restype=None, argtypes=(c_bool,))


def front(view_id: int) -> None:
    """A window opened from the menu: in front of the other apps' windows, with the focus. macOS 14+ may refuse
    to activate the app (cooperative activation); the window still comes to the front then."""
    activate()
    w = _window(view_id)
    send(w, 'makeKeyAndOrderFront:', None, restype=None, argtypes=(c_void_p,))
    send(w, 'orderFrontRegardless', restype=None)


def is_active() -> bool:
    return send(nsapp(), 'isActive', restype=c_bool)


def activation_policy() -> int:
    return send(nsapp(), 'activationPolicy', restype=c_long)


STATUS_LEVEL = 25  # NSStatusWindowLevel: above other apps' windows and the Dock (20), below menus
# NSWindowCollectionBehavior: CanJoinAllSpaces | Stationary | IgnoresCycle | FullScreenAuxiliary
ALL_SPACES = 1 << 0 | 1 << 4 | 1 << 6 | 1 << 8


def _window(view_id: int):
    return send(c_void_p(view_id), 'window')  # QWidget.winId() on macOS is the NSView


def overlay(view_id: int) -> None:
    """The subtitles' NSWindow: on every Space and over full-screen apps, above the Dock, never takes clicks,
    stays when the app is not active."""
    w = _window(view_id)
    send(w, 'setLevel:', STATUS_LEVEL, restype=None, argtypes=(c_long,))
    send(w, 'setCollectionBehavior:', ALL_SPACES, restype=None, argtypes=(c_ulong,))
    send(w, 'setIgnoresMouseEvents:', True, restype=None, argtypes=(c_bool,))
    send(w, 'setHidesOnDeactivate:', False, restype=None, argtypes=(c_bool,))


def window_state(view_id: int) -> dict:
    """What AppKit has for a widget's window (tests)."""
    w = _window(view_id)
    return {'level': send(w, 'level', restype=c_long), 'behavior': send(w, 'collectionBehavior', restype=c_ulong),
            'ignores_mouse': send(w, 'ignoresMouseEvents', restype=c_bool), 'hides': send(w, 'hidesOnDeactivate', restype=c_bool),
            'visible': send(w, 'isVisible', restype=c_bool), 'key': send(w, 'isKeyWindow', restype=c_bool),
            'class': objc.object_getClassName(w).decode(), 'style': send(w, 'styleMask', restype=c_ulong)}


# CoreFoundation, for the system's shortcuts and (tests) the window list -----------------------------------
cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
cf.CFArrayGetCount.restype, cf.CFArrayGetCount.argtypes = c_long, [c_void_p]
cf.CFArrayGetValueAtIndex.restype, cf.CFArrayGetValueAtIndex.argtypes = c_void_p, [c_void_p, c_long]
cf.CFDictionaryGetValue.restype, cf.CFDictionaryGetValue.argtypes = c_void_p, [c_void_p, c_void_p]
cf.CFNumberGetValue.restype, cf.CFNumberGetValue.argtypes = c_bool, [c_void_p, c_long, c_void_p]
cf.CFBooleanGetValue.restype, cf.CFBooleanGetValue.argtypes = c_bool, [c_void_p]
cf.CFRelease.argtypes = [c_void_p]
cf.CFStringCreateWithCString.restype, cf.CFStringCreateWithCString.argtypes = c_void_p, [c_void_p, ctypes.c_char_p, c_uint32]


def cfstr(s: str):
    """CFSTR("..."): kept for the process, like the compiler's constant strings."""
    return cf.CFStringCreateWithCString(None, s.encode(), 0x08000100)  # kCFStringEncodingUTF8


def cf_number(ref) -> int:
    v = c_int64()
    cf.CFNumberGetValue(ref, 4, byref(v))  # kCFNumberSInt64Type
    return v.value


def cf_items(array) -> list:
    return [cf.CFArrayGetValueAtIndex(array, i) for i in range(cf.CFArrayGetCount(array))]


# Carbon hotkey --------------------------------------------------------------------------------------------
carbon = ctypes.CDLL('/System/Library/Frameworks/Carbon.framework/Carbon')


class _HotKeyID(ctypes.Structure):
    _fields_ = [('signature', c_uint32), ('id', c_uint32)]


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [('eventClass', c_uint32), ('eventKind', c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(c_int32, c_void_p, c_void_p, c_void_p)
carbon.GetApplicationEventTarget.restype = c_void_p
carbon.InstallEventHandler.argtypes = [c_void_p, _HANDLER, c_ulong, POINTER(_EventTypeSpec), c_void_p, POINTER(c_void_p)]
carbon.RemoveEventHandler.argtypes = [c_void_p]
carbon.RegisterEventHotKey.argtypes = [c_uint32, c_uint32, _HotKeyID, c_void_p, c_uint32, POINTER(c_void_p)]
carbon.UnregisterEventHotKey.argtypes = [c_void_p]
carbon.GetEventParameter.argtypes = [c_void_p, c_uint32, c_uint32, c_void_p, c_ulong, c_void_p, c_void_p]
carbon.CopySymbolicHotKeys.argtypes = [POINTER(c_void_p)]


def fourcc(s: str) -> int:
    return int.from_bytes(s.encode(), 'big')


CMD, SHIFT, OPTION, CONTROL, FN = 0x100, 0x200, 0x800, 0x1000, 0x20000  # Carbon modifier bits
MODS = {'ctrl': CMD, 'meta': CONTROL, 'alt': OPTION, 'shift': SHIFT}  # Qt on macOS: Ctrl = ⌘, Meta = ⌃
SYMBOLS = ((FN, 'fn '), (CONTROL, '⌃'), (OPTION, '⌥'), (SHIFT, '⇧'), (CMD, '⌘'))  # Apple's order
EXCLUSIVE, EXISTS = 1, -9878  # kEventHotKeyExclusive, eventHotKeyExistsErr
# virtual key codes by position on the ANSI keyboard (Carbon kVK_*): the same physical key in any layout
# ponytail: letters by US position; an AZERTY or Dvorak user gets the key at that place, UCKeyTranslate if it matters
ANSI = "asdfhgzxcv§bqweryt123465=97-80]ou[ip\nlj'k;\\,/nm."
KEYS = {c: i for i, c in enumerate(ANSI) if c != '\n'} | {'`': 0x32}
KEYS |= {'Return': 0x24, 'Tab': 0x30, 'space': 0x31, 'BackSpace': 0x33, 'Escape': 0x35, 'Insert': 0x72, 'Help': 0x72,
         'Home': 0x73, 'Prior': 0x74, 'Delete': 0x75, 'End': 0x77, 'Next': 0x79,
         'Left': 0x7B, 'Right': 0x7C, 'Down': 0x7D, 'Up': 0x7E}
KEYS |= {f'F{i + 1}': c for i, c in enumerate((0x7A, 0x78, 0x63, 0x76, 0x60, 0x61, 0x62, 0x64, 0x65, 0x6D, 0x67, 0x6F,
                                               0x69, 0x6B, 0x71, 0x6A, 0x40, 0x4F, 0x50, 0x5A))}


def key_code(combo: str) -> tuple[int, int]:
    """'Ctrl+Shift+Space' -> (key code, Carbon modifiers)."""
    mods, sym = parse(combo)
    if sym not in KEYS:
        raise HotkeyError(f'cannot use the key {sym!r} of {combo!r} for the hotkey on macOS')
    return KEYS[sym], sum(MODS[m] for m in mods)


def native(code: int, mods: int) -> str:
    """⇧⌘Space, as macOS writes it."""
    name = next((k for k, c in KEYS.items() if c == code and k != 'Help'), str(code))
    return ''.join(s for bit, s in SYMBOLS if mods & bit) + {'space': 'Space', 'Prior': 'PageUp', 'Next': 'PageDown'}.get(
        name, name.upper() if len(name) == 1 else name)


SYMBOLIC_KEYS = [cfstr(n) for n in ('kHISymbolicHotKeyCode', 'kHISymbolicHotKeyModifiers', 'kHISymbolicHotKeyEnabled')]
# (CFSTR macros in the headers, not exported symbols)


def system_hotkeys() -> set[tuple[int, int]]:
    """(key code, Carbon modifiers) of every enabled macOS shortcut (System Settings → Keyboard → Keyboard Shortcuts)."""
    arr = c_void_p()
    if carbon.CopySymbolicHotKeys(byref(arr)) != 0 or not arr:
        return set()
    out = set()
    try:
        for d in cf_items(arr):
            code, mods, on = (cf.CFDictionaryGetValue(d, k) for k in SYMBOLIC_KEYS)
            if code and mods and on and cf.CFBooleanGetValue(on):
                out.add((cf_number(code), cf_number(mods) & (CMD | SHIFT | OPTION | CONTROL | FN)))  # fn+Q: Quick Note ...
    finally:
        cf.CFRelease(arr)
    return out


class MacHotkey:
    """Carbon hotkeys on the app's event target; the callback runs on the main thread (inside Qt's run loop)."""
    SIGNATURE = fourcc('PRSM')

    def __init__(self):
        self._grabs: dict[str, tuple[int, c_void_p, Callable[[], None]]] = {}  # combo -> id, EventHotKeyRef, callback
        self._next = 0
        self._handler = _HANDLER(self._on_event)  # kept: Carbon holds only the pointer
        self._ref = c_void_p()
        spec = _EventTypeSpec(fourcc('keyb'), 5)  # kEventClassKeyboard, kEventHotKeyPressed (no repeats while held)
        if err := carbon.InstallEventHandler(carbon.GetApplicationEventTarget(), self._handler, 1, byref(spec), None,
                                             byref(self._ref)):
            raise HotkeyError(f'cannot listen for hotkeys (Carbon error {err})')

    def register(self, combo: str, callback: Callable[[], None]) -> None:
        code, mods = key_code(combo)
        if not mods:
            raise HotkeyError(f'{native(code, mods)} alone would take the key from every app: add ⌘, ⌥, ⌃ or ⇧')
        try:
            taken = system_hotkeys()
        except Exception as e:  # unreadable: register anyway, as before the check
            log.warning('cannot read the macOS shortcuts: %s', e)
            taken = set()
        if (code, mods) in taken or (code, mods | FN) in taken:  # ⌃↑, ⌃← (Mission Control) come with the fn bit
            raise HotkeyError(f'{native(code, mods)} is a macOS shortcut (System Settings → Keyboard → Keyboard Shortcuts)')
        self._next += 1
        ref = c_void_p()
        err = carbon.RegisterEventHotKey(code, mods, _HotKeyID(self.SIGNATURE, self._next), carbon.GetApplicationEventTarget(),
                                         EXCLUSIVE, byref(ref))
        if err == EXISTS:
            raise HotkeyError(f'{native(code, mods)} is already taken by another application')
        if err:
            raise HotkeyError(f'cannot register {native(code, mods)} (Carbon error {err})')
        self._grabs[combo] = self._next, ref, callback

    def unregister(self, combo: str) -> None:
        if (g := self._grabs.pop(combo, None)) is not None:
            carbon.UnregisterEventHotKey(g[1])

    def _on_event(self, _call, event, _data) -> int:
        hk = _HotKeyID()
        if carbon.GetEventParameter(event, fourcc('----'), fourcc('hkid'), None, ctypes.sizeof(hk), None, byref(hk)):
            return -9874  # eventNotHandledErr
        for i, _, cb in list(self._grabs.values()):
            if hk.signature == self.SIGNATURE and hk.id == i:
                try:
                    cb()
                except Exception:  # an exception must not cross into Carbon
                    log.exception('hotkey')
                return 0
        return -9874

    def close(self) -> None:
        for combo in list(self._grabs):
            self.unregister(combo)
        if self._ref:
            carbon.RemoveEventHandler(self._ref)
            self._ref = c_void_p()


# window list (tests): what the window server shows for a process, no permission needed for these fields ------
cg = ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
cg.CGWindowListCopyWindowInfo.restype, cg.CGWindowListCopyWindowInfo.argtypes = c_void_p, [c_uint32, c_uint32]
cg.CGRectMakeWithDictionaryRepresentation.restype = c_bool
cg.CGRectMakeWithDictionaryRepresentation.argtypes = [c_void_p, POINTER(c_double * 4)]


def windows_of(pid: int) -> list[dict]:
    """On-screen windows of a process: layer (window level) and bounds (x, y, w, h); menu-bar items are layer 25."""
    arr = cg.CGWindowListCopyWindowInfo(1, 0)  # kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    keys = {n: c_void_p.in_dll(cg, n).value for n in ('kCGWindowOwnerPID', 'kCGWindowLayer', 'kCGWindowBounds')}
    out = []
    try:
        for d in cf_items(arr):
            if cf_number(cf.CFDictionaryGetValue(d, keys['kCGWindowOwnerPID'])) != pid:
                continue
            r = (c_double * 4)()
            cg.CGRectMakeWithDictionaryRepresentation(cf.CFDictionaryGetValue(d, keys['kCGWindowBounds']), byref(r))
            out.append({'layer': cf_number(cf.CFDictionaryGetValue(d, keys['kCGWindowLayer'])), 'bounds': tuple(r)})
    finally:
        cf.CFRelease(arr)
    return out

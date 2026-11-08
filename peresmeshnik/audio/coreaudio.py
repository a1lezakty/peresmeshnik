"""macOS Core Audio through ctypes: the devices as the system sees them now, and notifications when one goes.

PortAudio (sounddevice) reads the device list once, at its initialisation: a device plugged in later (AirPods
back) is not in it, and its default devices are the ones of that moment. So the names and the defaults come from
here, and `refresh_portaudio()` (only while no PortAudio stream is open) makes PortAudio read the list again.
`Watch` calls back once when a device goes (unplugged, Bluetooth gone, the sound server restarted): Core Audio's
property listeners, on its own notification thread; nothing polls.
"""
from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Callable

log = logging.getLogger('peresmeshnik.coreaudio')


def _cc(s: str) -> int:
    return int.from_bytes(s.encode(), 'big')


SYSTEM = 1  # kAudioObjectSystemObject
DEVICES, DEFAULT_OUTPUT, DEFAULT_INPUT, RUN_LOOP = _cc('dev#'), _cc('dOut'), _cc('dIn '), _cc('rnlp')
NAME, ALIVE, STREAMS, TRANSPORT, RATE = _cc('lnam'), _cc('livn'), _cc('stm#'), _cc('tran'), _cc('nsrt')
GLOBAL, INPUT, OUTPUT = _cc('glob'), _cc('inpt'), _cc('outp')
VIRTUAL, AGGREGATE = _cc('virt'), _cc('grup')  # kAudioDeviceTransportType…


class Address(ctypes.Structure):
    _fields_ = [('selector', ctypes.c_uint32), ('scope', ctypes.c_uint32), ('element', ctypes.c_uint32)]


LISTENER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(Address), ctypes.c_void_p)
_ca = _cf = None


def _lib():
    global _ca, _cf
    if _ca is None:
        ca = ctypes.CDLL('/System/Library/Frameworks/CoreAudio.framework/CoreAudio')
        cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        u32, ptr = ctypes.c_uint32, ctypes.c_void_p
        for name, args in (('AudioObjectGetPropertyDataSize', [u32, ctypes.POINTER(Address), u32, ptr, ctypes.POINTER(u32)]),
                           ('AudioObjectGetPropertyData', [u32, ctypes.POINTER(Address), u32, ptr, ctypes.POINTER(u32), ptr]),
                           ('AudioObjectSetPropertyData', [u32, ctypes.POINTER(Address), u32, ptr, u32, ptr]),
                           ('AudioObjectAddPropertyListener', [u32, ctypes.POINTER(Address), LISTENER, ptr]),
                           ('AudioObjectRemovePropertyListener', [u32, ctypes.POINTER(Address), LISTENER, ptr])):
            f = getattr(ca, name)
            f.argtypes, f.restype = args, ctypes.c_int32
        cf.CFStringGetCString.argtypes, cf.CFStringGetCString.restype = [ptr, ctypes.c_char_p, ctypes.c_long, u32], ctypes.c_bool
        cf.CFRelease.argtypes, cf.CFRelease.restype = [ptr], None
        # notifications on Core Audio's own thread, not on a run loop of the main thread (the CLI has none)
        none = ctypes.c_void_p(None)
        ca.AudioObjectSetPropertyData(SYSTEM, Address(RUN_LOOP, GLOBAL, 0), 0, None, ctypes.sizeof(none), ctypes.byref(none))
        _ca, _cf = ca, cf
    return _ca


def _get(obj: int, selector: int, ctype, scope: int = GLOBAL):
    v, size = ctype(), ctypes.c_uint32(ctypes.sizeof(ctype))
    err = _lib().AudioObjectGetPropertyData(obj, Address(selector, scope, 0), 0, None, ctypes.byref(size), ctypes.byref(v))
    return v.value if err == 0 else None


def _ids(obj: int, selector: int, scope: int = GLOBAL) -> list[int]:
    size = ctypes.c_uint32()
    a = Address(selector, scope, 0)
    if _lib().AudioObjectGetPropertyDataSize(obj, a, 0, None, ctypes.byref(size)) or not size.value:
        return []
    buf = (ctypes.c_uint32 * (size.value // 4))()
    if _lib().AudioObjectGetPropertyData(obj, a, 0, None, ctypes.byref(size), buf):
        return []
    return list(buf[:size.value // 4])


def _name(dev: int) -> str:
    ref = _get(dev, NAME, ctypes.c_void_p)
    if not ref:
        return ''
    buf = ctypes.create_string_buffer(1024)
    ok = _cf.CFStringGetCString(ref, buf, len(buf), 0x08000100)  # kCFStringEncodingUTF8
    _cf.CFRelease(ref)
    return buf.value.decode() if ok else ''


def devices() -> list[dict]:
    """[{'id', 'name', 'inputs', 'outputs' (stream counts), 'virtual' (BlackHole and the like, aggregates)}]"""
    out = []
    for d in _ids(SYSTEM, DEVICES):
        out.append({'id': d, 'name': _name(d), 'inputs': len(_ids(d, STREAMS, INPUT)), 'outputs': len(_ids(d, STREAMS, OUTPUT)),
                    'virtual': _get(d, TRANSPORT, ctypes.c_uint32) in (VIRTUAL, AGGREGATE)})
    return out


def default(output: bool = True) -> str:
    """Name of the system's default output (or input) now; '' when there is none."""
    d = _get(SYSTEM, DEFAULT_OUTPUT if output else DEFAULT_INPUT, ctypes.c_uint32)
    return _name(d) if d else ''


def device_id(name: str, output: bool = True) -> int | None:
    return next((d['id'] for d in devices() if d['name'] == name and d['outputs' if output else 'inputs']), None)


# Every PortAudio stream of the app on macOS opens with stream_args(): the device's own rate and one buffer size.
# PortAudio sets the device's IO buffer size (kAudioDevicePropertyBufferFrameSize, per process) from each stream's
# latency and block at every open; a second stream of ours on the same device (AirPods: the microphone and the
# headphones) with another size makes the device restart its IO, the first stream's render fails (-10863) and PortAudio
# stops it (CI, PortAudio 19.7 pa_mac_core.c). The same block and a latency below the device's own give every stream
# a buffer of exactly BLOCK frames, so nothing changes.
BLOCK = 1024  # ~21 ms at 48 kHz: room for a Python callback waiting for the GIL (5 ms switch interval)


def stream_args(name: str, output: bool = True) -> dict:
    """samplerate / blocksize / latency for a sounddevice stream on the device `name` (see BLOCK)."""
    return {'samplerate': rate(name, output), 'blocksize': BLOCK, 'latency': 0.0}


def rate(name: str, output: bool = True) -> int:
    """The device's own sample rate now. Streams open at it: PortAudio would otherwise change the device's rate (every
    other app on it hears that) or convert, and two streams of one device at two rates break each other (CI)."""
    dev = device_id(name, output)
    r = _get(dev, RATE, ctypes.c_double) if dev is not None else None
    if not r:
        raise RuntimeError(f'no {"output" if output else "input"} device {name!r}')
    return int(r)


def refresh_portaudio() -> None:
    """Make PortAudio read the device list again. It closes every PortAudio stream of the process: call it only
    while none is open (the app: at translation on, before the streams open)."""
    import sounddevice as sd
    sd._terminate()
    sd._initialize()


class Watch:
    """`on_gone(why)` once (on a thread of its own) when the device `dev` goes: it leaves the system's device list or
    stops being alive. close() removes the listeners; after it no call comes."""

    def __init__(self, dev: int, name: str, on_gone: Callable[[str], None]):
        self.dev, self.name, self.on_gone = dev, name, on_gone
        self._done = False
        self._lock = threading.Lock()
        self._cb = LISTENER(self._changed)  # kept: Core Audio holds only the C pointer
        self._addrs = [(SYSTEM, Address(DEVICES, GLOBAL, 0)), (dev, Address(ALIVE, GLOBAL, 0))]
        for obj, a in self._addrs:
            if err := _lib().AudioObjectAddPropertyListener(obj, a, self._cb, None):
                self.close()
                raise RuntimeError(f'cannot watch {name}: Core Audio error {err}')
        self._changed(0, 0, None, None)  # gone before the listeners were there

    def _changed(self, obj, n, addrs, data) -> int:
        try:
            if not self._done and (self.dev not in _ids(SYSTEM, DEVICES) or _get(self.dev, ALIVE, ctypes.c_uint32) == 0):
                with self._lock:
                    if self._done:
                        return 0
                    self._done = True
                threading.Thread(target=self.on_gone, args=(f'{self.name} went away',), name='device-gone', daemon=True).start()
        except Exception:  # never into Core Audio's thread
            log.exception('device watch')
        return 0

    def close(self) -> None:
        with self._lock:
            self._done = True
        for obj, a in self._addrs:
            _lib().AudioObjectRemovePropertyListener(obj, a, self._cb, None)


class Arrival:
    """`on_arrive()` once (on a thread of its own) when an output (or input) device named `name` is in the system's
    device list: the saved headphones are back. close() removes the listener."""

    def __init__(self, name: str, on_arrive: Callable[[], None], output: bool = True):
        self.name, self.on_arrive, self.output = name, on_arrive, output
        self._done = False
        self._lock = threading.Lock()
        self._cb = LISTENER(self._changed)
        self._addr = Address(DEVICES, GLOBAL, 0)
        if err := _lib().AudioObjectAddPropertyListener(SYSTEM, self._addr, self._cb, None):
            raise RuntimeError(f'cannot watch for {name}: Core Audio error {err}')

    def _changed(self, obj, n, addrs, data) -> int:
        try:
            if not self._done and device_id(self.name, self.output) is not None:
                with self._lock:
                    if self._done:
                        return 0
                    self._done = True
                threading.Thread(target=self.on_arrive, name='device-back', daemon=True).start()
        except Exception:
            log.exception('device arrival')
        return 0

    def close(self) -> None:
        with self._lock:
            self._done = True
        _lib().AudioObjectRemovePropertyListener(SYSTEM, self._addr, self._cb, None)

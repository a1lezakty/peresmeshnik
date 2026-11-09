"""macOS Core Audio through ctypes: the devices as the system sees them now, and notifications when one goes.

PortAudio (sounddevice) reads the device list once, at its initialisation: a device plugged in later (AirPods
back) is not in it, and its default devices are the ones of that moment. So the names and the defaults come from
here, and `refresh_portaudio()` makes PortAudio read the list again, with every registered stream closed around it.
`Watch` calls back once when a device goes (unplugged, Bluetooth gone, the sound server restarted): Core Audio's
property listeners, on its own notification thread; nothing polls. `Arrival` says when a device is back, `Running`
when a process starts or stops playing into a device.
"""
from __future__ import annotations

import ctypes
import itertools
import logging
import threading
import time
import weakref
from collections.abc import Callable

log = logging.getLogger('peresmeshnik.coreaudio')


def _cc(s: str) -> int:
    return int.from_bytes(s.encode(), 'big')


SYSTEM = 1  # kAudioObjectSystemObject
DEVICES, DEFAULT_OUTPUT, DEFAULT_INPUT, RUN_LOOP = _cc('dev#'), _cc('dOut'), _cc('dIn '), _cc('rnlp')
NAME, ALIVE, STREAMS, TRANSPORT, RATE = _cc('lnam'), _cc('livn'), _cc('stm#'), _cc('tran'), _cc('nsrt')
RUNNING_SOMEWHERE = _cc('gone')  # kAudioDevicePropertyDeviceIsRunningSomewhere
GLOBAL, INPUT, OUTPUT = _cc('glob'), _cc('inpt'), _cc('outp')
VIRTUAL, AGGREGATE = _cc('virt'), _cc('grup')  # kAudioDeviceTransportType…


class Address(ctypes.Structure):
    _fields_ = [('selector', ctypes.c_uint32), ('scope', ctypes.c_uint32), ('element', ctypes.c_uint32)]


LISTENER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(Address), ctypes.c_void_p)
_listeners: dict[int, _Listener] = {}  # registered with Core Audio, by their client data: alive until close()
_keys = itertools.count(1)


@LISTENER
def _dispatch(obj, n, addrs, key) -> int:
    """The one C callback of every listener, never freed: Core Audio holds only its pointer, and may call it while a
    listener is being removed (a freed callback was a SIGSEGV / SIGBUS on its 'HALC_ShellObject_Listener Queue')."""
    listener = _listeners.get(key or 0)
    return listener._changed(obj, n, addrs, key) if listener else 0


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


# Every object that owns PortAudio streams on macOS registers here (hold / release) and has suspend() / resume(): a
# refresh closes every PortAudio stream of the process, so all of them are closed and opened again around it.
_holders: weakref.WeakSet = weakref.WeakSet()
# one refresh at a time, and no stream opens while one runs (opening takes it too; re-entrant: resume() opens)
portaudio_lock = threading.RLock()


def hold(x) -> None:
    _holders.add(x)


def release(x) -> None:
    _holders.discard(x)


def refresh_portaudio() -> dict:
    """Make PortAudio read the device list again (a device plugged in since its start). It frees every PortAudio
    stream of the process: every registered holder is suspended first, all at once (PortAudio takes 0.1-0.25 s to
    stop each stream: pa_mac_core polls the unit with Pa_Sleep(100)), and resumed after, each on its own (a holder
    that fails to resume is reported, the others go on). Returns the phases in ms and the failures."""
    import sounddevice as sd
    with portaudio_lock:
        held = list(_holders)
        t = [time.perf_counter()]
        failed: list[str] = []

        def each(fn, x):
            try:
                fn()
            except Exception as e:
                log.exception('refresh: %s', x)
                failed.append(f'{type(x).__name__}: {e}')
        stops = [threading.Thread(target=each, args=(x.suspend, x), name='suspend') for x in held]
        for th in stops:
            th.start()
        for th in stops:
            th.join()
        t.append(time.perf_counter())
        sd._terminate()
        sd._initialize()
        t.append(time.perf_counter())
        for x in held:
            each(x.resume, x)
        t.append(time.perf_counter())
    ms = {k: round(1000 * (b - a)) for k, a, b in zip(('suspend', 'refresh', 'resume'), t, t[1:])}
    return ms | {'streams': len(held), 'failed': failed}


class _Listener:
    """Core Audio property listeners on (object, selector, scope); `_check()` runs on Core Audio's notification thread
    and returns what to report (or None). One-shot listeners fire once, on a thread of their own."""
    once = True

    def __init__(self, addrs: list[tuple[int, int, int]], what: str):
        self._done = False
        self._lock = threading.Lock()
        self._key = next(_keys)
        _listeners[self._key] = self
        self._addrs = [(obj, Address(sel, scope, 0)) for obj, sel, scope in addrs]
        for obj, a in self._addrs:
            if err := _lib().AudioObjectAddPropertyListener(obj, a, _dispatch, self._key):
                self.close()
                raise RuntimeError(f'cannot watch {what}: Core Audio error {err}')

    def _changed(self, obj, n, addrs, data) -> int:
        try:
            if not self._done and (why := self._check(obj)) is not None:
                with self._lock:
                    if self._done:
                        return 0
                    self._done = self.once
                threading.Thread(target=self._fire, args=(why,), name='coreaudio-event', daemon=True).start()
        except Exception:  # never into Core Audio's thread
            log.exception('coreaudio listener')
        return 0

    def _check(self, obj):
        raise NotImplementedError

    def _fire(self, why) -> None:
        raise NotImplementedError

    def close(self) -> None:
        with self._lock:
            self._done = True
        for obj, a in self._addrs:
            _lib().AudioObjectRemovePropertyListener(obj, a, _dispatch, self._key)
        _listeners.pop(self._key, None)


class Watch(_Listener):
    """`on_gone(why)` once (on a thread of its own) when the device `dev` goes: it leaves the system's device list or
    stops being alive. close() removes the listeners; after it no call comes."""

    def __init__(self, dev: int, name: str, on_gone: Callable[[str], None]):
        self.dev, self.name, self.on_gone = dev, name, on_gone
        super().__init__([(SYSTEM, DEVICES, GLOBAL), (dev, ALIVE, GLOBAL)], name)
        self._changed(0, 0, None, None)  # gone before the listeners were there

    def _check(self, obj):
        gone = self.dev not in _ids(SYSTEM, DEVICES) or _get(self.dev, ALIVE, ctypes.c_uint32) == 0
        return f'{self.name} went away' if gone else None

    def _fire(self, why) -> None:
        self.on_gone(why)


class Arrival(_Listener):
    """`on_arrive()` once (on a thread of its own) when an output (or input) device named `name` is there: it comes into
    the system's device list, or, when Core Audio has it already (but PortAudio did not open it: a Bluetooth profile
    still settling), when it changes (its streams, rate, aliveness) or the list does. close() removes the listeners."""

    def __init__(self, name: str, on_arrive: Callable[[], None], output: bool = True):
        self.name, self.on_arrive, self.output = name, on_arrive, output
        self.present = device_id(name, output)
        scope = OUTPUT if output else INPUT
        extra = [(self.present, STREAMS, scope), (self.present, RATE, GLOBAL), (self.present, ALIVE, GLOBAL)] if self.present else []
        super().__init__([(SYSTEM, DEVICES, GLOBAL), *extra], name)

    def _check(self, obj):
        return True if device_id(self.name, self.output) is not None else None

    def _fire(self, why) -> None:
        self.on_arrive()


class Running(_Listener):
    """`on_change(running)` (on a thread of its own) each time "some process runs IO on the device `name`" changes
    (kAudioDevicePropertyDeviceIsRunningSomewhere): a call app started or stopped playing into BlackHole 16ch."""
    once = False

    def __init__(self, name: str, on_change: Callable[[bool], None]):
        self.name, self.on_change = name, on_change
        if (dev := device_id(name, output=False)) is None:
            raise RuntimeError(f'no input device {name!r}')
        self.dev = dev
        super().__init__([(dev, RUNNING_SOMEWHERE, GLOBAL)], name)

    def running(self) -> bool:
        return bool(_get(self.dev, RUNNING_SOMEWHERE, ctypes.c_uint32))

    def _check(self, obj):
        return self.running()

    def _fire(self, running) -> None:
        self.on_change(running)

"""Windows "me -> them" devices through the WASAPI helper (native/windows/audio): VB-Cable as the virtual mic, the
microphone by endpoint id that follows a device going away and coming back, the loop "microphone -> CABLE Input" only
while a call app records CABLE Output, and what each costs in CPU.

On the CI runner (windows-latest, VB-Cable installed by the shared CI step, microphone access for desktop apps allowed
on that VM, the helper built) CABLE Output stands in for the microphone where the test needs one input only. The runner
has no second capture device (VB-CABLE A+B is sold, the free Hi-Fi Cable's 2014 installer does not install silently):
where CABLE Output is the call app's side, my voice comes from a file (the app's me side) or there is no microphone
(the loop's call / idle). PERESMESHNIK_REQUIRE_WINDOWS=1 (CI): a missing VB-Cable or helper fails instead of skipping.
Elsewhere only the checks that need no Windows run.
"""
import os
import subprocess
import sys
import threading
import time
import types

import numpy as np
import psutil
import pytest

from peresmeshnik.audio import windows

WIN = sys.platform == 'win32'
REQUIRE = os.environ.get('PERESMESHNIK_REQUIRE_WINDOWS') == '1'
SR_IN, SR_OUT = 24_000, 16_000  # TTS rate in, call-app rate out
SECOND = 'Hi-Fi Cable'  # the second cable's endpoints: "Hi-Fi Cable Input / Output (VB-Audio Hi-Fi Cable)"


def _ready() -> str:
    if not WIN:
        return 'Windows only'
    try:
        windows.helper()
        windows.cable()
    except Exception as e:  # not built, or no VB-Cable: on CI that is a failure, on a dev machine a skip
        if REQUIRE:
            raise
        return str(e)
    return ''


win = pytest.mark.skipif(bool(_ready()), reason=_ready() if WIN else 'Windows only')


def second() -> tuple[dict, dict] | None:
    """(render, capture) endpoints of a second cable (a dev machine with VB-Audio Hi-Fi Cable), else None."""
    devs = windows.devices()
    play = next((d for d in devs if d['outputs'] and SECOND in d['name']), None)
    rec = next((d for d in devs if d['inputs'] and SECOND in d['name']), None)
    return (play, rec) if play and rec else None


def say(*a) -> None:
    """print() that survives the runner's cp1252 console (the app's texts have arrows and ticks)."""
    enc = sys.stdout.encoding or 'utf-8'
    print(' '.join(map(str, a)).encode(enc, 'replace').decode(enc))


def cpu(p: subprocess.Popen) -> float:
    t = psutil.Process(p.pid).cpu_times()
    return t.user + t.system


def tone(seconds: float, sr: int, f: float = 440.0, level: float = 0.3) -> np.ndarray:
    return (level * np.sin(2 * np.pi * f * np.arange(int(sr * seconds)) / sr)).astype(np.float32)


class Feed:
    """A test process plays a tone into an endpoint, as long as the test needs (the stand-in microphone)."""

    def __init__(self, dev: str, f: float = 440.0):
        self.p = windows.spawn('play', '--to', dev, '--rate', str(SR_OUT), stdout=subprocess.DEVNULL)
        self.stop_ = threading.Event()
        chunk = tone(0.1, SR_OUT, f).tobytes()

        def run():
            t0 = time.monotonic()
            for i in range(10 ** 6):
                if self.stop_.wait(max(0.0, t0 + i * 0.1 - 0.2 - time.monotonic())):
                    break
                try:
                    self.p.stdin.write(chunk)
                    self.p.stdin.flush()
                except (BrokenPipeError, ValueError, OSError):
                    break
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def stop(self):
        self.stop_.set()
        self.t.join(5)
        self.p.kill()
        self.p.wait(5)


def lines(p: subprocess.Popen, out: list):
    def run():
        for b in p.stderr:
            out.append(b.decode(errors='replace').strip())
    threading.Thread(target=run, daemon=True).start()
    return out


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.05)
    return True


def _pwsh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', cmd], capture_output=True, text=True,
                          timeout=60)


def _policy(call):
    """IPolicyConfig, what the Sound control panel calls: undocumented but stable since Windows 7 (SoundVolumeView,
    AudioDeviceCmdlets). `call(p, vtbl)` with the object; released after."""
    import ctypes
    import uuid

    class GUID(ctypes.Structure):
        _fields_ = [('b', ctypes.c_ubyte * 16)]

    def guid(x):
        g = GUID()
        g.b[:] = uuid.UUID(x).bytes_le
        return g
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx(None, 0)  # MTA; already initialized on this thread is fine
    p = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(ctypes.byref(guid('870af99c-171d-4f9e-af0d-e63df40c2bc9')), None, 1,  # PolicyConfigClient
                                ctypes.byref(guid('f8679f50-850a-41cf-9c72-430f290290c8')), ctypes.byref(p))  # IPolicyConfig
    assert hr == 0 and p.value, f'IPolicyConfig: {hr & 0xFFFFFFFF:#x}'
    vtbl = ctypes.cast(p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    try:  # IUnknown (0-2), GetMixFormat 3, GetDeviceFormat 4, ResetDeviceFormat 5, SetDeviceFormat 6, ...,
        return call(p, vtbl)  # SetDefaultEndpoint 13, SetEndpointVisibility 14
    finally:
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])(p)


def endpoint_enabled(uid: str, on: bool) -> None:
    """Disable or enable an endpoint as the Sound control panel does: WASAPI then sees it go (DEVICE_STATE_DISABLED,
    its streams invalidated), as an unplugged headset's. Disable-PnpDevice is no stand-in: on the endpoint's node
    (SWD\\MMDEVAPI\\<id>) WASAPI still lists it active, and the cable's driver node is not stopped while it plays."""
    import ctypes
    hr = _policy(lambda p, v: ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)(v[14])(
        p, uid, int(on)))
    assert hr == 0, f'SetEndpointVisibility({uid}, {on}): {hr & 0xFFFFFFFF:#x}'


def device_rate(uid: str, rate: int) -> int:
    """The endpoint's "Default format" to `rate` (channels and sample type kept), as the Sound control panel's Advanced
    tab sets it; returns the rate it had."""
    import ctypes
    import struct

    def call(p, v):
        fn = ctypes.WINFUNCTYPE
        get = fn(ctypes.c_long, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(v[4])
        mix = fn(ctypes.c_long, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p))(v[3])
        put = fn(ctypes.c_long, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p)(v[6])
        fmts = []
        for ask in (lambda out: get(p, uid, 0, out), lambda out: mix(p, uid, out)):  # the device's, the engine's mix
            ptr = ctypes.c_void_p()
            hr = ask(ctypes.byref(ptr))
            assert hr == 0 and ptr.value, f'the format of {uid}: {hr & 0xFFFFFFFF:#x}'
            head = ctypes.string_at(ptr, 18)  # WAVEFORMATEX; cbSize more bytes follow (EXTENSIBLE: 22)
            fmts.append(bytearray(ctypes.string_at(ptr, 18 + struct.unpack_from('<H', head, 16)[0])))
            ctypes.windll.ole32.CoTaskMemFree(ptr)
        old = struct.unpack_from('<I', fmts[0], 4)[0]
        for f in fmts:  # nSamplesPerSec, nAvgBytesPerSec = rate x nBlockAlign
            struct.pack_into('<II', f, 4, rate, rate * struct.unpack_from('<H', f, 12)[0])
        bufs = [(ctypes.c_char * len(f)).from_buffer(f) for f in fmts]
        hr = put(p, uid, ctypes.addressof(bufs[0]), ctypes.addressof(bufs[1]))
        assert hr == 0, f'SetDeviceFormat({uid}, {rate}): {hr & 0xFFFFFFFF:#x}'
        return old
    return _policy(call)


# --- what needs no Windows

DEV = {'rate': 48000, 'default_input': False, 'default_output': False}
CABLE_IN = {**DEV, 'uid': '{0.0.0.00000000}.{in}', 'name': 'CABLE Input (VB-Audio Virtual Cable)', 'inputs': 0, 'outputs': 2,
            'transport': 'virt'}
CABLE_OUT = {**DEV, 'uid': '{0.0.1.00000000}.{out}', 'name': 'CABLE Output (VB-Audio Virtual Cable)', 'inputs': 2, 'outputs': 0,
             'transport': 'virt'}
HEADSET = {**DEV, 'uid': '{0.0.1.00000000}.{usb}', 'name': 'Microphone (USB Headset)', 'inputs': 1, 'outputs': 0,
           'transport': 'usb', 'default_input': True}
SPEAKERS = {**DEV, 'uid': '{0.0.0.00000000}.{hda}', 'name': 'Speakers (Realtek(R) Audio)', 'inputs': 0, 'outputs': 2,
            'transport': '', 'default_output': True}
MEETER = {**DEV, 'uid': '{0.0.1.00000000}.{vm}', 'name': 'VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)', 'inputs': 2,
          'outputs': 0, 'transport': 'virt'}
BROADCAST = {**DEV, 'uid': '{0.0.1.00000000}.{nv}', 'name': 'Microphone (NVIDIA Broadcast)', 'inputs': 1, 'outputs': 0,
             'transport': 'virt'}  # the real microphone, its noise taken out: the user's voice


def test_cable_and_virtual_inputs_are_no_microphone(monkeypatch):
    from peresmeshnik.ui import selftest
    devs = [HEADSET, SPEAKERS, CABLE_IN, CABLE_OUT, MEETER, BROADCAST]
    monkeypatch.setattr(windows, 'devices', lambda: devs)
    monkeypatch.setattr(selftest, 'HELPER', True)
    monkeypatch.setattr(selftest, 'WINDOWS', True)
    monkeypatch.setattr(sys, 'platform', 'win32')  # mic.native() picks audio/windows.py
    assert windows.cable(devs) == (CABLE_IN, CABLE_OUT)
    for key in (CABLE_OUT['uid'], CABLE_OUT['name']):
        with pytest.raises(ValueError, match='CABLE Output is the virtual microphone for the call app'):
            windows.not_virtual(key)
    with pytest.raises(ValueError, match='VoiceMeeter Output .* is a virtual device, not your microphone'):
        windows.not_virtual(MEETER['uid'])
    assert windows.not_virtual(HEADSET['uid']) == HEADSET['uid']
    assert windows.not_virtual(BROADCAST['uid']) == BROADCAST['uid']
    assert windows.not_virtual('') == ''  # the default input: the helper never takes a virtual one by itself
    assert windows.not_virtual(CABLE_OUT['name'], what='headphones') == CABLE_OUT['name']  # not this check's business
    assert selftest.devices('sources') == [(HEADSET['uid'], HEADSET['name']), (BROADCAST['uid'], BROADCAST['name'])]
    assert selftest.devices('sinks') == [(SPEAKERS['name'], SPEAKERS['name'])]  # by name: the headphones go through PortAudio
    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', MEETER['uid'])  # the tests' stand-in passes
    assert windows.not_virtual(MEETER['uid']) == MEETER['uid']
    assert (MEETER['uid'], MEETER['name']) in selftest.devices('sources')
    monkeypatch.delenv('PYTEST_CURRENT_TEST')  # the app itself: the tests' hook lets nothing through
    assert (MEETER['uid'], MEETER['name']) not in selftest.devices('sources')


def test_no_cable_says_how_to_install(monkeypatch):
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    QApplication = pytest.importorskip('PySide6.QtWidgets').QApplication  # noqa: N806
    QApplication.instance() or QApplication([])
    from peresmeshnik import config
    from peresmeshnik.ui import selftest, wizard
    devs, perm, asked = [HEADSET, SPEAKERS], ['granted'], []
    monkeypatch.setattr(windows, 'devices', lambda: list(devs))
    monkeypatch.setattr(windows, 'permission', lambda: asked.append(1) or perm[0])
    monkeypatch.setattr(wizard, 'WINDOWS', True)
    monkeypatch.setattr(wizard, 'MACOS', False)
    monkeypatch.setattr(selftest, 'HELPER', True)
    monkeypatch.setattr(selftest, 'WINDOWS', True)
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(windows.CableMissing, match='Run as administrator'):
        windows.cable()
    page = wizard.DevicesPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    text = page.vmic.text()
    assert 'not installed' in text and 'vb-audio.com/Cable' in text and 'Run as administrator' in text and 'restart' in text
    assert not page.recheck.isHidden()
    devs += [CABLE_IN, CABLE_OUT]  # installed meanwhile (and Windows restarted)
    page.recheck.click()
    assert page.vmic.text().startswith('✔ VB-Audio Virtual Cable is installed') and page.recheck.isHidden()
    assert 'CABLE Output' in page.vmic.text()
    assert [page.mic.itemData(i) for i in range(page.mic.count())] == ['', HEADSET['uid']]  # never the cable
    assert '⚠' not in page.vmic.text() and '✘' not in page.vmic.text()
    devs[:] = [HEADSET, {**SPEAKERS, 'default_output': False}, {**CABLE_IN, 'default_output': True},
               {**CABLE_OUT, 'default_input': True}]  # the cable as Windows' default devices, the microphone refused
    assert wait_for(lambda: QApplication.processEvents() or asked == [1])  # tried once the cable is there, in a worker
    perm[0] = 'denied'
    page.initializePage()  # back to the page: the devices again, the microphone not (it takes up to seconds)
    QApplication.processEvents()
    assert asked == [1] and windows.NO_PERMISSION not in page.vmic.text()
    assert page.vmic.text().count('⚠') == 2  # the cable as Windows' default devices
    page.recheck.click()  # "Check again" tries the microphone again
    assert wait_for(lambda: QApplication.processEvents() or windows.NO_PERMISSION in page.vmic.text())
    text = page.vmic.text()
    assert "CABLE Input (VB-Audio Virtual Cable)</b> is Windows' default output" in text and 'Sound → Output' in text
    assert "CABLE Output (VB-Audio Virtual Cable)</b> is Windows' default microphone" in text and 'Sound → Input' in text
    assert asked == [1, 1] and not page.recheck.isHidden()


def test_a_refused_microphone_says_so_at_once(monkeypatch):
    """No permission (the helper exits with 6 as it opens the microphone): record() raises why at once, not after its
    seconds of waiting for sound (the voice page's bar ran 50 s, then said "too quiet")."""
    from peresmeshnik.audio import mic

    class Refused:
        def __init__(self, device=None, sample_rate=16000, on_end=None, **k):
            self.on_end = on_end

        def start(self, on_frames):
            threading.Timer(0.2, self.on_end, [f'the microphone: {windows.NO_PERMISSION}']).start()

        def stop(self):
            pass
    monkeypatch.setattr(mic, 'Mic', Refused)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match='Privacy & security'):
        mic.record(45)
    assert time.monotonic() - t0 < 2


def test_cable_output_by_name_and_the_helpers_own_words(monkeypatch):
    other = {**CABLE_OUT, 'uid': '{0.0.1.00000000}.{16}', 'name': 'CABLE 16ch Output (VB-Audio Virtual Cable)'}
    assert windows.cable([CABLE_IN, other, CABLE_OUT]) == (CABLE_IN, CABLE_OUT)  # as play prefers "CABLE Input"
    monkeypatch.setattr(windows, 'helper', lambda: 'peresmeshnik-audio.exe')
    monkeypatch.setattr(windows.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(
        returncode=5, stdout=b'', stderr=b'peresmeshnik-audio: no audio device enumerator (is the Windows Audio service running?)\n'))
    with pytest.raises(RuntimeError, match=r'^no audio device enumerator \(is the Windows Audio service running\?\)$'):
        windows.devices()


class Proc:
    """A helper process: its stderr lines, then its exit code."""
    def __init__(self, lines=(), code=None):
        self.stderr, self.returncode, self.stdin = iter(lines), code, types.SimpleNamespace(close=lambda: None)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode if self.returncode is not None else 0

    def kill(self):
        pass


def test_a_call_app_on_another_microphone_is_hinted_once(monkeypatch):
    """Translation on, nobody records CABLE Output (the helper's watch: idle) and the user talks: one hint which
    microphone to choose in the call app; while the call app records it, none."""
    from peresmeshnik import app
    said = []
    w = app.CallWatch(CABLE_OUT['uid'], said.append, lambda *a, **k: Proc([b'idle\n']))
    assert wait_for(lambda: not w.reading)
    w.talked()
    assert said == []  # not yet HINT_S without a reader
    w.since -= app.CallWatch.HINT_S
    w.talked()
    w.talked()
    assert said == [app.CallWatch.HINT] and 'CABLE Output' in said[0]
    ok = app.CallWatch(CABLE_OUT['uid'], said.append, lambda *a, **k: Proc([b'idle\n', b'call\n']))
    assert wait_for(lambda: ok.reading and ok.p.stderr.__length_hint__() == 0)
    ok.since -= app.CallWatch.HINT_S
    ok.talked()
    assert said == [app.CallWatch.HINT]


def test_the_microphone_loop_starts_again_within_its_budget(monkeypatch):
    """Translation off: the loop "microphone -> CABLE Input" ends by itself: started again like a side, at most RESTARTS
    times a minute, then the tray says why; a refused microphone (6) is said at once."""
    from peresmeshnik import app
    from peresmeshnik.audio import mic
    spawned, states = [], []

    def spawn(*a, **k):
        spawned.append(a)
        return Proc([b'idle\n'], 5)
    monkeypatch.setattr(mic, 'native', lambda: types.SimpleNamespace(not_virtual=lambda key: key, spawn=spawn, NO_PERMISSION='no'))
    monkeypatch.setattr(sys, 'platform', 'win32')
    ctl = app.Controller(Overlay(), states.append, settings=app.config.Settings())
    ctl.devices, ctl._vmic = True, types.SimpleNamespace(device=CABLE_IN, call_mic=CABLE_OUT)
    try:
        ctl.submit(ctl._loopback, True).result(5)
        assert wait_for(lambda: len(spawned) == 1 + app.RESTARTS and states, 10), (spawned, states)
        assert states == ['error: the call will not hear your microphone while translation is off: the loop ended (5)']
    finally:
        ctl._pool.shutdown()


def test_them_to_me_translates_while_the_clone_loads(monkeypatch):
    """Translation on with the clone: them -> me starts before me -> them is made (the voice pack's runner loads for up
    to ~30 s), and the tray says the voice is loading."""
    from peresmeshnik import app
    order, states = [], []

    class FakeSide:
        def __init__(self, name):
            self.name, self.engine, self.inp = name, 'local', None

        def set_live(self, on):
            pass

        def start(self):
            order.append(f'{self.name} starts')

        def stop(self):
            pass

    def side(self, name, s):
        order.append(f'{name} made')
        return FakeSide(name)
    monkeypatch.setattr(app.Controller, '_side', side)
    s = app.config.Settings()
    s.me_to_them, s.local.voice = 'local', 'me'
    ctl = app.Controller(Overlay(), states.append, them_file='them.wav', me_file='me.wav', settings=s)
    try:
        ctl.submit(ctl._on).result(10)
    finally:
        ctl._pool.shutdown()
    assert order == ['them made', 'them starts', 'me made', 'me starts']
    assert 'warming up: them → me translates, your voice is loading' in states and ctl.on


# --- on Windows, with VB-Cable

@win
def test_devices():
    devs = windows.devices()
    say('\ndevices:', [(d['name'], d['transport'], d['inputs'], d['outputs'], d['rate']) for d in devs])
    play, rec = windows.cable(devs)
    assert play['transport'] == rec['transport'] == 'virt'  # a software device, from its KS filter
    assert rec['name'].startswith(windows.CABLE_OUT)
    from peresmeshnik.ui.selftest import devices
    assert not any(windows.CABLE in name for _, name in devices('sources') + devices('sinks'))


@win
def test_round_trip():
    """The translation into CABLE Input, recorded from CABLE Output like a call app: tone, level, length, real time."""
    from peresmeshnik.audio.virtual_mic_windows import VirtualMic
    from peresmeshnik.ui.selftest import Recorder
    vm = VirtualMic()
    rec = Recorder(vm.call_mic['uid'], SR_OUT)
    try:
        time.sleep(0.3)
        t = np.arange(int(SR_IN * 3)) / SR_IN
        x = (0.3 * np.sin(2 * np.pi * 440 * t) * (t < 1.5)).astype(np.float32)  # 1.5 s tone, 1.5 s silence
        t0 = time.perf_counter()
        for i in range(0, len(x), SR_IN // 5):  # 200 ms chunks, like TTS
            vm.play(x[i:i + SR_IN // 5], SR_IN)
        helper_cpu = cpu(vm._proc)
        vm.drain()
        wall = time.perf_counter() - t0
        time.sleep(0.3)
        y = rec.stop()
    finally:
        vm.remove()
    on = np.nonzero(np.abs(y) > 0.1)[0]
    assert len(on), 'nothing came out of CABLE Output'
    seg = y[on[0]:on[0] + SR_OUT]
    say(f'\nround trip: {len(y) / SR_OUT:.2f} s recorded, tone {(on[-1] - on[0]) / SR_OUT:.3f} s, '
          f'rms {np.sqrt(np.mean(seg ** 2)):.3f}, played in {wall:.2f} s, helper CPU {helper_cpu:.3f} s')
    assert np.argmax(np.abs(np.fft.rfft(seg))) * SR_OUT / len(seg) == pytest.approx(440, abs=3)
    assert (on[-1] - on[0]) / SR_OUT == pytest.approx(1.5, abs=0.05)  # whole tone, nothing dropped
    assert np.sqrt(np.mean(seg ** 2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.1)  # level kept
    # real time by the samples above (the whole tone, nothing dropped, its level); the wall time says only that it was
    # not played faster than that, and that drain() returned: a busy runner takes longer (3.64 and 4.9 s seen)
    assert 2.9 < wall < 15


class Timed:
    """The helper's recording of an endpoint, each chunk stamped when it arrives."""

    def __init__(self, dev: str, sr: int = SR_OUT):
        self.sr, self.chunks, self.stamps = sr, [], []
        self.p = windows.spawn('record', '--from', dev, '--rate', str(sr))

        def run():
            n = 0
            for b in iter(lambda: self.p.stdout.read(4 * sr // 100), b''):
                self.chunks.append(np.frombuffer(b[:len(b) // 4 * 4], '<f4'))
                n += len(self.chunks[-1])
                self.stamps.append((time.monotonic(), n))
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def at(self, i: int) -> float:
        return next(t for t, n in self.stamps if n > i)

    def stop(self) -> np.ndarray:
        self.p.stdin.close()
        self.p.wait(5)
        self.t.join(5)
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


@win
def test_flush_leaves_silence():
    """Barge-in and the mode switch drop queued speech (flush: the player is killed) while the call app keeps reading
    CABLE Output: from then on it hears silence (no last buffer left looping, as BlackHole did on macOS)."""
    from peresmeshnik.audio.virtual_mic_windows import VirtualMic
    vm = VirtualMic()
    rec = Timed(vm.call_mic['uid'])
    try:
        time.sleep(0.4)
        vm.play(tone(6.0, SR_IN), SR_IN)
        time.sleep(1.0)
        vm.flush()
        time.sleep(4.0)
        y = rec.stop()
    finally:
        vm.remove()
    # the tone's own samples, not its span or when they came: a busy runner leaves gaps in it (its span 1.8-2.3 s and
    # rms 0.13-0.15 over the span were seen), so the sound is counted, as record() counts it (PR #31)
    loud = np.flatnonzero(np.abs(y) > 0.01)
    heard, rms, rest = len(loud) / SR_OUT, np.sqrt(np.mean(y[loud] ** 2)), y[loud[-1] + 1:]
    say(f'\nflush: {heard:.2f} s of the 6 s tone heard (flushed after 1 s, span {(loud[-1] - loud[0]) / SR_OUT:.2f} s), '
        f'rms {rms:.3f}; then {len(rest) / SR_OUT:.1f} s of silence (max {np.abs(rest).max() if len(rest) else 0:.4f})')
    assert rms > 0.15  # it was playing, at its level (0.3 peak: 0.21)
    assert 0.3 < heard < 3.0  # it stopped at the flush: about 1 s of the 6 s tone (plus what the cable held)
    assert len(rest) > 1.5 * SR_OUT  # and nothing after it: no last buffer left looping


@win
def test_latency():
    """play() -> the sound at CABLE Output as a call app reads it, 5 clicks from silence (the first one starts the
    player). Upper bound: includes the reader's own 10 ms chunks."""
    from peresmeshnik.audio.virtual_mic_windows import VirtualMic
    vm = VirtualMic()
    rec = Timed(vm.call_mic['uid'])
    sent = []
    try:
        time.sleep(0.5)
        for _ in range(5):
            sent.append(time.monotonic())
            vm.play(np.concatenate([np.full(SR_IN // 50, 0.5, np.float32), np.zeros(SR_IN // 2, np.float32)]), SR_IN)
            time.sleep(0.6)
        y = rec.stop()
    finally:
        vm.remove()
    on = np.nonzero(np.abs(y) > 0.25)[0]
    starts = [i for k, i in enumerate(on) if k == 0 or i - on[k - 1] > SR_OUT // 10]
    lags = [rec.at(i) - t for i, t in zip(starts, sent)]
    say('\nplay -> CABLE Output, ms:', [round(x * 1000) for x in lags])
    assert len(starts) == 5
    assert max(lags[1:]) < 0.15


@win
def test_mic_follows_a_device_that_goes_and_comes_back(monkeypatch):
    """The saved microphone (CABLE Output here, fed a tone through CABLE Input) goes away (the endpoint disabled as in
    the Sound control panel, as a headset unplugged): the helper says so, takes the default input or none, comes back
    to the saved one when it returns; frames flow from it again. The translation's player into CABLE Input says why it
    stopped."""
    from peresmeshnik.audio.mic import Mic
    from peresmeshnik.audio.virtual_mic_windows import VirtualMic
    play, rec = windows.cable()
    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', rec['uid'])
    feed = Feed(play['uid'])
    frames, events = [], []
    m = Mic(rec['uid'], follow=True, on_device=lambda kind, name: events.append((kind, name)),
            on_end=lambda why: events.append(('end', why)))
    disabled = False
    try:
        m.start(frames.append)
        assert wait_for(lambda: ('using', rec['name']) in events), events
        assert wait_for(lambda: len(frames) > 25)
        assert np.abs(np.concatenate(frames[-10:])).max() > 0.2, 'no tone from the saved microphone'
        endpoint_enabled(rec['uid'], False)
        disabled = True
        assert wait_for(lambda: ('lost', rec['name']) in events, 15), (events, [d['name'] for d in windows.devices()])
        assert wait_for(lambda: events[-1][0] in ('using', 'none') and events[-1][1] != rec['name'], 10), events
        say('\nwithout the saved microphone:', events[-1])
        endpoint_enabled(rec['uid'], True)
        disabled = False
        assert wait_for(lambda: events[-1] == ('using', rec['name']), 15), events
        n = len(frames)
        assert wait_for(lambda: len(frames) > n + 25)
        assert np.abs(np.concatenate(frames[-10:])).max() > 0.2, 'no tone after it came back'
        assert m.source == rec['name'] and m.uid == rec['uid']
        assert not any(k == 'end' for k, _ in events)
        say('events:', events)
    finally:
        if disabled:
            endpoint_enabled(rec['uid'], True)
        m.stop()
        feed.stop()
        wait_for(lambda: bool(_ready() == ''), 15)

    vm = VirtualMic()  # CABLE Input goes while the translation plays: the next play() says why
    try:
        vm.play(tone(0.5, SR_IN), SR_IN)
        time.sleep(0.3)
        endpoint_enabled(play['uid'], False)
        assert wait_for(lambda: vm._proc.poll() is not None, 15)
        assert vm._proc.returncode == 4
        with pytest.raises(RuntimeError, match='the playback stopped .*went away'):
            vm.play(tone(0.2, SR_IN), SR_IN)
    finally:
        endpoint_enabled(play['uid'], True)
        vm.remove()
        assert wait_for(lambda: bool(_ready() == ''), 15)


@win
def test_a_format_change_of_the_cable_opens_its_streams_again(monkeypatch):
    """The user changes the "Default format" of CABLE Input, then of CABLE Output (Sound control panel → Advanced) while
    the translation plays, the call app records and the loop runs: Windows invalidates their streams, often without a
    word. play ends with 7 and VirtualMic plays the next clause on a new one; the recording and the loop go on."""
    from peresmeshnik.audio.mic import Mic
    from peresmeshnik.audio.virtual_mic_windows import VirtualMic
    play, rec = windows.cable()
    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', rec['uid'])
    new = 44100 if play['rate'] != 44100 else 48000
    vm, frames, events, rates = VirtualMic(), [], [], {}
    m = Mic(rec['uid'], follow=True, on_device=lambda kind, name: events.append((kind, name)))
    loop = windows.spawn('loop', '--from', '', '--to', play['uid'], stdout=subprocess.DEVNULL)  # no --watch: it runs
    looped = lines(loop, [])
    try:
        m.start(frames.append)
        assert wait_for(lambda: ('using', rec['name']) in events), events
        vm.play(tone(4.0, SR_IN), SR_IN)
        time.sleep(0.5)
        rates[play['uid']] = device_rate(play['uid'], new)
        assert wait_for(lambda: vm._proc.poll() is not None, 10), 'play went on after the format change (silent?)'
        assert vm._proc.returncode == 7, vm._proc.returncode
        assert wait_for(lambda: 'reopened' in looped, 5), f'the loop did not open its output again: {looped}'
        rates[rec['uid']] = device_rate(rec['uid'], new)
        time.sleep(1.5)  # a check a second at most
        n = len(frames)
        vm.play(tone(1.5, SR_IN), SR_IN)  # reopen = (7,): a new player at the new format
        assert wait_for(lambda: len(frames) > n + 60, 10)
        level = np.abs(np.concatenate(frames[n:])).max()
        say(f'\nformat change to {new} Hz: play exit {7}, the recording after it peaks {level:.2f}; events {events}; '
            f'loop {looped}')
        assert level > 0.2, 'no tone through the cable after its format changed'
        assert loop.poll() is None, f'the loop ended ({loop.returncode})'
    finally:
        for uid, rate in rates.items():
            device_rate(uid, rate)
        m.stop()
        vm.remove()
        loop.stdin.close()
        loop.wait(5)


@win
def test_watch_says_who_records_the_cable():
    """Translation on: `watch` says "call" while another process (the call app) records CABLE Output, "idle" when it
    stops (CallWatch hints the microphone of the call app from it)."""
    from peresmeshnik.ui.selftest import Recorder
    play, rec = windows.cable()
    w = windows.spawn('watch', '--watch', rec['uid'], stdout=subprocess.DEVNULL)
    ev = lines(w, [])
    try:
        assert wait_for(lambda: ev[:1] == ['idle']), ev
        call = Recorder(rec['uid'], SR_OUT)
        assert wait_for(lambda: ev[-1:] == ['call']), ev
        call.stop()
        assert wait_for(lambda: ev[-1:] == ['idle']), ev
    finally:
        w.stdin.close()
        w.wait(5)
    say(f'\nwatch: {ev}')
    assert w.returncode == 0


@win
def test_mic_cpu(monkeypatch):
    """CPU of the helper recording an endpoint at 16 kHz (the translation's microphone), per second of audio."""
    from peresmeshnik.audio.mic import Mic
    play, rec = windows.cable()
    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', rec['uid'])
    frames = []
    m = Mic(rec['uid'])
    m.start(frames.append)
    try:
        time.sleep(1)
        c0, t0, n0 = cpu(m._proc), time.monotonic(), len(frames)
        time.sleep(10)
        c, dt, n = cpu(m._proc) - c0, time.monotonic() - t0, len(frames) - n0
    finally:
        m.stop()
    say(f'\nmicrophone: {n * 0.02:.2f} s of frames in {dt:.2f} s, helper {c / dt:.4f} CPU s per s')
    assert n * 0.02 == pytest.approx(dt, rel=0.05)  # 20 ms frames in real time
    assert c / dt < 0.05


@win
def test_loop_only_while_a_call_records():
    """Translation off: the loop "microphone -> CABLE Input" opens the microphone only while another process (the call
    app) records CABLE Output (its WASAPI session is active); idle it costs no CPU and holds no microphone. The
    microphone: a second cable fed a chirp where there is one, else none (the runner): then only call / idle and CPU."""
    from peresmeshnik.ui.selftest import Recorder, chirp, find
    play, rec = windows.cable()
    two = second()
    mic_in, mic = two or ({'uid': ''}, {'uid': ''})
    loop = windows.spawn('loop', '--from', mic['uid'], '--to', play['uid'], '--watch', rec['uid'], stdout=subprocess.DEVNULL)
    ev = lines(loop, [])
    try:
        assert wait_for(lambda: 'idle' in ev), ev
        c0 = cpu(loop)
        time.sleep(10)
        idle = cpu(loop) - c0
        call = Recorder(rec['uid'], SR_OUT)  # the call app takes CABLE Output as its microphone
        opened = ('using ' + mic['uid']) if two else 'none'  # no microphone on the runner: the loop says so
        assert wait_for(lambda: 'call' in ev and any(e.startswith(opened) for e in ev)), ev
        time.sleep(0.5)
        ref = chirp(SR_OUT)
        if two:
            feed = windows.spawn('play', '--to', mic_in['uid'], '--rate', str(SR_OUT), stdout=subprocess.DEVNULL)
            feed.communicate(np.concatenate([np.zeros(SR_OUT // 2, np.float32), ref]).tobytes(), timeout=10)
        c1 = cpu(loop)
        time.sleep(10)
        active = cpu(loop) - c1
        y = call.stop()
        assert wait_for(lambda: ev[-1] == 'idle'), ev  # the call app let go: the microphone is closed again
    finally:
        loop.stdin.close()
        loop.wait(5)
    score, at = find(y, ref)
    say(f'\nloop: idle {idle / 10:.4f} CPU s per s, while a call records {active / 10:.4f}; chirp match {score:.2f}; events {ev}')
    assert score > 0.5 or not two
    assert idle / 10 < 0.002
    assert loop.returncode == 0


@win
def test_self_test(monkeypatch):
    from peresmeshnik.ui import selftest
    _, rec = windows.cable()
    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', rec['uid'])
    rs = selftest.run_all(mic=rec['uid'])
    say('\n' + '\n'.join(f'{r.ok} {r.name}: {r.detail}' for r in rs))
    assert rs[0].ok, rs[0]  # the virtual mic; the stand-in microphone is silent


class Fake:
    """"Translates" each second of input into a second of a 220 Hz tone (whole periods: seconds join cleanly)."""

    def __init__(self, speaks):
        import queue
        self.q, self.n, self.speaks = queue.Queue(), 0, speaks

    def feed(self, pcm):
        from peresmeshnik.engines import Speech
        self.n += len(pcm)
        if self.speaks and self.n >= 16000:
            self.n -= 16000
            self.q.put(Speech(tone(1.0, 24000, 220.0), 24000))

    def events(self):
        while (e := self.q.get()) is not None:
            yield e

    def close(self):
        self.q.put(None)


class Overlay:
    def side(self, name):
        return self

    def show(self, caption):
        pass

    def hide(self):
        pass


@win
def test_app_me_side(monkeypatch, tmp_path):
    """The app's me -> them on Windows (LiveSwitch) with a fake translation (220 Hz) and my voice from a file (300 Hz):
    what the call app hears from CABLE Output: translation, live voice, translation, then nothing once translation
    is off (a file run has no loop)."""
    import soundfile as sf

    from peresmeshnik import app, config
    from peresmeshnik.ui.selftest import Recorder
    play, rec = windows.cable()
    me = tmp_path / 'me.wav'
    sf.write(me, tone(12.0, 16000, 300.0), 16000)
    s = config.Settings()
    s.me_to_them = 'local'
    monkeypatch.setattr(app.Side, '_local', lambda self, s: Fake(self.name == 'me'))  # no models; the rest is real
    states = []
    ctl = app.Controller(Overlay(), states.append, me_file=str(me), settings=s)
    rec_ = None
    try:
        ctl.set_translation(True).result()
        assert any(x.name == 'me' for x in ctl.sides), states
        rec_ = Recorder(rec['uid'], 48_000)
        time.sleep(2.5)  # a second of input, then the translation speaks
        ctl.set_live(True)
        time.sleep(1.5)
        ctl.set_live(False)
        time.sleep(2.5)
        ctl.set_translation(False).result()
        time.sleep(1.0)
        x = rec_.stop()
        rec_ = None
    finally:
        if rec_:
            rec_.stop()
        ctl.quit()
    left = [p.name() for p in psutil.Process().children(recursive=True) if 'peresmeshnik-audio' in p.name()]
    say('\nstates:', states)
    assert not left, left
    sr, w = 48_000, 960
    f = np.fft.rfftfreq(w, 1 / sr)
    wins = x[:len(x) // w * w].reshape(-1, w)
    rms = np.sqrt(np.mean(wins ** 2, axis=1))
    peak = f[np.argmax(np.abs(np.fft.rfft(wins, axis=1)), axis=1)]
    kind = np.where(rms < 0.02, '.', np.where(np.abs(peak - 220) < 30, 't', np.where(np.abs(peak - 300) < 30, 'L', '?')))
    heard = [k for k in kind if k in 'tL']
    runs = ''.join(k for i, k in enumerate(heard) if i == 0 or k != heard[i - 1])
    say('call hears:', ''.join(kind))
    assert runs == 'tLt', runs  # translation, live voice, translation; off: silence



@pytest.mark.parametrize('code', [0, 5])
def test_call_sound_page_on_windows(monkeypatch, code):
    """The wizard's call page on Windows: S1's process loopback checked (a too old Windows says why, "Check again"),
    and the call app playing now named."""
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    QApplication = pytest.importorskip('PySide6.QtWidgets').QApplication  # noqa: N806
    QApplication.instance() or QApplication([])
    from peresmeshnik import config
    from peresmeshnik.audio import capture_windows as cw
    from peresmeshnik.ui import wizard
    why = 'Windows older than 10 2004 (build 19041) has no process loopback'
    monkeypatch.setattr(wizard, 'WINDOWS', True)
    monkeypatch.setattr(cw, 'check', lambda: (code, why if code else ''))
    monkeypatch.setattr(cw, 'guess_call_app', lambda: 'Telegram.exe')
    page = wizard.CallSoundPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    text = page.state.text()
    if code:
        assert text == f'✘ {why}' and not page.button.isHidden()
    else:
        assert text.startswith('✔ Windows gives Peresmeshnik') and 'Telegram.exe' in text and page.button.isHidden()


@win
def test_call_sound_page_on_the_runner():
    """The real check: S1's loopback helper (built by CI) on this Windows."""
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from peresmeshnik import config
    from peresmeshnik.ui import wizard
    page = wizard.CallSoundPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    say('\ncall page:', page.state.text())
    assert page.state.text().startswith('✔'), page.state.text()


@win
def test_voice_page_records_on_windows(monkeypatch, tmp_path):
    """The wizard's voice page records through the helper: a steady tone into CABLE Input, CABLE Output as the
    stand-in microphone; recorded for the asked 3 s, not saved, and the page says why."""
    pytest.importorskip('PySide6')
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from peresmeshnik import config, voice
    from peresmeshnik.ui.wizard import VoicePage
    play, rec = windows.cable()
    monkeypatch.setenv('PERESMESHNIK_TEST_MIC', rec['uid'])
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    s = config.Settings()
    s.me_to_them, s.ui.mic = 'local', rec['uid']
    page = VoicePage(type('W', (), {'s': s})())
    page.initializePage()
    feed = Feed(play['uid'], f=300.0)
    try:
        page.seconds = 3.0
        page.record()
        end = time.monotonic() + 20
        while page.running and time.monotonic() < end:
            QCoreApplication.processEvents()
            time.sleep(0.01)
    finally:
        feed.stop()
    text = page.out.text()
    say('\nvoice page:', text)
    assert text.startswith('<b>Record again: 3 s recorded, need 30-60 s; only 0 s of speech'), text
    assert 'Not saved' in text and page.voice is None

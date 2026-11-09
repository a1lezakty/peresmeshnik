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


def endpoint_enabled(uid: str, on: bool) -> None:
    """Disable or enable an endpoint as the Sound control panel does: WASAPI then sees it go (DEVICE_STATE_DISABLED,
    its streams invalidated), as an unplugged headset's. IPolicyConfig::SetEndpointVisibility is undocumented but
    stable since Windows 7 (what SoundVolumeView /Disable calls). Disable-PnpDevice is no stand-in: on the endpoint's
    node (SWD\\MMDEVAPI\\<id>) WASAPI still lists it active, and the cable's driver node is not stopped while it plays."""
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
    try:  # IUnknown (3), then GetMixFormat ... SetDefaultEndpoint (11): SetEndpointVisibility is slot 14
        hr = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)(vtbl[14])(p, uid, int(on))
    finally:
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])(p)
    assert hr == 0, f'SetEndpointVisibility({uid}, {on}): {hr & 0xFFFFFFFF:#x}'


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


def test_cable_and_virtual_inputs_are_no_microphone(monkeypatch):
    from peresmeshnik.ui import selftest
    devs = [HEADSET, SPEAKERS, CABLE_IN, CABLE_OUT, MEETER]
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
    assert windows.not_virtual('') == ''  # the default input: the helper never takes a virtual one by itself
    assert windows.not_virtual(CABLE_OUT['name'], what='headphones') == CABLE_OUT['name']  # not this check's business
    assert selftest.devices('sources') == [(HEADSET['uid'], HEADSET['name'])]
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
    devs = [HEADSET, SPEAKERS]
    monkeypatch.setattr(windows, 'devices', lambda: list(devs))
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
    assert 2.9 < wall < 4.0  # played in real time (3.64 s seen on a busy runner)


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
        vm.play(tone(3.0, SR_IN), SR_IN)
        time.sleep(1.0)
        t = time.monotonic()
        vm.flush()
        time.sleep(2.5)
        y = rec.stop()
    finally:
        vm.remove()
    cut = next(n for at, n in rec.stamps if at >= t + 0.3)  # chunks arrive with jitter: ~1 ms of tone was seen at +0.15 s
    before, after = y[:cut - SR_OUT // 5], y[cut:]
    say(f'\nflush: rms before {np.sqrt(np.mean(before[-SR_OUT // 2:] ** 2)):.3f}, after {np.sqrt(np.mean(after ** 2)):.4f} '
          f'(max {np.abs(after).max():.4f}) over {len(after) / SR_OUT:.1f} s')
    assert np.sqrt(np.mean(before[-SR_OUT // 2:] ** 2)) > 0.15  # it was playing
    assert np.abs(after).max() < 0.01


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

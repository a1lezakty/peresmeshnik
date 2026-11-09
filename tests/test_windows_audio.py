"""Windows call sound: the WASAPI process loopback helper and its Python wrapper, the mute spike (session volume to
0: the loopback keeps the tone, the device output goes silent), and the headphones through WASAPI and their
reopening. A virtual audio device (VB-Cable) stands in for the speakers/headphones, since the runner has none: its
"CABLE Input" is the output, its "CABLE Output" the recording side (what the output played).

PERESMESHNIK_LOOPBACK: the built helper (CI sets it). PERESMESHNIK_REQUIRE_WINDOWS=1 (CI): a missing helper or
virtual device fails the test instead of skipping it, so a green job means the tests ran.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows')
SR = 16000
CABLE_OUT = 'CABLE Output (VB-Audio Virtual Cable)'  # the input side: what the output ("CABLE Input") played
CABLE_IN = 'CABLE Input (VB-Audio Virtual Cable)'    # the default output in CI (the "speakers")
REQUIRE = os.environ.get('PERESMESHNIK_REQUIRE_WINDOWS') == '1'
TONE = os.path.join(os.path.dirname(__file__), 'windows', 'tone.py')


def need(ok: bool, why: str) -> None:
    if not ok:
        (pytest.fail if REQUIRE else pytest.skip)(why)


def helper_built() -> bool:
    try:
        from peresmeshnik.audio.capture_windows import helper
        return bool(helper())
    except RuntimeError:
        return False


def has_cable() -> bool:
    import sounddevice as sd
    names = {d['name'] for d in sd.query_devices()}
    return any(CABLE_OUT in n for n in names) and any(CABLE_IN in n for n in names)


@pytest.fixture
def helper_ok():
    need(helper_built(), 'needs the loopback helper: PERESMESHNIK_LOOPBACK or native/windows/loopback/build (build.cmd)')


@pytest.fixture
def cable():
    need(has_cable(), 'needs VB-Cable (CABLE Input / CABLE Output)')


def peak_hz(x: np.ndarray, sr: int) -> float:
    x = x[:sr]
    return float(np.abs(np.fft.rfft(x * np.hanning(len(x)))).argmax() * sr / len(x))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def mag_at(x: np.ndarray, hz: float, sr: int) -> float:
    x = x[:sr]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    b = int(round(hz * len(x) / sr))
    return float(spec[b]) if 0 <= b < len(spec) else 0.0


def until(cond, s: float = 5.0) -> bool:
    end = time.monotonic() + s
    while not (ok := bool(cond())) and time.monotonic() < end:
        time.sleep(0.05)
    return ok


class Frames:
    def __init__(self):
        self.got, self.lock = [], threading.Lock()

    def __call__(self, pcm):
        with self.lock:
            self.got.append(pcm.copy())

    def audio(self) -> np.ndarray:
        with self.lock:
            return np.concatenate(self.got) if self.got else np.zeros(0, np.float32)


def play_tone(device: str | None = None, hz: float = 440.0) -> subprocess.Popen:
    args = [sys.executable, TONE, device or '', str(hz)]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    line = p.stdout.readline()  # the "tone ... on ..." line: it is playing now (else the player died: show why)
    if not line:
        err = p.stderr.read().decode(errors='replace')
        raise AssertionError(f'the tone player did not start: {err}')
    return p


def set_default_output(name: str) -> None:
    """Best-effort: make `name` the default output if a tool is present (SoundVolumeView / nircmd). The tests name the
    device explicitly (loopback by pid, output by name), so none of them depends on the system default; VB-Cable's
    installer makes CABLE Input the default on its own, and CI no longer sets it."""
    for tool, args in (('SoundVolumeView.exe', ['/SetDefault', name, '0']), ('nircmd.exe', ['setdefaultsounddevice', name])):
        if subprocess.run(['where', tool], capture_output=True).returncode == 0:
            subprocess.run([tool, *args], capture_output=True)
            return


# ---- device-independent: the helper's capability check ------------------------------------------

def test_check_says_loopback_is_supported(helper_ok):
    from peresmeshnik.audio import capture_windows
    code, why = capture_windows.check()
    assert code == 0, why
    assert capture_windows.path() == 'loopback'


def test_check_output_shape(helper_ok):
    from peresmeshnik.audio.capture_windows import helper
    out = subprocess.run([helper(), 'check'], capture_output=True, text=True).stdout
    assert out.startswith('windows=') and 'loopback=yes' in out


def test_helper_runs_without_a_console_window(helper_ok):
    """The app is a windowed .exe and spawns the helper with CREATE_NO_WINDOW, so no console flashes and the
    long-lived tap/watch/mute hold no window the user could close. With the flag the child owns no console at all
    (GetConsoleWindow() == NULL): the helper reports console=0."""
    from peresmeshnik.audio.capture_windows import CREATE_NO_WINDOW, helper
    out = subprocess.run([helper(), 'console'], capture_output=True, text=True, creationflags=CREATE_NO_WINDOW).stdout
    assert 'console=0' in out, out


# ---- the process loopback (needs a device to play into) -----------------------------------------

@pytest.mark.usefixtures('helper_ok', 'cable')
def test_loopback_captures_only_the_target_process():
    from peresmeshnik.audio.capture_windows import ProcessLoopback
    set_default_output(CABLE_IN)
    player = play_tone(CABLE_IN, 440.0)
    other = play_tone(CABLE_IN, 880.0)  # a different process on the same device: must NOT leak into player's loopback
    frames = Frames()
    cap = ProcessLoopback(pid=player.pid)
    cap.start(frames)
    try:
        assert until(lambda: rms(frames.audio()) > 0.05, 6), 'the loopback heard no tone'
        audio = frames.audio()
        assert abs(peak_hz(audio, SR) - 440) < 10, peak_hz(audio, SR)
        assert mag_at(audio, 880, SR) < 0.25 * mag_at(audio, 440, SR), 'the other process leaked into the loopback'
    finally:
        cap.stop()
        for p in (player, other):
            p.kill()
            p.wait()


def session_volume(pid: int) -> float:
    from peresmeshnik.audio.capture_windows import helper
    got = subprocess.run([helper(), 'volume', '--pid', str(pid), '--tree'], capture_output=True, text=True).stdout
    vols = [json.loads(line)['volume'] for line in got.splitlines() if line.startswith('{')]
    return max(vols) if vols else -1.0


@pytest.mark.usefixtures('helper_ok', 'cable')
def test_mute_spike_zeroes_the_session_but_keeps_the_loopback():
    """Variant A: the process loopback takes the sound BEFORE the audio session volume. Setting the session volume to
    0 makes the call app contribute silence to the device mix (a Windows guarantee), while the loopback still has the
    tone. The runner has no real recording of the virtual cable, so the output silence is read back from the session
    volume; hearing the muted speakers on real hardware is a USER-CHECKS line."""
    from peresmeshnik.audio.capture_windows import ProcessLoopback, helper
    set_default_output(CABLE_IN)
    player = play_tone(CABLE_IN, 440.0)
    frames = Frames()
    cap = ProcessLoopback(pid=player.pid)
    cap.start(frames)
    try:
        assert until(lambda: rms(frames.audio()) > 0.05, 10), 'the loopback heard no tone'
        assert session_volume(player.pid) > 0.5, 'the session had no (full) volume to start with'
        subprocess.run([helper(), 'volume', '--pid', str(player.pid), '--tree', '--set', '0'], capture_output=True, check=True)
        assert session_volume(player.pid) < 0.01, 'the session volume was not set to 0'  # the output mix now gets silence
        frames.got.clear()
        assert until(lambda: rms(frames.audio()) > 0.05, 5), 'the loopback lost the tone after muting'  # taken before the volume
    finally:
        subprocess.run([helper(), 'volume', '--pid', str(player.pid), '--tree', '--set', '1'], capture_output=True)
        cap.stop()
        player.kill()
        player.wait()


@pytest.mark.usefixtures('helper_ok', 'cable')
def test_mute_helper_mutes_a_session_that_appears_after_it_starts():
    """The persistent `mute` helper mutes sessions created AFTER it starts (a call that begins, a new browser tab):
    OnSessionCreated in the target's tree -> volume 0. It targets the test's own tree; a tone player started after it
    is a child, so its new session is muted. Restores on stdin close."""
    from peresmeshnik.audio.capture_windows import CREATE_NO_WINDOW, helper
    set_default_output(CABLE_IN)
    muter = subprocess.Popen([helper(), 'mute', '--pid', str(os.getpid()), '--tree'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    player = None
    try:
        time.sleep(1.0)  # let the helper register its session notifications
        player = play_tone(CABLE_IN, 440.0)  # a new session, after the helper started
        deadline = time.monotonic() + 8
        vol = -1.0
        while time.monotonic() < deadline:
            vol = session_volume(player.pid)
            if 0 <= vol < 0.01:
                break
            time.sleep(0.4)
        assert 0 <= vol < 0.01, f'the new session was not muted: {vol}'
    finally:
        muter.stdin.close()  # restores every muted session and exits
        try:
            muter.wait(5)
        except subprocess.TimeoutExpired:
            muter.kill()
        if player:
            player.kill()
            player.wait()


def in_tree(session_pid: int, root_pid: int) -> bool:
    """`session_pid` is `root_pid` or a descendant of it (the audio session often sits in a child process: a
    browser's audio service, or the venv python trampoline's child on the runner)."""
    import psutil
    try:
        return session_pid == root_pid or root_pid in {a.pid for a in psutil.Process(session_pid).parents()}
    except psutil.Error:
        return False


@pytest.mark.usefixtures('helper_ok', 'cable')
def test_list_shows_the_playing_process():
    from peresmeshnik.audio.capture_windows import sessions
    set_default_output(CABLE_IN)
    player = play_tone(CABLE_IN, 440.0)
    try:
        assert until(lambda: any(in_tree(s['pid'], player.pid) and s['playing'] for s in sessions()), 8), \
            f'no playing session in the tree of {player.pid}: {sessions()}'
    finally:
        player.kill()
        player.wait()


# ---- the headphones through WASAPI (needs a device) ---------------------------------------------

@pytest.mark.usefixtures('cable')
def test_headphones_play_without_error():
    """The WASAPI output opens on the virtual cable, takes audio and drains it, without the device ending by itself.
    (The runner cannot record the cable back; that the translation is heard is a USER-CHECKS line.)"""
    from peresmeshnik.audio.out import WasapiOut
    ended = []
    out = WasapiOut(SR, CABLE_IN, on_end=ended.append)
    try:
        assert out.rate > 0 and out.device.startswith('CABLE Input')
        out.play((0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32), SR)
        assert until(lambda: out.queued_s() < 0.05, 3), f'the audio did not drain: {out.queued_s():.2f} s left'
        assert out.alive and not ended, ended
    finally:
        out.stop()


def _pwsh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', cmd], capture_output=True, text=True)


def cable_input_instance() -> str:
    r = _pwsh("(Get-PnpDevice -Class AudioEndpoint | Where-Object FriendlyName -like '*CABLE Input*' "
              "| Select-Object -First 1).InstanceId")
    return r.stdout.strip()


@pytest.mark.usefixtures('cable')
def test_headphones_reopen_when_the_device_goes(monkeypatch):
    """A device change refreshes PortAudio (its list is cached at init) before deciding; a device gone from the
    refreshed list cannot be reopened, so the output reports it and the owner reopens. Real path: disable the CABLE
    Input endpoint (Disable-PnpDevice), re-enabled after. If the runner has no admin PnP control, the gone device is
    simulated by hiding CABLE Input from PortAudio's query instead."""
    import sounddevice as sd

    from peresmeshnik.audio.out import WasapiOut
    ended = []
    out = WasapiOut(SR, CABLE_IN, on_end=ended.append)
    inst = cable_input_instance()
    disabled = bool(inst) and _pwsh(f"Disable-PnpDevice -InstanceId '{inst}' -Confirm:$false").returncode == 0
    try:
        if disabled:  # real unplug: the helper's watch fires on the state change, the output reopens
            assert until(lambda: bool(ended), 25), 'the output did not report the disabled endpoint'
        else:  # no PnP control on this runner: simulate the gone device at PortAudio's query
            real_query = sd.query_devices
            monkeypatch.setattr(sd, 'query_devices',
                                lambda device=None, **k: ([d for d in real_query() if not d['name'].startswith('CABLE Input')]
                                                          if device is None else real_query(device, **k)))
            out._on_device_event()
            assert until(lambda: bool(ended), 5), 'the output did not report the device going away'
        assert not out.alive
    finally:
        monkeypatch.undo()
        if disabled:
            _pwsh(f"Enable-PnpDevice -InstanceId '{inst}' -Confirm:$false")
            until(lambda: has_cable(), 25)  # back for any later test
        out.stop()


@pytest.mark.usefixtures('cable')
def test_headphones_follow_the_default_when_it_moves(monkeypatch):
    """Opened as the system default, the output follows the default when it moves to another device (out.py's
    _was_default branch): after the refresh the default is no longer ours -> reported -> the owner reopens on the new
    default. The runner has one render device, so the moved default is simulated at default_output()."""
    from peresmeshnik.audio import coreaudio, winaudio
    from peresmeshnik.audio.out import WasapiOut
    ended = []
    out = WasapiOut(SR, CABLE_IN, on_end=ended.append)
    out._was_default = True  # as if opened as the default, to exercise the follow-the-default branch
    try:
        monkeypatch.setattr(coreaudio, 'refresh_portaudio', lambda *a, **k: None)  # keep our stream; test the branch only
        monkeypatch.setattr(winaudio, 'output_names', lambda: {out.device, 'Another Output'})  # ours still present
        monkeypatch.setattr(winaudio, 'default_output', lambda: 'Another Output')  # the default moved away
        out._on_device_event()
        assert until(lambda: bool(ended), 4), 'the output did not follow the moved default'
        assert not out.alive
    finally:
        monkeypatch.undo()
        out.stop()


# ---- app.py wiring of them -> me on Windows ----------------------------------------------------

def test_app_wires_them_me_on_windows(helper_ok):
    """The building blocks app.py wires for them -> me on Windows: the loopback is the 'them' input, WasapiOut is the
    output class, and the mute spike + its restore are in place. The full installed-app path is release.yml
    (test_app_windows.py); the file -> them -> me CPU pipeline is the work-gate (test_pipeline_cpu.py)."""
    from peresmeshnik.audio import capture_windows, out
    assert out.Output is out.WasapiOut  # _output('them') -> Headphones on WasapiOut
    assert capture_windows.path() == 'loopback'  # _input('them') guards on this
    tap = capture_windows.ProcessLoopback(on_end=lambda why: None)  # the 'them' input app.py returns
    for m in ('start', 'retarget', 'stop'):
        assert callable(getattr(tap, m)), m
    for fn in ('mute_call', 'restore_call'):  # _route calls these (mute spike + restore on off/crash)
        assert callable(getattr(capture_windows, fn)), fn


# ---- round 2: no console window everywhere, microphone in the PortAudio registry ----------------

def test_every_helper_spawn_suppresses_the_console_window(monkeypatch):
    """Every place that starts the helper or the Piper runner passes CREATE_NO_WINDOW, so a windowed app flashes no
    console: capture_windows._run and the tap/mute Popens, winaudio's watch, and the Piper runner."""
    import io
    import subprocess as sp

    from peresmeshnik.audio import capture_windows as cw
    from peresmeshnik.audio import winaudio
    from peresmeshnik.local import tts
    seen = []

    class FakeProc:
        def __init__(self, *a, **k):
            seen.append(k.get('creationflags', 0))
            self.stdin, self.stdout, self.stderr = io.BytesIO(), io.BytesIO(b''), io.BytesIO(b'')
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self, *a, **k):
            return 0

        def kill(self):
            pass

    def fake_run(*a, **k):
        seen.append(k.get('creationflags', 0))
        return type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()

    monkeypatch.setattr(sp, 'Popen', FakeProc)
    monkeypatch.setattr(cw.subprocess, 'run', fake_run)
    monkeypatch.setattr(cw, 'helper', lambda: 'helper.exe')
    monkeypatch.setattr(cw, 'tree_root', lambda app: 123)
    monkeypatch.setattr(tts, 'runner_cmd', lambda: ['runner.exe'])
    monkeypatch.setattr(tts, 'voice_path', lambda v: 'voice.onnx')

    cw._run(['check'])                       # subprocess.run
    for spawn in (lambda: winaudio.DeviceWatch(lambda: None),   # watch Popen
                  lambda: cw.mute_call('X.exe'),                # mute Popen
                  lambda: cw.ProcessLoopback(pid=1).start(lambda pcm: None),  # tap Popen
                  lambda: tts.PiperTts('piper-ru_RU-dmitri-medium')):         # piper Popen
        try:
            spawn()
        except Exception:
            pass
    assert seen, 'no spawns recorded'
    assert all(f & subprocess.CREATE_NO_WINDOW for f in seen), seen


def test_microphone_is_held_in_the_portaudio_registry_on_windows(monkeypatch):
    """N1: the me -> them microphone stream is registered in the shared PortAudio holder registry on Windows (as on
    macOS), so a device refresh suspends/resumes it instead of freeing a live, unmanaged stream."""
    from peresmeshnik.audio import coreaudio
    from peresmeshnik.audio.mic import Mic
    m = Mic(device='whatever', sample_rate=SR)
    monkeypatch.setattr(Mic, '_open', lambda self, on_frames: setattr(self, '_stream', object()))  # no real device
    m.start(lambda pcm: None)
    try:
        assert m in coreaudio._holders, 'the microphone is not in the PortAudio registry (a refresh would free it)'
    finally:
        coreaudio.release(m)

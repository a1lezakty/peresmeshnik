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


class RawTap:
    """The helper's raw loopback of `pid` (48 kHz mono float32, no resampling), read all the time; take(s) is what
    came in the next s seconds."""

    def __init__(self, pid: int):
        import re

        from peresmeshnik.audio.capture_windows import CREATE_NO_WINDOW, helper
        self.p = subprocess.Popen([helper(), 'tap', '--pid', str(pid)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        self.rate = int(re.search(rb'rate=(\d+)', self.p.stdout.readline())[1])
        self.got, self.lock = [], threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while b := self.p.stdout.read(4096):
            with self.lock:
                self.got.append(np.frombuffer(b, '<f4'))

    def take(self, s: float) -> np.ndarray:
        with self.lock:
            self.got.clear()
        time.sleep(s)
        with self.lock:
            return np.concatenate(self.got) if self.got else np.zeros(0, np.float32)

    def close(self):
        self.p.stdin.close()
        self.p.wait(5)


def set_volume(pid: int, level: float) -> None:
    from peresmeshnik.audio.capture_windows import helper
    subprocess.run([helper(), 'volume', '--pid', str(pid), '--tree', '--set', f'{level:.4f}'], capture_output=True, check=True)
    time.sleep(0.6)  # the loopback's buffer (up to 200 ms) and the pipe hold sound from before the change


def lined_up(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The part of `ref` (a periodic source, longer than a period plus x) that `x` is: by the best integer lag."""
    n = len(ref) + len(x)
    c = np.fft.irfft(np.fft.rfft(ref, n) * np.conj(np.fft.rfft(x, n)), n)[:len(ref) - len(x) + 1]
    lag = int(np.argmax(c))
    return ref[lag:lag + len(x)]


def snr_db(ref: np.ndarray, x: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(ref ** 2) / max(float(np.sum((x - ref) ** 2)), 1e-30)))


SPEECH = os.path.join(os.path.dirname(__file__), 'data', 'call-en.ogg')


@pytest.mark.usefixtures('helper_ok', 'cable')
@pytest.mark.parametrize('source', ['tone', 'speech'])
def test_ducked_session_brought_back_up_matches_full_volume(source):
    """H-11 (stage 7): the process loopback is taken AFTER the session volume (at 0 it is silence), so the call app
    is ducked to DUCK, not muted, and the loopback brought back up by 1/DUCK. Brought back up, the sound matches the
    one at volume 1 within -40 dB (SNR >= 40 dB, the level error included) at every level 0.002..0.01: no threshold
    or step near DUCK. The speech loop and the tone are periodic, so the captures line up sample for sample."""
    from peresmeshnik.audio.capture_windows import DUCK
    player = play_tone(CABLE_IN, 440.0 if source == 'tone' else SPEECH)
    tap = None
    rows = []
    try:
        tap = RawTap(player.pid)
        set_volume(player.pid, 1.0)
        ref = tap.take(4.5)
        assert rms(ref) > 0.05, 'the loopback heard nothing at volume 1'
        x = tap.take(1.5)[: tap.rate]
        rows.append((1.0, rms(x) / rms(ref), snr_db(lined_up(ref, x), x)))  # the method's own floor
        for level in (0.1, 0.01, 0.007, DUCK, 0.003, 0.002):
            set_volume(player.pid, level)
            x = tap.take(1.5)[: tap.rate] / level
            rows.append((level, rms(x) / rms(ref), snr_db(lined_up(ref, x), x)))
    finally:
        if tap:
            tap.close()
        set_volume(player.pid, 1.0)
        player.kill()
        player.wait()
    print(f'\n{source}: level -> brought back up / volume 1, SNR dB: ' +
          '; '.join(f'{lv:g} -> {ratio:.4f}, {snr:.1f}' for lv, ratio, snr in rows))
    bad = [(lv, round(snr, 1)) for lv, ratio, snr in rows if lv <= 0.01 and snr < 40]
    assert not bad, f'below 40 dB after bringing it back up: {bad}'


@pytest.mark.usefixtures('helper_ok', 'cable')
def test_mute_helper_ducks_and_the_loopback_keeps_the_sound(monkeypatch):
    """The stage-4 spike, measured after the buffer from before the change is gone (it measured that buffer, so it
    passed though a session at 0 gives the loopback silence): the mute helper ducks the session to DUCK, and the
    ProcessLoopback brought back up (_gain, as mute_call sets it) has the tone at its full-volume level."""
    from peresmeshnik.audio import capture_windows as cw
    player = play_tone(CABLE_IN, 440.0)
    frames, muter = Frames(), None
    cap = cw.ProcessLoopback(pid=player.pid)
    cap.start(frames)
    try:
        assert until(lambda: player_sessions(cw.helper(), player), 8), 'the player has no session'
        set_volume(player.pid, 1.0)
        frames.got.clear()
        time.sleep(1.5)
        full = rms(frames.audio())
        assert full > 0.1, 'the loopback heard no tone'
        muter = subprocess.Popen([cw.helper(), 'mute', '--pid', str(player.pid), '--tree', '--level', f'{cw.DUCK:.4f}'],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 creationflags=cw.CREATE_NO_WINDOW)
        assert json.loads(muter.stdout.readline())['event'] == 'muted'
        monkeypatch.setattr(cw, '_gain', 1 / cw.DUCK)
        vols = [x['volume'] for x in player_sessions(cw.helper(), player)]
        assert vols and all(abs(v - cw.DUCK) < 1e-3 for v in vols), vols  # what the speakers get: -46 dB
        time.sleep(0.6)
        frames.got.clear()
        time.sleep(1.5)
        ducked = rms(frames.audio())
        set_volume(player.pid, 1.0)  # FIXES 1: raised by hand (or by the app) while translating: not 200 times into a clip
        frames.got.clear()
        time.sleep(1.5)
        raised = rms(frames.audio())
        print(f'\nloopback rms at volume 1 {full:.4f}, ducked to {cw.DUCK} and brought back up {ducked:.4f}, '
              f'raised to 1 meanwhile {raised:.4f}')
        assert abs(20 * np.log10(ducked / full)) < 0.5, (full, ducked)
        assert abs(20 * np.log10(raised / full)) < 0.5, ('a session raised to 1 was brought up into a clip', full, raised)
    finally:
        if muter:
            muter.stdin.close()
            muter.wait(5)
        cap.stop()
        set_volume(player.pid, 1.0)
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
        muter.stdin.close()  # mute-only: it stops muting and exits, the app restores
        try:
            muter.wait(5)
        except subprocess.TimeoutExpired:
            muter.kill()
        if player:  # else Windows keeps python.exe at 0 here and the next tone player starts silent
            subprocess.run([helper(), 'volume', '--pid', str(player.pid), '--tree', '--set', '1'], capture_output=True)
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


# ---- the helper's ends and the volume given back (stage 7: H-06, M-21, L-14) ---------------------------------

def player_sessions(helper_path: str, player) -> list[dict]:
    """The tone player's sessions (its tree): [{'pid', 'device', 'volume'}]."""
    got = subprocess.run([helper_path, 'volume', '--pid', str(player.pid), '--tree'], capture_output=True, text=True).stdout
    return [json.loads(line) for line in got.splitlines() if line.startswith('{')]


@pytest.mark.usefixtures('helper_ok', 'cable')
def test_tap_says_not_running_when_the_call_app_closes():
    """S2-L1: the call app closed while translating is "<app> is not running" (exit 3: the side is only un-muted),
    not a helper that died (exit 0 -> a restart, a crash report and "restarted after an error")."""
    from peresmeshnik.audio.capture_windows import ProcessLoopback, helper
    player = play_tone(CABLE_IN, 440.0)
    assert until(lambda: player_sessions(helper(), player), 8), 'the player has no session'
    subprocess.run([helper(), 'volume', '--pid', str(player.pid), '--tree', '--set', '1'], capture_output=True)
    ended, frames = [], Frames()
    cap = ProcessLoopback(pid=player.pid, on_end=ended.append)
    cap.start(frames)
    try:
        assert until(lambda: rms(frames.audio()) > 0.05, 10), 'the loopback heard no tone'
        player.kill()
        player.wait()
        assert until(lambda: bool(ended), 10), 'the loopback did not end with the app'
        assert ended == [f'{player.pid} is not running'], ended
    finally:
        cap.stop()


@pytest.mark.usefixtures('helper_ok', 'cable')
def test_volume_and_mute_name_the_device():
    """S2-07: Windows keeps an app's volume per device, so the app saves where it muted and what it gave back."""
    from peresmeshnik.audio.capture_windows import CREATE_NO_WINDOW, helper
    player = play_tone(CABLE_IN, 440.0)
    muter = None
    try:
        assert until(lambda: player_sessions(helper(), player), 8), 'the player has no session'
        devices = {x['device'] for x in player_sessions(helper(), player)}
        assert all(d.startswith('{') for d in devices), devices  # endpoint ids: {0.0.0.00000000}.{guid}
        muter = subprocess.Popen([helper(), 'mute', '--pid', str(player.pid), '--tree'], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        ev = json.loads(muter.stdout.readline())
        assert ev['event'] == 'muted' and ev['device'] in devices, ev
    finally:
        if muter:
            muter.stdin.close()
            muter.wait(5)
        subprocess.run([helper(), 'volume', '--pid', str(player.pid), '--tree', '--set', '1'], capture_output=True)
        player.kill()
        player.wait()


def keeper(helper_path: str, exe: str, level: float, devices=(), seconds: int = 60) -> tuple[subprocess.Popen, list]:
    """A `keep` helper (no stdin, as the app starts it: it outlives the app) and the list its "restored" lines go to."""
    from peresmeshnik.audio.capture_windows import CREATE_NO_WINDOW
    args = [helper_path, 'keep', '--exe', exe, '--level', str(level), '--for', str(seconds)]
    for d in devices:
        args += ['--device', d]
    proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=CREATE_NO_WINDOW)
    events = []
    threading.Thread(target=lambda: [events.append(json.loads(x)) for x in proc.stdout if x.startswith(b'{')],
                     daemon=True).start()
    return proc, events


def end(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(5)


@pytest.mark.usefixtures('helper_ok', 'cable')
def test_keep_gives_a_session_at_0_its_level_back():
    """F1-02: the call app was muted and could not get its level back (it closed first). `keep` finds its session at
    0 (Windows starts it at the persisted 0) and sets the saved level; the line says so."""
    import psutil

    from peresmeshnik.audio.capture_windows import helper
    player = play_tone(CABLE_IN, 440.0)
    proc = None
    try:
        assert until(lambda: player_sessions(helper(), player), 8), 'the player has no session'
        subprocess.run([helper(), 'volume', '--pid', str(player.pid), '--tree', '--set', '0'], capture_output=True, check=True)
        device = player_sessions(helper(), player)[0]['device']
        exe = psutil.Process(player_sessions(helper(), player)[0]['pid']).name()
        proc, events = keeper(helper(), exe, 0.7, [device])
        mine = lambda: [e for e in events if in_tree(e['pid'], player.pid)]  # noqa: E731
        assert until(mine, 8), f'no "restored" for the player: {events}'
        assert mine()[0]['device'] == device and mine()[0]['was'] < 0.01, mine()
        vols = [x['volume'] for x in player_sessions(helper(), player)]
        assert vols and all(abs(v - 0.7) < 0.02 for v in vols), vols
    finally:
        if proc:
            end(proc)
        subprocess.run([helper(), 'volume', '--pid', str(player.pid), '--tree', '--set', '1'], capture_output=True)
        player.kill()
        player.wait()


def ps_player(wav) -> subprocess.Popen:
    """A call app that is not python.exe (this test process and its tone players are): Windows PowerShell 5 playing a
    wav on the default output (CABLE Input on the runner) in a loop. The CI step runs in pwsh, not powershell.exe."""
    return subprocess.Popen(['powershell.exe', '-NoProfile', '-Command',
                             f"$p = New-Object Media.SoundPlayer '{wav}'; $p.PlayLooping(); Start-Sleep 60"])


@pytest.mark.usefixtures('helper_ok', 'cable')
def test_keep_catches_the_apps_next_session_and_ends_when_done(tmp_path):
    """H-06: the call app left ducked starts again (Peresmeshnik may be gone by then: the helper reads no stdin). Its
    new session is caught as it appears (the runner opens it at the persisted 0: printed), given its level back, and
    the helper ends by itself once that device is done."""
    import soundfile as sf

    from peresmeshnik.audio.capture_windows import helper
    wav = tmp_path / 'tone.wav'
    sf.write(wav, (0.3 * np.sin(2 * np.pi * 440 * np.arange(48000 * 2) / 48000)).astype(np.float32), 48000)
    first = ps_player(wav)
    player = proc = None
    try:
        assert until(lambda: player_sessions(helper(), first), 15), 'powershell.exe has no session'
        device = player_sessions(helper(), first)[0]['device']
        subprocess.run([helper(), 'volume', '--pid', str(first.pid), '--tree', '--set', '0'], capture_output=True, check=True)
        first.kill()
        first.wait()
        proc, events = keeper(helper(), 'powershell.exe', 0.6, [device])
        time.sleep(1.5)
        assert proc.poll() is None, 'the helper ended with nothing done (no stdin is no reason to go)'
        player = ps_player(wav)  # a new session, after the helper started
        assert until(lambda: [e for e in events if e['pid'] == player.pid], 15), f'no "restored" for the new session: {events}'
        ev = next(e for e in events if e['pid'] == player.pid)
        print(f'\nthe new session opened at {ev["was"]:.2f} (a persisted 0: {ev["was"] < 0.01})')
        assert until(lambda: proc.poll() is not None, 5), 'the helper did not end once its device was done'
        vols = [x['volume'] for x in player_sessions(helper(), player)]
        assert vols and min(vols) >= 0.01, vols
    finally:
        if proc:
            end(proc)
        for p in (first, player):
            if p:
                subprocess.run([helper(), 'volume', '--pid', str(p.pid), '--tree', '--set', '1'], capture_output=True)
                p.kill()
                p.wait()


@pytest.mark.skipif(os.environ.get('PERESMESHNIK_AUDIOSRV_RESTART') != '1',
                    reason='restarts Windows Audio: its own CI step (PERESMESHNIK_AUDIOSRV_RESTART=1)')
@pytest.mark.usefixtures('helper_ok', 'cable')
def test_tap_ends_when_windows_audio_restarts():
    """S2-06: Windows Audio restarted under a running capture (a driver update, a crash of the service). A process
    loopback is not told (no error, no event: before the fix the helper waited forever, the side silent). The helper
    watches the service, waits for it to run again and exits 7; ProcessLoopback starts it again and hears the app
    (or, if Windows is not ready yet, the side ends with a reason and restarts). The app here is this process, its
    PortAudio opened anew after the restart, as a call app's audio would be."""
    import sounddevice as sd

    from peresmeshnik.audio import winaudio
    from peresmeshnik.audio.capture_windows import ProcessLoopback

    def tone_stream():
        dev = winaudio.device_index(CABLE_IN)
        sr = int(sd.query_devices(dev)['default_samplerate'])
        phase = [0.0]

        def cb(out, frames, t, status):
            k = phase[0] + 2 * np.pi * 440 / sr * np.arange(frames)
            out[:] = (0.3 * np.sin(k)).astype(np.float32)[:, None]
            phase[0] = float(k[-1] + 2 * np.pi * 440 / sr) % (2 * np.pi)
        s = sd.OutputStream(device=dev, channels=1, samplerate=sr, dtype='float32', callback=cb)
        s.start()
        return s

    frames, ended = Frames(), []
    cap = ProcessLoopback(pid=os.getpid(), on_end=ended.append)
    cap.start(frames)
    stream = None
    try:
        stream = tone_stream()
        assert until(lambda: rms(frames.audio()[-SR:]) > 0.05, 8), 'the loopback heard no tone before the restart'
        stream.close()
        subprocess.run(['powershell', '-NoProfile', '-Command', 'Restart-Service Audiosrv -Force'], check=True, timeout=90)
        sd._terminate()
        sd._initialize()
        assert until(has_cable, 30), 'VB-Cable did not come back'
        stream = tone_stream()
        frames.got.clear()
        ok = until(lambda: bool(ended) or rms(frames.audio()[-SR:]) > 0.05, 45)
        print(f'\nafter Windows Audio restarted: restarts {len(cap._restarts)}, ended {ended}, '
              f'heard again {rms(frames.audio()[-SR:]) > 0.05}; {cap.status!r}')
        assert ok, f'the capture stays silent 45 s after Windows Audio restarted: {cap.status!r}'
    finally:
        if stream:
            stream.close()
        cap.stop()


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


@pytest.mark.usefixtures('cable')
def test_headphones_reopen_when_the_device_goes():
    """A device change refreshes PortAudio (its list is cached at init) before deciding; a device gone from the
    refreshed list cannot be reopened, so the output reports it and the owner reopens. Real path: the CABLE Input
    endpoint disabled as the Sound control panel does (WASAPI sees DEVICE_STATE_DISABLED), re-enabled after.
    Disable-PnpDevice on the endpoint's node is no stand-in: WASAPI keeps it active (test_virtual_mic_windows)."""
    from test_virtual_mic_windows import endpoint_enabled

    from peresmeshnik.audio import windows
    from peresmeshnik.audio.out import WasapiOut
    uid = next(d['uid'] for d in windows.devices() if d['name'] == CABLE_IN)
    ended = []
    out = WasapiOut(SR, CABLE_IN, on_end=ended.append)
    endpoint_enabled(uid, False)
    try:
        assert until(lambda: bool(ended), 25), 'the output did not report the disabled endpoint'
        assert not out.alive
    finally:
        endpoint_enabled(uid, True)
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


def test_microphone_is_no_portaudio_stream_on_windows(monkeypatch):
    """The me -> them microphone goes through the WASAPI helper (S2, audio/windows.py), not PortAudio: a device refresh
    (refresh_portaudio) has nothing of it to free, so it is not in the holder registry."""
    import types

    from peresmeshnik.audio import coreaudio, windows
    from peresmeshnik.audio.mic import Mic
    spawned = []
    proc = types.SimpleNamespace(stdout=types.SimpleNamespace(read=lambda n: b''), stderr=iter(()), wait=lambda *a: 0,
                                 returncode=0, terminate=lambda: None)
    monkeypatch.setattr(windows, 'devices', lambda: [])
    monkeypatch.setattr(windows, 'spawn', lambda *a, **k: spawned.append(a) or proc)
    m = Mic(device='whatever', sample_rate=SR)
    m.start(lambda pcm: None)
    try:
        assert spawned and spawned[0][:3] == ('record', '--from', 'whatever'), spawned
        assert m not in coreaudio._holders
    finally:
        m.stop()

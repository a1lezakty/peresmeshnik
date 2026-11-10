"""The mute-spike state (capture_windows): one original level per call app in a file, with the audio devices where
the app was muted and not yet given back (Windows keeps an app's volume per device). The mute helper is mute-only;
Python owns the restore. The saved non-zero level is never overwritten by a near-zero (a crash leaves the app at 0
and Windows persists it), other apps keep their own entries, restore_call clears an entry only when it was applied on
every device it was muted on, and a `keep` helper puts the rest back on the app's next session there. mute_call
restores a leftover first. Pure Python (the helpers are fakes): runs on every OS.
"""
import io
import json
import subprocess
import threading
import time

import pytest

from peresmeshnik.audio import capture_windows as cw


@pytest.fixture
def state(tmp_path, monkeypatch):
    f = tmp_path / 'muted-session.json'
    monkeypatch.setattr(cw, '_mute_state', lambda: f)
    monkeypatch.setattr(cw, '_muter', None, raising=False)  # no live helper: the file path
    monkeypatch.setattr(cw, '_keepers', {}, raising=False)
    monkeypatch.setattr(cw, '_keep', lambda app, e: None, raising=False)  # no keep helper unless a test wants one
    return f


def saved(state, keeper: bool = False) -> dict:
    """The file as the app reads it (under its lock: the helpers' pumps write it meanwhile; Windows: no replace while
    it is open); without the keep helpers' pids unless asked."""
    with cw._state_lock:
        d = json.loads(state.read_text())
    return d if keeper else {app: {k: v for k, v in e.items() if k != 'keeper'} for app, e in d.items()}


def test_remember_never_overwrites_a_nonzero_with_near_zero(state):
    cw._remember('Zoom.exe', 0.8)
    cw._remember('Zoom.exe', 0.0)   # a crash-reopened session reads 0: must not clobber 0.8
    cw._remember('Zoom.exe', 0.005)
    assert saved(state)['Zoom.exe']['level'] == 0.8


def test_remember_keeps_each_app_separately_and_the_louder_original(state):
    cw._remember('Zoom.exe', 0.5)
    cw._remember('Zoom.exe', 0.8)      # keep the max
    cw._remember('Discord.exe', 0.4)
    assert {app: e['level'] for app, e in saved(state).items()} == {'Zoom.exe': 0.8, 'Discord.exe': 0.4}


def test_remember_records_each_device_once(state):
    cw._remember('Zoom.exe', 0.8, 'HEADSET')
    cw._remember('Zoom.exe', 0.0, 'SPEAKERS')  # a session already at 0 there: still to be given back there
    cw._remember('Zoom.exe', 0.6, 'HEADSET')
    assert saved(state) == {'Zoom.exe': {'level': 0.8, 'devices': ['HEADSET', 'SPEAKERS']}}


def test_restore_puts_each_level_back_and_clears(state, monkeypatch):
    state.write_text(json.dumps({'Zoom.exe': {'level': 0.8, 'devices': ['A']}, 'Discord.exe': {'level': 0.5, 'devices': ['A']}}))
    calls = []
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: (calls.append((app, round(level, 3))), ['A'])[1])
    cw.restore_call()
    assert sorted(calls) == [('Discord.exe', 0.5), ('Zoom.exe', 0.8)]
    assert not state.exists()


def test_restore_reads_the_file_of_an_older_version(state, monkeypatch):
    """0.3.0 saved the bare level per app (no devices): any session given back clears it."""
    state.write_text(json.dumps({'Zoom.exe': 0.8}))
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: ['A'])
    cw.restore_call()
    assert not state.exists()


def test_restore_keeps_the_entry_when_the_app_is_not_running(state, monkeypatch):
    state.write_text(json.dumps({'Zoom.exe': {'level': 0.8, 'devices': ['A']}}))
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: [])  # app gone: nothing applied
    cw.restore_call()
    assert saved(state) == {'Zoom.exe': {'level': 0.8, 'devices': ['A']}}, 'the level was lost though it could not be restored'


def test_restore_keeps_a_device_that_is_not_there_now(state, monkeypatch):
    """S2-07: muted on the headset, the headset went flat, Zoom moved to the speakers. Translation off gives the
    speakers back; the headset's 0 stays in the file (Windows keeps it per device) instead of being forgotten."""
    cw._remember('Zoom.exe', 0.8, 'HEADSET')
    cw._remember('Zoom.exe', 0.8, 'SPEAKERS')
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: ['SPEAKERS'])
    cw.restore_call()
    assert saved(state) == {'Zoom.exe': {'level': 0.8, 'devices': ['HEADSET']}}


def test_restore_of_one_app_leaves_another_apps_leftover(state, monkeypatch):
    state.write_text(json.dumps({'Zoom.exe': 0.8, 'Discord.exe': 0.5}))
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: ['A'] if app == 'Zoom.exe' else [])  # only Zoom runs
    cw.restore_call()
    assert saved(state) == {'Discord.exe': {'level': 0.5, 'devices': []}}


def test_mute_restores_a_leftover_before_muting(state, monkeypatch):
    """A crash left Zoom muted (file has its 0.8). Translating again restores the true level first; it does not start
    the helper on a 0 session and save that 0."""
    state.write_text(json.dumps({'Zoom.exe': 0.8}))
    restored = []
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: (restored.append((app, round(level, 3))), ['A'])[1])
    monkeypatch.setattr(cw, 'tree_root', lambda app: None)  # the app is not running: mute_call returns after restore
    cw.mute_call('Zoom.exe')
    assert restored == [('Zoom.exe', 0.8)]
    assert not state.exists()


# ---- the keep helper: a level not given back yet goes back on the app's next session ------------------------------

class FakeKeeper:
    """A `keep` helper: its args are recorded; it says "restored" for each device in `events` once started; ended by
    kill() (it reads no stdin: it outlives the app)."""
    spawned: list = []
    events: list = []

    def __init__(self, args, **kw):
        FakeKeeper.spawned.append(self)
        self.args, self.kw = args, kw
        self.pid = 4000 + len(FakeKeeper.spawned)
        self.closed = threading.Event()
        self.stdin = None
        self.stdout = self._lines(list(FakeKeeper.events))
        self.returncode = None

    def _lines(self, events):
        for d in events:
            yield json.dumps({'event': 'restored', 'pid': 7, 'device': d}).encode() + b'\n'
        self.closed.wait(10)  # a live helper: its stdout ends when it is stopped

    def wait(self, *a):
        return 0

    def kill(self):
        self.closed.set()


@pytest.fixture
def keeper(state, monkeypatch):
    monkeypatch.undo()  # the real _keep (the fixture above stubbed it), the same state file
    monkeypatch.setattr(cw, '_mute_state', lambda: state)
    monkeypatch.setattr(cw, '_muter', None, raising=False)
    monkeypatch.setattr(cw, '_keepers', {}, raising=False)
    monkeypatch.setattr(cw, 'helper', lambda: 'helper.exe')
    monkeypatch.setattr(cw.subprocess, 'Popen', FakeKeeper)
    FakeKeeper.spawned, FakeKeeper.events = [], []
    return FakeKeeper


def until(cond, s=3.0):
    end = time.monotonic() + s
    while not cond() and time.monotonic() < end:
        time.sleep(0.02)
    return cond()


def test_a_level_not_given_back_is_watched_for_and_cleared_when_the_app_plays_again(state, keeper, monkeypatch):
    """F1-02: the call ended and the call app closed before translation was turned off. Nothing to set now; a keep
    helper waits for the app's next session on that device (Windows opens it at the persisted 0) and puts 0.8 back;
    its "restored" clears the entry and the helper is stopped."""
    cw._remember('Zoom.exe', 0.8, 'HEADSET')
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: [])  # Zoom is not running
    keeper.events = ['HEADSET']
    cw.restore_call()
    assert len(keeper.spawned) == 1
    args = keeper.spawned[0].args
    assert args[1:] == ['keep', '--exe', 'Zoom.exe', '--level', '0.8000', '--for', str(cw.KEEP_S), '--device', 'HEADSET'], args
    assert keeper.spawned[0].kw['stdin'] is subprocess.DEVNULL, 'a keep helper on our stdin dies with the app'
    assert until(lambda: not state.exists()), 'the entry was not cleared by the helper saying "restored"'
    assert until(keeper.spawned[0].closed.is_set), 'the keep helper was not stopped after it was done'


def test_the_keep_helper_stays_for_a_device_still_to_come(state, keeper, monkeypatch):
    cw._remember('Zoom.exe', 0.8, 'HEADSET')
    cw._remember('Zoom.exe', 0.8, 'SPEAKERS')
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: [])
    keeper.events = ['SPEAKERS']
    cw.restore_call()
    assert until(lambda: saved(state) == {'Zoom.exe': {'level': 0.8, 'devices': ['HEADSET']}})
    time.sleep(0.1)
    assert not keeper.spawned[0].closed.is_set(), 'stopped though the headset is still to be given back'


def test_muting_the_app_again_stops_its_keep_helper(state, keeper, monkeypatch):
    """A keep helper of the app being muted would fight the mute (0 -> level on each new session)."""
    cw._remember('Zoom.exe', 0.8, 'HEADSET')
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: [])
    cw.restore_call()
    assert len(keeper.spawned) == 1 and not keeper.spawned[0].closed.is_set()
    monkeypatch.setattr(cw, 'tree_root', lambda app: None)  # mute_call then returns after its restore
    cw.mute_call('Zoom.exe')
    assert keeper.spawned[0].closed.is_set()
    assert len(keeper.spawned) == 1, 'a new keep helper was started for the app being muted'


def test_the_keep_helper_outlives_the_app_and_the_next_run_ends_it(state, keeper, monkeypatch):
    """H-06: the user quits before the call app plays again: the keep helper (no stdin) goes on and gives the level
    back without Peresmeshnik. Its pid is in the file; the next run ends it first (it would undo the next duck), only
    if that pid is still our helper (a pid can be reused)."""
    cw._remember('Zoom.exe', 0.8, 'HEADSET')
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: [])
    cw.restore_call()
    pid = keeper.spawned[0].pid
    assert saved(state, keeper=True)['Zoom.exe']['keeper'] == pid
    killed = []

    class Proc:
        def __init__(self, p):
            self.p = p

        def name(self):
            return 'peresmeshnik-loopback.exe' if self.p in (pid, 78) else 'notepad.exe'

        def cmdline(self):  # 78: our helper of the new run (watch), on a pid the keep helper had
            return ['peresmeshnik-loopback.exe', 'keep' if self.p == pid else 'watch']

        def kill(self):
            killed.append(self.p)
    import psutil
    monkeypatch.setattr(psutil, 'Process', Proc)
    monkeypatch.setattr(cw, '_keepers', {})  # a new run: it knows the helper only from the file
    monkeypatch.setattr(cw, 'tree_root', lambda app: None)  # the app is not running: mute_call stops after its restore
    cw.mute_call('Zoom.exe')
    assert killed == [pid]
    with cw._state_lock:
        state.write_text(json.dumps({'Zoom.exe': {'level': 0.8, 'devices': ['HEADSET'], 'keeper': 77}}))
    cw.restore_call()
    assert killed == [pid], 'killed a reused pid that is not our helper'
    with cw._state_lock:  # circle 2, 4: the pid is now another helper of ours (watch, tap): not a keep helper
        state.write_text(json.dumps({'Zoom.exe': {'level': 0.8, 'devices': ['HEADSET'], 'keeper': 78}}))
    cw.restore_call()
    assert killed == [pid], 'killed our own watch helper on a reused pid'


class FakeMuter:
    """The `mute` helper: prints `lines`, then waits for its stdin to close."""
    def __init__(self, lines):
        self.closed = threading.Event()
        self.stdin = self
        self.stdout = self._out(lines)

    def _out(self, lines):
        yield from (json.dumps(x).encode() + b'\n' for x in lines)
        self.closed.wait(10)

    def close(self):
        self.closed.set()

    def wait(self, *a):
        return 0

    def kill(self):
        self.closed.set()


@pytest.mark.parametrize('lines, gain', [
    ([{'event': 'muted', 'pid': 1, 'level': 0.8, 'device': 'A'}], 1 / cw.DUCK),  # ducked: brought back up
    ([{'event': 'failed', 'pid': 1, 'device': 'A', 'hr': 5}], 1.0),           # SetMasterVolume failed: as it is
    ([], 1.0)])                                                                 # nothing ducked yet
def test_the_loopback_is_brought_back_up_only_once_a_session_is_ducked(state, monkeypatch, lines, gain):
    """FIXES 1: the gain follows the helper's word that a session is ducked, not its start; translation off (and
    the helper's end) takes it back."""
    muter = FakeMuter(lines)
    monkeypatch.setattr(cw, 'helper', lambda: 'helper.exe')
    monkeypatch.setattr(cw, 'tree_root', lambda app: 123)
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: [])
    monkeypatch.setattr(cw.subprocess, 'Popen', lambda *a, **k: muter)
    monkeypatch.setattr(cw, '_gain', 1.0)
    cw.mute_call('Zoom.exe')
    time.sleep(0.2)
    assert cw._gain == gain
    cw.restore_call()
    assert cw._gain == 1.0


def test_a_mute_helper_that_dies_leaves_the_gain_until_translation_off(state, monkeypatch):
    """Circle 2, 2: the helper died after ducking: it gives nothing back, the sessions stay at DUCK, so the sound is
    still brought up (else the call is heard nowhere); translation off takes the gain back and restores the levels."""
    muter = FakeMuter([{'event': 'muted', 'pid': 1, 'level': 0.8, 'device': 'A'}])
    muter.poll = lambda: 1
    monkeypatch.setattr(cw, 'helper', lambda: 'helper.exe')
    monkeypatch.setattr(cw, 'tree_root', lambda app: 123)
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: [])
    monkeypatch.setattr(cw.subprocess, 'Popen', lambda *a, **k: muter)
    cw.mute_call('Zoom.exe')
    assert until(lambda: cw._gain != 1.0)
    muter.closed.set()  # its stdout ends: it died
    time.sleep(0.2)
    assert cw._gain == 1 / cw.DUCK, 'the sessions are still ducked: the sound must still be brought up'
    restored = []
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: restored.append((app, level)) or ['A'])
    cw.restore_call()
    assert cw._gain == 1.0 and restored == [('Zoom.exe', 0.8)]


def test_only_a_block_as_quiet_as_a_ducked_one_is_brought_up(monkeypatch):
    """FIXES 1: a block louder than 2*DUCK is not ducked (the duck not taken yet, raised by hand, a helper gone):
    passed as it is, not clipped."""
    import numpy as np
    monkeypatch.setattr(cw, '_gain', 1 / cw.DUCK)
    sr = 48000
    t = np.arange(sr // 4) / sr
    quiet = (cw.DUCK * 0.8 * np.sin(2 * np.pi * 440 * t)).astype('<f4')
    loud = (0.3 * np.sin(2 * np.pi * 440 * t)).astype('<f4')
    head = b'peresmeshnik-loopback 1 rate=48000 channels=1 format=f32le\n'
    proc = type('P', (), {'stdout': io.BytesIO(head + quiet.tobytes() + loud.tobytes()), 'wait': lambda *a: 0,
                          'returncode': 0})()
    got = []
    cap = cw.ProcessLoopback(pid=1, on_end=lambda why: None)
    cap._proc = proc
    cap._read(proc, got.append)
    x = np.concatenate(got)
    n = len(x) // 2
    first, second = float(np.abs(x[n // 4:n - n // 4]).max()), float(np.abs(x[n + n // 4:-n // 4]).max())
    assert 0.7 < first < 0.9, first   # 0.004 brought up to ~0.8
    assert 0.25 < second < 0.35, second  # 0.3 left as it is, not 60 clipped to 1


def blocks_out(blocks: list) -> list:
    """ProcessLoopback._read over 48 kHz blocks of 32 ms (as the helper sends them): the peak of each block out."""
    import numpy as np
    head = b'peresmeshnik-loopback 1 rate=48000 channels=1 format=f32le\n'
    proc = type('P', (), {'stdout': io.BytesIO(head + b''.join(np.asarray(b, '<f4').tobytes() for b in blocks)),
                          'wait': lambda *a: 0, 'returncode': 0})()
    got = []
    cap = cw.ProcessLoopback(pid=1, on_end=lambda why: None)
    cap._proc = proc
    cap._read(proc, got.append)
    return [float(np.abs(x).max(initial=0.0)) for x in got]


def test_a_raised_session_is_not_brought_up_in_its_pauses(monkeypatch):
    """Circle 2, 1: once the duck took, a block too loud to be ducked means the session was raised (by hand, by the
    app): the quiet blocks after it (pauses, breath, call noise) stay as they are for HOLD_S, not 200 times up; after
    HOLD_S of quiet the sound is taken for ducked again. The loud sound from before the duck (the first blocks after
    the gain came on) holds nothing."""
    import numpy as np
    monkeypatch.setattr(cw, '_gain', 1 / cw.DUCK)
    t = np.arange(1536) / 48000
    quiet, loud = 0.003 * np.sin(2 * np.pi * 500 * t), 0.3 * np.sin(2 * np.pi * 500 * t)
    hold = round(cw.HOLD_S * 1000 / 32)
    seq = [loud] * 6 + [quiet] * 6 + [loud] * 6 + [quiet] * (hold + 10)
    out = blocks_out(seq)
    assert all(p < 0.31 for p in out[1:6]), out[:6]  # before the duck: as it is (not clipped), no hold after it
    assert all(0.5 < p < 0.7 for p in out[8:12]), out[6:12]  # ducked: 0.003 brought up to 0.6
    after = out[18:]
    assert all(p < 0.004 for p in after[2:hold - 1]), after[:hold]  # a pause of a raised session: as it is
    assert all(0.5 < p < 0.7 for p in after[hold + 2:]), after[hold:]  # HOLD_S of quiet: ducked again


def test_a_restore_gives_back_only_sessions_still_down(monkeypatch):
    """Circle 2, 5: a session already up (a keep helper gave it back after the app quit, the user set it since) keeps
    its volume: the helper sets only those below 0.01."""
    seen = []
    monkeypatch.setattr(cw, 'tree_root', lambda app: 123)
    monkeypatch.setattr(cw, '_run', lambda args: seen.append(args) or subprocess.CompletedProcess(args, 0, '', ''))
    cw.set_session_volume('Telegram.exe', 0.6)
    assert seen == [['volume', '--pid', '123', '--tree', '--set', '0.6000', '--below', '0.01']]


def test_an_app_still_quiet_is_said_in_the_tray(state, monkeypatch):
    """Circle 2, 3: translation off (or the next start) with a level not given back: restore_call names the app (not
    the one about to be muted), and the tray says where its volume is and that it comes back."""
    from types import SimpleNamespace

    from peresmeshnik import app as appmod
    cw._remember('Telegram.exe', 0.8, 'HEADSET')
    cw._remember('Zoom.exe', 0.7, 'HEADSET')
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: [])
    assert cw.restore_call(muting='Zoom.exe') == ['Telegram.exe']
    note = cw.quiet_note(['Telegram.exe'])
    assert note.startswith('Telegram stays quiet') and '14 days' in note and 'Volume mixer' in note
    assert cw.quiet_note([]) == ''
    states = []
    for left in (['Telegram.exe'], []):
        monkeypatch.setattr(cw, 'restore_call', lambda left=left: left)
        appmod.Controller._call_volume_back(SimpleNamespace(on_state=states.append))
    assert states == [f'notice: {note}']

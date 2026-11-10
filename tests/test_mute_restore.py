"""The mute-spike state (capture_windows): one original level per call app in a file, with the audio devices where
the app was muted and not yet given back (Windows keeps an app's volume per device). The mute helper is mute-only;
Python owns the restore. The saved non-zero level is never overwritten by a near-zero (a crash leaves the app at 0
and Windows persists it), other apps keep their own entries, restore_call clears an entry only when it was applied on
every device it was muted on, and a `keep` helper puts the rest back on the app's next session there. mute_call
restores a leftover first. Pure Python (the helpers are fakes): runs on every OS.
"""
import json
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


def saved(state) -> dict:
    with cw._state_lock:  # as the app reads it: the helpers' pumps write it meanwhile (Windows: no replace while open)
        return json.loads(state.read_text())


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
    """A `keep` helper: its args are recorded; it says "restored" for each device in `events` once started."""
    spawned: list = []
    events: list = []

    def __init__(self, args, **kw):
        FakeKeeper.spawned.append(self)
        self.args = args
        self.closed = threading.Event()
        self.stdin = self
        self.stdout = self._lines(list(FakeKeeper.events))
        self.returncode = None

    def _lines(self, events):
        for d in events:
            yield json.dumps({'event': 'restored', 'pid': 7, 'device': d}).encode() + b'\n'
        self.closed.wait(10)  # a live helper: its stdout ends when it is stopped

    def close(self):  # stdin closed: the helper's sign to stop
        self.closed.set()

    def wait(self, *a):
        return 0

    def kill(self):
        pass


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
    assert args[1:] == ['keep', '--exe', 'Zoom.exe', '--level', '0.8000', '--device', 'HEADSET'], args
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

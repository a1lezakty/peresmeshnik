"""The mute-spike state (capture_windows): one original level per call app in a file. The mute helper is mute-only;
Python owns the restore. The saved non-zero level is never overwritten by a near-zero (a crash leaves the app at 0
and Windows persists it), other apps keep their own entries, restore_call clears an entry only when it is actually
applied, and mute_call restores a leftover first. Pure Python (the helper is not spawned): runs on every OS.
"""
import json

import pytest

from peresmeshnik.audio import capture_windows as cw


@pytest.fixture
def state(tmp_path, monkeypatch):
    f = tmp_path / 'muted-session.json'
    monkeypatch.setattr(cw, '_mute_state', lambda: f)
    monkeypatch.setattr(cw, '_muter', None, raising=False)  # no live helper: the file path
    return f


def test_remember_never_overwrites_a_nonzero_with_near_zero(state):
    cw._remember('Zoom.exe', 0.8)
    cw._remember('Zoom.exe', 0.0)   # a crash-reopened session reads 0: must not clobber 0.8
    cw._remember('Zoom.exe', 0.005)
    assert json.loads(state.read_text())['Zoom.exe'] == 0.8


def test_remember_keeps_each_app_separately_and_the_louder_original(state):
    cw._remember('Zoom.exe', 0.5)
    cw._remember('Zoom.exe', 0.8)      # keep the max
    cw._remember('Discord.exe', 0.4)
    assert json.loads(state.read_text()) == {'Zoom.exe': 0.8, 'Discord.exe': 0.4}


def test_restore_puts_each_level_back_and_clears(state, monkeypatch):
    state.write_text(json.dumps({'Zoom.exe': 0.8, 'Discord.exe': 0.5}))
    calls = []
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: (calls.append((app, round(level, 3))), True)[1])
    cw.restore_call()
    assert sorted(calls) == [('Discord.exe', 0.5), ('Zoom.exe', 0.8)]
    assert not state.exists()


def test_restore_keeps_the_entry_when_the_app_is_not_running(state, monkeypatch):
    state.write_text(json.dumps({'Zoom.exe': 0.8}))
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: False)  # app gone: nothing applied
    cw.restore_call()
    assert json.loads(state.read_text()) == {'Zoom.exe': 0.8}, 'the level was lost though it could not be restored'


def test_restore_of_one_app_leaves_another_apps_leftover(state, monkeypatch):
    state.write_text(json.dumps({'Zoom.exe': 0.8, 'Discord.exe': 0.5}))
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: app == 'Zoom.exe')  # only Zoom is running
    cw.restore_call()
    assert json.loads(state.read_text()) == {'Discord.exe': 0.5}


def test_mute_restores_a_leftover_before_muting(state, monkeypatch):
    """A crash left Zoom muted (file has its 0.8). Translating again restores the true level first; it does not start
    the helper on a 0 session and save that 0."""
    state.write_text(json.dumps({'Zoom.exe': 0.8}))
    restored = []
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: (restored.append((app, round(level, 3))), True)[1])
    monkeypatch.setattr(cw, 'tree_root', lambda app: None)  # the app is not running: mute_call returns after restore
    cw.mute_call('Zoom.exe')
    assert restored == [('Zoom.exe', 0.8)]
    assert not state.exists()

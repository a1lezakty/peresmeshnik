"""The mute-spike crash recovery (capture_windows): the original session levels are saved to a file so a crash does
not leave the call app muted; restore_call puts them back, keeps the file when the call app is not running yet (so a
later restore can still apply it), and clears it on success. mute_call restores a leftover first, so it never saves
the 0 a previous crash left as the "original". Pure Python (the helper is not spawned): runs on every OS.
"""
import json

import pytest

from peresmeshnik.audio import capture_windows as cw


@pytest.fixture
def state(tmp_path, monkeypatch):
    f = tmp_path / 'muted-session.json'
    monkeypatch.setattr(cw, '_mute_state', lambda: f)
    monkeypatch.setattr(cw, '_muter', None, raising=False)  # no live helper: the file (crash-recovery) path
    return f


def test_restore_puts_the_saved_level_back_and_clears_the_file(state, monkeypatch):
    state.write_text(json.dumps({'app': 'Zoom.exe', 'levels': {'100': 0.8, '200': 0.6}}))
    calls = []
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: (calls.append((app, level)), True)[1])
    cw.restore_call()
    assert calls == [('Zoom.exe', 0.8)]  # the max original goes back on the tree (crash-recovery path)
    assert not state.exists()


def test_restore_keeps_the_file_when_the_app_is_not_running(state, monkeypatch):
    state.write_text(json.dumps({'app': 'Zoom.exe', 'levels': {'100': 0.8}}))
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: False)  # app gone: nothing to set
    cw.restore_call()
    assert state.exists(), 'the file was deleted though the restore did not apply: the level would be lost'


def test_restore_without_a_file_is_a_noop(state, monkeypatch):
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: True)
    cw.restore_call()  # no file, no live helper
    assert not state.exists()


def test_mute_restores_a_leftover_before_muting(state, monkeypatch):
    """A crash left Zoom muted (file has its 0.8 original). Translating again must restore the true level first, not
    start the helper on a session already at 0 and save that 0 as the original."""
    state.write_text(json.dumps({'app': 'Zoom.exe', 'levels': {'100': 0.8}}))
    restored = []
    monkeypatch.setattr(cw, 'set_session_volume', lambda app, level: (restored.append((app, level)), True)[1])
    monkeypatch.setattr(cw, 'tree_root', lambda app: None)  # the app is not running now: mute_call returns after restore
    cw.mute_call('Zoom.exe')
    assert restored == [('Zoom.exe', 0.8)], 'the leftover was not restored before muting'
    assert not state.exists()

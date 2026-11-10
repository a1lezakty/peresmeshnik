"""The Python side of the call-sound helpers on macOS and Windows, with the helpers faked: runs on every OS.

- The tray's "Call app" menu never offers our own helpers (peresmeshnik-audio plays the translation into the virtual
  cable: picking it would mute or tap our own voice) — F1-L08.
- A capture helper that asks to be started again (exit 7) and cannot be (Popen fails, the app was replaced on disk)
  ends the side with a reason instead of killing the capture thread silently — S2-L3.
- Windows device events come in bursts (a headset brings several state lines and a default change): one PortAudio
  refresh per burst, none for a microphone default change — S2-05.
"""
import io
import json
import threading
import time
import types

import pytest

from peresmeshnik.audio import capture_macos, capture_windows, winaudio


def test_windows_call_apps_leave_our_helpers_out(monkeypatch):
    monkeypatch.setattr(capture_windows, 'sessions', lambda: [
        {'pid': 1, 'name': 'peresmeshnik-audio.exe', 'playing': True},
        {'pid': 2, 'name': 'Peresmeshnik-Loopback.exe', 'playing': True},
        {'pid': 3, 'name': 'Peresmeshnik.exe', 'playing': True},
        {'pid': 4, 'name': 'Zoom.exe', 'playing': True}])
    assert [p['key'] for p in capture_windows.players()] == ['Zoom.exe']


def test_macos_call_apps_leave_our_helpers_out(monkeypatch):
    monkeypatch.setattr(capture_macos, 'processes', lambda: [
        {'pid': 11, 'bundle': 'org.peresmeshnik.audio', 'name': 'peresmeshnik-audio', 'playing': True},
        {'pid': 12, 'bundle': 'io.github.a1lezakty.peresmeshnik', 'name': 'Peresmeshnik', 'playing': True},
        {'pid': 13, 'bundle': 'us.zoom.xos', 'name': 'zoom.us', 'playing': True}])
    assert [p['key'] for p in capture_macos.players()] == ['us.zoom.xos']


def restart_proc(head: bytes) -> types.SimpleNamespace:
    """A helper that printed `head` and exited 7 ("start me again")."""
    return types.SimpleNamespace(stdout=io.BytesIO(head), stderr=io.BytesIO(b''), wait=lambda *a: 7, returncode=7)


@pytest.mark.parametrize('kind', ['windows', 'macos'])
def test_a_restart_that_cannot_spawn_ends_the_side_with_a_reason(kind, monkeypatch):
    mod, cls = (capture_windows, capture_windows.ProcessLoopback) if kind == 'windows' else (capture_macos, capture_macos.ProcessTap)
    ended = []
    cap = cls(pid=4242, on_end=ended.append)
    cap._on_frames = on_frames = lambda pcm: None
    cap._proc = proc = restart_proc(b'no header\n')

    def gone():
        raise RuntimeError('no helper at /old/place')
    monkeypatch.setattr(mod, 'helper', gone)
    cap._read(proc, on_frames)  # the capture thread's body: it must not raise out of the thread
    assert ended and 'no helper at /old/place' in ended[0], ended
    assert cap._proc is None


class FakeWatch:
    """The `watch` helper: prints `lines` (bursts separated by a pause of `gap` s), then waits for stdin to close."""

    def __init__(self, bursts, gap):
        self._bursts, self._gap = bursts, gap
        self.stdin = types.SimpleNamespace(close=lambda: None)
        self.stdout = self._lines()

    def _lines(self):
        for i, burst in enumerate(self._bursts):
            if i:
                time.sleep(self._gap)
            for ev in burst:
                yield json.dumps(ev).encode() + b'\n'

    def wait(self, *a):
        return 0

    def kill(self):
        pass


def watch(monkeypatch, bursts, gap=0.0) -> list:
    calls = []
    monkeypatch.setattr(capture_windows, 'helper', lambda: 'helper.exe')
    monkeypatch.setattr(winaudio.subprocess, 'Popen', lambda *a, **k: FakeWatch(bursts, gap))
    done = threading.Event()
    w = winaudio.DeviceWatch(lambda: calls.append(time.monotonic()))
    done.wait(1.5)
    w.close()
    return calls


def test_a_burst_of_device_events_refreshes_once(monkeypatch):
    burst = [{'event': 'state', 'id': 'BT-stereo', 'state': 1}, {'event': 'state', 'id': 'BT-handsfree', 'state': 1},
             {'event': 'default', 'flow': 'render', 'id': 'BT-stereo'}, {'event': 'state', 'id': 'BT-mic', 'state': 1}]
    assert len(watch(monkeypatch, [burst])) == 1


def test_a_microphone_default_change_is_no_output_event(monkeypatch):
    assert watch(monkeypatch, [[{'event': 'default', 'flow': 'capture', 'id': 'webcam-mic'}]]) == []


def test_two_bursts_apart_refresh_twice(monkeypatch):
    burst = [{'event': 'default', 'flow': 'render', 'id': 'X'}]
    assert len(watch(monkeypatch, [burst, burst], gap=0.8)) == 2

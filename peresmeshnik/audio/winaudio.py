"""Windows audio: the device list and default as Windows sees them now, and notifications when one changes.

sounddevice (PortAudio) reads the device list once, at init: a device plugged in later (a headset back) is not in
it. So the default and the rates come from here (sounddevice's WASAPI host API, which matches the system), and a
device change is heard through the native helper's `watch` (IMMNotificationClient), not by polling. The PortAudio
refresh machinery (close every stream, re-init, reopen) is shared with macOS (`coreaudio`): identical logic, so it is
reused rather than copied. `DeviceWatch` calls back when the default render device or a device's state changes.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
from collections.abc import Callable

# the PortAudio holder registry and refresh are platform-neutral (no Core Audio calls): reuse them
from peresmeshnik.audio.coreaudio import hold, portaudio_lock, refresh_portaudio, release  # noqa: F401

log = logging.getLogger('peresmeshnik.winaudio')
SETTLE_S = 0.3  # device events within this of the first are one change (a headset brings several at once)
BLOCK = 1024  # as macOS: one buffer size for every stream (room for a Python callback waiting for the GIL)


def _wasapi_hostapi() -> int | None:
    import sounddevice as sd
    for i, h in enumerate(sd.query_hostapis()):
        if 'WASAPI' in h['name']:
            return i
    return None


def default_output() -> str:
    """Name of the system's default output now ('' when there is none). Prefers the WASAPI default (it follows the
    Windows default device), else PortAudio's default output."""
    import sounddevice as sd
    with portaudio_lock:
        api = _wasapi_hostapi()
        if api is not None:
            d = sd.query_hostapis(api).get('default_output_device', -1)
            if d is not None and d >= 0:
                return sd.query_devices(d)['name']
        d = sd.default.device[1]
        return sd.query_devices(d)['name'] if d is not None and d >= 0 else ''


def outputs() -> list[dict]:
    """[{'name', 'rate'}] of the output devices (WASAPI where it lists them, else all), one per name."""
    import sounddevice as sd
    with portaudio_lock:
        api = _wasapi_hostapi()
        seen = {}
        for d in sd.query_devices():
            if d['max_output_channels'] > 0 and (api is None or d['hostapi'] == api):
                seen.setdefault(d['name'], {'name': d['name'], 'rate': int(d['default_samplerate'])})
        return list(seen.values())


def output_names() -> set[str]:
    return {d['name'] for d in outputs()}


def device_index(name: str, output: bool = True) -> int:
    """PortAudio's index of the device `name` on the WASAPI host API (a VB-Cable name is the same across MME /
    DirectSound / WASAPI, so the host API has to be pinned, else PortAudio picks one at random)."""
    import sounddevice as sd
    key = 'max_output_channels' if output else 'max_input_channels'
    with portaudio_lock:
        api = _wasapi_hostapi()
        devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['name'] == name and d[key] > 0 and (api is None or d['hostapi'] == api):
            return i
    from peresmeshnik.audio.out import NotInPortAudio
    raise NotInPortAudio(f'no {"output" if output else "input"} device {name!r} on WASAPI')


def rate(name: str) -> int:
    """The output device's sample rate (its WASAPI shared-mode mix rate). Streams open at it, so two streams of one
    device do not fight over the rate (the macOS lesson, and WASAPI shared mode has one mix rate per device)."""
    import sounddevice as sd
    with portaudio_lock:
        api = _wasapi_hostapi()
        for d in sd.query_devices():
            if d['name'] == name and d['max_output_channels'] > 0 and (api is None or d['hostapi'] == api):
                return int(d['default_samplerate'])
    raise RuntimeError(f'no output device {name!r}')


def stream_args(name: str) -> dict:
    """samplerate / blocksize for a sounddevice output stream on `name` (one buffer size, as macOS)."""
    return {'samplerate': rate(name), 'blocksize': BLOCK}


class DeviceWatch:
    """`on_change()` (on a thread of its own) when the default render device changes or a device's state does: the
    native helper's `watch` emits one JSON line per event (no polling). A burst (a headset: its stereo and hands-free
    endpoints and a new default) is one call, SETTLE_S after its first event, so the output is refreshed once, not
    once per line; a change of the default microphone is none of the output's business. close() ends it; after it no
    call comes."""

    def __init__(self, on_change: Callable[[], None]):
        from peresmeshnik.audio.capture_windows import CREATE_NO_WINDOW, helper
        self.on_change = on_change
        self._stopped = False
        self._pending = threading.Lock()  # held from a burst's first event until its call
        self._proc = subprocess.Popen([helper(), 'watch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        threading.Thread(target=self._read, name='winaudio-watch', daemon=True).start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            if self._stopped:
                return
            try:
                ev = json.loads(line.decode(errors='replace'))
            except ValueError:
                continue
            if ev.get('event') == 'state' or (ev.get('event') == 'default' and ev.get('flow') == 'render'):
                if self._pending.acquire(blocking=False):  # else a call is due already: this event is in it
                    threading.Timer(SETTLE_S, self._fire).start()

    def _fire(self) -> None:
        self._pending.release()  # an event from now on is a new change (the refresh below may not have seen it)
        if not self._stopped:
            self.on_change()

    def close(self) -> None:
        self._stopped = True
        if self._proc.stdin:
            try:
                self._proc.stdin.close()  # the helper ends when stdin closes
            except OSError:
                pass
        try:
            self._proc.wait(5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

"""Linux capture of the call sound: a null sink "Peresmeshnik Call", a recording of its monitor, and the
call app's playback streams moved into it while translation is on.

Works with PipeWire (pipewire-pulse) and PulseAudio. The reader threads block on the pipes from `parec`
and `pactl subscribe` and only hand on what they read; nothing polls.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from collections.abc import Callable

import numpy as np

from peresmeshnik.engines import SAMPLE_RATE

log = logging.getLogger('peresmeshnik.call')
CALL_SINK = 'peresmeshnik_call'
CLIENT = 'Peresmeshnik'  # application.name of our own streams (headphones, virtual mic): never moved
SETTLE_S = 1.0  # WirePlumber may still link a new stream to its default this long after our move: move it again
# found by themselves when no call app is chosen yet (matched in application.process.binary / application.name)
CALL_APPS = ('telegram', 'zoom', 'discord', 'skype', 'teams', 'slack', 'webex', 'signal', 'element', 'jitsi', 'whatsapp', 'viber')
# one playback stream carries every tab: moving it moves YouTube in the next tab too, so never picked by itself
BROWSERS = ('chrome', 'chromium', 'firefox', 'brave', 'opera', 'vivaldi', 'edge', 'yandex', 'epiphany', 'librewolf', 'floorp',
            'zen', 'falkon', 'midori', 'qutebrowser')


def pactl(*args: str) -> str:
    return subprocess.run(['pactl', *args], check=True, capture_output=True, text=True, timeout=10).stdout


class PipeWireCapture:
    """`AudioIn` from the monitor of `sink` (parec, 20 ms). `make_sink`: load the null sink on start() and
    unload it on stop() (the CLI); the app keeps its own sink for its whole run and passes False.
    `on_end(why)` runs once when parec ends by itself (the sound server restarted, the sink went away)."""

    def __init__(self, sink: str = CALL_SINK, description: str = 'Peresmeshnik Call', frame: int = 512,
                 make_sink: bool = True, on_end: Callable[[str], None] = lambda why: None):
        self.sink, self.description, self.frame, self.make_sink, self.on_end = sink, description, frame, make_sink, on_end
        self._module = self._proc = self._thread = None
        self._stopping = False

    def start(self, on_frames: Callable[[np.ndarray], None]) -> None:
        if self.make_sink:
            for line in pactl('list', 'short', 'modules').splitlines():  # a sink left by a crashed run goes first
                if 'module-null-sink' in line and f'sink_name={self.sink}' in line.split():
                    pactl('unload-module', line.split()[0])
            # priority 0: never preferred over a real output device as the default sink
            self._module = pactl('load-module', 'module-null-sink', f'sink_name={self.sink}',
                                 f'sink_properties=\'device.description="{self.description}"'
                                 ' priority.session=0 priority.driver=0\'').strip()
        self._stopping = False
        # dont-reconnect: PipeWire would move the recording to the next new device (e.g. the virtual mic's sink)
        self._proc = subprocess.Popen(['parec', f'--device={self.sink}.monitor', '--raw', '--format=float32le',
                                       f'--rate={SAMPLE_RATE}', '--channels=1', '--latency-msec=20',
                                       '--property=node.dont-reconnect=true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._read, args=(on_frames, self._proc), name='capture', daemon=True)
        self._thread.start()

    def _read(self, on_frames, proc):
        n = self.frame * 4
        while len(b := proc.stdout.read(n)) == n:
            on_frames(np.frombuffer(b, '<f4'))
        if not self._stopping:
            why = proc.stderr.read().decode(errors='replace').strip().splitlines()
            proc.wait()
            self.on_end(f'parec ended ({proc.returncode}){": " + why[-1] if why else ""}')

    def stop(self) -> None:
        """Stop recording and remove the sink (also when parec does not go quietly)."""
        self._stopping = True
        try:
            if self._proc:
                self._proc.terminate()
                try:
                    self._proc.wait(5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                self._thread.join(5)
                self._proc = None
        finally:
            if self._module:
                pactl('unload-module', self._module)
                self._module = None


def app_key(props: dict) -> str:
    """What a chosen call app is remembered by: the program (Electron apps all say "Chromium" as their name)."""
    return props.get('application.process.binary') or props.get('application.name') or ''


def is_browser(key: str) -> bool:
    return any(b in key.lower() for b in BROWSERS)


def players() -> list[dict]:
    """Playback streams of other apps, ours (the headphones, the virtual mic, loopbacks) left out:
    [{'index', 'key', 'name', 'sink' (name), 'corked'}]."""
    sinks = {str(s['index']): s['name'] for s in json.loads(pactl('-f', 'json', 'list', 'sinks') or '[]')}
    out = []
    for si in json.loads(pactl('-f', 'json', 'list', 'sink-inputs') or '[]'):
        p = si.get('properties', {})
        if si.get('owner_module') is not None or p.get('application.name') == CLIENT or not app_key(p):
            continue
        out.append({'index': str(si['index']), 'key': app_key(p), 'name': p.get('application.name') or app_key(p),
                    'sink': sinks.get(str(si['sink']), ''), 'corked': bool(si.get('corked'))})
    return out


def apps(streams: list[dict] | None = None) -> list[tuple[str, str]]:
    """[(key, label)] of the apps playing now, one per app; a browser's label says that all of it moves."""
    seen = {}
    for p in players() if streams is None else streams:
        seen.setdefault(p['key'], f'{p["name"]} (all its sound, every tab)' if is_browser(p['key']) else p['name'])
    return sorted(seen.items(), key=lambda kv: kv[1].lower())


def guess_call_app(streams: list[dict] | None = None) -> str:
    """The one known call app (not a browser) playing now, else ''."""
    keys = {k for k, _ in apps(streams) if not is_browser(k) and any(c in k.lower() for c in CALL_APPS)}
    return keys.pop() if len(keys) == 1 else ''


def default_sink() -> str:
    return pactl('get-default-sink').strip()


def send_back(index: str, sink: str) -> None:
    """Move a stream to `sink` if it exists, else to the default sink. Moving to the default sink also makes
    PipeWire / PulseAudio forget the stream's saved target (it follows the default again): WirePlumber
    remembers every move per app (restore-stream), so a stream left in our sink would go there next time."""
    names = {s['name'] for s in json.loads(pactl('-f', 'json', 'list', 'sinks') or '[]')}
    if sink not in names or sink.startswith('peresmeshnik_'):
        sink = default_sink()
    if sink.startswith('peresmeshnik_'):
        raise RuntimeError(f'the default output is our own {sink}: cannot give the call its sound back')
    subprocess.run(['pactl', 'move-sink-input', index, sink], capture_output=True, timeout=10)


def return_strays(sink: str = CALL_SINK) -> list[str]:
    """Streams of other apps sitting in our call sink while nothing moved them there (a killed run, or
    WirePlumber restoring an old move): back to the default output. Returns the apps moved."""
    moved = []
    for p in players():
        if p['sink'] == sink:
            send_back(p['index'], '')
            moved.append(p['name'])
    return moved


class CallRouter:
    """While translation is on: the call app's playback streams (`app` = `app_key`) go into the call sink, the ones
    playing now and every new one (the call reconnects, a second call); stop() sends each back where it was,
    or to the default output when that device is gone. New streams are seen from `pactl subscribe` (a blocked
    read of the sound server's events), not by polling. `on_move(name)` runs after each stream moved in."""

    def __init__(self, app: str, sink: str = CALL_SINK, on_move: Callable[[str], None] = lambda name: None):
        self.app, self.sink, self.on_move = app, sink, on_move
        self.moved: dict[str, str] = {}  # stream index -> the sink it came from ('' = unknown: the default)
        self._at: dict[str, float] = {}  # when it was moved
        self._lock = threading.Lock()  # the event thread and stop()
        self._proc = self._thread = None

    def start(self) -> None:
        # subscribe first, then move what plays now: a stream that appears in between is still seen
        self._proc = subprocess.Popen(['pactl', 'subscribe'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._events, args=(self._proc,), name='call-router', daemon=True)
        self._thread.start()
        self._move()

    def _events(self, proc) -> None:
        for line in proc.stdout:  # "Event 'new' on sink-input #123"; 'change' too: at 'new' the stream may not
            if line.startswith(("Event 'new' on sink-input #", "Event 'change' on sink-input #")):  # be listed yet
                index = line.rsplit('#', 1)[1].strip()
                if index in self.moved and time.monotonic() - self._at[index] > SETTLE_S:  # ours; later moves are the user's
                    continue
                try:
                    self._move(index)
                except Exception:
                    log.exception('moving a new stream of %s', self.app)

    def _move(self, index: str | None = None) -> None:
        with self._lock:
            if self._proc is None:  # stopped meanwhile
                return
            for p in players():
                if p['key'] != self.app or (index is not None and p['index'] != index):
                    continue
                again = p['index'] in self.moved
                if again and p['sink'] == self.sink:
                    continue
                # a new stream that WirePlumber linked to the default right after our move (a race in its policy:
                # 2 of 30 here) is moved again; it came from where it was the first time
                if p['sink'] != self.sink and subprocess.run(['pactl', 'move-sink-input', p['index'], self.sink],
                                                             capture_output=True, timeout=10).returncode:
                    continue  # gone meanwhile
                if not again:
                    self.moved[p['index']] = '' if p['sink'] == self.sink else p['sink']  # already there: a restored old move
                    self._at[p['index']] = time.monotonic()
                    self.on_move(p['name'])
                log.info('call sound: %s stream %s %s -> %s%s', p['name'], p['index'], p['sink'], self.sink, ' (again)' if again else '')

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            moved, self.moved = self.moved, {}
        if proc:
            proc.terminate()
            proc.wait(5)
            self._thread.join(5)
        here = {p['index'] for p in players() if p['sink'] == self.sink}  # gone streams and ones the user moved stay
        for index, sink in moved.items():
            if index in here:
                send_back(index, sink)

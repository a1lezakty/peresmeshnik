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

from peresmeshnik.audio import c_locale
from peresmeshnik.engines import SAMPLE_RATE

log = logging.getLogger('peresmeshnik.call')
CALL_SINK = 'peresmeshnik_call'
CLIENT = 'Peresmeshnik'  # application.name of our own streams (headphones, virtual mic): never moved
# never the default output in place of a real one (the system's sound would go silent into our sink). Negative,
# not 0: WirePlumber 0.4 counts a device without a priority as -1, so 0 wins over it (S2 measured; as the virtual mic)
LOW_PRIORITY = 'priority.session=-10 priority.driver=0'
SETTLE_S = 1.0  # WirePlumber may still link a new stream to its default this long after our move: move it again
# found by themselves when no call app is chosen yet (matched in application.process.binary / application.name)
# and in the bundle ids on macOS (ru.keepcoder.Telegram, us.zoom.xos, Cisco-Systems.Spark is Webex)
CALL_APPS = ('telegram', 'zoom', 'discord', 'skype', 'teams', 'slack', 'webex', 'cisco-systems.spark', 'signal', 'element',
             'jitsi', 'whatsapp', 'viber')
# one playback stream carries every tab: moving it moves YouTube in the next tab too, so never picked by itself
# WirePlumber's restore-stream remembers a stream's target under the first of these the stream has
RESTORE_KEYS = ('media.role', 'application.id', 'application.name', 'media.name', 'node.name')
BROWSERS = ('chrome', 'chromium', 'firefox', 'brave', 'opera', 'vivaldi', 'edge', 'yandex', 'epiphany', 'librewolf', 'floorp',
            'zen', 'falkon', 'midori', 'qutebrowser', 'webkit', 'thebrowser')  # macOS: com.apple.WebKit (Safari), Arc


def pactl(*args: str) -> str:
    return subprocess.run(['pactl', *args], check=True, capture_output=True, text=True, timeout=10, env=c_locale()).stdout


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
            self._module = pactl('load-module', 'module-null-sink', f'sink_name={self.sink}',
                                 f'sink_properties=\'device.description="{self.description}" {LOW_PRIORITY}\'').strip()
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
    [{'index', 'key', 'name', 'sink' (name), 'corked', 'restore' (WirePlumber's key for it, {property: value})}]."""
    sinks = {str(s['index']): s['name'] for s in json.loads(pactl('-f', 'json', 'list', 'sinks') or '[]')}
    out = []
    for si in json.loads(pactl('-f', 'json', 'list', 'sink-inputs') or '[]'):
        p = si.get('properties', {})
        if si.get('owner_module') is not None or p.get('application.name') == CLIENT or not app_key(p):
            continue
        out.append({'index': str(si['index']), 'key': app_key(p), 'name': p.get('application.name') or app_key(p),
                    'sink': sinks.get(str(si['sink']), ''), 'corked': bool(si.get('corked')),
                    'restore': next(({k: p[k]} for k in RESTORE_KEYS if p.get(k)), {})})
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


def sink_of(index: str) -> str | None:
    """The sink a stream plays into now ('' while it is not linked), None once it is gone."""
    sinks = {str(s['index']): s['name'] for s in json.loads(pactl('-f', 'json', 'list', 'sinks') or '[]')}
    si = next((si for si in json.loads(pactl('-f', 'json', 'list', 'sink-inputs') or '[]') if str(si['index']) == index), None)
    return None if si is None else sinks.get(str(si['sink']), '')


def move(index: str, sink: str, tries: int = 3) -> bool:
    """Move a stream into `sink` and see it there: True, or False when the stream (or the sink) is gone. WirePlumber
    may drop a move while pactl says yes (a stream or a sink a moment old: no event, the stream stays; 1 run of ~5
    of the PipeWire tests here), so a move not seen within 0.5 s goes again; raises if it never lands."""
    for _ in range(tries):
        if subprocess.run(['pactl', 'move-sink-input', index, sink], capture_output=True, timeout=10).returncode:
            return False
        for _ in range(10):
            now = sink_of(index)
            if now is None:
                return False
            if now == sink:
                return True
            time.sleep(0.05)
    raise RuntimeError(f'stream {index} never reached {sink}: the sound server dropped the move {tries} times')


def send_back(index: str, sink: str) -> None:
    """Move a stream to `sink` if it exists, else to the default sink. Moving to the default sink also makes
    PipeWire / PulseAudio forget the stream's saved target (it follows the default again): WirePlumber
    remembers every move per app (restore-stream), so a stream left in our sink would go there next time."""
    names = {s['name'] for s in json.loads(pactl('-f', 'json', 'list', 'sinks') or '[]')}
    if sink not in names or sink.startswith('peresmeshnik_'):
        sink = default_sink()
    if sink.startswith('peresmeshnik_'):
        raise RuntimeError(f'the default output is our own {sink}: cannot give the call its sound back')
    move(index, sink)


def return_strays(sink: str = CALL_SINK) -> list[str]:
    """Streams of other apps sitting in our call sink while nothing moved them there (a killed run, or
    WirePlumber restoring an old move): back to the default output. Returns the apps moved."""
    moved = []
    for p in players():
        if p['sink'] == sink:
            send_back(p['index'], '')
            moved.append(p['name'])
    return moved


def forget(restore: dict, sink: str = CALL_SINK) -> None:
    """WirePlumber keeps a moved stream's target after the stream ends (restore-stream, per `restore` key): the app's
    next stream (the next call, its ringtone) would go into our sink while translation is off. A silent stream under
    the same key, started in our sink and moved to the default output, makes it forget, as a stream given back does."""
    args = [f'--property={k}={v}' for k, v in restore.items()]
    probe = subprocess.Popen(['pacat', '--playback', '--raw', '--format=s16le', '--rate=8000', '--channels=1', f'--device={sink}',
                              '--stream-name=Peresmeshnik probe', '--property=peresmeshnik.probe=1', *args],
                             stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):  # ponytail: a bounded wait (2 s) for it to be listed, once per off; no event stream for one stream
            index = next((str(si['index']) for si in json.loads(pactl('-f', 'json', 'list', 'sink-inputs') or '[]')
                          if si.get('properties', {}).get('peresmeshnik.probe') == '1'), None)
            if index is not None:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f'cannot clear the remembered move of {restore}: the probe stream never appeared')
        target = default_sink()
        if target.startswith('peresmeshnik_'):
            raise RuntimeError(f'the default output is our own {target}')
        move(index, target)
        time.sleep(0.3)  # WirePlumber saves the new target from the move while the stream is still there
    finally:
        probe.kill()
        probe.wait()


class CallRouter:
    """While translation is on: the call app's playback streams (`app` = `app_key`) go into the call sink, the ones
    playing now and every new one (the call reconnects, a second call); stop() sends each back where it was,
    or to the default output when that device is gone. New streams are seen from `pactl subscribe` (a blocked
    read of the sound server's events), not by polling. `on_move(name)` runs after each stream moved in."""

    def __init__(self, app: str, sink: str = CALL_SINK, on_move: Callable[[str], None] = lambda name: None):
        self.app, self.sink, self.on_move = app, sink, on_move
        self.moved: dict[str, str] = {}  # stream index -> the sink it came from ('' = unknown: the default)
        self._at: dict[str, float] = {}  # when it was moved
        self._keys: dict[str, dict] = {}  # its restore key: forgotten at stop() when the stream is gone
        self._lock = threading.Lock()  # the event thread and stop()
        self._proc = self._thread = None

    def start(self) -> None:
        # subscribe first, then move what plays now: a stream that appears in between is still seen
        self._proc = subprocess.Popen(['pactl', 'subscribe'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                      env=c_locale())
        self._thread = threading.Thread(target=self._events, args=(self._proc,), name='call-router', daemon=True)
        self._thread.start()
        self._move()

    def _events(self, proc) -> None:
        for line in proc.stdout:  # "Event 'new' on sink-input #123"; 'change' too: at 'new' the stream may not
            if line.startswith(("Event 'new' on sink-input #", "Event 'change' on sink-input #")):  # be listed yet
                index = line.rsplit('#', 1)[1].strip()
                at = self._at.get(index)  # set by _move under the lock; a read is atomic
                if at is not None and time.monotonic() - at > SETTLE_S:  # ours; later moves are the user's
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
                if index is not None and p['index'] != index:
                    continue
                if p['key'] != self.app:
                    if p['sink'] == self.sink:  # WirePlumber restored our move under a shared name (Electron apps and the
                        send_back(p['index'], '')  # Chromium browser are all "Chromium"): not the call app, back it goes
                        log.info('call sound: %s stream %s is not %s: back to the default output', p['name'], p['index'], self.app)
                    continue
                again = p['index'] in self.moved
                if again and p['sink'] == self.sink:
                    continue
                # a new stream that WirePlumber linked to the default right after our move (a race in its policy:
                # 2 of 30 here) is moved again; it came from where it was the first time
                try:
                    if p['sink'] != self.sink and not move(p['index'], self.sink):
                        continue  # gone meanwhile
                except RuntimeError as e:
                    log.warning('call sound: %s', e)
                    continue
                if not again:
                    self._at[p['index']] = time.monotonic()
                    self._keys[p['index']] = p['restore']
                    self.moved[p['index']] = '' if p['sink'] == self.sink else p['sink']  # already there: a restored old move
                    self.on_move(p['name'])
                log.info('call sound: %s stream %s %s -> %s%s', p['name'], p['index'], p['sink'], self.sink, ' (again)' if again else '')

    def stop(self) -> None:
        """Every moved stream still in our sink goes back; for those gone (the call ended while translating) WirePlumber
        forgets the move. Each step on its own: one that fails does not keep the others; the errors are raised after."""
        with self._lock:
            proc, self._proc = self._proc, None
            moved, self.moved = self.moved, {}
            keys, self._keys = self._keys, {}
        if proc:
            proc.terminate()
        errors = []
        try:
            now = {p['index']: p for p in players()}
            back = set()
            for index, sink in moved.items():
                if index in now and now[index]['sink'] == self.sink:  # the ones the user moved elsewhere stay
                    try:
                        send_back(index, sink)
                        back.add(tuple(keys[index].items()))
                    except Exception as e:
                        errors.append(str(e))
            gone = {tuple(keys[i].items()) for i in moved if i not in now and keys.get(i)}
            for key in gone - back:  # a stream given back has already reset it (to where it came from)
                try:
                    forget(dict(key), self.sink)
                except Exception as e:
                    errors.append(str(e))
        finally:
            if proc:
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                self._thread.join(5)
        if errors:
            raise RuntimeError('; '.join(errors))

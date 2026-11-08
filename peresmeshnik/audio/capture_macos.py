"""macOS capture of the call sound ("them -> me").

macOS 14.4+: a Core Audio process tap of the call app, made by the helper `peresmeshnik-tap` (native/macos/tap): the app's
sound comes to us and is muted on the speakers while the tap runs (mutedWhenTapped), so only the translation is heard.
The tap exists only while translation is on: off, the call sounds as usual. The helper is $PERESMESHNIK_TAP (the .app
sets it), else native/macos/tap/build/peresmeshnik-tap of this checkout. The first tap asks for "System Audio Recording"
(macOS shows its dialog once); denied, the helper exits with its own code and the reason says where to allow it.

Older macOS (no process taps): the call app's speaker is BlackHole 16ch. The app records it while translating
(`BlackHoleIn`), and while translation is off plays it on to the headphones itself (`PassThrough`), else the call
would not be heard at all. `PERESMESHNIK_CALL_PATH=blackhole` takes this path on a newer macOS too (tests).

The call app is remembered by its bundle id (`Ui.call_app`); its helper processes (com.google.Chrome.helper) go with it.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from peresmeshnik.audio.resample import Resampler
from peresmeshnik.engines import SAMPLE_RATE

log = logging.getLogger('peresmeshnik.call')
CALL_DEVICE = 'BlackHole 16ch'  # the call app's speaker on macOS < 14.4
BUILT = Path(__file__).resolve().parents[2] / 'native' / 'macos' / 'tap' / 'build' / 'peresmeshnik-tap'
BUILD = 'swiftc -O -target arm64-apple-macos12 native/macos/tap/*.swift -o native/macos/tap/build/peresmeshnik-tap'
ALLOW = ('allow Peresmeshnik in System Settings > Privacy & Security > Screen & System Audio Recording > '
         'System Audio Recording Only, then turn translation on again')
EXIT = {3: 'the call app is not running', 4: f'no permission to record the call app\'s sound: {ALLOW}',
        5: f'macOS {platform.mac_ver()[0]} has no process taps (14.4+): set the call app\'s speaker to {CALL_DEVICE}',
        6: 'Core Audio could not tap the call app', 7: 'the tap\'s sample rate changed'}
RESTART = 7  # the helper's "start me again": the tapped app moved to a device at another rate (AirPods in a call)


def helper() -> str:
    path = os.environ.get('PERESMESHNIK_TAP') or str(BUILT)
    if not os.access(path, os.X_OK):
        raise RuntimeError(f'no tap helper at {path}: build it with `{BUILD}`')
    return path


def tap_supported() -> bool:
    v = platform.mac_ver()[0]
    return sys.platform == 'darwin' and tuple(int(x) for x in (v.split('.') + ['0'])[:2]) >= (14, 4)


def path() -> str:
    """'tap' (macOS 14.4+) or 'blackhole' (older; PERESMESHNIK_CALL_PATH=blackhole forces it, for tests)."""
    return os.environ.get('PERESMESHNIK_CALL_PATH') or ('tap' if tap_supported() else 'blackhole')


def check() -> tuple[int, str]:
    """(0, '') when a tap can work now; else the helper's exit code and the reason to show."""
    r = subprocess.run([helper(), 'check'], capture_output=True, text=True, timeout=10)
    log.info('tap helper: %s', r.stdout.strip())
    return r.returncode, '' if r.returncode == 0 else EXIT.get(r.returncode, r.stderr.strip() or f'tap helper exit {r.returncode}')


def request() -> tuple[int, str]:
    """Ask macOS for "System Audio Recording" now (its dialog, once; blocks until answered, run it off the GUI thread):
    (0, '') allowed, else the helper's code and the reason. The first tap asks by itself too."""
    r = subprocess.run([helper(), 'request'], capture_output=True, text=True)
    return r.returncode, '' if r.returncode == 0 else EXIT.get(r.returncode, r.stderr.strip() or f'tap helper exit {r.returncode}')


def processes() -> list[dict]:
    """The processes Core Audio knows: [{'pid', 'bundle', 'name', 'playing'}]."""
    r = subprocess.run([helper(), 'list'], capture_output=True, text=True, timeout=10)
    if r.returncode:
        raise RuntimeError(EXIT.get(r.returncode, r.stderr.strip()))
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]


def app_of(bundle: str) -> str:
    """The app a helper process belongs to: com.google.Chrome.helper.renderer -> com.google.Chrome."""
    return re.sub(r'\.helper(\..*)?$', '', bundle, flags=re.IGNORECASE)


def players() -> list[dict]:
    """The apps playing now, one per app, ours left out: [{'key' (bundle id), 'name'}] (capture_pipewire.apps shape)."""
    seen = {}
    for p in processes():
        if p['playing'] and p['bundle'] and p['pid'] != os.getpid():
            key = app_of(p['bundle'])
            seen.setdefault(key, {'key': key, 'name': re.sub(r' Helper( \(.*\))?$', '', p['name']) or key})
    return list(seen.values())


def apps() -> list[tuple[str, str]]:
    """[(bundle id, label)] for the tray's "Call app" menu, as on Linux (a browser's label says all of it is taken)."""
    from peresmeshnik.audio.capture_pipewire import apps as label
    return label(players())


def guess_call_app() -> str:
    from peresmeshnik.audio.capture_pipewire import guess_call_app as guess
    return guess(players())


class ProcessTap:
    """`AudioIn` from a process tap of the app `app` (bundle id; `pid` for tests), 16 kHz mono. `retarget(app)` taps
    another app ('' = none: the helper goes, the call is heard as usual) without a new start(). `on_end(why)` runs once
    when the helper ends by itself (no permission, the process ended, Core Audio failed)."""

    def __init__(self, app: str = '', pid: int | None = None, on_end: Callable[[str], None] = lambda why: None):
        self.app, self.pid, self.on_end = app, pid, on_end
        self.rate = 0  # the tap's own rate (from the helper's header)
        self.status = ''  # the helper's last message
        self._on_frames = self._proc = None
        self._lock = threading.Lock()

    def start(self, on_frames: Callable[[np.ndarray], None]) -> None:
        self._on_frames = on_frames
        if self.app or self.pid:
            self._spawn()

    def retarget(self, app: str) -> None:
        with self._lock:
            if app and app == self.app and self._proc is not None and self._proc.poll() is None:
                return  # tapping it already: a new helper would let the original through for a moment
            old, self._proc, self.app = self._proc, None, app
        if old:
            self._end(old)
        if app and self._on_frames:
            self._spawn()

    def stop(self) -> None:
        self.retarget('')
        self._on_frames = None

    def _spawn(self) -> None:
        target = ['--pid', str(self.pid)] if self.pid else ['--bundle', self.app]
        proc = subprocess.Popen([helper(), 'tap', *target], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with self._lock:
            self._proc = proc
        threading.Thread(target=self._messages, args=(proc,), name='tap-messages', daemon=True).start()
        threading.Thread(target=self._read, args=(proc, self._on_frames), name='capture', daemon=True).start()

    def _messages(self, proc) -> None:
        for line in proc.stderr:  # "peresmeshnik-tap: tapping 812 at 48000 Hz", "waiting for … to use sound"
            self.status = line.decode(errors='replace').strip().removeprefix('peresmeshnik-tap: ')
            log.info('call tap: %s', self.status)

    def _read(self, proc, on_frames) -> None:
        head = proc.stdout.readline().decode(errors='replace')
        if m := re.match(r'peresmeshnik-tap 1 rate=(\d+) channels=1 format=f32le', head):
            self.rate = int(m[1])
            rs, n = Resampler(self.rate, SAMPLE_RATE), self.rate * 32 // 1000 * 4  # 32 ms, as the Linux capture
            while len(b := proc.stdout.read(n)) == n:
                on_frames(rs(np.frombuffer(b, '<f4')))
        proc.wait()
        with self._lock:
            ours = proc is self._proc  # not stopped or retargeted
            if ours:
                self._proc = None
        if ours and proc.returncode == RESTART and on_frames is self._on_frames:
            log.info('call tap: %s; starting it again', self.status)
            self._spawn()  # the new rate comes in the new header
            return
        if ours:
            why = EXIT.get(proc.returncode) or f'the tap helper ended ({proc.returncode}): {self.status}'
            if proc.returncode == 3:
                why = f'{self.app or self.pid} is not running'
            self.on_end(why)

    @staticmethod
    def _end(proc) -> None:
        proc.stdin.close()  # the helper's sign to remove the tap: the app is heard as usual again
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class BlackHoleIn:
    """`AudioIn` from BlackHole 16ch (the call app's speaker on macOS < 14.4): the channels that carry sound mixed to
    mono (a stereo app plays into 1-2, a mono one into the centre channel 3: CI), at `rate` (16 kHz for the
    translation). The stream runs at the device's own rate (coreaudio.rate), resampled here. `on_end(why)` when the
    stream ends by itself. Registered with coreaudio while it runs (a PortAudio refresh suspends and resumes it)."""

    def __init__(self, device: str = CALL_DEVICE, on_end: Callable[[str], None] = lambda why: None, rate: int = SAMPLE_RATE):
        self.device, self.on_end, self.rate = device, on_end, rate
        self._stream = self._on_frames = None
        self._stopping = False

    def start(self, on_frames: Callable[[np.ndarray], None]) -> None:
        from peresmeshnik.audio import coreaudio
        with coreaudio.portaudio_lock:  # not while PortAudio is being refreshed
            self._start(on_frames)

    def _start(self, on_frames) -> None:
        import sounddevice as sd

        from peresmeshnik.audio import coreaudio
        from peresmeshnik.audio.out import portaudio_index
        self._on_frames = on_frames
        args = coreaudio.stream_args(self.device, output=False)
        rs = Resampler(args['samplerate'], self.rate)
        index = portaudio_index(self.device, output=False)
        channels = sd.query_devices(index)['max_input_channels']

        def callback(indata, frames, time_info, status):  # PortAudio's thread: mix, resample, hand on
            on = np.any(indata, axis=0)  # an unused channel of BlackHole is exact zeros
            on_frames(rs(indata[:, on].mean(axis=1, dtype=np.float32) if on.any() else np.zeros(frames, np.float32)))

        def finished():
            if not self._stopping:
                self.on_end(f'{self.device}: the recording ended')
        self._stopping = False
        self._stream = sd.InputStream(device=index, channels=channels, dtype='float32', callback=callback,
                                      finished_callback=finished, **args)
        self._stream.start()
        coreaudio.hold(self)

    def stop(self) -> None:
        from peresmeshnik.audio import coreaudio
        self._stopping = True
        coreaudio.release(self)
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.abort()  # nothing to drain on an input
            stream.close()

    def suspend(self) -> None:
        """PortAudio is about to be refreshed (coreaudio.refresh_portaudio): close the stream, resume() opens it again."""
        self.stop()

    def resume(self) -> None:
        if self._on_frames is not None:
            self.start(self._on_frames)


class PassThrough:
    """macOS < 14.4 with translation off: BlackHole 16ch (the call app's speaker) on to the headphones, else the call is
    not heard; only while a call app plays into it. Core Audio says when some process starts IO on BlackHole 16ch
    (coreaudio.Running): then a BlackHoleIn at the device's rate feeds a SoundDeviceOut on `pick()` (the headphones).
    ZERO_S of exact silence on the input closes both; when the device still runs (the call app's IO is on, nobody
    talks) they open again and the next window is twice as long, up to MAX_ZERO_S. With no call app playing nothing
    runs. `on_end(why)` once when the output goes or a stream ends by itself; `on_unknown(name)` when PortAudio does
    not see the headphones (the owner refreshes it; another output plays meanwhile).
    ponytail: two device clocks, no drift correction: the queue is cut back to nothing past MAX_QUEUE_S (a click every
    ~30 min at 100 ppm); resample adaptively if that is heard."""
    MAX_QUEUE_S = 0.2
    ZERO_S, MAX_ZERO_S = 5.0, 60.0

    def __init__(self, pick: Callable[..., str], on_end: Callable[[str], None] = lambda why: None,
                 on_unknown: Callable[[str], None] = lambda name: None):
        self.pick, self.on_end, self.on_unknown = pick, on_end, on_unknown
        self.output = ''  # the headphones while the streams are open
        self.opened = 0  # how many times the streams opened (tests, BENCH)
        self._in = self._out = self._running = None
        self._lock = threading.RLock()
        self._stopped = self._ended = False
        self._zeros, self._window = 0.0, self.ZERO_S

    def start(self) -> None:
        from peresmeshnik.audio import coreaudio
        self._running = coreaudio.Running(CALL_DEVICE, lambda running: running and self._open())
        if self._running.running():
            self._open()

    def _open(self) -> None:
        from peresmeshnik.audio import coreaudio
        from peresmeshnik.audio.out import NotInPortAudio, SoundDeviceOut
        with self._lock:
            if self._stopped or self._in is not None:
                return
            rate = coreaudio.rate(CALL_DEVICE, output=False)
            name = self.pick()
            try:
                out = SoundDeviceOut(rate, name, on_end=self._gone)
            except NotInPortAudio:
                self.on_unknown(name)
                name = self.pick(exclude=name)
                out = SoundDeviceOut(rate, name, on_end=self._gone)
            self._out, self.output, self._zeros = out, name, 0.0
            self._in = BlackHoleIn(on_end=self._gone, rate=rate)
            try:
                self._in.start(lambda pcm: self._on_frames(pcm, out, rate))
            except BaseException:
                self._close()
                raise
            self.opened += 1
            log.info('call sound: %s -> %s while translation is off', CALL_DEVICE, name)

    def _on_frames(self, pcm, out, rate) -> None:  # PortAudio's thread
        if pcm.any():
            self._zeros, self._window = 0.0, self.ZERO_S
        else:
            self._zeros += len(pcm) / rate
            if self._zeros >= self._window:
                self._zeros = -1e9  # once
                threading.Thread(target=self._quiet, name='pass-through-quiet', daemon=True).start()
        if out.queued_s() > self.MAX_QUEUE_S:
            out.flush()
        out.play(pcm, rate)

    def _quiet(self) -> None:
        """The call app plays silence: close; open again at once if it still runs IO on BlackHole 16ch."""
        with self._lock:
            if self._stopped:
                return
            self._close()
            if self._running.running():
                self._window = min(self._window * 2, self.MAX_ZERO_S)
                self._open()
            else:
                self._window = self.ZERO_S
                log.info('call sound: nothing plays into %s; the pass-through waits', CALL_DEVICE)

    def _gone(self, why: str) -> None:
        if not self._ended and not self._stopped:
            self._ended = True
            self.on_end(why)

    def _close(self) -> None:
        cap, out, self._in, self._out, self.output = self._in, self._out, None, None, ''
        if out is not None:
            out.flush()
        # both at once: PortAudio takes 0.1-0.25 s to stop each stream (it polls the unit with Pa_Sleep(100))
        stops = [threading.Thread(target=x.stop) for x in (cap, out) if x is not None]
        for t in stops:
            t.start()
        for t in stops:
            t.join()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            running, self._running = self._running, None
            self._close()
        if running is not None:
            running.close()

"""Microphone capture, and "translation <-> live voice" for the me -> them direction.

`Mic`: on Linux a PipeWire / PulseAudio source by name (the names of `pactl list sources`, as in the
wizard and the settings) through `parec`; on macOS a Core Audio device by UID or name, on Windows a WASAPI endpoint
by id or name, through the helper (`audio/macos.py`, `audio/windows.py`), which can also follow the device itself
(`follow`: the saved one while it is there, else the default input, back to the saved one when it returns); both read
on a thread blocked on the pipe. Elsewhere PortAudio (sounddevice). The helper, not PortAudio, on Windows too: it
sees endpoints come and go by itself (PortAudio only after a refresh that reopens every stream of the process).
The reader / device callback only hands frames on (engines.py rules): `on_frames` must just enqueue. The stream
exists only between start() and stop(), so when translation is off no device is open. Our own devices
(`peresmeshnik_*`, BlackHole 2ch, CABLE Output) are refused: the app would hear itself.

`LiveSwitch`: the me -> them `Direction` with `set_live` (see its docstring).
"""
from __future__ import annotations

import collections
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import numpy as np

from peresmeshnik.audio.resample import Resampler
from peresmeshnik.engines import SAMPLE_RATE, Speech

OURS = 'peresmeshnik_'  # every PipeWire device the app makes starts with this
FADE_MS = 30  # crossfade of the live switch: >= 2 periods of a low voice (85 Hz), well under a syllable (BENCH)
LEAD_MS = 60  # output kept ahead of the input clock: room for scheduling jitter on top of the device's 40 ms
START_S = 5.0  # record(): how much longer than asked it waits for the microphone's sound
LEVEL_DB = -26.0  # LiveSwitch: my speech level assumed until the microphone has carried some (a call's usual level)
MATCH_DB = (-30.0, 10.0)  # the gain that brings the translation to the live voice's level stays within this
PEAK = 0.95  # ... and never takes it above this
RISE_DB = 20.0  # ... and goes back up at most this fast (dB a second) after a peak or a louder level held it down


class Level:
    """Speech level of a stream (dBFS): the power mean of its 32 ms frames of clear speech, so pauses, breaths and room
    noise do not count (a power mean, as BS.1770: a mean of dB moved with where the quiet frames were cut off, and the
    microphone's cut is not the speech's). A frame counts when it is above `FLOOR_DB` and within `RANGE_DB` of the level;
    with `noise_s` (a microphone) also `SPEECH_DB` above the noise floor, the quietest frame of the last `noise_s` (else a
    minute of room noise before my first word was learnt as my level: the first clauses went out 13-18 dB quiet). A plain
    mean over the first `1 / alpha` such frames, then it follows a change over as many. None until `hold_s` of them; with
    `follow_s` (the translation's side, where the cloud's voice gives way to a local one) that much of a voice all under
    the range starts it over."""
    FLOOR_DB, SPEECH_DB, RANGE_DB, FRAME_S = -60.0, 12.0, 15.0, 0.032

    def __init__(self, sample_rate: int, alpha: float, noise_s: float = 0.0, hold_s: float = 0.0,
                 follow_s: float = math.inf):
        self.frame, self.alpha = round(sample_rate * self.FRAME_S), alpha
        self.hold, self.follow = hold_s / self.FRAME_S, follow_s / self.FRAME_S
        self._noise = collections.deque(maxlen=round(noise_s / self.FRAME_S)) if noise_s else None
        self._p, self._n, self._low = 0.0, 0, 0  # the power mean, frames in it, frames under the range in a row
        self._rest = np.zeros(0, np.float32)

    @property
    def db(self) -> float | None:
        return 10 * math.log10(self._p) if self._n and self._n >= self.hold else None

    def __call__(self, x: np.ndarray) -> None:
        x = np.concatenate([self._rest, x])
        k = len(x) // self.frame
        self._rest = x[k * self.frame:]
        for d in (10 * np.log10(np.mean(x[:k * self.frame].reshape(k, self.frame) ** 2, axis=1) + 1e-12)).tolist():
            if self._noise is not None:
                self._noise.append(d)
            if d <= (self.FLOOR_DB if self._noise is None else max(min(self._noise), self.FLOOR_DB) + self.SPEECH_DB):
                continue
            if self._n and d < 10 * math.log10(self._p) - self.RANGE_DB:
                self._low += 1
                if self._low < self.follow:
                    continue
                self._n = 0  # a new, quieter voice: start over from this frame
            self._n, self._low = self._n + 1, 0
            self._p += max(self.alpha, 1 / self._n) * (10 ** (d / 10) - self._p)


def _pactl(*args: str) -> str:
    return subprocess.run(['pactl', *args], check=True, capture_output=True, text=True, timeout=10).stdout


def sources() -> list[str]:
    """Names of the PipeWire / PulseAudio sources (monitors of sinks included)."""
    return [line.split('\t')[1] for line in _pactl('list', 'short', 'sources').splitlines() if '\t' in line]


def native():
    """The helper's platform module: audio/macos.py or audio/windows.py."""
    if sys.platform == 'win32':
        from peresmeshnik.audio import windows
        return windows
    from peresmeshnik.audio import macos
    return macos


def not_ours(name: str) -> str:
    if name.startswith(OURS):
        raise ValueError(f'{name} is a Peresmeshnik device, not a microphone: choose your real microphone in the '
                         'settings or the system sound settings')
    return name


class Mic:
    """`AudioIn` from a microphone. `device`: Linux: a source name, '' / None = the default source; macOS: a Core
    Audio UID or name, '' / None = the default input; elsewhere a PortAudio name or index. `on_end(why)`: the stream
    ended by itself (the device went away, the sound server restarted, no permission), called on the reader thread;
    stop() does not call it. macOS: `follow` makes the helper keep to the saved device or the default one by itself,
    and `on_device(event, name)` hears of it: 'using' (opened, again after a loss), 'lost', 'none' (no microphone)."""

    def __init__(self, device: int | str | None = None, sample_rate: int = SAMPLE_RATE, frame_ms: int = 20,
                 on_end: Callable[[str], None] | None = None, follow: bool = False,
                 on_device: Callable[[str, str], None] | None = None):
        self.device, self.sample_rate, self.on_end = device or None, sample_rate, on_end
        self.follow, self.on_device = follow, on_device
        self.blocksize = sample_rate * frame_ms // 1000
        self.pipewire = sys.platform.startswith('linux') and bool(shutil.which('parec'))
        self.helper = sys.platform in ('darwin', 'win32')
        self.source = ''  # the device being recorded (Linux, macOS)
        self.uid = ''  # macOS: its Core Audio UID; Windows: its endpoint id
        self._stream = self._proc = self._thread = None
        self._stopping = False

    def start(self, on_frames: Callable[[np.ndarray], None]) -> None:
        self._stopping = False
        if self.pipewire:
            self._start_parec(on_frames)
            return
        if self.helper:
            self._start_helper(on_frames)
            return
        import sounddevice as sd  # PortAudio is loaded only when a stream opens

        def callback(indata, frames, time_info, status):
            on_frames(indata[:, 0].copy())

        def finished():
            if not self._stopping and self.on_end:
                self.on_end('the input stream ended')

        self._stream = sd.InputStream(device=self.device, channels=1, samplerate=self.sample_rate, blocksize=self.blocksize,
                                      dtype='float32', callback=callback, finished_callback=finished)
        self._stream.start()

    def _start_parec(self, on_frames) -> None:
        name = not_ours(self.device or _pactl('get-default-source').strip())
        if name not in sources():
            raise RuntimeError(f'no microphone {name!r} (unplugged?): see pactl list short sources')
        self.source = name
        # dont-reconnect: when the device goes, PipeWire would move the recording to another source (maybe a monitor
        # of our own); we would rather know and choose (on_end)
        self._proc = subprocess.Popen(
            ['parec', f'--device={name}', '--raw', '--format=float32le', f'--rate={self.sample_rate}', '--channels=1',
             '--latency-msec=20', '--client-name=Peresmeshnik', '--stream-name=Peresmeshnik microphone',
             '--property=node.dont-reconnect=true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._read(on_frames, lambda: self._proc.stderr.read().decode(errors='replace').strip().splitlines())

    def _read(self, on_frames, why) -> None:
        proc, name = self._proc, self.source

        def read():
            n = self.blocksize * 4
            while len(b := proc.stdout.read(n)) == n:
                on_frames(np.frombuffer(b, '<f4'))
            proc.wait()
            if not self._stopping and self.on_end:
                lines = why()
                self.on_end(f'{name or "the microphone"}: {lines[-1] if lines else f"the recording exited with {proc.returncode}"}')
        self._thread = threading.Thread(target=read, name='mic', daemon=True)
        self._thread.start()

    def _start_helper(self, on_frames) -> None:
        helper = native()
        self.source = helper.not_virtual(self.device or '')
        self._proc = proc = helper.spawn('record', '--from', self.source, '--rate', str(self.sample_rate), *(['--follow'] * self.follow))
        errors: list[str] = []
        done = threading.Event()

        def events():  # the helper's stderr: device events, else why it ended
            for raw in proc.stderr:
                line = raw.decode(errors='replace').strip()
                kind, _, rest = line.partition(' ')
                if kind in ('using', 'lost', 'none'):
                    uid, _, name = rest.partition('\t')
                    name = name or rest
                    if kind == 'using':
                        self.source, self.uid = name, uid
                    if self.on_device and not self._stopping:
                        self.on_device(kind, name)
                elif line:
                    errors.append(line.removeprefix('peresmeshnik-audio: '))
            done.set()

        def why():
            done.wait(5)
            return [helper.NO_PERMISSION] if proc.returncode == 6 else errors
        threading.Thread(target=events, name='mic-events', daemon=True).start()
        self._read(on_frames, why)

    def stop(self) -> None:
        self._stopping = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            if self._thread is not threading.current_thread():  # on_end may stop it from its own reader
                self._thread.join(5)
            self._proc = None


def followed(saved: str, on_state: Callable[[str], None] = lambda text: None) -> Mic:
    """macOS, Windows: the microphone of me -> them, kept by the helper on the saved device, else the default input
    (never a virtual one), back on the saved one when it returns. The helper takes another one at once: the tray says which one
    plays meanwhile, and again when the saved one is back."""
    away = {'name': '', 'none': False}  # the saved microphone while it is away; no microphone at all

    def on_device(kind: str, name: str) -> None:
        if kind == 'lost':
            away['name'] = away['name'] or name
        elif kind == 'none':
            if not away['none']:
                on_state('error: no microphone: plug one in or choose it in the settings')
            away['none'] = True
        elif kind == 'using' and (away['name'] or away['none']):
            away['none'] = False
            if saved:
                back = saved in (m.uid, name)
            else:  # the default input: back when the one that went comes back (or a first one after none)
                back = not away['name'] or name == away['name']
            if back:
                away['name'] = ''
                on_state(f'translating (the microphone is back: {name})')
            else:
                on_state(f'translating ({away["name"] or "your microphone"} went away: {name} meanwhile)')
    m = Mic(saved, follow=True, on_device=on_device, on_end=lambda why: on_state(f'error: the microphone stopped: {why}'))
    return m


def record(seconds: float, device: int | str | None = None, sample_rate: int = SAMPLE_RATE,
           on_progress: Callable[[float], None] | None = None) -> np.ndarray:
    """Blocking recording of `seconds` of sound from the microphone (enrollment); `on_progress(seconds so far)` once a
    second of it. Counted in sound, not wall time: the helper takes ~0.5 s to start (macOS runner: 2.5 s of 3 came by
    wall time). A microphone that gives less within `START_S` extra (it went away) returns what came; one that ends
    before any sound came raises with why (no permission: the helper says so at once, not after the wait)."""
    frames: list[np.ndarray] = []
    got = threading.Condition()
    n, ended = [0], []

    def take(pcm: np.ndarray) -> None:
        with got:
            frames.append(pcm)
            n[0] += len(pcm)
            got.notify()

    def end(why: str) -> None:
        with got:
            ended.append(why)
            got.notify()
    mic = Mic(device, sample_rate, on_end=end)
    mic.start(take)
    try:
        end_at = time.monotonic() + seconds + START_S
        for i in range(1, math.ceil(seconds) + 1):
            want = min(i, seconds)
            with got:
                if not got.wait_for(lambda: n[0] >= want * sample_rate or ended, max(0.0, end_at - time.monotonic())) \
                        or n[0] < want * sample_rate:  # noqa: B023
                    break
            if on_progress:
                on_progress(want)
    finally:
        mic.stop()
    if not frames and ended:
        raise RuntimeError(ended[0])
    return np.concatenate(frames)[:int(seconds * sample_rate)] if frames else np.zeros(0, np.float32)


class LiveSwitch:
    """The me -> them `Direction` with "translation <-> live voice": wraps a local or cloud direction; `set_live(True)`
    sends the microphone out as is and stops feeding the translation, `set_live(False)` goes back. Each switch is an
    equal-power crossfade over `fade_ms`, so no click (BENCH: the jump at the seam stays at the signals' own level).

    The output is clocked by the input: every input frame puts out the same duration of sound (translated speech queued
    here, the live microphone, or both while fading); a stretch of output starts with `lead_ms` of silence, a cushion
    against scheduling jitter. So the queue of translated speech stays here, where it can be faded, not in the device's
    buffer. With nothing to say nothing goes out; when the input stalls or ends (a file, a microphone being reopened)
    the queued speech goes out unpaced after `STALL_S`, as without the switch.

    The translation is brought to the level of my live voice (`Level` of the microphone's frames against that of the
    speech, ~3 s of speech each to follow; LEVEL_DB until the microphone has carried 1 s of clear speech), so the
    switch does not jump in loudness either: the clone spoke ~9 dB louder than a microphone at a call's level, with
    peaks near full scale (BENCH, stage 5).
    ponytail: the microphone's clock drives the output: in live voice a ~100 ppm drift against the virtual mic adds
    ~6 ms of latency per minute; resample against the sink's clock if live stretches get long.
    """
    supports_live = True
    STALL_S = 0.5

    def __init__(self, inner, sample_rate: int = 24_000, fade_ms: float = FADE_MS, lead_ms: float = LEAD_MS):
        self.inner, self.sample_rate = inner, sample_rate
        self.fade = max(1, round(sample_rate * fade_ms / 1000))
        self.lead = round(sample_rate * lead_ms / 1000)
        self.live = False
        self._up = Resampler(SAMPLE_RATE, sample_rate)  # the microphone, 16 kHz
        self._rs: dict = {}  # the inner direction's speech at another rate
        self._q: queue.Queue = queue.Queue()
        # events thread only:
        self._buf = np.zeros(0, np.float32)  # translated speech not yet put out
        self._g, self._to = 0.0, 0.0  # gain of the live voice now and where it is going (0 translation .. 1 live)
        self._out_end = 0.0  # monotonic time when the output put out so far has played
        self._quiet = True  # nothing to put out: feed() does not wake the events thread
        self._closing = False
        self._mic = Level(SAMPLE_RATE, 0.01, noise_s=3.0, hold_s=1.0)  # feed()'s thread
        self._tts, self._gain = Level(sample_rate, 0.01, follow_s=1.0), None
        threading.Thread(target=self._pump, name='me-live-pump', daemon=True).start()

    # Direction
    def feed(self, pcm: np.ndarray) -> None:
        self._mic(pcm)
        if not self.live:
            self.inner.feed(pcm)
        if self.live or not self._quiet:
            self._q.put(('in', pcm))

    def set_live(self, on: bool) -> None:
        if on == self.live:
            return
        if on:  # a pause closes the clause in progress, so the translation resumes on a fresh one afterwards
            self.inner.feed(np.zeros(SAMPLE_RATE // 2, np.float32))
        self.live = on
        self._q.put(('live', on))

    def events(self):
        while True:
            try:
                item = self._q.get(timeout=None if self._quiet else self.STALL_S)
            except queue.Empty:  # no input clock: say the rest unpaced, there is nothing to fade against
                if len(self._buf) and self._g == 0 == self._to:
                    yield Speech(self._buf, self.sample_rate)
                    self._buf = np.zeros(0, np.float32)
                self._quiet = True
                continue
            if item is None:  # the inner direction ended: after finish() (a file) the rest is said, after close() not
                if len(self._buf) and not self._closing and self._g < 1:
                    yield Speech(self._buf, self.sample_rate)
                return
            kind, x = item
            if kind == 'error':
                raise x
            if kind == 'caption':
                yield x
            elif kind == 'speech':
                if self._to == 0:  # translation that comes while live (or going live) is not said
                    rs = self._rs.get(x.sample_rate) or self._rs.setdefault(x.sample_rate, Resampler(x.sample_rate, self.sample_rate))
                    self._buf = self._match(self._buf, rs(x.pcm))
                    self._quiet = False
            elif kind == 'live':
                self._to = 1.0 if x else 0.0
                self._quiet = False
            elif (out := self._step(x)) is not None:
                yield Speech(out, self.sample_rate)

    def close(self) -> None:
        self._closing = True
        self.inner.close()  # its events end, so do ours

    # internals
    def _pump(self) -> None:
        try:
            for e in self.inner.events():
                self._q.put(('speech' if isinstance(e, Speech) else 'caption', e))  # captions, Latency: passed on
        except Exception as e:  # a dead session: the app's consumer reports it
            self._q.put(('error', e))
        self._q.put(None)

    def _match(self, buf: np.ndarray, x: np.ndarray) -> np.ndarray:
        """`buf` (queued, not put out yet) + `x`, translated speech, at the live voice's level and under PEAK. A lower
        gain (a louder peak or level) ramps the end of `buf` down over up to 5 ms; a higher one comes back at RISE_DB,
        a limiter's release, so the cloud's short chunks, each with its own peak, do not pump the level."""
        self._tts(x)
        if not (peak := float(np.abs(x).max()) if len(x) else 0.0):
            return np.concatenate([buf, x])  # silence (the clone's pause after a sentence) stays as it is
        tts = self._tts.db if self._tts.db is not None else 10 * math.log10(float(np.mean(x ** 2)))  # < 1 frame yet
        g = min(10 ** (float(np.clip((self._mic.db or LEVEL_DB) - tts, *MATCH_DB)) / 20), PEAK / peak)
        g0 = g if self._gain is None else self._gain
        if g > g0:
            ramp = (10 ** np.minimum(math.log10(g), math.log10(g0) + RISE_DB / 20 / self.sample_rate
                                     * np.arange(1, len(x) + 1))).astype(np.float32)
            g = float(ramp[-1])
        else:
            ramp = np.float32(g)
            if g < g0 and (n := min(len(buf), self.sample_rate // 200)):
                buf = np.concatenate([buf[:-n], buf[-n:] * np.linspace(1, g / g0, n + 1, dtype=np.float32)[1:]])
        self._gain = g
        return np.concatenate([buf, x * ramp])

    def _step(self, pcm: np.ndarray) -> np.ndarray | None:
        """One input frame -> the same duration of output, after the lead when the output starts from silence."""
        live = self._up(pcm)
        if self._g == 0 == self._to and not len(self._buf):
            self._quiet = True
            return None
        n = len(live)
        k = min(n, len(self._buf))
        tts = np.concatenate([self._buf[:k], np.zeros(n - k, np.float32)])
        self._buf = self._buf[k:]
        if self._g != self._to:
            g = np.clip(self._g + np.sign(self._to - self._g) / self.fade * np.arange(1, n + 1), 0.0, 1.0)
            self._g = float(g[-1])
        else:
            g = self._g
        if self._g == 1.0:
            self._buf = np.zeros(0, np.float32)  # live: the rest of the translation is not said
        out = (tts * np.cos(g * np.pi / 2) + live * np.sin(g * np.pi / 2)).astype(np.float32)
        now = time.monotonic()
        if now >= self._out_end:  # the output had played out (or never started): a new stretch, with the cushion
            out = np.concatenate([np.zeros(self.lead, np.float32), out])
        self._out_end = max(self._out_end, now) + len(out) / self.sample_rate
        return out


def click_ratio(x: np.ndarray, at: int, sr: int, refs: list[np.ndarray], half_ms: float = 5.0) -> float:
    """How much the biggest sample-to-sample jump within `half_ms` of `at` exceeds the biggest jump inside the signals
    themselves (`refs`: the two inputs, each on its own). A hard cut between two signals jumps by up to their amplitude:
    ratio >> 1; a clean crossfade stays ~1."""
    h = int(sr * half_ms / 1000)
    seam = np.abs(np.diff(x[max(0, at - h):at + h]))
    own = max(float(np.abs(np.diff(r)).max()) for r in refs)
    return float(seam.max() / max(own, 1e-9)) if len(seam) else 0.0

"""Translated speech to the headphones. Linux: a PulseAudio/PipeWire sink through `pacat` (`PulseOut`); macOS: a
Core Audio device through PortAudio (`SoundDeviceOut`). `Output` is the one of this OS, both with one interface.

Linux (`PulseOut`):

play() only enqueues; one playback thread writes to the device in short slices, blocking on its buffer.
flush() (barge-in, mode switch) ends that pacat at once: a stream that runs dry keeps its last bit below
the prebuffer and would play it later, in front of the next speech; the next chunk opens a new one.
A thread per pacat waits on its exit and calls `on_end` when the device went away, so a dead output is
known at once, not at the next chunk. With a named device one `pactl subscribe` also watches for its removal:
since WirePlumber 0.5 a `node.dont-reconnect` stream may outlive its device without ending.
"""
from __future__ import annotations

import collections
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import numpy as np

from peresmeshnik.audio import c_locale

PIPE_BYTES = 16384  # pacat's stdin: ~0.19 s at 22.05 kHz float32 (the Linux default 64 KB holds 0.74 s)


def default_sink() -> str:
    """Name of the current default sink. Resolve it before creating our own sinks: on a box without a real
    output device the new null sink becomes the default."""
    return subprocess.run(['pactl', 'get-default-sink'], check=True, capture_output=True, text=True).stdout.strip()


class PulseOut:
    """`on_end(why)` runs once (on a thread of its own) when the device went away (a Bluetooth profile
    switch, unplugged; `node.dont-reconnect` keeps it from following another device: pacat ends, or the
    named sink's removal event comes first) or the sound server stopped.
    stop() and flush() do not call it. The owner reopens (app.Headphones)."""
    slice_s = 0.05

    def __init__(self, sample_rate: int, device: str | None = None, latency_ms: int = 40,
                 on_end: Callable[[str], None] = lambda why: None):
        self.sample_rate, self.latency_s, self.on_end = sample_rate, latency_ms / 1000, on_end
        self._cmd = ['pacat', '--playback', '--raw', '--format=float32le', f'--rate={sample_rate}', '--channels=1',
                     f'--latency-msec={latency_ms}', '--client-name=Peresmeshnik', '--stream-name=Peresmeshnik translation',
                     # PipeWire moves a stream to a newly created sink (ours!) despite its target unless told not to
                     '--property=node.dont-reconnect=true'] + ([f'--device={device}'] if device else [])
        self.alive = True  # False once the device is gone
        self._stopping = False
        self._lock = threading.Lock()  # flush() from another thread swaps the process under the writer
        self._gen = 0  # flush() count: a chunk queued before the last flush is not written on
        self._q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue()
        self._end = 0.0  # monotonic time when the audio written so far has played
        self._events = self._subscribe(device) if device else None  # before pacat: a removal right after is seen
        try:
            self._proc = self._open()  # at once: a missing device shows now, not at the first word
        except BaseException:
            self._end_events()
            raise
        self._thread = threading.Thread(target=self._run, name='playback', daemon=True)
        self._thread.start()

    def _open(self) -> subprocess.Popen:
        p = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            import fcntl  # Linux (pacat); the module imports on every OS
            fcntl.fcntl(p.stdin, getattr(fcntl, 'F_SETPIPE_SZ', 1031), PIPE_BYTES)
        except (ImportError, OSError):
            pass  # the default pipe: more audio ahead of the device
        threading.Thread(target=self._watch, args=(p,), name='playback-watch', daemon=True).start()
        return p

    def _subscribe(self, device: str) -> subprocess.Popen:
        sub = subprocess.Popen(['pactl', 'subscribe'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=c_locale())
        try:
            sinks = subprocess.run(['pactl', 'list', 'short', 'sinks'], capture_output=True, text=True, timeout=10,
                                   env=c_locale()).stdout
        except BaseException:
            sub.kill()
            sub.wait()
            raise
        index = next((line.split('\t')[0] for line in sinks.splitlines() if line.split('\t')[1:2] == [device]), None)

        def watch():
            for line in sub.stdout:  # until stop() kills it, or the sound server goes (pacat ends with it)
                if (ev := re.match(r"Event 'remove' on sink #(\d+)", line)) and ev[1] == index:
                    self._lost(f'{device} was removed')
                    return
        if index is not None:  # else pacat fails on the missing device and _watch tells
            threading.Thread(target=watch, name='playback-events', daemon=True).start()
        return sub

    def _lost(self, why: str) -> None:
        with self._lock:
            if self._stopping or not self.alive:
                return
            self.alive = False
            proc = self._proc
        if proc is not None:
            proc.kill()  # its _watch sees alive False: on_end stays once
        self.on_end(why)

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        assert sample_rate == self.sample_rate, f'{sample_rate} != {self.sample_rate}'
        self._q.put((self._gen, pcm))

    def flush(self) -> None:
        """Drop the queued audio and what pacat still holds (within ~50 ms)."""
        with self._lock:
            self._gen += 1
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            old, self._proc, self._end = self._proc, None, 0.0
        if old is not None:
            old.kill()

    def stop(self) -> None:
        """Play what is queued, then close the device."""
        self._stopping = True
        self._end_events()
        self._q.put(None)
        self._thread.join()
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _end_events(self) -> None:
        if self._events is not None:
            self._events.kill()
            self._events.wait()

    def _run(self):
        step = max(1, int(self.slice_s * self.sample_rate))
        while (item := self._q.get()) is not None:
            gen, pcm = item
            data = np.ascontiguousarray(pcm, '<f4')
            for i in range(0, len(data), step):
                with self._lock:
                    if gen != self._gen or not self.alive:
                        break
                    if self._proc is None:  # flushed: a new stream for the speech after it
                        self._proc = self._open()
                    proc = self._proc
                part = data[i:i + step]
                self._end = max(self._end, time.monotonic()) + len(part) / self.sample_rate
                try:
                    proc.stdin.write(part.tobytes())
                    proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):  # flushed meanwhile, or gone (_watch tells the owner)
                    break
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        # pacat spins a core draining after EOF (0.64 s CPU for a 0.7 s tail), so close only once it has played
        if self.alive:
            time.sleep(max(0.0, self._end - time.monotonic()) + self.latency_s)  # the last write still sits in the buffer
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def _watch(self, proc: subprocess.Popen):
        err = proc.stderr.read().decode(errors='replace').strip().splitlines()  # until pacat exits
        proc.wait()
        with self._lock:
            ours = proc is self._proc and not self._stopping and self.alive  # it ended by itself
            if ours:
                self.alive = False
        if ours:
            self.on_end(f'pacat ended ({proc.returncode}){": " + err[-1] if err else ""}')


class NotInPortAudio(RuntimeError):
    """The device is there, but PortAudio has not seen it (plugged in after its last refresh)."""


def portaudio_index(name: str, output: bool = True) -> int:
    """PortAudio's index of the device `name` (exact). PortAudio lists the devices of its start only: a device plugged
    in since is not there until coreaudio.refresh_portaudio()."""
    import sounddevice as sd
    key = 'max_output_channels' if output else 'max_input_channels'
    for i, d in enumerate(sd.query_devices()):
        if d['name'] == name and d[key] > 0:
            return i
    raise NotInPortAudio(f'no {"output" if output else "input"} device {name!r} for PortAudio (plugged in after its start)')


class SoundDeviceOut:
    """macOS: the headphones through PortAudio on a device by name (None / '' = the system's default output now), at the
    device's own rate (`rate`; play() resamples from `sample_rate`). play() only enqueues; PortAudio's callback takes
    from that queue (silence when it is empty). `on_end(why)` runs
    once (on a thread of its own) when the device goes: Core Audio says it left the device list or is not alive
    (coreaudio.Watch: PortAudio itself would only go quiet), or the stream ended by itself. stop() and flush() do
    not call it. The owner reopens (app.Headphones)."""

    def __init__(self, sample_rate: int, device: str | None = None, latency_ms: int = 40,
                 on_end: Callable[[str], None] = lambda why: None):
        import sounddevice as sd

        from peresmeshnik.audio import coreaudio
        self.sample_rate, self.on_end = sample_rate, on_end
        self.device = device or coreaudio.default()
        dev = coreaudio.device_id(self.device)
        if dev is None:
            raise RuntimeError(f'no output device {self.device!r}')
        index = portaudio_index(self.device)
        self.channels = min(2, sd.query_devices(index)['max_output_channels'])  # mono on both ears
        args = coreaudio.stream_args(self.device)
        self.rate = args['samplerate']
        from peresmeshnik.audio.resample import Resampler
        self._rs = Resampler(sample_rate, self.rate)
        self.alive = True  # False once the device is gone
        self._stopping = False
        self._lock = threading.Lock()  # play() / flush() and the callback
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._pos = 0  # samples of _chunks[0] already played
        self._empty = threading.Event()
        self._empty.set()
        self._watch = coreaudio.Watch(dev, self.device, self._lost)
        try:
            self._stream = sd.OutputStream(device=index, channels=self.channels, dtype='float32', callback=self._fill,
                                           finished_callback=self._finished, **args)  # latency_ms: BLOCK on macOS
            self._stream.start()
        except BaseException:
            self._watch.close()
            raise

    def _fill(self, outdata, frames, time_info, status) -> None:  # PortAudio's thread
        n = 0
        with self._lock:
            while n < frames and self._chunks:
                c = self._chunks[0]
                k = min(frames - n, len(c) - self._pos)
                outdata[n:n + k] = c[self._pos:self._pos + k, None]
                n += k
                self._pos += k
                if self._pos == len(c):
                    self._chunks.popleft()
                    self._pos = 0
            if not self._chunks:
                self._empty.set()
        outdata[n:] = 0

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        assert sample_rate == self.sample_rate, f'{sample_rate} != {self.sample_rate}'
        if not len(pcm):
            return
        pcm = self._rs(np.ascontiguousarray(pcm, np.float32))  # the caller's thread, one at a time (as PulseOut's writer)
        with self._lock:
            if self.alive:
                self._chunks.append(pcm)
                self._empty.clear()

    def flush(self) -> None:
        """Drop the queued audio (what PortAudio holds, ~latency, still plays)."""
        with self._lock:
            self._chunks.clear()
            self._pos = 0
            self._empty.set()

    def take(self) -> list[np.ndarray]:
        """The audio not played yet, taken out, at `rate` (suspend: it plays on the next device)."""
        with self._lock:
            chunks = list(self._chunks)
            if chunks:
                chunks[0] = chunks[0][self._pos:]
            self._chunks.clear()
            self._pos = 0
            self._empty.set()
        return chunks

    def queued_s(self) -> float:
        with self._lock:
            return (sum(len(c) for c in self._chunks) - self._pos) / self.rate

    def stop(self, drain: bool = True) -> None:
        """Play what is queued (drain), then close the device."""
        if self.alive and drain:
            self._empty.wait(self.queued_s() + 2.0)
        self._stopping = True
        self._watch.close()
        if self.alive and drain:
            self._stream.stop()  # returns once PortAudio's buffers have played
        else:
            self._stream.abort()
        self._stream.close()

    def _finished(self) -> None:  # PortAudio's thread: the stream stopped (ours: stop(); else it ended by itself)
        if not self._stopping:
            threading.Thread(target=self._lost, args=(f'{self.device}: the output stream ended',), daemon=True).start()

    def _lost(self, why: str) -> None:
        with self._lock:
            if self._stopping or not self.alive:
                return
            self.alive = False
            self._chunks.clear()
            self._empty.set()
        self.on_end(why)


Output = PulseOut if sys.platform.startswith('linux') else SoundDeviceOut

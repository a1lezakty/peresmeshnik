"""Linux virtual microphone "Peresmeshnik Mic" on PipeWire (pipewire-pulse) or PulseAudio.

    null sink  peresmeshnik_mic_sink  <- our translated speech (pacat)
    remap source  peresmeshnik_mic  ("Peresmeshnik Mic") on the sink's monitor -> the call app's mic

Call apps hide monitor sources, hence the remap source. The device lives while the app runs
(`remove()` on exit; a stale one left by a crash is replaced on the next start). There is no
playback stream until the first play(); from then until stop()/drain() one `pacat` process,
fed by a writer thread, so play() never blocks.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time

import numpy as np

SINK = 'peresmeshnik_mic_sink'
SOURCE = 'peresmeshnik_mic'
NAME = 'Peresmeshnik Mic'


def _pactl(*args: str) -> str:
    return subprocess.run(['pactl', *args], check=True, capture_output=True, text=True).stdout


def exists() -> bool:
    return any(line.split('\t')[1:2] == [SOURCE] for line in _pactl('list', 'short', 'sources').splitlines())


def _remove_stale() -> None:
    mine = [line.split('\t') for line in _pactl('list', 'short', 'modules').splitlines()]
    mine = [m[0] for m in mine if len(m) > 2 and (f'sink_name={SINK}' in m[2] or f'source_name={SOURCE}' in m[2])]
    for idx in sorted(mine, key=int, reverse=True):  # the remap source before its sink
        _pactl('unload-module', idx)


class VirtualMic:
    """AudioOut into "Peresmeshnik Mic"."""

    def __init__(self, latency_ms: int = 40):
        self.latency_ms = latency_ms
        self._modules = self._make()
        self._q: queue.Queue = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._sr = 0
        self._end = 0.0  # monotonic time when the audio written so far finishes playing
        self._lock = threading.Lock()  # play / flush / drain from different threads (barge-in, mode switch)
        self._writer = threading.Thread(target=self._write, name='virtual-mic', daemon=True)
        self._writer.start()

    def _make(self) -> list[str]:
        if not shutil.which('pactl') or not shutil.which('pacat'):
            raise RuntimeError('pactl/pacat not found: install pulseaudio-utils (PipeWire: pipewire-pulse)')
        _remove_stale()
        # never the default output or input in place of another device (the system's sound would go into the call,
        # other apps would record our translation). Negative, not 0: WirePlumber 0.4 counts a device without a
        # priority as -1, so 0 would win over it (measured); real ALSA / Bluetooth devices have ~1000-2000
        low = 'priority.session=-10'
        return [
            _pactl('load-module', 'module-null-sink', f'sink_name={SINK}', 'channels=1',
                   f"sink_properties='device.description=\"{NAME} (feed)\" {low}'").strip(),
            _pactl('load-module', 'module-remap-source', f'master={SINK}.monitor', f'source_name={SOURCE}', 'channels=1',
                   f"source_properties='device.description=\"{NAME}\" {low}'").strip(),
        ]

    def _open(self, sample_rate: int) -> None:
        self._proc = subprocess.Popen(
            ['pacat', '--playback', f'--device={SINK}', '--format=float32le', f'--rate={sample_rate}', '--channels=1',
             '--raw', f'--latency-msec={self.latency_ms}', '--client-name=Peresmeshnik', f'--stream-name={NAME}',
             '--property=node.dont-reconnect=true'],  # else PipeWire moves it to the next new sink (e.g. the call capture)
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._sr = sample_rate

    def _write(self) -> None:
        while (item := self._q.get()) is not None:
            proc, data, seconds = item
            try:
                if data is None:  # end of this stream. pacat spins a core if EOF comes while it still
                    time.sleep(max(0.0, self._end - time.monotonic()))  # holds audio: close once played
                    proc.stdin.close()
                    proc.wait()
                else:
                    self._end = max(self._end, time.monotonic()) + seconds
                    proc.stdin.write(data)
                    proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):  # that pacat was flushed or stopped
                pass

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        data = np.ascontiguousarray(pcm, dtype='<f4').tobytes()
        with self._lock:
            if self._proc is not None and sample_rate != self._sr:  # new rate: the old stream plays out first
                self._q.put((self._proc, None, 0.0))
                self._proc = None
            if self._proc is None:
                self._open(sample_rate)
            self._q.put((self._proc, data, len(pcm) / sample_rate))

    def flush(self) -> None:
        """Drop queued audio, including what pacat already buffered."""
        with self._lock:
            procs = [self._proc] if self._proc else []
            try:
                while True:
                    if item := self._q.get_nowait():
                        procs.append(item[0])
            except queue.Empty:
                pass
            for p in {id(p): p for p in procs}.values():  # the current stream and any still queued
                self._stop_proc(p)
            self._proc, self._end = None, 0.0

    @staticmethod
    def _stop_proc(p: subprocess.Popen) -> None:
        p.kill()
        p.wait()

    def drain(self) -> None:
        """Block until everything queued so far has been played (end of a file run)."""
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is not None:
                self._q.put((proc, None, 0.0))
        if proc is not None:
            proc.wait()

    def stop(self) -> None:
        """Close the playback stream; the device stays for the call app."""
        self.flush()

    def remove(self) -> None:
        """Stop and delete the device (app exit)."""
        self.stop()
        self._q.put(None)
        self._writer.join()
        for idx in reversed(self._modules):
            subprocess.run(['pactl', 'unload-module', idx], capture_output=True)
        self._modules = []

"""Linux virtual microphone "Peresmeshnik Mic" on PipeWire (pipewire-pulse) or PulseAudio.

    null sink  peresmeshnik_mic_sink  <- our translated speech (pacat)
    remap source  peresmeshnik_mic  ("Peresmeshnik Mic") on the sink's monitor -> the call app's mic

Call apps hide monitor sources, hence the remap source. The device lives while the app runs
(`remove()` on exit; a stale one left by a crash is replaced on the next start). Playback runs
only between the first play() and stop(): one `pacat` process fed by a writer thread, so
play() never blocks and an idle direction has no open stream.
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
        if not shutil.which('pactl') or not shutil.which('pacat'):
            raise RuntimeError('pactl/pacat not found: install pulseaudio-utils (PipeWire: pipewire-pulse)')
        self.latency_ms = latency_ms
        _remove_stale()
        self._modules = [
            _pactl('load-module', 'module-null-sink', f'sink_name={SINK}', 'channels=1',
                   f"sink_properties='device.description=\"{NAME} (feed)\"'").strip(),
            _pactl('load-module', 'module-remap-source', f'master={SINK}.monitor', f'source_name={SOURCE}', 'channels=1',
                   f"source_properties='device.description=\"{NAME}\"'").strip(),
        ]
        self._q: queue.Queue = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._sr = 0
        self._end = 0.0  # monotonic time when the audio written so far finishes playing
        self._writer = threading.Thread(target=self._write, name='virtual-mic', daemon=True)
        self._writer.start()

    def _open(self, sample_rate: int) -> None:
        self._close_proc()
        self._proc = subprocess.Popen(
            ['pacat', '--playback', f'--device={SINK}', '--format=float32le', f'--rate={sample_rate}', '--channels=1',
             '--raw', f'--latency-msec={self.latency_ms}', '--client-name=Peresmeshnik', f'--stream-name={NAME}'],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._sr = sample_rate

    def _write(self) -> None:
        while (item := self._q.get()) is not None:
            proc, data, seconds = item
            try:
                if data is None:  # drain(). pacat spins a core if EOF comes while it still holds audio,
                    time.sleep(max(0.0, self._end - time.monotonic()))  # so close only when it is played
                    proc.stdin.close()
                else:
                    self._end = max(self._end, time.monotonic()) + seconds
                    proc.stdin.write(data)
                    proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):  # that pacat was flushed or stopped
                pass

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        if self._proc is None or sample_rate != self._sr:
            self._open(sample_rate)
        self._q.put((self._proc, np.ascontiguousarray(pcm, dtype='<f4').tobytes(), len(pcm) / sample_rate))

    def flush(self) -> None:
        """Drop queued audio, including what pacat already buffered."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._close_proc()
        self._end = 0.0

    def _close_proc(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def drain(self) -> None:
        """Block until everything queued so far has been played (end of a file run)."""
        proc, self._proc = self._proc, None
        if proc is not None:
            self._q.put((proc, None, 0.0))
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

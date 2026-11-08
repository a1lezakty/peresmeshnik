"""Stock-voice TTS: Piper in its own process (GPL, piper-runner/), voices with free data licenses."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import struct
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from peresmeshnik.engines import Voice

RUNNER = Path(__file__).resolve().parents[2] / 'piper-runner' / 'piper_runner.py'


def voice_path(voice_id: str) -> Path:
    """The .onnx of a registry voice `piper-<name>` (its .onnx.json sits next to it)."""
    from peresmeshnik import models
    return models.path(voice_id) / f"{voice_id.removeprefix('piper-')}.onnx"


def runner_cmd() -> list[str]:
    if os.environ.get('PERESMESHNIK_PIPER_CMD'):
        return shlex.split(os.environ['PERESMESHNIK_PIPER_CMD'], posix=os.name != 'nt')
    return [shutil.which('uv') or 'uv', 'run', '--no-project', '--with', 'piper-tts==1.8.0', 'python', str(RUNNER)]


class PiperTts:
    """One runner process per voice; synth() yields one float32 chunk per sentence."""

    def __init__(self, voice_id: str = 'piper-ru_RU-dmitri-medium', threads: int = 2):
        self.voice_id = voice_id
        onnx = voice_path(voice_id)
        self._p = subprocess.Popen([*runner_cmd(), str(onnx), '--threads', str(threads)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.sample_rate = self._u32()

    def _read(self, n: int) -> bytes:
        b = self._p.stdout.read(n)
        if len(b) < n:
            raise RuntimeError(f'piper runner exited ({self._p.poll()})')
        return b

    def _u32(self) -> int:
        return struct.unpack('<I', self._read(4))[0]

    def synth(self, text: str, lang: str, voice: Voice | None = None, speed: float = 1.0) -> Iterator[np.ndarray]:
        """`speed` > 1 speaks faster (Piper's length scale; the pipeline uses it when the voice falls behind)."""
        self._p.stdin.write(json.dumps({'text': text, 'speed': speed}).encode() + b'\n')  # ASCII JSON: no console code page issues
        self._p.stdin.flush()
        try:
            while n := self._u32():
                yield np.frombuffer(self._read(n), '<f4')
        except GeneratorExit:  # the caller stopped early: read the rest, or it would be the next request's audio
            while n := self._u32():
                self._read(n)
            raise

    def close(self) -> None:
        self._p.stdin.close()
        try:
            self._p.wait(5)
        except subprocess.TimeoutExpired:
            self._p.kill()

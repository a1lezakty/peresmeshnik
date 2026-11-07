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

VOICES = {'en': 'en/en_US/john/medium/en_US-john-medium',  # public domain
          'ru': 'ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium'}  # CC0
PIPER_VOICES = ('rhasspy/piper-voices', 'e7229f7d33a10e145e109b886d661984201ac3c4')
RUNNER = Path(__file__).resolve().parents[2] / 'piper-runner' / 'piper_runner.py'


def voice_path(lang: str) -> Path:
    from huggingface_hub import hf_hub_download
    repo, rev = PIPER_VOICES
    hf_hub_download(repo, VOICES[lang] + '.onnx.json', revision=rev)
    return Path(hf_hub_download(repo, VOICES[lang] + '.onnx', revision=rev))


def runner_cmd() -> list[str]:
    if os.environ.get('PERESMESHNIK_PIPER_CMD'):
        return shlex.split(os.environ['PERESMESHNIK_PIPER_CMD'])
    return [shutil.which('uv') or 'uv', 'run', '--no-project', '--with', 'piper-tts==1.8.0', 'python', str(RUNNER)]


class PiperTts:
    """One runner process per voice; synth() yields one float32 chunk per sentence."""

    def __init__(self, lang: str, voice_onnx: str | Path | None = None, threads: int = 2):
        self.lang = lang
        onnx = voice_onnx or voice_path(lang)
        self._p = subprocess.Popen([*runner_cmd(), str(onnx), '--threads', str(threads)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.sample_rate = self._u32()

    def _read(self, n: int) -> bytes:
        b = self._p.stdout.read(n)
        if len(b) < n:
            raise RuntimeError(f'piper runner exited ({self._p.poll()})')
        return b

    def _u32(self) -> int:
        return struct.unpack('<I', self._read(4))[0]

    def synth(self, text: str, lang: str, voice: Voice | None = None) -> Iterator[np.ndarray]:
        self._p.stdin.write(json.dumps({'text': text}).encode() + b'\n')  # ASCII JSON: no console code page issues
        self._p.stdin.flush()
        while n := self._u32():
            yield np.frombuffer(self._read(n), '<f4')

    def close(self) -> None:
        self._p.stdin.close()
        try:
            self._p.wait(5)
        except subprocess.TimeoutExpired:
            self._p.kill()

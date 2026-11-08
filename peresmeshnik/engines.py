"""Engine interfaces: the contract between pipeline stages, for local and cloud engines alike.

Rules every implementation follows (docs/ARCHITECTURE.md, "CPU and memory"):
- audio callbacks only enqueue frames; all methods here run on worker threads;
- every engine takes an explicit thread count, never "all cores";
- nothing polls or spins: when translation is off, no stream is open and no engine thread is busy.
Internal audio is mono float32 PCM at 16 kHz unless a sample rate is given.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000


@dataclass
class Caption:
    """Subtitle update: original text from ASR, translation once MT has it. `clause` numbers the
    clauses of a direction (-1: not known): a partial of clause n + 1 can come before the final of n."""
    original: str
    translation: str = ''
    final: bool = False
    clause: int = -1


@dataclass
class Speech:
    """Translated audio to play (headphones or the virtual microphone)."""
    pcm: np.ndarray
    sample_rate: int


@dataclass
class Voice:
    """The user's cloned voice: made once from a 30-60 s recording, stored in the user data dir."""
    name: str
    engine: str
    path: Path
    meta: dict = field(default_factory=dict)


class AudioIn(Protocol):
    """Microphone, WASAPI loopback, macOS process tap or PipeWire monitor."""
    def start(self, on_frames: Callable[[np.ndarray], None]) -> None: ...  # 16 kHz mono; callback only enqueues
    def stop(self) -> None: ...  # closes the device stream


class AudioOut(Protocol):
    """Headphones or the virtual microphone."""
    def play(self, pcm: np.ndarray, sample_rate: int) -> None: ...  # enqueue and return
    def flush(self) -> None: ...  # drop queued audio (mode switch, barge-in)
    def stop(self) -> None: ...


class Vad(Protocol):
    def feed(self, frame: np.ndarray) -> Iterator[tuple[str, np.ndarray]]: ...  # ('start' | 'speech' | 'end', pcm)


class AsrStream(Protocol):
    def feed(self, pcm: np.ndarray) -> str | None: ...  # new partial text of the current segment, or None
    def finish(self) -> str: ...  # final text of the segment


class Asr(Protocol):
    def stream(self, lang: str) -> AsrStream: ...


class Translator(Protocol):
    def translate(self, text: str, src: str, tgt: str, context: tuple[str, ...] = ()) -> str: ...


class Tts(Protocol):
    sample_rate: int
    def synth(self, text: str, lang: str, voice: Voice | None = None) -> Iterator[np.ndarray]: ...  # audio chunks


class VoiceCloner(Protocol):
    def enroll(self, pcm: np.ndarray, sample_rate: int, transcript: str | None = None) -> Voice: ...


class Direction(Protocol):
    """One translation direction ("them -> me" or "me -> them") behind one interface:
    local = Vad + Asr + Translator + Tts composed by the pipeline; cloud = one Gemini Live session."""
    def feed(self, pcm: np.ndarray) -> None: ...
    def events(self) -> Iterator[Caption | Speech]: ...  # blocks until the next event
    def close(self) -> None: ...  # free models / close the session; idle again

    supports_live: bool = False  # True where set_live() works; the app checks getattr(d, 'supports_live', False)

    def set_live(self, on: bool) -> None:
        """Hotkey "translation <-> live voice". on=True: the direction passes its input through as `Speech`
        (crossfaded, no click) and stops translating; False: back to translation. The app keeps feeding it."""
        raise NotImplementedError


class Overlay(Protocol):
    def show(self, caption: Caption) -> None: ...  # repaint only when the text changed
    def hide(self) -> None: ...


class Hotkey(Protocol):
    def register(self, combo: str, callback: Callable[[], None]) -> None: ...
    def unregister(self, combo: str) -> None: ...

"""Clause joins (pipeline._speak): each clause is synthesized on its own and the clauses play back to back, so a sentence
cut by length (`max_clause_s`) had a pause and a jump in loudness in its middle that the same sentence said whole has not
(research 2026-11-16 VOICE §1.3, B.5; measured again by bench/joins.py: Pocket, 3 voices x 8 sentences). `Joins` takes each
clause's chunks as they come:

- its synthesized silence is cut to EDGE_S at its start and, where the sentence goes on (the clause was cut by length), at
  its end; at a sentence's end to PAUSE_S. Silence is held back only until more speech or the end shows what it was: speech
  is never held up by it, and the pause before the first word no longer waits in the queue (Pocket-en: ~0.3 s of it).
- a clause that goes on with the sentence of the one before starts at the loudness that one ended at, plus RESET_DB (as
  in a sentence said whole), and comes back to its own over RAMP_S. Not the clauses' whole loudness: that is the same
  within ~1 dB; the jump is each clause's own fall from its loud start to its end (6-9 dB). For it LOOK_S of the clause's
  speech is held back first, mostly while the clause before still plays.

No crossfade: the joins have no clicks (the night's metric), they fall in silence.
"""
from __future__ import annotations

import itertools
import re

import numpy as np

FRAME_S = 0.01  # silence is judged in frames this long
# a frame is sound when it is NOISE_DB above the clause's quietest (Pocket's silence is its sample's room noise, or -80 dBFS)
# or within QUIET_DB of its loudest (before its first sound: of the clause before, TOP_DB at first): a soft word end keeps
NOISE_DB, QUIET_DB, TOP_DB = 20.0, 30.0, -15.0
EDGE_S = 0.05  # silence kept at a join inside a sentence, each side: ~0.1 s between the clauses (whole: 0-0.1 s)
PAUSE_S = 0.25  # ... after a sentence's end
LOOK_S = 0.4  # the start of a clause that goes on: its loudness from this much of its speech (the night's metric window)
RESET_DB = 1.5  # it starts this much louder than the clause before ended (a sentence said whole: +1.5..+1.9 dB, median)
RAMP_S = 1.0  # ... and is back at its own loudness after this long
MAX_DB = 6.0  # it is turned down at most this much, never up: a clause that ended in a whisper (-35 dBFS) does not make
# the next one start as quiet (bench/joins.py: at 12 dB such starts fell under the speech of their own clause)
FADE_S = 0.005  # where silence is cut the sound fades in / out: Pocket's silence is room noise, cut short it clicked


def goes_on(text: str) -> str:
    """The text of a clause cut by length for the TTS: the sentence goes on, so no full stop (MT may add one; Pocket ends a
    text that ends in a word with one, local/pocket.prompt). A trailing comma says so to every engine."""
    core = re.sub(r'[\s.,;:…]+$', '', text)
    return core if not core or core[-1] in '!?' else core + ','


def speech_db(x: np.ndarray, sr: int, head: bool, n: int) -> float | None:
    """Loudness (dB) of the first (`head`) or last `n` samples of the speech in `x`: its 20 ms frames within 30 dB of its
    loudest, as the night's join metric (research r2/stitch.py)."""
    f = sr // 50
    k = len(x) // f
    if not k:
        return None
    fr = x[:k * f].reshape(k, f)
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    v = fr[e > e.max() - 30].reshape(-1)
    v = v[:n] if head else v[-n:]
    return float(10 * np.log10(np.mean(v ** 2) + 1e-12)) if len(v) else None


def fade(x: np.ndarray, sr: int, start: bool) -> np.ndarray:
    """`x` faded in (`start`) or out over FADE_S."""
    n = min(round(FADE_S * sr), len(x))
    x = x.copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    if start:
        x[:n] *= ramp
    else:
        x[len(x) - n:] *= ramp[::-1]
    return x


class Joins:
    """One per direction: the clauses of its voice in the order said."""

    def __init__(self):
        self.end_db: float | None = None  # how loud the last clause ended (its last LOOK_S of speech, as played)
        self.cut = False  # the last clause was cut by length: the next one goes on with its sentence
        self.top = TOP_DB  # the last clause's loudest 10 ms frame, dBFS

    def __call__(self, chunks, sr: int, cut: bool = False):
        """The chunks of one clause (float32 at `sr`), joined to the clause before. `cut`: this clause was cut by length."""
        goes, self.cut = self.cut, cut
        start, self.end_db = self.end_db, None  # a clause broken off (barge-in, close) leaves no level to join to
        said = []
        for c in self._level(self._trim(chunks, sr, EDGE_S if cut else PAUSE_S), sr, start if goes else None):
            said.append(c)
            yield c
        if said:
            self.end_db = speech_db(np.concatenate(said), sr, False, round(LOOK_S * sr))

    def _trim(self, chunks, sr: int, tail_s: float):
        """The chunks with the silence before the first sound cut to EDGE_S and after the last to `tail_s`; silence in
        between stays (held back until sound follows). A clause with no sound at all goes out as it came."""
        f, lead, tail = round(sr * FRAME_S), round(sr * EDGE_S), round(sr * tail_s)
        x = np.zeros(0, np.float32)  # held back: judged silence since the last sound, then less than a frame not judged
        judged, floor, top, spoke, came, cut = 0, np.inf, -np.inf, False, [], False
        try:
            for c in chunks:
                if not spoke:
                    came.append(c)
                x = np.concatenate([x, np.asarray(c, np.float32)])
                k = (len(x) - judged) // f
                if not k:
                    continue
                db = 10 * np.log10(np.mean(x[judged:judged + k * f].reshape(k, f) ** 2, axis=1) + 1e-12)
                if (live := db[db > -100]).size:  # digital silence is no noise floor
                    floor = min(floor, float(live.min()))
                loud = db > min(floor + NOISE_DB, (top if spoke else self.top) - QUIET_DB)
                if loud.any():
                    top = max(top, float(db.max()))
                    a = judged + int(np.argmax(loud)) * f  # the first frame of sound
                    b = judged + (int(np.flatnonzero(loud)[-1]) + 1) * f  # the end of the last
                    y = x[0 if spoke else max(a - lead, 0):b]
                    if not spoke and (cut or a > lead):
                        y = fade(y, sr, True)
                    yield y
                    x, judged, spoke, came = x[b:], judged + k * f - b, True, []
                else:
                    judged += k * f
                    if not spoke and judged > lead:  # before the first sound only the last EDGE_S is kept
                        x, judged, cut = x[judged - lead:], lead, True
        finally:
            getattr(chunks, 'close', lambda: None)()  # broken off: the engine reads away the rest (local/tts.py)
        if spoke:
            self.top = top
            if tail and len(x):
                yield fade(x[:tail], sr, False) if len(x) > tail else x
        else:
            yield from came

    @staticmethod
    def _level(chunks, sr: int, end_db: float | None):
        """`chunks` as they come; with `end_db` (the clause before ended this loud, the sentence goes on) its start is
        turned down to end_db + RESET_DB and back over RAMP_S."""
        if end_db is None:
            yield from chunks
            return
        look, head, n = round(LOOK_S * sr), [], 0
        for c in chunks:
            head.append(np.asarray(c, np.float32))
            n += len(c)
            if n >= look:
                break
        if not head:
            return
        x = np.concatenate(head)
        start = speech_db(x, sr, True, look)
        g0 = 0.0 if start is None else min(0.0, max(-MAX_DB, end_db + RESET_DB - start))
        ramp, at = round(RAMP_S * sr), 0
        for c in itertools.chain([x], chunks):
            if g0 and at < ramp:
                t = np.arange(at, at + len(c))
                c = c * (10 ** (g0 * np.clip(1 - t / ramp, 0, 1) / 20)).astype(np.float32)
            at += len(c)
            yield c

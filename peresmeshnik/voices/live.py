"""Their voice on the fly (them -> me): the translation in the voice of the person talking, only in my headphones.

RESEARCH-clone §4. The them side's pipeline hands on each clause the ASR heard words in (`LocalDirection.on_clause`):
its 16 kHz audio, after the echo gate. A clause goes into their voice sample only if it is at least MIN_CLAUSE_S long,
passes voice.check's level, clipping and noise limits, and nothing we said into the call played around it: the echo
gate zeroes our echo alone but lets double talk through, and that carries my own (cloned) voice. A worker thread of its
own, not the pipeline's, does the rest: it loads the clone engine and the speaker vector model (so the side starts as
fast as before), gives each clause its speaker vector, finds whose it is, and makes that person's voice (the engine's
state) at READY_S[0] seconds of their speech and again at the next ones (the cleanest BEST_S of it), then keeps it.

`TheirVoice` is the side's `Tts`. Each clause is said in the voice of the person who said it: the worker knows whose
it is before the clause has been translated (a clause too short or unclear for a vector: the person of the clause
before), and the speak worker waits up to WAIT_S for that. Until their voice is made, the stock voice says it, not
someone else's. The voice is taken when a clause starts, so a voice made again never changes a clause half said. A
voice kept from an earlier call (`voice.save_them`, with its speaker vector) is known by that vector and speaks from
their first clause. Nothing here is written to disk: `them()` hands the voice of the person who spoke last to the
tray, which keeps it only when the user asks and the other person agreed.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from peresmeshnik.audio.resample import Resampler
from peresmeshnik.engines import SAMPLE_RATE, Voice

log = logging.getLogger('peresmeshnik.live')
MIN_CLAUSE_S = 1.0  # shorter clauses: their speaker vector is unreliable (S3's threshold was measured on 3-8 s)
READY_S = (3.0, 5.0, 10.0, 20.0)  # their clean speech when the voice is made, and made again; then it stays (§4.2)
BEST_S = 10.0  # the sample: their cleanest clauses up to this long (longer sounds no closer; Pocket breaks at ~40 s)
# cosine of a clause's speaker vector to a person's mean vector: the same person. ponytail: S3's threshold (5 s live vs
# a file, same-gender strangers pass <= 0.64 %); clauses are shorter but of one call, BENCH stage 8 S2 checks it
SAME = 0.45
WAIT_S = 0.5  # the speak worker waits this long to know whose clause it is (the worker is making a voice), then guesses
OUT_SR = 24_000  # the side's audio: the clone engines' (Pocket, XTTS); the stock voice is resampled to it
GAP = np.zeros(SAMPLE_RATE // 10, np.float32)  # between clauses in the sample
RTF_AFTER_S = 4.0  # their cloned voice's speed is checked once, after this much of it (me -> them: a ~4 s test phrase)


@dataclass
class LiveProfile(Voice):
    """A voice made during the call: no folder, the engine's state in memory (local/pocket.py reads `state`)."""
    state: np.ndarray | None = None


@dataclass
class Person:
    vec: np.ndarray  # speaker vectors, summed weighted by seconds (a kept voice: its own)
    seconds: float = 0.0  # their speech taken
    clips: list = field(default_factory=list)  # (snr dB, 16 kHz pcm) of their clean clauses, until the voice stays
    voice: LiveProfile | None = None
    made_at: float = 0.0  # seconds when the voice was made
    kept: bool = False  # from an earlier call: it stays as it is

    def similarity(self, v: np.ndarray) -> float:
        return float(v @ self.vec / (np.linalg.norm(self.vec) + 1e-12))


class Clause:
    """Whose clause it is, once the worker knows (`person`: None for nobody yet)."""

    def __init__(self):
        self.person, self.known = None, threading.Event()


def clean(pcm: np.ndarray) -> float | None:
    """The clause's speech above its background (dB), or None when it is too quiet, clipped or noisy for a sample
    (voice.check's limits; its length limits are for a recording, not a clause)."""
    from peresmeshnik import voice
    r = voice.check(pcm, SAMPLE_RATE)
    if r.clipped > voice.MAX_CLIPPED or r.level_db < voice.QUIET_DB or r.snr_db < voice.NOISY_DB:
        return None
    return r.snr_db


class LiveVoice:
    """Collects the voices of the call (see the module). `make_engine()` and `make_speaker()` run on its worker:
    the clone engine (`condition(pcm, sr) -> state`, `synth(text, lang, voice)`) and voice.Speaker. `kept`: voices
    from earlier calls, [(speaker vector, engine state, seconds)]. `echo`: the side's audio.echo.EchoGate. `note(why)`:
    said once when the clone fails and the stock voice takes over.
    Clauses go to the worker in order (those without audio for the sample too): who said one is decided in order."""

    def __init__(self, make_engine, make_speaker, engine_id: str = '', lang: str = '', kept=(), echo=None, note=None):
        self.engine, self.engine_id, self.lang, self.dead, self.note = None, engine_id, lang, '', note
        self.people: list[Person] = [Person(np.asarray(v, np.float32), s, voice=LiveProfile('them', engine_id, None, {
            'kind': 'them'}, state=st), made_at=s, kept=True) for v, st, s in kept]
        self.speaking: Person | None = None  # who said the last clause the worker went through
        self.speaker = None
        self.ref, self.pad = (echo.ref, float(echo.delays[-1]) + 0.2) if echo is not None else (None, 0.0)
        self.stats = {'clauses': 0, 'short': 0, 'echo': 0, 'noisy': 0, 'made': 0, 'late': 0}
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._work, args=(make_engine, make_speaker), name='their-voice',
                                        daemon=True)
        self._thread.start()

    # the pipeline's listen worker: only checks and enqueues
    def offer(self, pcm: np.ndarray, text: str, t0: float, t1: float) -> Clause:
        """A clause the ASR heard words in -> its Clause, the `voice` the pipeline hands to synth() for it."""
        c = Clause()
        self.stats['clauses'] += 1
        if len(pcm) < MIN_CLAUSE_S * SAMPLE_RATE:
            self.stats['short'] += 1
            pcm = None
        elif self.ref is not None:  # we said something into the call around it: their audio may carry it back
            from peresmeshnik.audio.echo import HOP, LOUD
            if self.ref.features(np.arange(t0 - self.pad, t1, HOP / SAMPLE_RATE)).max() > LOUD:
                self.stats['echo'] += 1
                pcm = None
        self._q.put((pcm, c))
        return c

    # the speak worker
    def voice(self, clause: Clause | None = None) -> LiveProfile | None:
        """The voice to say `clause` in: its person's; None: theirs is not made yet (the stock voice speaks it)."""
        if self.dead:
            return None
        if clause is not None and not clause.known.wait(WAIT_S):
            self.stats['late'] += 1
        p = clause.person if clause is not None and clause.known.is_set() else self.speaking
        return p.voice if p is not None else None

    def them(self) -> tuple[dict, str, float] | None:
        """The voice of the person who spoke last, for the tray's "keep the voice" (voice.save_them): ({model id: (state, sha256 of its
        weights)}: the engine's and the speaker vector's, their language, seconds of their speech); None: none yet, or
        an engine of the voice pack (NVIDIA), whose voice is the runner's for this call only (COMMON-stage8 23:49)."""
        from peresmeshnik import languages, models
        from peresmeshnik.voice import SPEAKER
        p = self.speaking
        if p is None or p.voice is None or self.dead or languages.needs_gpu(self.engine_id):
            return None
        return ({self.engine_id: (p.voice.state, models.get(self.engine_id).fingerprint),
                 SPEAKER: (p.vec / (np.linalg.norm(p.vec) + 1e-12), models.get(SPEAKER).fingerprint)},
                self.lang, round(p.seconds, 1))

    def fail(self, why) -> None:
        """The clone failed: the stock voice for the rest of the call."""
        if not self.dead:
            self.dead = str(why) or type(why).__name__
            log.warning('their voice: the clone failed, the stock voice speaks: %s', why)
            if self.note is not None:
                self.note(self.dead)

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(5)
        if self.engine is not None and hasattr(self.engine, 'close'):
            self.engine.close()

    # the worker
    def _work(self, make_engine, make_speaker) -> None:
        try:
            self.engine, self.speaker = make_engine(), make_speaker()
            if not hasattr(self.engine, 'condition'):  # a voice only from a folder (the voice pack before its condition)
                self.engine.close()
                self.engine = None
                self.fail(f'{self.engine_id} cannot take a voice during a call')
        except Exception as e:  # weights broken, onnxruntime refuses: the stock voice speaks, as before
            self.fail(e)
        while (item := self._q.get()) is not None:
            pcm, c = item
            try:
                if pcm is not None and not self.dead:
                    self._add(pcm, c)
            except Exception as e:
                self.fail(e)
            finally:  # nobody new: the person of the clause before
                if not c.known.is_set():
                    c.person = self.speaking
                    c.known.set()

    def _add(self, pcm: np.ndarray, c: Clause | None = None) -> None:
        if (snr := clean(pcm)) is None:
            self.stats['noisy'] += 1
            return
        v, sec = self.speaker.embed(pcm, SAMPLE_RATE), len(pcm) / SAMPLE_RATE
        p = max(self.people, key=lambda q: q.similarity(v), default=None)
        if p is None or p.similarity(v) < SAME:
            p = Person(np.zeros_like(v))
            self.people.append(p)
        self.speaking = p
        if c is not None:  # known before their voice is made again: the clause need not wait for that
            c.person = p
            c.known.set()
        if not p.kept and p.seconds < READY_S[-1]:
            p.vec += sec * v
            p.seconds += sec
            p.clips.append((snr, pcm))
            if any(p.made_at < s <= p.seconds for s in READY_S):
                self._make(p)

    def _make(self, p: Person) -> None:
        """Their voice from their cleanest BEST_S (in the order said); handed over whole, between clauses."""
        best, n = set(), 0.0
        for i in sorted(range(len(p.clips)), key=lambda i: -p.clips[i][0]):
            if n >= BEST_S:
                break
            best.add(i)
            n += len(p.clips[i][1]) / SAMPLE_RATE
        sample = np.concatenate([x for i, (_, pcm) in enumerate(p.clips) if i in best for x in (pcm, GAP)])
        state = self.engine.condition(sample, SAMPLE_RATE)
        p.voice, p.made_at = LiveProfile('them', self.engine_id, None, {'kind': 'them'}, state=state), p.seconds
        self.stats['made'] += 1
        if p.seconds >= READY_S[-1]:
            p.clips = []  # it stays: their speech is not kept any longer
        log.info('their voice made from %.1f s of speech (%d people in the call)', p.seconds, len(self.people))


def faster(chunks, speed: float, sr: int = OUT_SR):
    """The speech of `chunks` `speed` times faster at the same pitch (WSOLA): 40 ms Hann frames at half overlap, each
    taken from the input where it best continues the one before (within 5 ms of where the pace puts it). The clone
    has no speed of its own and speaks at its sample's pace (BENCH stage 8: 37 % slower than Piper); the pipeline asks
    for this when the voice is behind (pipeline.MAX_SPEED)."""
    if speed <= 1.0:
        yield from chunks
        return
    n = sr // 25
    hs, tol = n // 2, sr // 200
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)  # periodic: halves sum to 1
    x, x0, k, nat, tail = np.zeros(0, np.float32), 0, 0, None, np.zeros(hs, np.float32)  # x[0] is input sample x0
    for c in chunks:
        x = np.concatenate([x, np.asarray(c, np.float32)])
        while (pos := round(k * hs * speed)) + tol + hs + n <= x0 + len(x):
            lo = max(pos - tol, 0)
            at = lo
            if nat is not None:  # where the input looks most like the natural continuation of the last frame
                seg = x[lo - x0:pos + tol + n - x0]
                at = lo + int(np.argmax(np.correlate(seg, nat, 'valid')))
            frame = x[at - x0:at + n - x0] * win
            nat = x[at + hs - x0:at + hs + n - x0]
            yield tail + frame[:hs]
            tail, k = frame[hs:], k + 1
            cut = max(round(k * hs * speed) - tol, 0) - x0  # nothing before the next search is needed again
            x, x0 = x[cut:], x0 + cut
    yield tail


class TheirVoice:
    """The them side's `Tts`: their cloned voice once one is made, the stock voice until then and when the clone
    fails. Never into the call: app.Side refuses an output that goes there (`them`)."""
    them = True
    sample_rate = OUT_SR

    def __init__(self, stock, live: LiveVoice):
        self.stock, self.live = stock, live
        self._rtf: list | None = [0.0, 0.0]  # their voice's synthesis and audio seconds until checked (None: checked)

    def synth(self, text: str, lang: str, voice=None, speed: float = 1.0):
        """`voice`: the clause's Clause (LiveVoice.offer, through the pipeline); None: a clause of whoever spoke last."""
        if (v := self.live.voice(voice if isinstance(voice, Clause) else None)) is not None:
            said, eng = False, self.live.engine
            try:
                for chunk in self._at(faster(self._timed(eng.synth(text, lang, v), eng.sample_rate), speed,
                                             eng.sample_rate), eng.sample_rate):
                    said = True
                    yield chunk
                return
            except Exception as e:  # GeneratorExit (barge-in) is no Exception: it goes through
                self.live.fail(e)
                if said:  # the rest of this clause is lost, the next ones are said in the stock voice
                    return
        yield from self._at(self.stock.synth(text, lang, **({'speed': speed} if speed != 1.0 else {})),
                            self.stock.sample_rate)

    def _timed(self, chunks, sr: int):
        """The clone's chunks, its synthesis time counted until RTF_AFTER_S of speech: slower than app.MAX_CLONE_RTF,
        the computer cannot keep up with it and the stock voice speaks the rest of the call, said (as me -> them)."""
        t = time.perf_counter()
        for c in chunks:
            if self._rtf is not None:
                self._rtf[0] += time.perf_counter() - t
                self._rtf[1] += len(c) / sr if np.any(c) else 0  # not the pause after it (app.clone_rtf)
            yield c
            t = time.perf_counter()
        if self._rtf is not None and self._rtf[1] >= RTF_AFTER_S:
            from peresmeshnik.app import MAX_CLONE_RTF
            rtf, self._rtf = self._rtf[0] / self._rtf[1], None
            if rtf > MAX_CLONE_RTF:
                self.live.fail(f'this computer is too slow for their cloned voice: {rtf:.1f} s of synthesis per second '
                               'of speech')

    @staticmethod
    def _at(chunks, sr: int):
        rs = Resampler(sr, OUT_SR)
        for c in chunks:
            yield rs(np.asarray(c, np.float32))

    def close(self) -> None:
        self.live.close()
        if hasattr(self.stock, 'close'):
            self.stock.close()

    @property
    def _p(self):  # the Piper process: pipeline.cpu_s counts it
        return getattr(self.stock, '_p', None)


def attach(d, s, echo=None, note=None):
    """Their voice on the them side `d` (a LocalDirection with its stock voice), when `[local] their_voice` is on and a
    clone engine speaks the language here (languages.clone_engine) with its weights and the speaker vector model
    downloaded; else `d` as it was (logged: the stock voice speaks, as before). `note(why)`: LiveVoice's."""
    from peresmeshnik import languages, models
    from peresmeshnik.local.clone import make_engine
    if not s.local.their_voice:
        return d
    engine_id, why = languages.clone_engine(s.my_lang, 'them_to_me')
    if why:
        log.info('their voice: the stock voice speaks (%s)', why)
        return d
    try:
        from peresmeshnik.voice import SPEAKER, Speaker
        model_dir, spk_dir = models.path(engine_id), models.path(SPEAKER)
    except (models.ModelMissing, ImportError) as e:
        log.info('their voice: the stock voice speaks (%s)', e)
        return d
    threads = s.local.threads
    live = LiveVoice(lambda: make_engine(engine_id, model_dir, threads), lambda: Speaker(spk_dir), engine_id,
                     s.their_lang, kept(engine_id), echo, note)
    d.tts, d.on_clause = TheirVoice(d.tts, live), live.offer
    return d


def kept(engine_id: str, root=None) -> list:
    """The contacts' voices kept from earlier calls that this engine (these weights) and the speaker vector model can use:
    [(speaker vector, engine state, seconds)]."""
    from peresmeshnik import models, voice
    out, eng, spk = [], models.get(engine_id).fingerprint, models.get(voice.SPEAKER).fingerprint
    for v in voice.profiles(root):
        if v.meta.get('kind') == 'them':
            try:
                st, vec = voice.engine_state(v, engine_id, eng), voice.engine_state(v, voice.SPEAKER, spk)
            except (OSError, ValueError):  # a broken file: that voice is made again in the call
                continue
            if st is not None and vec is not None:
                out.append((vec, st, float(v.meta.get('seconds') or READY_S[-1])))
    return out

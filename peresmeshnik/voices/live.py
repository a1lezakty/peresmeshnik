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
MIN_CLAUSE_S = 1.0  # shorter clauses: their speaker vector is unreliable (S3's threshold was measured on 3-8 s): such a
# clause goes into nobody's sample and makes nobody new, it is only matched to the people known
MIN_EMBED_S = 0.25  # shorter still: no speaker vector at all, the stock voice says it
READY_S = (3.0, 5.0, 10.0, 20.0)  # their clean speech when the voice is made, and made again; then it stays (§4.2)
BEST_S = 10.0  # the sample: their cleanest clauses up to this long (longer sounds no closer; Pocket breaks at ~40 s)
# cosine of a clause's speaker vector to a person's mean vector: the same person. ponytail: S3's threshold (5 s live vs
# a file, same-gender strangers pass <= 0.64 %); clauses are shorter but of one call, BENCH stage 8 S2 checks it
SAME = 0.45
# a clause shorter than MIN_CLAUSE_S: matched only above this (review r3 4; C, LibriSpeech dev-clean, 6 speakers, 0.3-0.9 s
# pieces against a person's 20 s mean vector: another person passes 0.45 in 0.15-0.46 % of 2635 pairs, 0.55 in none;
# the same person passes 0.55 in 11 % under 0.6 s and 55 % at 0.6-1 s), below it the stock voice says it
SHORT_SAME = 0.55
WAIT_S = 0.5  # the speak worker waits this long to know whose clause it is (the worker is making a voice), then the
# stock voice says it
OUT_SR = 24_000  # the side's audio: the clone engines' (Pocket, XTTS); the stock voice is resampled to it
GAP = np.zeros(SAMPLE_RATE // 10, np.float32)  # between clauses in the sample
CLOSE_WAIT_S = 5.0  # translation off: waited for the worker this long; one still loading the engine frees it itself
START_WAIT_S = 180.0  # a new call's worker waits for the last call's to free its engine (a runner's start_s and more)
OURS = -4.5  # echo.bands' loudest band of a block where we spoke into the call (speech at -40 dBFS: most blocks above;
# the microphone's noise up to ~-47 dBFS below; echo.LOUD, -7, is noise at -60 dBFS already)
OURS_S = 0.25  # this much of our speech in a clause's echo window: it may carry our voice, not into their sample
KEEP_S = 10.0  # their clean speech before "keep their voice" keeps it (a voice of 3 s is a first guess)
RESET = object()  # LiveVoice.reset() in its worker's queue
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
    said: float = 0.0  # seconds of their clauses matched to them in this call (any, not only those for the sample)

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

    _closing: threading.Thread | None = None  # a closed call's worker still freeing its engine: ours loads after it

    def __init__(self, make_engine, make_speaker, engine_id: str = '', lang: str = '', kept=(), echo=None, note=None,
                 raw=None, me: str = ''):
        self.engine, self.engine_id, self.lang, self.dead, self.note = None, engine_id, lang, '', note
        self.me, self._my_vec = me, None  # my own voice profile: a voice like it is never kept as theirs
        self._kept = list(kept)
        self.people: list[Person] = self._known()
        self.speaking: Person | None = None  # who said the last clause the worker went through
        self.speaker = None
        self.gate, self.raw = echo, raw  # raw(): my microphone goes into the call as it is, not through our output
        self.rtf: list | None = [0.0, 0.0]  # their voice's synthesis and audio seconds until checked (None: checked)
        self._since = -np.inf  # reset() at: a clause heard before it is the call before's
        self.stats = {'clauses': 0, 'short': 0, 'echo': 0, 'noisy': 0, 'made': 0, 'late': 0, 'raw': 0}
        self._q: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._work, args=(make_engine, make_speaker, LiveVoice._closing),
                                        name='their-voice', daemon=True)
        self._thread.start()

    def _known(self) -> list[Person]:
        return [Person(np.asarray(v, np.float32), s, voice=LiveProfile('them', self.engine_id, None, {'kind': 'them'},
                                                                         state=st), made_at=s, kept=True)
                for v, st, s in self._kept]

    def reset(self) -> None:
        """A new call on the same engines (a server's warm side): the people of the last one, their speech and their
        voices go (the voice pack's runner forgets its states too), and so does the clone's failure or slowness: the next
        call checks it again; the engine stays loaded. A clause heard before this (one the pause after the last call
        closes, offered later) is nobody's."""
        self._since = time.perf_counter()
        self._q.put(RESET)

    # the pipeline's listen worker: only checks and enqueues
    def offer(self, pcm: np.ndarray, text: str, t0: float, t1: float) -> Clause:
        """A clause the ASR heard words in -> its Clause, the `voice` the pipeline hands to synth() for it."""
        c = Clause()
        self.stats['clauses'] += 1
        sample = True
        if t1 <= self._since:  # the call before's, closed by its pause after reset(): not into this call's people
            self._q.put((None, c, False))
            return c
        if len(pcm) < MIN_CLAUSE_S * SAMPLE_RATE:
            self.stats['short'] += 1
            sample = False
        elif self._ours(t0, t1):  # it may carry our own voice: matched to who said it, not into their sample
            self.stats['echo'] += 1
            sample = False
        self._q.put((pcm if len(pcm) >= MIN_EMBED_S * SAMPLE_RATE else None, c, sample))
        return c

    def _ours(self, t0: float, t1: float) -> bool:
        """We spoke into the call around this clause of theirs (times on the echo gate's clock): what came back may be
        in it (the gate zeroes our echo alone, double talk passes). The window is where our speech can sound in it: at
        the gate's delay once it has found one, else over all the delays it searches; or my raw microphone goes there."""
        if self.raw is not None and self.raw():
            self.stats['raw'] += 1
            return True
        g = self.gate
        if g is None:
            return False
        from peresmeshnik.audio.echo import HOP
        tail = 0.2  # the delay's jitter and the far room's reverb
        lo, hi = (t0 - g.delay - tail, t1 - g.delay + tail) if g.rho >= g.rho_on and g.gain is not None else (
            t0 - float(g.delays[-1]) - tail, t1 - float(g.delays[0]))
        f = g.ref.features(np.arange(lo, hi, HOP / SAMPLE_RATE)).max(axis=-1)
        return int((f > OURS).sum()) * HOP / SAMPLE_RATE >= OURS_S

    # the speak worker
    def voice(self, clause: Clause | None = None) -> LiveProfile | None:
        """The voice to say `clause` in: its person's; None (the stock voice says it): theirs is not made yet, nobody
        knows whose it is (a new person's short word), or not within WAIT_S. No voice made yet, or the engine not loaded
        yet (a kept voice: the worker loading it, or waiting for the last call's to be freed): no wait at all."""
        if self.dead or self.engine is None or not any(p.voice is not None for p in list(self.people)):
            return None
        if clause is None:
            p = self.speaking
        elif clause.known.wait(WAIT_S):
            p = clause.person
        else:
            self.stats['late'] += 1
            p = None
        return p.voice if p is not None else None

    def them(self) -> tuple[tuple[dict, str, float] | None, str]:
        """For the tray's "keep their voice" (voice.save_them), taken once the user said they agreed: (({model id:
        (state, sha256 of its weights)}: the engine's and the speaker vector's, their language, seconds of their speech),
        '') or (None, why not, for the user). Only in a call with one person on their side, after KEEP_S of their speech,
        not a voice like mine, not an engine of the voice pack (NVIDIA: its runner's for this call only, 23:49)."""
        from peresmeshnik import languages, models
        from peresmeshnik.voice import SPEAKER
        if self.dead:
            return None, f'their cloned voice stopped in this call ({self.dead})'
        if languages.needs_gpu(self.engine_id):
            return None, ('with the NVIDIA voice pack their voice lives only during the call, in the pack\'s own memory: '
                          'it cannot be kept')
        here = [p for p in list(self.people) if p.said >= READY_S[0]]
        if not here and self.stats['raw'] and self.raw is not None and self.raw():
            return None, ('your microphone goes into the call as it is (me → them is not running): their voice takes no '
                          'sample meanwhile')
        if len(here) > 1:
            return None, f'{len(here)} people speak in this call: a voice can be kept in a call with one of them only'
        p = here[0] if here else None
        if p is not None and p.kept:
            return None, 'their voice is kept already (Voices…)'
        if p is None or p.voice is None or p.seconds < KEEP_S:
            return None, f'not enough of their speech yet: {p.seconds if p else 0:.0f} of {KEEP_S:.0f} s'
        if (mine := self._mine()) is not None and p.similarity(mine) >= SAME:
            return None, 'this voice is too close to your own: it is not kept as theirs'
        return ({self.engine_id: (p.voice.state, models.get(self.engine_id).fingerprint),
                 SPEAKER: (p.vec / (np.linalg.norm(p.vec) + 1e-12), models.get(SPEAKER).fingerprint)},
                self.lang, round(p.seconds, 1)), ''

    def _mine(self) -> np.ndarray | None:
        """My own voice's speaker vector (from my profile's recording, once), None without one."""
        if self._my_vec is None and self.me and self.speaker is not None:
            from peresmeshnik import voice
            try:
                x, sr = voice.read_audio(voice.load(self.me).path / 'reference.wav')
                self._my_vec = self.speaker.embed(x, sr)
            except Exception:  # no profile, a damaged one: nothing to compare with
                self._my_vec = False
        return None if self._my_vec is None or self._my_vec is False else self._my_vec

    def fail(self, why) -> None:
        """The clone failed: the stock voice for the rest of the call."""
        if not self.dead:
            self.dead = str(why) or type(why).__name__
            log.warning('their voice: the clone failed, the stock voice speaks: %s', why)
            if self.note is not None:
                self.note(self.dead)

    def close(self) -> None:
        """Translation off: the worker drops the clauses left and frees the engine (a voice pack's runner) on its way
        out, also one it is still loading when this returns: nothing of the call outlives it."""
        self._closed = True
        self._q.put(None)
        self._thread.join(CLOSE_WAIT_S)
        if self._thread.is_alive():  # still loading: the next call's worker waits for it (one runner of theirs at a time)
            LiveVoice._closing = self._thread

    # the worker
    def _work(self, make_engine, make_speaker, last: threading.Thread | None = None) -> None:
        if last is not None:  # translation off and on again quickly: one runner of theirs on the card, not two
            last.join(START_WAIT_S)
        try:
            if not self._closed:
                self.engine = make_engine()
                if not hasattr(self.engine, 'condition'):  # a voice only from a folder (the voice pack before its condition)
                    raise RuntimeError(f'{self.engine_id} cannot take a voice during a call')
                self.speaker = make_speaker()
        except Exception as e:  # weights broken, onnxruntime refuses: the stock voice speaks, as before
            if (engine := self.engine) is not None:  # the speaker model failed after it: nothing of it is kept
                self.engine = None
                if hasattr(engine, 'close'):
                    engine.close()
            self.fail(e)
        while (item := self._q.get()) is not None:
            if item is RESET:
                self._forget()
                continue
            pcm, c, sample = item
            try:
                if pcm is not None and not self.dead and not self._closed:
                    self._add(pcm, c, sample)
            except Exception as e:
                self.fail(e)
            finally:  # not matched to anyone: nobody's voice, the stock one says it
                c.known.set()
        if self._closed and (engine := self.engine) is not None:
            self.engine = None
            if hasattr(engine, 'close'):
                engine.close()

    def _forget(self) -> None:
        """reset() in the worker, after the call before's clauses."""
        self.people, self.speaking, self.rtf = self._known(), None, [0.0, 0.0]
        if self.engine is not None:  # loaded: a failure in the call before (too slow, a crash in it) is checked again;
            self.dead = ''  # one of loading stays
            if hasattr(self.engine, 'forget_states'):  # the voice pack: the runner holds their states
                try:
                    self.engine.forget_states()
                except Exception as e:  # a dead runner: nothing of them in it any more
                    self.fail(e)

    def _add(self, pcm: np.ndarray, c: Clause | None = None, sample: bool = True) -> None:
        """Whose clause it is, and its audio into that person's sample when `sample` (long enough, none of our voice)
        and it is clean. A clause that is not for a sample is only matched: it makes nobody new."""
        if sample and (snr := clean(pcm)) is None:
            self.stats['noisy'] += 1
        sample = sample and snr is not None
        v, sec = self.speaker.embed(pcm, SAMPLE_RATE), len(pcm) / SAMPLE_RATE
        p = max(self.people, key=lambda q: q.similarity(v), default=None)
        if p is None or p.similarity(v) < (SAME if sec >= MIN_CLAUSE_S else SHORT_SAME):
            p = Person(np.zeros_like(v)) if sample else None
            if p is not None:
                self.people.append(p)
        self.speaking = p
        if p is not None:
            p.said += sec
        if c is not None:  # known before their voice is made again: the clause need not wait for that
            c.person = p
            c.known.set()
        if sample and not p.kept and p.seconds < READY_S[-1]:
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
    has no speed of its own and speaks at its sample's pace (BENCH stage 8: about Piper's); the pipeline asks
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
            if (r := self.live.rtf) is not None:
                r[0] += time.perf_counter() - t
                r[1] += len(c) / sr if np.any(c) else 0  # not the pause after it (app.clone_rtf)
            yield c
            t = time.perf_counter()
        if (r := self.live.rtf) is not None and r[1] >= RTF_AFTER_S:
            from peresmeshnik.app import MAX_CLONE_RTF
            rtf, self.live.rtf = r[0] / r[1], None
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


def attach(d, s, echo=None, note=None, raw=None):
    """Their voice on the them side `d` (a LocalDirection with its stock voice), when `[local] their_voice` is on and a
    clone engine speaks the language here (languages.clone_engine) with its weights and the speaker vector model
    downloaded; else `d` as it was (logged: the stock voice speaks, as before), whatever goes wrong here: the side
    starts all the same. `note(why)`, `raw()`: LiveVoice's."""
    try:
        from peresmeshnik import models
        from peresmeshnik.local.clone import make_engine
        engine_id, why = engine(s)
        if not engine_id:
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
                         s.their_lang, kept(engine_id), echo, note, raw, s.local.voice)
    except Exception:  # a broken profile folder, the registry: their voice is a nicety, the side is not
        log.exception('their voice: the stock voice speaks')
        return d
    d.tts, d.on_clause = TheirVoice(d.tts, live), live.offer
    return d


def forget(d) -> None:
    """A side's direction `d` is kept for another call (remote/server.py's warm side): their voices of this one go."""
    if (lv := getattr(getattr(d, 'tts', None), 'live', None)) is not None:
        lv.reset()


def engine(s) -> tuple[str | None, str]:
    """(the clone engine their voice is made in with settings `s` on this computer, '') or (None, why not): the side
    (attach), languages.needed_models and the Components page ask it, nobody else (languages.clone_engine's answer).
    With my own voice cloned here too, its engine goes `beside`: both must fit the NVIDIA card (mine Qwen3-TTS: on 8 GB
    theirs Pocket TTS on the CPU, on 12 GB XTTS-v2). Me -> them in the cloud or on my server speaks in my clone here
    when that fails (cloud/fallback.local_voice, if it is ready): it goes beside too."""
    from peresmeshnik import languages
    if not s.local.their_voice:
        return None, 'their voice is off ([local] their_voice)'
    me, beside = languages.direction(s, 'me_to_them'), ''
    if s.me_to_them == 'local':
        beside = me.tts if me.voice == 'clone' else ''
    elif me.voice == 'clone' and not languages.cloud_subtitles(s, 'me_to_them'):  # a local path to fall back on
        from peresmeshnik.cloud import fallback
        i = fallback.local_voice('me', s)[0]
        beside = i if i and fallback._is_clone(i) else ''
    return languages.clone_engine(s.my_lang, 'them_to_me', beside=beside)


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

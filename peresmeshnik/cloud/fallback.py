"""A cloud side that goes local when the cloud fails, and back when the cloud answers again (ARCHITECTURE §3).

`Fallback` is the side's `Direction`. While the cloud works, Gemini speaks and the local path (VAD + ASR + MT: the
side's subtitles, unless `cloud.subtitles`) runs beside it with its voice muted: the TTS wrapper `Muted` says a clause
only when it was translated while the side was local. When the cloud fails (`CloudDown`: the connection was lost and one
new one failed, an API error, the quota, no translated voice `cloud.fallback_after_s` after a phrase began):
- the local voice takes over at once: the clause in progress and the ones after it, and the muted clauses of the last
  seconds the cloud had not said yet. The cloud's voice runs its measured lag behind the speech, so a clause whose speech
  ended before "its last voiced chunk - lag" counts as said (an estimate within ~0.5 s: one short clause at most is
  said twice or not at all);
- with cloud subtitles there is no local path yet: it is made then and gets the kept input from that point on;
- one message in the tray; no local path here (models, the voice pack, no NVIDIA GPU for the clone): the reason.
Back, without a timer: when a phrase starts (a local partial subtitle) and the pause since the last try has passed (30 s,
doubling to 15 min while the cloud stays down), a new session is opened and nothing is sent to it; when the next phrase
starts and the session is up, the input goes to the cloud again from the phrase's first syllable (the last seconds of
input are kept for that), the local voice is muted from that clause on, one message. With no local path at all there are
no phrases to wait for: the next input after the pause tries, and a session that is up is used at once.

Threads: `feed()` (the capture's thread) only hands on; one thread per inner direction pumps its events into one queue;
`events()` (the side's thread) makes every decision.
"""
from __future__ import annotations

import collections
import importlib.util
import inspect
import logging
import queue
import shutil
import statistics
import threading
import time

from peresmeshnik.audio.resample import Resampler
from peresmeshnik.cloud.gemini import OUT_RATE, CloudDown
from peresmeshnik.engines import Caption, Latency, Speech

log = logging.getLogger(__name__)
SETTLE_S = 0.45  # a clause is translated this long after its speech ended (VAD pause 0.3 s, final ASR, MT: BENCH)
MARGIN_S = 0.5  # what ended this close after the estimated end of the cloud's voice counts as said
LAG_S = 3.5  # the cloud's voice behind the speech until this session measured it (done after end, 3.1-3.7 s, BENCH)
KEEP_S = 15.0  # input kept: for a new local path, and for the first syllables of a phrase going back to the cloud
PRE_S = 1.0  # back to the cloud: the input from this long before the phrase's first partial subtitle
RETRY_S, RETRY_MAX_S = 30.0, 900.0
ARROW = {'them': 'them → me', 'me': 'me → them'}
_END, _SAY = object(), object()


class Muted:
    """The local path's TTS beside a working cloud: `synth` says a clause only when `want(text)`. The real TTS (`make()`)
    is made at its first use, or ahead by `load()` (Piper: cheap, so the first fallback speaks at once). Its audio is
    resampled to one rate: the pipeline reads `sample_rate` once."""
    sample_rate = OUT_RATE

    def __init__(self, make):
        self.make, self.want, self.on_error = make, lambda text: False, lambda e: None
        self.lock = threading.Lock()  # one synthesis at a time: the speak worker's, a fallback's re-say of muted clauses
        self._tts, self.error = None, None

    def load(self) -> None:
        with self.lock:
            if self._tts is None and self.error is None:
                try:
                    self._get()
                except RuntimeError:  # said at the fallback (`error`)
                    pass

    def _get(self):  # under the lock
        if self._tts is None and self.error is None:
            try:
                self._tts = self.make()
            except Exception as e:  # a missing model or voice pack: the reason goes to the tray
                log.warning('local voice: %s', e)
                self.error = e
        if self._tts is None:
            raise RuntimeError(f'no local voice: {self.error}')
        return self._tts

    def synth(self, text, lang, voice=None, speed=1.0):
        if self.want(text):
            with self.lock:
                try:
                    yield from self.say(text, lang, voice, speed)
                except RuntimeError as e:  # no local voice after all (the clone failed to load): not the pipeline's death
                    self.on_error(e)

    def say(self, text, lang, voice=None, speed=1.0):  # under the lock
        tts = self._get()
        rs = Resampler(tts.sample_rate, self.sample_rate)
        kw = {'speed': speed} if speed != 1.0 and 'speed' in inspect.signature(tts.synth).parameters else {}
        for chunk in tts.synth(text, lang, voice, **kw):
            yield rs(chunk)

    def close(self) -> None:
        tts, self._tts = self._tts, None
        if tts is not None and hasattr(tts, 'close'):
            tts.close()


class _Noted:
    """The local path's MT: each clause's translation is noted (`made`) before the pipeline captions and speaks it."""

    def __init__(self, mt, made):
        self.mt, self.made = mt, made

    def translate(self, text, src, tgt, context=()):
        out = self.mt.translate(text, src, tgt, context)
        self.made(out)
        return out

    def close(self) -> None:
        if hasattr(self.mt, 'close'):
            self.mt.close()


class Fallback:
    """`make_cloud()`: a fail_fast GeminiDirection (opens its session); `make_local(tts)`: a LocalDirection speaking
    through `tts` (raises when its models are missing); `make_tts()`: the side's local voice; `voice_why`: why there is
    none here ('' = there is); `preload`: make it now (Piper), not at the first fallback (the clone, ~4 GB)."""
    supports_live = False

    def __init__(self, name: str, tgt: str, cloud_subtitles: bool, make_cloud, make_local, make_tts, voice=None,
                 voice_why: str = '', preload: bool = False, on_state=lambda text: None):
        self.name, self.tgt, self.voice, self.on_state = name, tgt, voice, on_state
        self.cloud_subtitles, self.make_cloud, self.make_local = cloud_subtitles, make_cloud, make_local
        self.muted = Muted(make_tts)
        self.muted.want = self._want
        self.muted.on_error = lambda e: self._q.put((self, e))
        self.voice_why = voice_why
        self.mode = 'cloud'  # 'cloud' | 'local' (the local voice speaks) | 'down' (no local voice here)
        self._lock = threading.Lock()  # mode, notes, input, targets: feed (capture), the local MT and speech, events
        self._made: collections.deque = collections.deque(maxlen=64)  # [translation, time, say it] not spoken yet
        self._muted: collections.deque = collections.deque(maxlen=8)  # (translation, time) muted while the cloud spoke
        self._input: collections.deque = collections.deque()  # (time, pcm) of the last KEEP_S
        self._warming = False
        self._q: queue.Queue = queue.Queue()
        self._closing = False
        self.heard_at = 0.0  # when the cloud's last voiced chunk came
        self.retry_s, self.retry_at = RETRY_S, 0.0
        self.back_at = 0.0  # when the side last went back to the cloud
        self.probe = None
        self.clause, self.final_at = None, 0.0  # the local path's last clause seen, and its last final subtitle's time
        self.local, self.local_why = None, ''
        # cloud subtitles, back in the cloud: local paths saying what is left (their voice passes until their end),
        # and those not closed yet (closed after that, or by close())
        self._leaving: set = set()
        self._to_close: set = set()
        self.switched: list[tuple[str, float]] = []  # (what, time): 'down', 'back' (tests, the bench)
        if not cloud_subtitles:
            from peresmeshnik.models import ModelMissing
            try:
                self.local = self._new_local()
            except ModelMissing as e:  # the cloud's own subtitles then; any other error stops the side (a broken model)
                self.local_why = str(e)
                log.warning('%s: no local subtitles (%s): the cloud has no fallback here', name, e)
        if preload and not voice_why:
            threading.Thread(target=self.muted.load, name=f'{name}-local-voice', daemon=True).start()
        try:
            self.cloud = make_cloud()
        except BaseException:
            if self.local:
                self.local.close()
            raise
        for d in (self.cloud, self.local):
            if d:
                self._pump(d)

    # Direction --------------------------------------------------------------------------------------
    def feed(self, pcm) -> None:
        now = time.monotonic()
        with self._lock:
            self._input.append((now, pcm))
            while now - self._input[0][0] > KEEP_S:
                self._input.popleft()
            to = [d for d in (self.cloud if self.mode == 'cloud' else None, self.local) if d]
            if self.mode != 'cloud' and self.local is None and self.probe is None and now >= self.retry_at:
                self.retry_at = float('inf')  # one try at a time; the events thread sets the next
                self._q.put((self, 'try'))
            elif self.mode != 'cloud' and self.local is None and self.probe and self.probe.ready.is_set():
                self._q.put((self, 'back'))
        for d in to:
            d.feed(pcm)

    def events(self):
        while (item := self._q.get()) is not None:
            src, e = item
            if src is self:  # from feed() (no local path to wait for a phrase with) or from Muted
                if isinstance(e, BaseException) and self.mode == 'local':
                    self._down(str(e), 'its voice failed to start')
                elif e == 'try' and self.probe is None:
                    self._try()
                elif e == 'back' and self.probe is not None and self.mode != 'cloud':
                    self._back(self.probe, time.monotonic())
                continue
            if isinstance(e, BaseException) or e is _END:
                if self._closing or src in self._leaving:  # the ends that close() and _back make
                    self._leaving.discard(src)
                    continue
                if src is self.probe:
                    self._probe_failed(e)
                elif src is self.cloud:
                    self._cloud_failed(e if isinstance(e, BaseException) else CloudDown('the cloud session ended'))
                elif src is self.local and not self._closing:  # a worker died: the side reports it
                    self.local = None
                    if isinstance(e, BaseException):
                        raise e
                    src.finish()  # raises with the worker's error
                    raise RuntimeError('the local path ended')
                continue
            if isinstance(e, Speech):
                if src is self.cloud and float(abs(e.pcm).max(initial=0)) > 0.02:
                    self.heard_at = time.monotonic()
                # the local path's voice always: Muted lets through only the clauses translated while it was local,
                # so the rest of one going on as the side goes back to the cloud is not cut
                if src is _SAY or src is self.local or src in self._leaving or (src is self.cloud and self.mode == 'cloud'):
                    yield e
            elif isinstance(e, Caption):
                if src is self.local:
                    if e.final:
                        self.final_at = time.monotonic()
                    elif e.clause != self.clause:
                        self.clause = e.clause
                        self._phrase()
                if src is (self.local or self.cloud):  # subtitles: the local path's when there is one
                    yield e
            elif isinstance(e, Latency) and src in (self.local or self.cloud, self.cloud if self.mode == 'cloud' else self.local):
                yield e

    def interrupt(self) -> None:
        if self.local:
            self.local.interrupt()

    def close(self) -> None:
        self._closing = True
        with self._lock:
            leaving, self._to_close = list(self._to_close), set()
        for d in (self.probe, self.cloud, self.local, *leaving):
            if d:
                d.close()
        self.muted.close()
        self._q.put(None)

    # the local path --------------------------------------------------------------------------------
    def _new_local(self):
        d = self.make_local(self.muted)
        d.mt = _Noted(d.mt, self._noted)
        self._warming = True
        try:
            d.warm_up()
        finally:
            self._warming = False
        return d

    def _noted(self, text: str) -> None:  # the local MT's thread
        if self._warming:  # its warm-up phrase: never said, never said again
            return
        with self._lock:
            self._made.append([text, time.monotonic(), self.mode == 'local'])

    def _want(self, text: str) -> bool:  # the local speak worker's thread: say this clause?
        with self._lock:
            while self._made:
                t, at, say = self._made.popleft()  # clauses the pipeline dropped (too late) went without a call
                if t == text:
                    if not say:
                        self._muted.append((t, at))
                    return say
        return False

    def _pump(self, d) -> None:
        def run():
            try:
                for e in d.events():
                    self._q.put((d, e))
                self._q.put((d, _END))
            except BaseException as e:
                self._q.put((d, e))
        threading.Thread(target=run, name=f'{self.name}-{type(d).__name__}', daemon=True).start()

    # switching -------------------------------------------------------------------------------------
    def _cloud_failed(self, e: BaseException) -> None:
        old, self.cloud = self.cloud, None
        threading.Thread(target=old.close, daemon=True).start()
        why = str(e) if isinstance(e, CloudDown) else f'{type(e).__name__}: {e}'
        log.warning('%s: the cloud failed: %s', self.name, why)
        if self.heard_at < self.back_at:  # back in the cloud, and no voice from it since: the next try waits longer
            self.retry_s = min(self.retry_s * 2, RETRY_MAX_S)
        else:  # it spoke since (or this is its first fault): RETRY_S again
            self.retry_s = RETRY_S
        self.retry_at = time.monotonic() + self.retry_s
        self.switched.append(('down', time.monotonic()))
        lags = [s for k, s in getattr(old, 'latency', ()) if k == 'cloud.translate.done_after_end']
        cut = self.heard_at - (statistics.median(lags) if lags else LAG_S) + MARGIN_S  # speech that ended later: unsaid
        reason, local = self.voice_why or self.local_why or (str(self.muted.error) if self.muted.error else ''), None
        if not reason and self.local is None:  # cloud subtitles: the local path is made now
            try:
                local = self._new_local()
            except Exception as ex:
                reason = self.local_why = str(ex)
        if reason:
            self._down(reason, f'the cloud failed ({why})')
            return
        self.muted.lock.acquire()  # before a clause is marked to be said: the muted ones the cloud had not said go first
        try:
            with self._lock:
                self.mode = 'local'
                if local:  # it hears the input the cloud had not said yet, then the live input
                    self.local = local
                    for t, pcm in self._input:
                        if t >= cut:
                            local.feed(pcm)
                for r in self._made:  # not spoken yet: say the ones the cloud had not said
                    r[2] = r[1] - SETTLE_S > cut
                again = [t for t, at in self._muted if at - SETTLE_S > cut]
                self._muted.clear()
            if local:
                self._pump(local)
        except BaseException:
            self.muted.lock.release()
            raise
        self._say(again)  # releases the lock
        self.on_state(f'note: {ARROW[self.name]}: the cloud failed ({why}); translating locally until it answers again')

    def _down(self, reason: str, what: str) -> None:
        self.mode = 'down'
        live = '; the hotkey gives the call your own voice' if self.name == 'me' else ''
        self.on_state(f'error: {ARROW[self.name]}: {what} and there is no local path here: {reason}{live}')

    def _say(self, texts: list[str]) -> None:
        """Muted clauses the cloud had not said, before the next one the pipeline speaks: called with `muted.lock`
        held (taken before any clause was marked to be said), released once they are said."""
        if not texts:
            self.muted.lock.release()
            return

        def run():
            try:
                for text in texts:
                    for chunk in self.muted.say(text, self.tgt, self.voice):
                        self._q.put((_SAY, Speech(chunk, Muted.sample_rate)))
            except Exception as e:
                self._q.put((self, e))
            finally:
                self.muted.lock.release()
        threading.Thread(target=run, name=f'{self.name}-say', daemon=True).start()

    def _phrase(self) -> None:
        """A phrase starts (the local path's first partial subtitle of a clause): the time to try the cloud again."""
        if self.mode == 'cloud':
            return
        if self.probe is not None and self.probe.ready.is_set():
            self._back(self.probe, max(self.final_at, time.monotonic() - PRE_S))
        elif self.probe is None and time.monotonic() >= self.retry_at:
            self._try()

    def _try(self) -> None:
        try:
            self.probe = self.make_cloud()
        except Exception as e:
            self._probe_failed(e)
            return
        self._pump(self.probe)

    def _probe_failed(self, e) -> None:
        old, self.probe = self.probe, None
        if old is not None:
            threading.Thread(target=old.close, daemon=True).start()
        self.retry_at, self.retry_s = time.monotonic() + self.retry_s, min(self.retry_s * 2, RETRY_MAX_S)
        log.info('%s: the cloud is still down (%s); the next try in %.0f s or later', self.name, e, self.retry_s / 2)

    def _back(self, cloud, since: float) -> None:
        with self._lock:
            self.mode, self.cloud, self.probe = 'cloud', cloud, None
            for t, pcm in self._input:  # the phrase from its start
                if t >= since:
                    cloud.feed(pcm)
        self.retry_at, self.back_at = 0.0, time.monotonic()  # retry_s: RETRY_S again once it speaks (_cloud_failed)
        self.switched.append(('back', time.monotonic()))
        if self.cloud_subtitles and self.local:  # the subtitles come from the cloud again; it says what is left first
            old, self.local = self.local, None
            self._leaving.add(old)
            self._to_close.add(old)

            def leave():
                try:
                    old.finish()
                except Exception as e:
                    log.info('%s: the local path on its way out: %s', self.name, e)
                with self._lock:
                    mine = old in self._to_close  # else close() has closed it
                    self._to_close.discard(old)
                if mine:
                    old.close()
            threading.Thread(target=leave, name=f'{self.name}-local-leaves', daemon=True).start()
        self.on_state(f'note: {ARROW[self.name]}: back in the cloud')


def _direction(name: str, s):
    from peresmeshnik import languages
    return languages.direction(s, 'them_to_me' if name == 'them' else 'me_to_them')


def stock(name: str, s) -> bool:
    """A Piper voice: them -> me always, me -> them when the settings say so (voice.STOCK: an 8 GB Mac) or the clone
    cannot speak the language (languages.direction)."""
    return _direction(name, s).voice != 'clone'


def voice_why(name: str, s) -> str:
    """Why this side has no local voice to fall back to ('' = it has one), without loading anything."""
    from peresmeshnik import models
    d = _direction(name, s)
    try:
        if not d.local:
            return '; '.join(d.why)
        if d.voice == 'stock':
            models.path(d.tts)
            return ''
        if not s.local.voice:
            return 'no voice enrolled (setup wizard, "Your voice")'
        if not shutil.which('nvidia-smi'):
            return 'your cloned voice is not live without an NVIDIA GPU'
        from peresmeshnik.local.clone import voice_cmd
        if voice_cmd() is None and importlib.util.find_spec('torch') is None:  # the pack's runner, or torch here
            return 'the voice pack is not installed (setup wizard, "Your voice")'
        models.path(s.local.clone)
    except models.ModelMissing as e:
        return str(e)
    return ''


def make(name: str, s, key: str, on_state=lambda text: None, on_respawn=lambda e: None) -> Fallback:
    """The side's cloud direction with its local fallback, from the settings."""
    from peresmeshnik import languages
    from peresmeshnik.cloud.gemini import GeminiDirection
    from peresmeshnik.pipeline import MAX_LAG_S, make_local
    d = _direction(name, s)
    src, tgt = d.src, d.tgt
    voice, piper = None, d.voice == 'stock'
    if d.voice == 'clone' and s.local.voice:
        from peresmeshnik import voice as profiles
        try:
            voice = profiles.load(s.local.voice)
        except Exception as e:
            log.warning('me: voice %s: %s', s.local.voice, e)

    def tts():
        if not d.tts:
            raise RuntimeError('; '.join(d.why))
        if piper:  # crash.Respawn: the runner dies, only it starts again
            from peresmeshnik import crash
            from peresmeshnik.local.tts import PiperTts
            return crash.Respawn(lambda: PiperTts(d.tts, threads=s.local.threads), on_respawn)
        from peresmeshnik import models
        from peresmeshnik.local.clone import make_clone  # the voice pack's runner, or XTTS in this process
        return make_clone(models.path(s.local.clone), threads=s.local.threads)

    def local(t):  # no local path for this pair (models.toml langs): the reason, not a KeyError
        if not d.subtitles:
            raise RuntimeError('; '.join(d.why))
        return make_local(src, tgt, tts=t, settings=s, voice=voice, max_lag_s=MAX_LAG_S if name == 'them' else None)
    return Fallback(
        name, tgt, languages.cloud_subtitles(s, d.side),
        make_cloud=lambda: GeminiDirection(src, tgt, key, s.cloud, fail_fast=True),
        make_local=local,
        make_tts=tts, voice=voice, voice_why=voice_why(name, s), preload=piper, on_state=on_state)

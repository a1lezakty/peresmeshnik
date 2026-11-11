"""Their voice on the fly (voices/live.py) on fake engines: which clauses make the sample, the voice at 3 / 5 / 10 / 20 s,
one voice per call and the switch, a kept voice known at once, the stock voice until then and after a failure, never into
the call. No models."""
import time

import numpy as np
import pytest

from peresmeshnik import app, config
from peresmeshnik.audio.echo import Reference
from peresmeshnik.engines import SAMPLE_RATE
from peresmeshnik.pipeline import LocalDirection, run
from peresmeshnik.voices import live
from tests.test_pipeline import EnergyVad, ToneTts, UpperMt, WordAsr, silence, tone


def speech(s: float, amp: float, seed: int = 0) -> np.ndarray:
    """Speech-like: 200 ms syllables of a tone, 100 ms quiet noise between them (a background for voice.check)."""
    n = int(s * SAMPLE_RATE)
    t = np.arange(n)
    on = (t % 4800) < 3200
    noise = np.random.default_rng(seed).normal(0, 0.001, n)
    return (np.where(on, amp * np.sin(t * 0.07), 0) + noise).astype(np.float32)


A, B, C = np.eye(3, dtype=np.float32)


class AmpSpeaker:
    """The 'speaker vector' of a clause is its loudness: 0.3 is A, 0.1 is B, 0.6 is C (orthogonal vectors)."""
    def embed(self, pcm, sr):
        peak = np.abs(pcm).max()
        return C if peak > 0.45 else A if peak > 0.2 else B


class Gate:
    """audio.echo.EchoGate as LiveVoice reads it: its reference, the delays it searches, the one it found (if any)."""
    def __init__(self, ref, delay=None):
        self.ref, self.delays, self.rho_on = ref, np.array([-0.1, 2.0]), 0.5
        self.delay, self.rho, self.gain = (delay or 0.0), (0.9 if delay is not None else 0.0), (0.0 if delay else None)


class FakeClone:
    sample_rate = 24_000

    def __init__(self):
        self.made, self.said, self.broken = [], [], False

    def condition(self, pcm, sr):
        self.made.append(len(pcm) / sr)
        return np.full(3, len(self.made), np.float32)

    def synth(self, text, lang, voice=None):
        self.said.append((text, int(voice.state[0])))
        if self.broken:
            raise RuntimeError('onnxruntime: out of memory')
        yield np.ones(2400, np.float32)


class FakeStock:
    sample_rate = 22_050

    def __init__(self):
        self.said = []

    def synth(self, text, lang, voice=None, speed=1.0):
        self.said.append((text, speed))
        yield np.zeros(2205, np.float32)


def voice_of(engine=None, kept=(), echo=None, note=None, raw=None, engine_id='pocket-tts-ru'):
    """A LiveVoice whose worker has loaded its engines and stopped: clauses go through _add() right here."""
    engine = engine or FakeClone()
    v = live.LiveVoice(lambda: engine, AmpSpeaker, engine_id, 'en', kept, echo, note, raw)
    v._q.put(None)
    v._thread.join()
    return v


def test_each_clause_with_words_hands_on_its_own_audio():
    got = []
    d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), ToneTts(), 'en', 'ru')
    d.realtime, d.on_clause = False, lambda pcm, text, t0, t1: got.append((len(pcm), text, t0 <= t1))
    pcm = np.concatenate([tone(12.0), silence(0.6), tone(2.0), silence(0.6)])
    run(d, pcm)
    d.finish()
    assert len(got) >= 4 and all(ok for *_, ok in got)
    # the 12 s cut at 5 s, then 2 s: the clauses together are the speech, nothing lost or said twice
    assert abs(sum(n for n, *_ in got) - 14.0 * SAMPLE_RATE) <= 2 * 512
    assert abs(got[-1][0] - 2.0 * SAMPLE_RATE) <= 2 * 512  # to a VAD frame


def test_the_voice_is_made_at_3_5_10_and_20_s_then_stays():
    eng = FakeClone()
    v = voice_of(eng)
    assert v.voice() is None
    for i in range(2):
        v._add(speech(1.2, 0.3, i))
    assert v.voice() is None and eng.made == []  # 2.4 s: not yet, the stock voice speaks
    v._add(speech(1.2, 0.3, 2))
    assert len(eng.made) == 1 and v.voice().state[0] == 1  # 3.6 s
    for i in range(3, 20):
        v._add(speech(1.2, 0.3, i))
    assert len(eng.made) == 4  # 3, 5, 10, 20 s
    assert max(eng.made) <= live.BEST_S + 1.2 + 0.1 * 9  # the sample: the cleanest 10 s (and the gaps between clauses)
    assert v.people[0].clips == []  # it stays: their speech is not kept
    v._add(speech(1.2, 0.3, 99))
    assert len(eng.made) == 4 and v.speaking.seconds == pytest.approx(20.4)


def test_short_noisy_and_echoed_clauses_stay_out_of_the_sample():
    """Each goes to the worker in order with its audio, to be matched to who said it; only long clean clauses without
    our voice around them go into a sample, and only they make someone new."""
    ref = Reference()
    v = voice_of(echo=Gate(ref))
    v.offer(speech(0.8, 0.3), 'hi', 0.0, 0.8)
    pcm, _, sample = v._q.get()
    assert v.stats['short'] == 1 and pcm is not None and not sample
    v.offer(speech(0.2, 0.3), 'm', 0.0, 0.2)
    assert v._q.get()[0] is None  # too short for a speaker vector at all
    ref.push(speech(1.0, 0.3), SAMPLE_RATE)  # we spoke into the call just now
    now = time.perf_counter()
    v.offer(speech(1.5, 0.3), 'their words', now - 0.5, now + 1.0)
    assert v.stats['echo'] == 1 and not v._q.get()[2]
    v.offer(speech(1.5, 0.3), 'their words', now + 10, now + 11.5)  # long after: no echo
    assert v._q.get()[2]
    v._add(np.random.default_rng(1).normal(0, 0.1, 2 * SAMPLE_RATE).astype(np.float32))  # noise: no speech above it
    assert v.stats['noisy'] == 1 and v.people == []
    v._add(speech(1.5, 0.3), sample=False)  # an echoed clause of someone unknown: nobody new
    assert v.people == [] and v.speaking is None


def test_the_echo_window_is_where_our_speech_can_sound_and_noise_is_not_speech():
    """Live voice into the call (the reference is my microphone, noise between my words): their clauses still make a
    sample; with the gate's delay found, only our speech at that delay counts, not all of the 2 s it searches."""
    ref = Reference()
    noise = np.random.default_rng(2).normal(0, 10 ** (-55 / 20), 3 * SAMPLE_RATE).astype(np.float32)
    now = time.perf_counter()
    ref.push(noise, SAMPLE_RATE, now)  # my microphone's noise, played into the call up to now
    v = voice_of(echo=Gate(ref))
    v.offer(speech(1.5, 0.3), 'their words', now - 1.5, now)
    assert v._q.get()[2] and v.stats['echo'] == 0  # echo.LOUD alone would have taken the noise for us
    end = time.perf_counter()
    ref.push(speech(0.5, 0.3), SAMPLE_RATE, end)  # I said a word: it played over the last 0.5 s
    far = voice_of(echo=Gate(ref))  # no delay found yet: anything we said within 2 s before the clause counts
    far.offer(speech(1.5, 0.3), 'their words', end + 1.0, end + 2.5)
    assert not far._q.get()[2]
    near = voice_of(echo=Gate(ref, delay=0.1))  # the echo comes back after 0.1 s: our word is done by then
    near.offer(speech(1.5, 0.3), 'their words', end + 1.0, end + 2.5)
    assert near._q.get()[2]
    raw = voice_of(echo=Gate(Reference()), raw=lambda: True)  # my microphone into the call as it is (loopback)
    raw.offer(speech(1.5, 0.3), 'their words', end + 10, end + 11.5)
    assert not raw._q.get()[2] and raw.stats['echo'] == 1


def test_each_clause_in_the_voice_of_who_said_it_stock_until_theirs_is_made():
    eng, ref = FakeClone(), Reference()
    v = live.LiveVoice(lambda: eng, AmpSpeaker, 'pocket-tts-ru', 'en', echo=Gate(ref))  # its worker runs: offer() as
    a = [v.offer(speech(1.2, 0.3, i), 'a', 0, 1.2) for i in range(3)]  # the pipeline does
    b = v.offer(speech(1.5, 0.1, 10), 'b', 0, 1.5)
    short = v.offer(speech(0.5, 0.1, 11), 'uh', 0, 0.5)  # B's short word: matched to B, not taken as A's
    a2 = v.offer(speech(1.2, 0.3, 12), 'a', 0, 1.2)
    yes = v.offer(speech(0.5, 0.6, 13), 'yes', 0, 0.5)  # C's first word, too short to make anyone: nobody's voice
    ref.push(speech(1.0, 0.3), SAMPLE_RATE)  # we spoke into the call, and C talks over it: not into a sample
    now = time.perf_counter()
    echoed = v.offer(speech(1.5, 0.6, 14), 'c', now - 0.5, now + 1.0)
    assert v.voice(a2) is not None and v.voice(a2) is v.people[0].voice  # the worker is past A's 3.6 s: made
    assert v.voice(a[0]) is v.people[0].voice  # A's voice as it is when the clause is said
    assert v.voice(b) is None and v.voice(short) is None  # B's: not made yet, the stock voice (not A's)
    assert short.person is v.people[1] and v.stats['short'] == 2
    assert v.voice(yes) is None and yes.person is None  # the stock voice, not A's who spoke before
    assert v.voice(echoed) is None and echoed.person is None and len(v.people) == 2
    v.close()


def test_a_clause_not_known_in_time_or_before_any_voice_is_stock():
    """While someone's voice is being made, a clause queued behind it waits WAIT_S, then the stock voice says it (not
    the voice of whoever spoke last); before anyone's voice is made it does not wait at all (the engine loading)."""
    class Slow(FakeClone):
        def condition(self, pcm, sr):
            time.sleep(1.0)
            return super().condition(pcm, sr)
    v = live.LiveVoice(lambda: time.sleep(0.5) or Slow(), AmpSpeaker, 'pocket-tts-ru', 'en')
    first = v.offer(speech(1.2, 0.3), 'a', 0, 1.2)
    t = time.perf_counter()
    assert v.voice(first) is None and time.perf_counter() - t < 0.1  # still loading: no wait
    v.close()
    kept = [(B, np.full(3, 7.0, np.float32), 14.0)]  # B's voice is there from the start
    v = live.LiveVoice(Slow, AmpSpeaker, 'pocket-tts-ru', 'en', kept)
    a = [v.offer(speech(1.2, 0.3, i), 'a', 0, 1.2) for i in range(3)]  # A's third clause: A's voice made, 1 s
    behind = v.offer(speech(1.2, 0.3, 9), 'a', 0, 1.2)
    a[2].known.wait(5)
    t = time.perf_counter()
    assert v.voice(behind) is None and live.WAIT_S <= time.perf_counter() - t < live.WAIT_S + 0.4
    assert v.stats['late'] == 1
    v.close()


def test_a_kept_voice_speaks_from_their_first_clause():
    eng = FakeClone()
    v = voice_of(eng, kept=[(B, np.full(3, 7.0, np.float32), 14.0)])
    v._add(speech(1.2, 0.1))  # B, kept from an earlier call
    assert v.voice().state[0] == 7 and eng.made == []  # as kept: not made again
    v._add(speech(1.2, 0.3))  # someone new: a person of their own
    assert len(v.people) == 2 and not v.speaking.kept and v.voice() is None  # their voice is not made yet


def test_their_voice_speaks_stock_until_made_then_clone_and_stock_after_a_failure():
    eng, stock = FakeClone(), FakeStock()
    v = voice_of(eng)
    tts = live.TheirVoice(stock, v)
    out = list(tts.synth('one', 'ru', speed=1.2))
    assert stock.said == [('one', 1.2)] and len(out[0]) == pytest.approx(2400, abs=2)  # 22.05 -> 24 kHz
    for i in range(3):
        v._add(speech(1.2, 0.3, i))
    assert [len(c) for c in tts.synth('two', 'ru')] == [2400] and eng.said == [('two', 1)]
    eng.broken = True
    list(tts.synth('three', 'ru'))  # the clone fails before a word: the stock voice says it
    assert v.dead and stock.said[-1] == ('three', 1.0)
    list(tts.synth('four', 'ru'))
    assert eng.said[-1][0] == 'three' and stock.said[-1][0] == 'four'


def test_a_clone_too_slow_for_this_computer_gives_way_to_the_stock_voice_said_once(monkeypatch):
    """As me -> them (app.MAX_CLONE_RTF), timed on the call's own clauses: their voice's synthesis slower than real time
    -> the stock voice for the rest of the call, and the tray hears why, once; a clone fast enough stays."""
    class Slow(FakeClone):
        def synth(self, text, lang, voice=None):
            time.sleep(0.15)  # for its 0.1 s
            yield from super().synth(text, lang, voice)
    monkeypatch.setattr(live, 'RTF_AFTER_S', 0.1)
    for eng in (FakeClone(), Slow()):
        notes, stock = [], FakeStock()
        v = voice_of(eng, note=notes.append)
        for i in range(3):
            v._add(speech(1.2, 0.3, i))
        tts = live.TheirVoice(stock, v)
        list(tts.synth('one', 'ru'))
        assert eng.said == [('one', 1)] and v.rtf is None  # checked once
        list(tts.synth('two', 'ru'))
        if isinstance(eng, Slow):
            assert 'too slow for their cloned voice' in v.dead and notes == [v.dead] and stock.said == [('two', 1.0)]
        else:
            assert not v.dead and not notes and eng.said[-1][0] == 'two'


def test_their_voice_never_goes_into_the_call(monkeypatch):
    class Them:
        tts = live.TheirVoice(FakeStock(), voice_of())
        closed = False

        def close(self):
            self.closed = True
    d = Them()
    monkeypatch.setattr(app.Side, '_local', lambda self, s: d)
    call = type('Call', (), {'to_call': True, 'play': lambda *a: None})()
    with pytest.raises(RuntimeError, match='only in your headphones'):
        app.Side('them', config.Settings(), None, call, None)
    assert d.closed
    app.Side('them', config.Settings(), None, type('Phones', (), {'play': lambda *a: None})(), None)


def test_their_engine_shares_the_card_with_my_cloned_voice(monkeypatch):
    """FIXES-s1-addon 5 and its reserve (a runner per side): both directions cloned here, mine Qwen3-TTS. On an 8 GB card
    theirs is Pocket TTS on the CPU (no NVIDIA engine fits beside), on 12 GB XTTS-v2, on 24 GB Qwen3 too, and the same in
    what is downloaded; my voice in the cloud: the same while its fallback here is my clone, Qwen3 when it is a stock
    voice."""
    from peresmeshnik import languages, models
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'pack_here', lambda: True)
    monkeypatch.setattr(languages, 'addon_here', lambda: True)
    monkeypatch.setattr(models, 'is_fetched', lambda m, base=None: True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(8188 / 1024, (572, 83), 8.6))  # 8 GB as reported
    s = config.Settings(my_lang='ru', their_lang='en', them_to_me='local', me_to_them='local')
    assert languages.direction(s, 'me_to_them').tts == 'qwen3-tts' and live.engine(s) == ('pocket-tts-ru', '')
    need = languages.needed_models(s)
    assert {'qwen3-tts', 'pocket-tts-ru', 'wespeaker-resnet34'} <= set(need) and 'xtts-v2' not in need
    from peresmeshnik.cloud import fallback
    s.me_to_them = 'cloud'  # my clone speaks here when the cloud fails, if it is ready: on the same card
    monkeypatch.setattr(fallback, 'local_voice', lambda name, s: ('qwen3-tts', ''))
    assert live.engine(s) == ('pocket-tts-ru', '')
    monkeypatch.setattr(fallback, 'local_voice', lambda name, s: ('piper-en_US-john-medium', ''))  # its stock voice
    assert live.engine(s) == ('qwen3-tts', '')
    s.me_to_them = 'local'
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(12.0, (572, 83), 8.6))
    assert live.engine(s) == ('xtts-v2', '') and 'xtts-v2' in languages.needed_models(s)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(24.0, (572, 83), 8.6))
    assert live.engine(s) == ('qwen3-tts', '')
    s.local.their_voice = False
    assert live.engine(s)[0] is None


def test_attach_leaves_the_stock_voice_when_off_or_not_downloaded(monkeypatch, tmp_path):
    from peresmeshnik import models
    monkeypatch.setattr(models, 'root', lambda: tmp_path)
    d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), ToneTts(), 'en', 'ru')
    s = config.Settings()
    s.local.their_voice = False
    assert live.attach(d, s) is d and d.on_clause is None
    s.local.their_voice = True
    stock = d.tts
    assert live.attach(d, s) is d and d.tts is stock and d.on_clause is None  # pocket-tts-ru not downloaded here
    d.close()


def test_the_tray_gets_the_voice_speaking_now_or_why_not():
    """FIXES-s2 r2 6, 7: kept only after KEEP_S of their speech, in a call with one person on their side, not a voice
    like mine; otherwise (and with the NVIDIA voice pack, a dead clone, them -> me elsewhere) the tray says why."""
    from types import SimpleNamespace
    v = voice_of()
    them = SimpleNamespace(name='them', engine='local',
                           directions=[(SimpleNamespace(tts=live.TheirVoice(FakeStock(), v)), True, True)])
    me = SimpleNamespace(name='me', engine='local', directions=[(SimpleNamespace(tts=FakeStock()), True, True)])
    ctl = SimpleNamespace(sides=[me, them], settings=config.Settings())
    assert app.Controller.them_voice(ctl) == (None, 'not enough of their speech yet: 0 of 10 s')
    for i in range(3):
        v._add(speech(1.2, 0.3, i))
    assert app.Controller.them_voice(ctl)[1] == 'not enough of their speech yet: 4 of 10 s'  # a first voice only
    for i in range(6):
        v._add(speech(1.2, 0.3, 3 + i))
    states, lang, seconds, sample = app.Controller.them_voice(ctl)[0]
    assert set(states) == {'pocket-tts-ru', 'wespeaker-resnet34'} and lang == 'en' and seconds == 10.8 and sample is None
    v._my_vec = A  # A is too close to my own voice (my echo made it, say)
    assert 'too close to your own' in app.Controller.them_voice(ctl)[1]
    v._my_vec = None
    for i in range(3):
        v._add(speech(1.2, 0.1, 20 + i))  # B speaks too: whose would it be?
    assert app.Controller.them_voice(ctl) == (None, '2 people speak in this call: a voice can be kept in a call with '
                                                    'one of them only')
    v.dead = 'onnxruntime: out of memory'
    assert 'stopped in this call (onnxruntime' in app.Controller.them_voice(ctl)[1]
    them.engine, them.directions = 'cloud', [(SimpleNamespace(tts=FakeStock()), True, True)]
    assert 'in the cloud: their voice is made only when it runs on this computer' in app.Controller.them_voice(ctl)[1]
    them.engine = 'local'
    ctl.settings.local.their_voice = False
    assert app.Controller.them_voice(ctl)[1] == ('their voice is not made on this computer: their voice is off ([local] '
                                                 'their_voice)')
    assert app.Controller.them_voice(SimpleNamespace(sides=[me])) == (None, 'them → me is not running')


def test_a_voice_kept_from_the_tray_is_theirs_from_the_first_clause_of_the_next_call(tmp_path):
    """S3's profile: them() -> voice.save_them (the tray's RememberDialog.keep) -> live.kept: the same state and speaker
    vector, `kind = them`, no recording of their speech; the next call speaks it from their first clause."""
    from peresmeshnik import voice
    v = voice_of()
    for i in range(9):
        v._add(speech(1.2, 0.1, i))  # B: 10.8 s, their voice is made
    (states, lang, seconds, sample), why = v.them()
    assert why == '' and sample is None  # Pocket: its state, no recording of their speech
    saved = voice.save_them('Anna', lang, seconds, states, tmp_path)
    assert saved.meta['kind'] == 'them' and sorted(p.name for p in saved.path.iterdir()) == [
        'pocket-tts-ru.npy', 'voice.json', 'wespeaker-resnet34.npy']
    [(vec, state, s)] = live.kept('pocket-tts-ru', tmp_path)
    assert np.array_equal(state, v.speaking.voice.state) and vec @ B > 0.99 and s == seconds
    assert live.kept('pocket-tts-en', tmp_path) == []  # another engine: made again in that call
    eng = FakeClone()
    w = voice_of(eng, kept=live.kept('pocket-tts-ru', tmp_path))
    w._add(speech(1.2, 0.1, 9))
    assert np.array_equal(w.voice().state, state) and eng.made == []


def test_a_kept_voice_deleted_during_the_call_is_forgotten_at_once(tmp_path, monkeypatch):
    """S3 tail: Voices… deletes a contact's kept voice during the call (the tray: Controller.their_voices_deleted): the
    running them side forgets that person at once, their next clause is someone new's; another kept voice stays."""
    from types import SimpleNamespace

    from peresmeshnik import models, voice
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path)
    fp = models.get('pocket-tts-ru').fingerprint, models.get(voice.SPEAKER).fingerprint
    anna, _ = (voice.save_them(n, 'en', 14.0, {'pocket-tts-ru': (np.full(3, k, np.float32), fp[0]),
                                                voice.SPEAKER: (vec, fp[1])}, tmp_path)
               for n, k, vec in (('Anna', 7.0, B), ('Bob', 8.0, C)))
    v = live.LiveVoice(FakeClone, AmpSpeaker, 'pocket-tts-ru', 'en', live.kept('pocket-tts-ru'))
    assert sorted(p.voice.state[0] for p in v.people) == [7, 8]
    them = SimpleNamespace(name='them', engine='local',
                           directions=[(SimpleNamespace(tts=live.TheirVoice(FakeStock(), v)), True, True)])
    me = SimpleNamespace(name='me', engine='local', directions=[(SimpleNamespace(tts=FakeStock()), True, True)])
    voice.discard(anna.name, tmp_path)
    app.Controller.their_voices_deleted(SimpleNamespace(sides=[me, them]))
    v.close()  # the worker has gone through its queue
    assert [p.voice.state[0] for p in v.people] == [8] and len(v._known()) == 1  # a reset() does not bring Anna back
    v._add(speech(1.2, 0.1))  # Anna (B) speaks: someone new, the stock voice until their voice is made in this call
    assert not v.speaking.kept and v.voice() is None


def test_a_voice_pack_voice_speaks_in_the_call_and_is_kept_only_as_pockets_is():
    """NVIDIA (the voice pack's Qwen3-TTS / XTTS): their voice speaks in the call as Pocket's does; "keep" (the user,
    11.11) only after KEEP_S of their speech, with one person on their side, not a voice like mine."""
    v = voice_of(engine_id='xtts-v2')
    for i in range(3):
        v._add(speech(1.2, 0.3, i))
    assert v.voice() is not None and v.them() == (None, 'not enough of their speech yet: 4 of 10 s')
    for i in range(6):
        v._add(speech(1.2, 0.3, 3 + i))
    assert v.them()[1] == ''
    v._my_vec = A
    assert 'too close to your own' in v.them()[1]
    v._my_vec = None
    for i in range(3):
        v._add(speech(1.2, 0.1, i))  # B too
    assert v.them() == (None, '2 people speak in this call: a voice can be kept in a call with one of them only')


def test_with_the_voice_pack_10_s_of_their_clean_speech_is_kept_and_speaks_from_the_first_clause_next_time(tmp_path):
    """BRIEF-s2-keep-nvidia: their voice is an id in that call's runner, so "keep" writes at most voice.THEM_S of their
    cleanest clauses (whole, none with our speech around it) and their speaker vector, no state; the next call's runner
    makes their voice from it for their first clause, which speaks in it. Another file there, or a Pocket call:
    not used, the voice is made again in the call."""
    from peresmeshnik import voice
    v = voice_of(engine_id='xtts-v2')
    clean = [speech(1.2, 0.3, i) for i in range(12)]
    for i, pcm in enumerate(clean):
        v._add(pcm)
        if i == 4:
            v._add(speech(1.5, 0.3, 77), sample=False)  # we spoke into the call around it (offer(): _ours)
    (states, lang, seconds, (pcm, sr)), why = v.them()
    assert why == '' and list(states) == [voice.SPEAKER] and sr == SAMPLE_RATE and seconds == 14.4
    n = len(clean[0]) + len(live.GAP)
    assert len(pcm) <= voice.THEM_S * SAMPLE_RATE and len(pcm) // n == 7  # whole clauses only: 7 of 1.2 s and the gaps
    assert all(any(np.array_equal(pcm[k * n:k * n + len(clean[0])], c) for c in clean) for k in range(7))
    saved = voice.save_them('Anna', lang, seconds, states, tmp_path, (pcm, sr))
    assert sorted(p.name for p in saved.path.iterdir()) == ['sample.wav', 'voice.json', 'wespeaker-resnet34.npy']
    assert saved.meta['sample']['seconds'] == round(len(pcm) / sr, 2)
    [(vec, st, s, (back, bsr))] = live.kept('xtts-v2', tmp_path)
    assert st is None and vec @ A > 0.99 and s == seconds and bsr == sr and np.allclose(back, pcm, atol=1e-4)
    assert live.kept('pocket-tts-ru', tmp_path) == []  # a Pocket call: its voice is made in that call
    eng = FakeClone()
    w = voice_of(eng, kept=live.kept('xtts-v2', tmp_path), engine_id='xtts-v2')
    assert eng.made == []  # not before they speak
    w._add(speech(1.2, 0.3, 50))  # their first clause: their voice made from the recording for it
    assert w.speaking.kept and w.voice().state[0] == 1 and eng.made == [pytest.approx(len(pcm) / sr)]
    w._forget()  # reset() in the worker, a new call on warm engines: made again (the runner forgot it)
    w._add(speech(1.2, 0.3, 51))
    assert len(eng.made) == 2 and w.people[0].voice.state[0] == 2
    long = voice.save_them('Bob', 'en', 30.0, states, tmp_path, (np.full(12 * sr, 0.1, np.float32), sr))
    assert long.meta['sample']['seconds'] == voice.THEM_S  # at most THEM_S on disk, whoever hands more
    (saved.path / 'sample.wav').write_bytes((long.path / 'sample.wav').read_bytes())  # another file there
    assert [k[2] for k in live.kept('xtts-v2', tmp_path)] == [30.0]  # Anna's not used: only Bob's


def test_a_kept_recording_the_runner_refuses_is_made_again_in_the_call_the_others_speak(caplog):
    """FIXES-s2-39 1: one contact's recording refused by the runner is logged and that person made again in the call
    as anyone new; another contact and someone new speak in their own voices, the clone goes on."""
    class Picky(FakeClone):
        def condition(self, pcm, sr):
            if np.isclose(pcm.mean(), 0.5):
                raise RuntimeError('the voice pack: ValueError: a bad sample')
            return super().condition(pcm, sr)
    eng = Picky()
    rec = (np.full(SAMPLE_RATE * 9, 0.5, np.float32), SAMPLE_RATE), (np.full(SAMPLE_RATE * 9, 0.25, np.float32), SAMPLE_RATE)
    v = voice_of(eng, kept=[(B, None, 12.0, rec[0]), (C, None, 12.0, rec[1])], engine_id='xtts-v2')
    v._add(speech(1.2, 0.1))  # B: their recording refused
    assert not v.dead and not v.speaking.kept and v.voice() is None and 'could not be made' in caplog.text
    v._add(speech(1.2, 0.6))  # C: their voice from their recording
    assert v.speaking.kept and v.voice().state[0] == 1
    for i in range(3):
        v._add(speech(1.2, 0.3, i))  # someone new: made at 3 s
        v._add(speech(1.2, 0.1, i))  # and B, from their speech in this call
    assert [p.voice.state[0] for p in v.people] == [1, 2, 3] and not v.dead


def test_kept_recordings_hold_up_no_clause_and_a_voice_made_again_leaves_no_state_behind():
    """FIXES-s2-39 2: a contact's voice is made from their recording when they first speak, not all of them before the
    first clause (each clause of the call's start would wait WAIT_S and be said in the stock voice); a voice made again
    replaces its state in the runner, which keeps no copies of it."""
    class Slow(FakeClone):
        def __init__(self):
            super().__init__()
            self.forgot = []

        def condition(self, pcm, sr):
            time.sleep(0.2)
            return super().condition(pcm, sr)

        def forget_states(self, ids=None):
            self.forgot.append([int(i[0]) for i in ids])
    eng, rec = Slow(), (np.full(SAMPLE_RATE * 9, 0.25, np.float32), SAMPLE_RATE)
    v = live.LiveVoice(lambda: eng, AmpSpeaker, 'xtts-v2', 'en', [(B, None, 12.0, rec), (C, None, 12.0, rec),
                                                                   (np.float32([0, .6, .8]), None, 12.0, rec)])
    while v.engine is None:
        time.sleep(0.01)
    c = v.offer(speech(1.2, 0.3), 'a', 0, 1.2)  # someone new speaks first
    t = time.perf_counter()
    assert v.voice(c) is None and time.perf_counter() - t < 0.1 and eng.made == []
    c = v.offer(speech(1.2, 0.1), 'b', 0, 1.2)  # then a contact: their voice, made now (0.2 s)
    assert v.voice(c).state[0] == 1 and len(eng.made) == 1
    for i in range(1, 5):
        v.offer(speech(1.2, 0.3, i), 'a', 0, 1.2)  # 6 s of A: made at 3.6 and again at 6
    v._q.put(None)  # the worker goes through them all, then stops
    v._thread.join(10)
    a = next(p for p in v.people if not p.kept)
    assert len(eng.made) == 3 and a.voice.state[0] == 3 and eng.forgot == [[2]]  # the 3.6 s state forgotten


def test_an_engine_still_loading_when_translation_goes_off_is_freed_after_it(monkeypatch):
    """No voice pack runner outlives the call: close() waits CLOSE_WAIT_S, and a worker still loading the engine then
    frees it itself instead of going through the clauses left."""
    class Engine(FakeClone):
        closed = False

        def close(self):
            Engine.closed = True
    eng = Engine()
    monkeypatch.setattr(live, 'CLOSE_WAIT_S', 0.01)
    v = live.LiveVoice(lambda: time.sleep(0.3) or eng, AmpSpeaker, 'xtts-v2', 'en')
    for i in range(5):
        v.offer(speech(1.2, 0.3, i), 'a', 0, 1.2)
    v.close()
    assert v._thread.is_alive() and not Engine.closed  # still loading
    v._thread.join(5)
    assert Engine.closed and v.engine is None and eng.made == []


def test_an_engine_that_takes_a_voice_only_from_a_folder_is_closed_at_once():
    class Pack:  # XTTS through the voice pack: no condition()
        closed = False

        def close(self):
            Pack.closed = True
    v = live.LiveVoice(Pack, AmpSpeaker, 'xtts-v2', 'en')
    v.close()
    assert Pack.closed and 'cannot take a voice' in v.dead and v.engine is None
    assert v.voice(v.offer(speech(1.2, 0.3), 'a', 0, 1.2)) is None  # the stock voice


def test_faster_keeps_the_pitch_and_shortens_by_the_speed():
    sr = 24_000
    x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(3 * sr) / sr)).astype(np.float32)
    chunks = [x[i:i + 1920] for i in range(0, len(x), 1920)]  # Pocket's 80 ms frames
    assert list(live.faster(iter(chunks), 1.0)) == chunks
    y = np.concatenate(list(live.faster(iter(chunks), 1.25)))
    assert abs(len(y) / sr - 3 / 1.25) < 0.1
    f = np.fft.rfftfreq(len(y[sr // 10:-sr // 10]), 1 / sr)[np.argmax(np.abs(np.fft.rfft(y[sr // 10:-sr // 10])))]
    assert abs(f - 220) < 3  # the same pitch, not a faster tape
    mid = y[sr // 10:-sr // 10]
    assert np.abs(mid).max() < 0.33 and np.sqrt(np.mean(mid ** 2)) > 0.19  # frames add up without dips or clicks


def test_the_them_side_speaks_at_its_voices_rate_not_the_stock_one(monkeypatch, tmp_path):
    """FIXES-s2 r2 1: live.attach puts TheirVoice (24 kHz) in after the pipeline's workers started on the stock voice's
    rate: every Speech goes out at 24 kHz, the stock voice's clauses resampled, not labelled 22.05 kHz."""
    from peresmeshnik import models
    from peresmeshnik.engines import Speech
    from peresmeshnik.local import clone
    monkeypatch.setattr(live, 'engine', lambda s: ('pocket-tts-ru', ''))
    monkeypatch.setattr(models, 'path', lambda mid: tmp_path)
    monkeypatch.setattr(clone, 'make_engine', lambda *a, **k: FakeClone())
    monkeypatch.setattr(live, 'kept', lambda engine_id, root=None: [])  # not the voices of this computer's user
    monkeypatch.setattr('peresmeshnik.voice.Speaker', lambda path: AmpSpeaker())

    class Piper(FakeStock):
        def synth(self, text, lang, voice=None, speed=1.0):
            yield from (np.zeros(2205, np.float32) for _ in text.split())
    d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), Piper(), 'en', 'ru')
    d.realtime = False
    assert live.attach(d, config.Settings()) is d and isinstance(d.tts, live.TheirVoice)
    run(d, np.concatenate([tone(1.5), silence(0.6), tone(1.5), silence(0.6)]))
    d.finish()
    rates = [e.sample_rate for e in d.events() if isinstance(e, Speech)]
    assert rates and set(rates) == {24_000}
    d.close()


def test_a_speaker_model_that_fails_frees_the_engine_and_attach_never_stops_the_side(monkeypatch):
    """FIXES-s2 r2 12, 13: the engine loaded, then the speaker model failed: the engine is closed, the stock voice
    speaks; anything failing in attach itself leaves the side as it was."""
    class Engine(FakeClone):
        closed = False

        def close(self):
            Engine.closed = True

    def broken():
        raise OSError('wespeaker: no such file')
    v = live.LiveVoice(Engine, broken, 'pocket-tts-ru', 'en')
    v.close()
    assert Engine.closed and v.engine is None and 'wespeaker' in v.dead
    d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), ToneTts(), 'en', 'ru')
    stock = d.tts
    monkeypatch.setattr(live, 'engine', lambda s: 1 / 0)
    assert live.attach(d, config.Settings()) is d and d.tts is stock and d.on_clause is None
    d.close()


def test_translation_off_and_on_again_waits_for_the_last_runner(monkeypatch):
    """FIXES-s2 r2 14: the last call's worker still loading its engine when translation went off: the new call's
    engine loads only once that one is freed (one runner of theirs on the card)."""
    monkeypatch.setattr(live, 'CLOSE_WAIT_S', 0.01)
    times = {}

    def slow():
        time.sleep(0.4)
        return FakeClone()

    def stamped(name, make):
        def f():
            times[name] = time.perf_counter()
            return make()
        return f
    old = live.LiveVoice(slow, AmpSpeaker, 'xtts-v2', 'en')
    old.close()
    assert old._thread.is_alive() and live.LiveVoice._closing is old._thread
    t = time.perf_counter()
    new = live.LiveVoice(stamped('new', FakeClone), AmpSpeaker, 'xtts-v2', 'en')
    new._q.put(None)
    new._thread.join(5)
    assert times['new'] - t >= 0.3 and not old._thread.is_alive()


def test_saved_headphones_are_checked_as_the_default_is(monkeypatch):
    """FIXES-s2 r2 15: a saved output that is one of ours (Linux) or a BlackHole device (macOS) is refused like a
    default one would be: the translation would go into the call."""
    import sys
    from types import SimpleNamespace

    import peresmeshnik.audio as audio
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(app, 'pactl', lambda *a: '1\tperesmeshnik_mic_sink\tx\n2\talsa_output.usb\tx\n' if a[0] == 'list'
                        else 'alsa_output.usb\n')
    assert app.Headphones('alsa_output.usb').resolve() == 'alsa_output.usb'
    with pytest.raises(RuntimeError, match='saved output is our own device peresmeshnik_mic_sink'):
        app.Headphones('peresmeshnik_mic_sink').resolve()
    monkeypatch.setattr(sys, 'platform', 'darwin')
    core = SimpleNamespace(devices=lambda: [{'name': n, 'outputs': 2, 'virtual': n.startswith('BlackHole')}
                                            for n in ('BlackHole 2ch', 'AirPods')], default=lambda: 'AirPods')
    cap = SimpleNamespace(CALL_DEVICE='BlackHole 16ch', path=lambda: 'tap')
    for name, mod in (('coreaudio', core), ('capture_macos', cap)):
        monkeypatch.setattr(audio, name, mod, raising=False)
        monkeypatch.setitem(sys.modules, f'peresmeshnik.audio.{name}', mod)
    assert app.Headphones('AirPods').resolve() == 'AirPods'
    with pytest.raises(RuntimeError, match='saved output is BlackHole 2ch, a device of the call'):
        app.Headphones('BlackHole 2ch').resolve()


def test_a_new_call_on_warm_engines_checks_the_clone_again_and_drops_the_last_calls_clauses(monkeypatch):
    """FIXES-s2 r3 3, 2: reset() lets a clone found too slow (or failed) in the call before be checked again, its speed
    counted anew; a clause heard before the reset (the pause after the last call closes it, offered later) is nobody's."""
    class Slow(FakeClone):
        def synth(self, text, lang, voice=None):
            time.sleep(0.15)
            yield from super().synth(text, lang, voice)
    monkeypatch.setattr(live, 'RTF_AFTER_S', 0.1)
    v, stock = voice_of(Slow()), FakeStock()
    for i in range(3):
        v._add(speech(1.2, 0.3, i))
    tts = live.TheirVoice(stock, v)
    list(tts.synth('one', 'ru'))
    assert 'too slow' in v.dead and v.rtf is None
    v._forget()
    assert v.dead == '' and v.rtf == [0.0, 0.0] and v.people == []
    w = live.LiveVoice(FakeClone, AmpSpeaker, 'pocket-tts-ru', 'en')
    before = time.perf_counter()
    w.reset()
    old = w.offer(speech(1.5, 0.3), 'their last words', before - 1.5, before)  # A's, closed after the reset
    new = [w.offer(speech(1.2, 0.3, i), 'b', w._since + 1, w._since + 2.2) for i in range(3)]
    new[-1].known.wait(5)
    assert old.known.is_set() and old.person is None and len(w.people) == 1 and w.people[0].seconds == pytest.approx(3.6)
    w.close()


def test_a_short_word_like_someone_known_is_not_said_in_their_voice():
    """FIXES-s2 r3 4: under MIN_CLAUSE_S a match needs SHORT_SAME (0.55; measured), not SAME: a new person's "yes" at 0.5
    to A is said in the stock voice, while a clause of 1.2 s at 0.5 is A's."""
    near = (0.5 * A + np.sqrt(0.75) * C).astype(np.float32)  # 0.5 to A

    class Near(AmpSpeaker):
        def embed(self, pcm, sr):
            return near if np.abs(pcm).max() > 0.45 else super().embed(pcm, sr)
    eng = FakeClone()
    v = live.LiveVoice(lambda: eng, Near, 'pocket-tts-ru', 'en')
    v._q.put(None)
    v._thread.join()
    for i in range(3):
        v._add(speech(1.2, 0.3, i))  # A: made
    yes = live.Clause()
    v._add(speech(0.5, 0.6), yes, sample=False)
    assert yes.person is None and v.voice(yes) is None
    long = live.Clause()
    v._add(speech(1.2, 0.6), long, sample=False)
    assert long.person is v.people[0]


def test_a_kept_voice_waits_for_nothing_while_the_engine_loads():
    """FIXES-s2 r3 5: with a kept contact a voice exists from the start, but the engine is not loaded yet: no wait, the
    stock voice says the clause, and nothing fails."""
    v = live.LiveVoice(lambda: time.sleep(0.5) or FakeClone(), AmpSpeaker, 'pocket-tts-ru', 'en',
                       [(B, np.full(3, 7.0, np.float32), 14.0)])
    c = v.offer(speech(1.2, 0.1), 'b', 0, 1.2)
    stock = FakeStock()
    t = time.perf_counter()
    list(live.TheirVoice(stock, v).synth('hello', 'ru', c))
    assert time.perf_counter() - t < 0.1 and stock.said == [('hello', 1.0)] and not v.dead
    v.close()


def test_my_raw_microphone_in_the_call_is_said_as_the_reason():
    """FIXES-s2 r3 6: me -> them did not start, my microphone goes into the call as it is the whole call: no sample, and
    "keep their voice" says why instead of "not enough of their speech"."""
    v = voice_of(raw=lambda: True)
    for i in range(5):
        v.offer(speech(1.2, 0.3, i), 'a', 0, 1.2)
    assert v.stats['raw'] == 5
    assert v.them() == (None, 'your microphone goes into the call as it is (me → them is not running): their voice '
                              'takes no sample meanwhile')

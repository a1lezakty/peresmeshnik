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


class AmpSpeaker:
    """The 'speaker vector' of a clause is its loudness: 0.3 is A, 0.1 is B (orthogonal vectors)."""
    def embed(self, pcm, sr):
        return np.array([1.0, 0.0], np.float32) if np.abs(pcm).max() > 0.2 else np.array([0.0, 1.0], np.float32)


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


def voice_of(engine=None, kept=(), echo=None, note=None):
    """A LiveVoice whose worker has loaded its engines and stopped: clauses go through _add() right here."""
    engine = engine or FakeClone()
    v = live.LiveVoice(lambda: engine, AmpSpeaker, 'pocket-tts-ru', 'en', kept, echo, note)
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
    ref = Reference()
    gate = type('Gate', (), {'ref': ref, 'delays': np.array([0.0, 2.0])})()
    v = voice_of(echo=gate)
    v.offer(speech(0.8, 0.3), 'hi', 0.0, 0.8)
    assert v.stats['short'] == 1 and v._q.get()[0] is None  # in order, without its audio
    ref.push(speech(1.0, 0.3), SAMPLE_RATE)  # we spoke into the call just now
    now = time.perf_counter()
    v.offer(speech(1.5, 0.3), 'their words', now - 0.5, now + 1.0)
    assert v.stats['echo'] == 1 and v._q.get()[0] is None
    v.offer(speech(1.5, 0.3), 'their words', now + 10, now + 11.5)  # long after: no echo
    assert v._q.get()[0] is not None
    v._add(np.random.default_rng(1).normal(0, 0.1, 2 * SAMPLE_RATE).astype(np.float32))  # noise: no speech above it
    assert v.stats['noisy'] == 1 and v.people == []


def test_each_clause_in_the_voice_of_who_said_it_stock_until_theirs_is_made():
    eng = FakeClone()
    v = live.LiveVoice(lambda: eng, AmpSpeaker, 'pocket-tts-ru', 'en')  # its worker runs: offer() as the pipeline does
    a = [v.offer(speech(1.2, 0.3, i), 'a', 0, 1.2) for i in range(3)]
    b = v.offer(speech(1.5, 0.1, 10), 'b', 0, 1.5)
    short = v.offer(speech(0.5, 0.3, 11), 'uh', 0, 0.5)  # too short to tell: the person before (B)
    a2 = v.offer(speech(1.2, 0.3, 12), 'a', 0, 1.2)
    assert v.voice(a2) is not None and v.voice(a2) is v.people[0].voice  # the worker is past A's 3.6 s: made
    assert v.voice(a[0]) is v.people[0].voice  # A's voice as it is when the clause is said
    assert v.voice(b) is None and v.voice(short) is None  # B's: not made yet, the stock voice (not A's)
    assert short.person is v.people[1] and len(v.people) == 2 and v.stats['short'] == 1
    v.close()


def test_a_kept_voice_speaks_from_their_first_clause():
    eng = FakeClone()
    v = voice_of(eng, kept=[(np.array([0.0, 1.0], np.float32), np.full(3, 7.0, np.float32), 14.0)])
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
        assert eng.said == [('one', 1)] and tts._rtf is None  # checked once
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


def test_the_tray_gets_the_voice_speaking_now():
    from types import SimpleNamespace
    v = voice_of()
    them = SimpleNamespace(directions=[(SimpleNamespace(tts=live.TheirVoice(FakeStock(), v)), True, True)])
    me = SimpleNamespace(directions=[(SimpleNamespace(tts=FakeStock()), True, True)])
    ctl = SimpleNamespace(sides=[me, them])
    assert app.Controller.them_voice(ctl) is None  # nothing made yet: the tray says so
    for i in range(3):
        v._add(speech(1.2, 0.3, i))
    states, lang, seconds = app.Controller.them_voice(ctl)
    assert set(states) == {'pocket-tts-ru', 'wespeaker-resnet34'} and lang == 'en' and seconds == 3.6
    assert app.Controller.them_voice(SimpleNamespace(sides=[me])) is None


def test_a_voice_kept_from_the_tray_is_theirs_from_the_first_clause_of_the_next_call(tmp_path):
    """S3's profile: them() -> voice.save_them (the tray's RememberDialog.keep) -> live.kept: the same state and speaker
    vector, `kind = them`, no recording of their speech; the next call speaks it from their first clause."""
    from peresmeshnik import voice
    v = voice_of()
    for i in range(3):
        v._add(speech(1.2, 0.1, i))  # B: 3.6 s, their voice is made
    states, lang, seconds = v.them()
    saved = voice.save_them('Anna', lang, seconds, states, tmp_path)
    assert saved.meta['kind'] == 'them' and sorted(p.name for p in saved.path.iterdir()) == [
        'pocket-tts-ru.npy', 'voice.json', 'wespeaker-resnet34.npy']
    [(vec, state, s)] = live.kept('pocket-tts-ru', tmp_path)
    assert np.array_equal(state, v.speaking.voice.state) and vec @ [0, 1] > 0.99 and s == seconds
    assert live.kept('pocket-tts-en', tmp_path) == []  # another engine: made again in that call
    eng = FakeClone()
    w = voice_of(eng, kept=live.kept('pocket-tts-ru', tmp_path))
    w._add(speech(1.2, 0.1, 9))
    assert np.array_equal(w.voice().state, state) and eng.made == []


def test_a_voice_pack_voice_speaks_in_the_call_and_is_not_offered_to_keep():
    """NVIDIA (the voice pack's Qwen3-TTS / XTTS): their voice is an id in that call's runner, so the tray does not offer
    to keep it (COMMON-stage8 23:49); in the call it speaks as Pocket's does."""
    v = live.LiveVoice(FakeClone, AmpSpeaker, 'xtts-v2', 'en')
    v._q.put(None)
    v._thread.join()
    for i in range(3):
        v._add(speech(1.2, 0.3, i))
    assert v.voice() is not None and v.them() is None


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

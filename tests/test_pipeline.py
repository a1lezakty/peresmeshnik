"""LocalDirection on fake engines: clause segmentation, event order, latency log, silence skips ASR. No models."""
import json
import logging
import threading
import time

import numpy as np
import pytest

from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech
from peresmeshnik.pipeline import LocalDirection, run

FRAME = 512


class EnergyVad:
    """Silero's event contract on an energy threshold."""
    min_silence_frames = 10  # 320 ms

    def __init__(self):
        self.pos, self.busy_s, self.speaking, self.silent = 0, 0.0, False, 0

    def feed(self, pcm):
        for i in range(0, len(pcm), FRAME):
            f = pcm[i:i + FRAME]
            self.pos += len(f)
            loud = float(np.abs(f).max()) > 0.1
            if not self.speaking:
                if loud:
                    self.speaking, self.silent = True, 0
                    yield 'start', f
                continue
            self.silent = 0 if loud else self.silent + 1
            if self.silent >= self.min_silence_frames:
                self.speaking = False
                yield 'end', f
            else:
                yield 'speech', f


class WordAsr:
    """One word per 0.5 s of audio; no punctuation, so long clauses are cut by length only."""
    def __init__(self):
        self.streams = 0

    def stream(self, lang):
        self.streams += 1
        return WordStream()


class WordStream:
    def __init__(self):
        self.pcm, self.offset, self.words = np.zeros(0, np.float32), 0, 0

    @property
    def duration(self):
        return len(self.pcm) / SAMPLE_RATE

    def text(self, n):
        return ' '.join(f'w{self.words + i}' for i in range(n))

    def feed(self, pcm, partial=True):
        self.pcm = np.concatenate([self.pcm, pcm])
        return self.text(int(self.duration * 2)) if len(self.pcm) % (10 * FRAME) == 0 else None

    cut_at = 'word'

    def cut(self, force=False, comma=False):
        if not force:
            return None
        n = int(self.duration * 2) - 1  # keep the last word
        at = n * SAMPLE_RATE // 2
        text, self.words = self.text(n), self.words + n
        self.pcm, self.offset = self.pcm[at:], self.offset + at
        return text

    def finish(self, settled_s=0.0):
        text = self.text(max(1, int(self.duration * 2)))
        self.offset += len(self.pcm)
        self.pcm = np.zeros(0, np.float32)
        return text


class UpperMt:
    def translate(self, text, src, tgt, context=()):
        return text.upper()


class ToneTts:
    sample_rate = 8000

    def synth(self, text, lang, voice=None):
        for _ in text.split():
            yield np.zeros(800, np.float32)


def tone(s):
    return (0.5 * np.sin(np.arange(int(s * SAMPLE_RATE)) * 0.1)).astype(np.float32)


def silence(s):
    return np.zeros(int(s * SAMPLE_RATE), np.float32)


def translate(pcm, caplog=None):
    asr = WordAsr()
    d = LocalDirection(EnergyVad(), asr, UpperMt(), ToneTts(), 'en', 'ru')
    d.realtime = False
    run(d, pcm)
    d.finish()
    return asr, list(d.events()), d.summary()


def test_clauses_events_and_log(caplog):
    caplog.set_level(logging.INFO, 'peresmeshnik.latency')
    asr, events, summary = translate(np.concatenate([silence(0.5), tone(1.5), silence(0.6), tone(1.0), silence(0.6)]))
    finals = [e for e in events if isinstance(e, Caption) and e.final]
    assert [c.translation for c in finals] == [c.original.upper() for c in finals]
    assert len(finals) == 2 and asr.streams == 2
    # one chunk per word; speech starts after the first caption, captions do not wait for speech
    kinds = ''.join('C' if isinstance(e, Caption) and e.final else 'S' for e in events if isinstance(e, Speech) or e.final)
    assert kinds.count('S') == sum(len(c.original.split()) for c in finals) and kinds.index('S') > kinds.index('C')
    rows = [json.loads(r.getMessage()) for r in caplog.records]
    assert [r['reason'] for r in rows] == ['pause', 'pause']
    assert {'asr_final_ms', 'mt_ms', 'tts_first_ms', 'tts_ms'} <= rows[0].keys()
    assert 'voice_lag_s' not in rows[0]  # not real time, so no lag numbers
    assert 2.0 < rows[0]['end_s'] < 2.2  # speech ended at 2.0 s; the trailing silence is not counted
    assert summary['clauses'] == 2 and summary['audio_s'] > 4


def test_silence_runs_no_asr():
    asr, events, _ = translate(silence(3.0))
    assert asr.streams == 0 and events == []


def test_long_speech_is_cut():
    caplog_rows = []
    handler = logging.Handler()
    handler.emit = lambda r: caplog_rows.append(json.loads(r.getMessage()))
    log = logging.getLogger('peresmeshnik.latency')
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        _, events, _ = translate(np.concatenate([tone(12.0), silence(0.6)]))
    finally:
        log.removeHandler(handler)
    finals = [e.original for e in events if isinstance(e, Caption) and e.final]
    words = ' '.join(finals).split()
    assert words == [f'w{i}' for i in range(len(words))]  # nothing lost or repeated across cuts
    assert [r['reason'] for r in caplog_rows][:-1] == ['max'] * (len(finals) - 1)
    assert len(finals) >= 3


def test_long_speech_gets_translated_partials():
    """Stage 5: before a long clause ends, the words two partials agree on are translated for the subtitle (a
    caption without the original, not final); the final caption of the clause comes after them. The translator keeps up
    with the words, as in a call (audio at 1x, MT in tens of ms): fed as fast as it goes, a busy machine would let every
    partial be skipped behind the next one (3-4 runs of 20 under load before)."""
    done = threading.Event()

    class Mt(UpperMt):
        def translate(self, *a, **k):
            try:
                return super().translate(*a, **k)
            finally:
                done.set()
    d = LocalDirection(EnergyVad(), WordAsr(), Mt(), ToneTts(), 'en', 'ru')
    d.realtime, put = False, d._clauses.put

    def paced(c, *a, **k):
        done.clear()
        put(c, *a, **k)
        if c is not None and c.get('partial'):
            assert done.wait(10)
    d._clauses.put = paced
    run(d, np.concatenate([tone(6.0), silence(0.6)]))
    d.finish()
    events = list(d.events())
    caps = [e for e in events if isinstance(e, Caption)]
    early = [i for i, e in enumerate(caps) if not e.final and e.translation]
    assert early, caps
    for i in early:
        c = caps[i]
        final = next(j for j, e in enumerate(caps) if e.final and e.clause == c.clause)
        assert i < final and c.original == '' and caps[final].translation.startswith(c.translation)


def test_early_cut_at_a_comma():
    """With `early_s` a long clause is cut at a comma too, with at least MIN_PART_S of speech before it."""
    toks, ts = [' a', ' b', ',', ' c', ' d', ' e'], [0.2, 0.5, 0.7, 1.5, 2.0, 2.5]
    assert stream_of(ScriptedAsr(Result(toks, ts)), 4.0).cut() is None  # no sentence end
    s = stream_of(ScriptedAsr(Result(toks, ts)), 4.0)
    assert s.cut(comma=True) == 'a b,' and s.cut_at == 'comma'
    early = [0.2, 0.4, 0.5, 0.9, 2.0, 2.5]  # the comma 0.9 s in: too short a part
    assert stream_of(ScriptedAsr(Result(toks, early)), 4.0).cut(comma=True) is None


@pytest.mark.filterwarnings('ignore::pytest.PytestUnhandledThreadExceptionWarning')
def test_dead_worker_does_not_hang():
    class BrokenMt:
        def translate(self, *a, **k):
            raise RuntimeError('boom')

    d = LocalDirection(EnergyVad(), WordAsr(), BrokenMt(), ToneTts(), 'en', 'ru')
    run(d, np.concatenate([tone(1.0), silence(0.6), tone(1.0), silence(0.6)]))
    with pytest.raises(RuntimeError, match='boom'):
        d.finish()  # returns (no hang) and says why
    assert not [e for e in d.events() if isinstance(e, Caption) and e.final]
    d.close()


class SlowTts(ToneTts):
    def synth(self, text, lang, voice=None):
        for _ in text.split():
            time.sleep(0.2)
            yield np.zeros(800, np.float32)


def two_clauses():
    return np.concatenate([tone(1.5), silence(0.6), tone(1.5), silence(0.6)])


def test_captions_do_not_wait_for_speech():
    d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), SlowTts(), 'en', 'ru')
    d.realtime = False
    run(d, two_clauses())
    threading.Thread(target=d.finish).start()
    arrived = [(time.perf_counter(), e) for e in d.events()]
    finals = [t for t, e in arrived if isinstance(e, Caption) and e.final]
    words0 = len(next(e for _, e in arrived if isinstance(e, Caption) and e.final).original.split())
    speech = [t for t, e in arrived if isinstance(e, Speech)]
    assert [e.clause for _, e in arrived if isinstance(e, Caption) and e.final] == [0, 1]
    assert finals[1] < speech[words0 - 1]  # the second caption is out before the first clause is spoken


def test_close_drops_queued_speech():
    d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), SlowTts(), 'en', 'ru')
    d.realtime = False
    run(d, np.concatenate([two_clauses(), two_clauses()]))
    time.sleep(0.3)
    t = time.perf_counter()
    d.close()  # translation off: no more speech, at once
    assert time.perf_counter() - t < 0.5
    assert sum(isinstance(e, Speech) for e in d.events()) < 6  # of ~14 words, one chunk per 0.2 s


# OnnxAsrStream (cuts, context, dedup, empty finals) on a fake model with fixed tokens and timestamps
class Result:
    def __init__(self, toks=(), ts=()):
        self.tokens, self.timestamps, self.text = list(toks), list(ts), ''.join(toks).strip()


class ScriptedAsr:
    partial_every = SAMPLE_RATE // 10  # a partial every 0.1 s, so feed() returns one per chunk below

    def __init__(self, *results):
        self.results, self.calls = list(results), []

    def recognize(self, pcm, lead=True, lang=None):
        self.calls.append((round(len(pcm) / SAMPLE_RATE, 2), lead))
        return self.results.pop(0) if self.results else Result()


def stream_of(asr, seconds, lang='en'):
    from peresmeshnik.local.asr import OnnxAsrStream
    s = OnnxAsrStream(asr, lang)
    s.feed(np.zeros(int(seconds * SAMPLE_RATE), np.float32))
    return s


def test_cut_at_sentence_end_and_context():
    asr = ScriptedAsr(Result([' Hello', ' there', '.', ' How', ' are', ' you'], [0.2, 0.6, 0.9, 1.2, 1.6, 2.0]),
                      # next window: 1 s of context in front; 'there.' (t < 1 s) belongs to the previous clause
                      Result([' there', '.', ' How', ' are', ' you', ' today'], [0.3, 0.6, 1.0, 1.4, 1.8, 2.3]))
    s = stream_of(asr, 3.0)
    assert asr.calls == [(3.0, True)]  # an onset window gets the lead-in
    assert s.cut() == 'Hello there.'
    assert s.offset == int(1.05 * SAMPLE_RATE) and s.ctx == SAMPLE_RATE  # cut between '.' and 'How', 1 s kept
    assert s.feed(np.zeros(SAMPLE_RATE // 10, np.float32)) == 'How are you today'
    assert asr.calls[-1][1] is False  # mid-speech window: no lead-in


def test_no_cut_inside_abbreviation_and_forced_cut():
    asr = ScriptedAsr(Result([' The', ' U', '.', 'N', '.', ' also', ' hopes'], [0.2, 0.5, 0.6, 0.7, 0.8, 1.0, 1.4]))
    assert stream_of(asr, 3.0).cut() is None  # "U.N. also" is not a sentence end
    asr = ScriptedAsr(Result([' a', ' b', ',', ' c', ' d', ' e', ' f'], [0.2, 0.5, 0.8, 1.0, 3.0, 3.5, 4.6]))
    s = stream_of(asr, 5.0)
    assert s.cut(force=True) == 'a b, c d'  # the comma is in the first half; 'f' is too near the window end


def test_dedup_across_cut_and_empty_final_retry():
    asr = ScriptedAsr(Result([' one', ' two', ' three'], [0.2, 0.8, 1.5]),
                      Result([' two', ' four'], [1.0, 1.5]),  # repeats 'two' right after the context
                      Result(), Result(), Result([' five'], [0.2]))  # partial and final come back empty, the retry has it
    s = stream_of(asr, 3.0)
    assert s.cut(force=True) == 'one two'
    assert s.finish() == 'four'
    s = stream_of(asr, 1.0)
    assert s.finish() == 'five' and asr.calls[-2:] == [(1.0, True), (1.0, False)]
    # the window's timestamps ran ahead: the next window hears 'cope with' after the cut, not in its context
    asr = ScriptedAsr(Result([' to', ' cope', ' with', ' the'], [0.2, 0.8, 1.2, 2.0]),
                      Result([' warming', ' cope', ' with', ' the', ' impacts.'], [0.5, 1.0, 1.3, 1.6, 1.9]))
    s = stream_of(asr, 3.0)
    assert s.cut(force=True) == 'to cope with'
    assert s.finish() == 'the impacts.'


def test_cuts_in_languages_without_spaces_or_case():
    """ja / zh: no spaces, so any token boundary is a word boundary and a repeat is counted in characters; ar, ko,
    ja, zh: no capitals, so a sentence end is punctuation before any letter."""
    asr = ScriptedAsr(Result(['今日は', '晴れ', 'です', '。', '明日', 'も', '晴れ'], [0.2, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]),
                      Result(['です', '。', '明日', 'も', '晴れ', 'でしょう', '。'], [0.3, 0.6, 1.1, 1.3, 1.5, 1.8, 2.1]))
    s = stream_of(asr, 3.0, 'ja')
    assert s.cut() == '今日は晴れです。'
    assert s.finish() == '明日も晴れでしょう。'
    # a forced cut between tokens, the next window repeats its last characters: dropped
    asr = ScriptedAsr(Result(['我们', '需要', '应对', '气候', '变化'], [0.2, 0.6, 1.0, 1.4, 2.6]),
                      Result(['应对', '气候', '变化', '的', '影响'], [1.1, 1.3, 1.6, 1.9, 2.2]))
    s = stream_of(asr, 3.0, 'zh')
    assert s.cut(force=True) == '我们需要应对'  # '变化' is too near the window end, so is the cut before it
    assert s.finish() == '气候变化的影响'  # the next window heard '应对' after the cut, not in its context
    asr = ScriptedAsr(Result([' مرحبا', ' بكم', '.', ' كيف', ' حالكم', ' اليوم'], [0.2, 0.6, 0.9, 1.2, 1.6, 2.0]))
    assert stream_of(asr, 3.0, 'ar').cut() == 'مرحبا بكم.'
    asr = ScriptedAsr(Result([' 안녕하세요', '?', ' 저는', ' 학생', '입니다'], [0.2, 0.6, 1.0, 1.4, 2.0]))
    assert stream_of(asr, 3.0, 'ko').cut() == '안녕하세요?'
    asr = ScriptedAsr(Result([' مرحبا', '،', ' كيف', ' حالكم', ' اليوم', ' يا', ' اصدقاء'], [0.2, 0.6, 1.2, 1.6, 2.6, 3.0, 4.6]))
    assert stream_of(asr, 5.0, 'ar').cut(force=True) == 'مرحبا، كيف حالكم اليوم'  # no Arabic comma in the second half


def test_a_model_without_timestamps_cuts_at_the_quietest_moment():
    """Whisper (text only): no cut at punctuation; a forced cut at the quietest 0.1 s of the second half, its words
    from one more run on the audio before it; the rest is the next clause, its repeat of the cut's words dropped."""
    from peresmeshnik.local.asr import OnnxAsrStream

    class Text:
        def __init__(self, text):
            self.text, self.tokens, self.timestamps = text, None, None
    asr = ScriptedAsr(Text('one two. three four'), Text('one two three'), Text('three four five'))
    asr.timed = False
    s = OnnxAsrStream(asr, 'en')
    x = np.ones(int(5.0 * SAMPLE_RATE), np.float32)
    x[int(3.3 * SAMPLE_RATE):int(3.4 * SAMPLE_RATE)] = 0  # the pause between words
    assert s.feed(x) == 'one two. three four'
    assert s.cut() is None  # no times: a sentence end in the text says nothing about where it is in the audio
    assert s.cut(force=True) == 'one two three'
    assert asr.calls[-1] == (3.35, True) and s.offset == int(3.35 * SAMPLE_RATE) and s.ctx == 0
    assert s.finish() == 'four five'


def test_a_phrase_said_twice_across_a_cut_stays():
    """'Thank you. Thank you very much.' without a pause: the next window hears the first 'Thank you.' in its
    context before the cut, so the second one is said again, not a repeat of the cut's audio."""
    asr = ScriptedAsr(Result([' Thank', ' you.', ' Thank', ' you', ' very', ' much.'], [0.2, 0.5, 1.0, 1.3, 2.5, 2.8]),
                      Result([' Thank', ' you.', ' Thank', ' you', ' very', ' much.'], [0.2, 0.5, 1.0, 1.3, 1.6, 1.9]))
    s = stream_of(asr, 3.0)
    assert s.cut() == 'Thank you.'
    assert s.finish() == 'Thank you very much.'


# queue policy and barge-in: real-time feeds (a few seconds each), so the lags are real
class LongTts(ToneTts):
    """Speech twice as long as its clause (0.5 s of audio per word -> 1 s of speech), as a translation into a
    longer language on continuous speech; `speed` shortens it."""
    def __init__(self, speeds=False):
        self.speeds = []
        if speeds:
            self.synth = self._synth_speed

    def synth(self, text, lang, voice=None):
        for _ in text.split():
            yield np.zeros(self.sample_rate, np.float32)

    def _synth_speed(self, text, lang, voice=None, speed=1.0):
        self.speeds.append(speed)
        for _ in text.split():
            yield np.zeros(int(self.sample_rate / speed), np.float32)


def voice_rows(tts, max_lag_s, seconds=4.0):
    rows = []
    handler = logging.Handler()
    handler.emit = lambda r: rows.append(json.loads(r.getMessage()))
    log = logging.getLogger('peresmeshnik.latency')
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), tts, 'en', 'ru', max_clause_s=1.0, max_lag_s=max_lag_s)
        threading.Thread(target=lambda: list(d.events()), daemon=True).start()
        run(d, np.concatenate([tone(seconds), silence(0.6)]), realtime=True)
        d.finish()
    finally:
        log.removeHandler(handler)
    return rows


def test_queue_policy_caps_the_voice_lag():
    free = voice_rows(LongTts(), None)
    assert max(r['voice_lag_s'] for r in free) > 2.5 and not any('dropped' in r for r in free)  # falls ever further behind
    capped = voice_rows(LongTts(), 1.5)
    spoken = [r for r in capped if 'dropped' not in r]
    assert spoken and max(r['voice_lag_s'] for r in spoken) < 1.5 + 0.3  # + one TTS first-chunk time
    assert any('dropped' in r for r in capped)  # and it costs whole clauses (their captions stay)
    tts = LongTts(speeds=True)
    voice_rows(tts, 3.0)
    assert max(tts.speeds) > 1.0 and all(1.0 <= s <= 1.4 for s in tts.speeds)  # behind: faster, within the cap


def test_barge_in_drops_what_is_not_heard_yet():
    """The user talks over the voice: the clause being spoken stops, a clause that ends later is spoken."""
    d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), SlowTts(), 'en', 'ru')
    got = []

    def consume():
        for e in d.events():
            if isinstance(e, Speech):
                got.append(time.perf_counter())
                if len(got) == 1:
                    d.interrupt()
    threading.Thread(target=consume, daemon=True).start()
    run(d, np.concatenate([tone(1.5), silence(1.5), tone(1.5), silence(0.6)]), realtime=True)
    d.finish()
    time.sleep(0.1)
    assert len(got) == 1 + 3  # clause 0: one chunk of its 3 words, then cut; clause 1: all 3 words
    assert d.stats['dropped'] == 0  # nothing was queued at the cut; clause 0's tail broke off mid-synthesis


def test_call_app_choice():
    """A known call app is taken by itself only when it is the one playing; a browser never (all its tabs would move)."""
    from peresmeshnik.audio.capture_pipewire import apps, guess_call_app, is_browser
    tg, web = {'key': 'Telegram', 'name': 'TelegramDesktop'}, {'key': 'chrome', 'name': 'Google Chrome'}
    zoom = {'key': 'zoom', 'name': 'ZOOM'}
    assert guess_call_app([tg, web, web]) == 'Telegram'
    assert guess_call_app([tg, zoom]) == '' and guess_call_app([web]) == ''
    assert apps([web, tg, web]) == [('chrome', 'Google Chrome (all its sound, every tab)'), ('Telegram', 'TelegramDesktop')]
    assert is_browser('firefox-esr') and not is_browser('telegram-desktop')


def test_barge_in_on_my_words_not_on_a_backchannel():
    """My side's partial captions cut their voice once I say a few words; "yeah" / "ага" while listening does not."""
    from peresmeshnik.app import Side
    me = object.__new__(Side)
    me.name, me.live, me.overlay, me.on_error = 'me', False, type('O', (), {'show': lambda self, c: None})(), print
    talks = []
    me.on_talk = lambda: talks.append(1)
    events = [Caption('ага', clause=0), Caption('ага', 'yeah', True, 0), Caption('да, но я', clause=1), Caption('да, но я думаю', clause=1)]
    me._consume(type('D', (), {'events': lambda self: iter(events)})(), True, False)
    assert len(talks) == 2  # the two partials of clause 1, none for the back-channel


def test_blas_threads_pinned_before_numpy_whatever_imports_first():
    """The app imports numpy before the pipeline: the BLAS pool must be pinned by the package itself
    (it took every core: 3.0 instead of 0.8 CPU s per audio s for them -> me on a 16-core box)."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k not in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    code = 'import peresmeshnik.app, os, sys; print(os.environ["OPENBLAS_NUM_THREADS"], os.environ["OMP_NUM_THREADS"])'
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.split() == ['1', '1'], r.stdout + r.stderr


def test_latency_events_in_real_time_and_through_the_live_switch():
    """Stage 5: a real-time direction puts out Latency per clause and link; the me -> them live switch passes
    them on (it took every event that is not a caption for speech)."""
    from peresmeshnik.audio.mic import LiveSwitch
    from peresmeshnik.engines import Latency
    d = LocalDirection(EnergyVad(), WordAsr(), UpperMt(), ToneTts(), 'en', 'ru')
    sw = LiveSwitch(d, 8000)
    got = []
    t = threading.Thread(target=lambda: got.extend(sw.events()))
    t.start()
    run(d, np.concatenate([tone(1.0), silence(0.6)]), realtime=True)
    d.finish()
    t.join(5)
    lat = {e.what: e.s for e in got if isinstance(e, Latency)}
    assert set(lat) == {'vad', 'asr', 'caption', 'tts_first', 'voice'}, got
    assert 0.3 <= lat['vad'] <= lat['asr'] <= lat['caption'] <= lat['tts_first'] <= lat['voice'] < 2.0, lat
    assert any(isinstance(e, Speech) for e in got) and any(isinstance(e, Caption) and e.final for e in got)


def test_whisper_decodes_a_capped_number_of_tokens():
    """A window Whisper hears as a loop would take 448 tokens; the language guess (3 tokens) stays as asked."""
    from peresmeshnik.local.asr import WHISPER_MAX_TOKENS, cap_whisper

    class Inner:
        def _decoding(self, enc, toks, max_length=448):
            return max_length
    model = type('M', (), {})()
    model.asr = Inner()
    cap_whisper(model)
    assert model.asr._decoding(None, None) == WHISPER_MAX_TOKENS and model.asr._decoding(None, None, 3) == 3


def test_a_stream_that_will_not_move_back_stops_nothing(monkeypatch):
    """send_back() raises when the sound server drops the move three times: the router still moves the call's stream,
    return_strays() still gives the others back and says which failed."""
    from peresmeshnik.audio import capture_pipewire as cp
    ps = [{'index': '1', 'key': 'chromium', 'name': 'Chromium', 'sink': cp.CALL_SINK, 'restore': {}},
          {'index': '2', 'key': 'zoom', 'name': 'Zoom', 'sink': 'alsa', 'restore': {'k': 'v'}},
          {'index': '3', 'key': 'player', 'name': 'Player', 'sink': cp.CALL_SINK, 'restore': {}}]
    back = []

    def send_back(index, sink):
        if index == '1':
            raise RuntimeError('stream 1 never reached alsa')
        back.append(index)
    monkeypatch.setattr(cp, 'players', lambda: ps)
    monkeypatch.setattr(cp, 'send_back', send_back)
    monkeypatch.setattr(cp, 'move', lambda index, sink: True)
    r = cp.CallRouter('zoom')
    r._proc = object()  # started
    r._move()
    assert r.moved == {'2': 'alsa'} and back == ['3']
    with pytest.raises(RuntimeError, match=r'Chromium: stream 1 never.*\(given back: Player\)'):
        cp.return_strays()


def test_a_recording_is_read_without_copies_of_itself(tmp_path):
    """L-17: `run --in` / `app --them-file`: a long stereo 48 kHz recording is read in blocks, down-mixed and resampled
    on the way; memory stays near the 16 kHz mono result, not three copies of the file (~95 MB per minute)."""
    import tracemalloc

    import soundfile as sf

    from peresmeshnik.pipeline import read_audio
    sr, n = 48_000, 48_000 * 20
    t = np.arange(n) / sr
    sf.write(tmp_path / 'call.wav', np.stack([np.sin(2 * np.pi * 440 * t), np.zeros(n)], 1).astype(np.float32) * 0.5, sr)
    tracemalloc.start()
    x = read_audio(str(tmp_path / 'call.wav'))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert x.dtype == np.float32 and abs(len(x) - 20 * SAMPLE_RATE) <= 1
    assert peak < 3 * x.nbytes, f'{peak >> 10} KB for {x.nbytes >> 10} KB of audio'
    ref = np.sin(2 * np.pi * 440 * np.arange(len(x)) / SAMPLE_RATE) * 0.25  # the mix of the two channels
    assert np.abs(x[100:-100] - ref[100:-100]).max() < 0.01

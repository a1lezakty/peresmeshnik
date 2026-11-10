"""Enrollment checks and the clone engine's text/length guards (no torch, no model needed)."""
import json
import os
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from peresmeshnik import voice
from peresmeshnik.local.clone import SR, XttsClone, budget_samples, cap_pauses, guard, split_text
from peresmeshnik.local.xtts import FADE_IN, SENTENCE_END, SENTENCE_PAUSE_S, closed, ending, normalize, shape

REF_RU = Path(os.environ.get('BENCH_DATA', 'bench-data')) / 'set' / 'ref_ru.wav'  # LibriVox reader (bench/data.py), if present


def speech_like(seconds, level=0.1, noise=0.001, sr=16_000, talk=0.8, seed=0):
    """Noise 'phrases' with pauses: `talk` share of every 2 s is speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    env = (t % 2) < 2 * talk
    return (level * env * rng.standard_normal(len(t)) + noise * rng.standard_normal(len(t))).astype(np.float32)


def test_good_recording_passes():
    r = voice.check(speech_like(45), 16_000)
    assert r.ok, r.errors
    assert r.speech_s == pytest.approx(36, abs=2)
    assert r.level_db == pytest.approx(-20, abs=1.5) and r.snr_db > 35


def test_the_clone_reads_its_checkpoint_with_the_apps_reader(monkeypatch, tmp_path):
    """M-19 (review of PR #22): xtts.Engine loads the model with `_read_checkpoint` (writes nothing), not coqui-tts's
    loader (writes model.pth again): checked on stand-ins of torch and coqui-tts, so it runs without them."""
    from peresmeshnik.local import xtts
    used = []

    class Model:
        def load_checkpoint(self, cfg, checkpoint_dir, eval):
            used.append(self._load_checkpoint)  # what coqui's load_checkpoint calls to read the file

        def to(self, device):
            pass
    mods = {'torch': types.SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None,
                                           cuda=types.SimpleNamespace(is_available=lambda: False)),
            'TTS.tts.configs.xtts_config': types.SimpleNamespace(XttsConfig=type('C', (), {'load_json': lambda s, p: None})),
            'TTS.tts.models.xtts': types.SimpleNamespace(Xtts=types.SimpleNamespace(init_from_config=lambda cfg: Model()))}
    for name in ('TTS', 'TTS.tts', 'TTS.tts.configs', 'TTS.tts.models'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    xtts.Engine(tmp_path)
    assert used == [xtts._read_checkpoint]


def test_a_dc_offset_is_not_noise():
    """Stage 7 (C's finding): a microphone with a DC offset (here -40 dBFS) is not "noisy" nor louder: the offset is
    no sound, the check measures without it."""
    x = speech_like(45)
    r, dc = voice.check(x, 16_000), voice.check(x + np.float32(0.01), 16_000)
    assert dc.ok and dc.snr_db == pytest.approx(r.snr_db, abs=1) and dc.level_db == pytest.approx(r.level_db, abs=0.5), dc


@pytest.mark.parametrize('pcm, sr, problem', [
    (speech_like(20), 16_000, 'need 30-60 s'),
    (speech_like(45, talk=0.2), 16_000, 'of speech'),
    (np.clip(speech_like(45) * 20, -1, 1), 16_000, 'clipping'),
    (speech_like(45, noise=0.05), 16_000, 'background'),
    (speech_like(45, level=0.001, noise=0.00001), 16_000, 'too quiet'),
    (speech_like(45, sr=8000), 8000, 'sample rate'),
])
def test_bad_recordings_fail(pcm, sr, problem):
    r = voice.check(pcm, sr)
    assert not r.ok and any(problem in e for e in r.errors), r.errors


def test_telephone_band_warns():
    x = speech_like(45)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / 16_000)
    X[(f < 300) | (f > 3400)] = 0
    r = voice.check(np.fft.irfft(X, len(x)).astype(np.float32), 16_000)
    assert r.ok and r.band_db < voice.BAND_DB and any('above 4 kHz' in w for w in r.warnings), r
    assert voice.verdict(r).startswith('Good enough: no sound above 4 kHz')
    assert voice.check(x, 16_000).band_db > -10  # white-noise "speech" has it all


def test_clone_check_thresholds():
    class Clone:
        sample_rate = 24_000

        def synth(self, text, lang, v):
            yield np.zeros(2400, np.float32)

        def similarity(self, v, pcm, sr):
            return 0.3

    class Asr:
        def recognize(self, pcm):
            return type('R', (), {'text': 'good morning can you hear me well'})()
    r = voice.check_clone(Clone(), None, 'en', Asr())
    assert r.wer > voice.MAX_WER and r.similarity == 0.3 and not r.ok
    assert voice.clone_verdict(r).startswith('Record again: the clone is hard to understand')
    assert any('somewhat like you' in w for w in r.warnings)
    assert voice.wer('Ёлка, раз два!', 'елка раз три') == pytest.approx(1 / 3)


def test_test_phrases_in_every_language_the_clone_speaks():
    """check_clone says TEST_PHRASES[their language]: a language of the clone without them was a KeyError (stage 6, S1).
    The wizard's recording text: one for each of the 14."""
    from peresmeshnik import languages, models
    spoken = {lang for m in models.registry().values() if m.role == 'clone' for lang in m.langs}
    assert 'ja' in spoken and spoken <= set(voice.TEST_PHRASES) and all(len(voice.TEST_PHRASES[x]) == 2 for x in spoken)
    assert set(languages.NAMES) == set(voice.PROMPTS) and len(voice.PROMPTS) == 14


def test_wer_of_the_test_phrases_in_arabic_chinese_japanese():
    """An ASR writes no Arabic vowel marks (the tanwin of مرحبًا would split the word); zh / ja go by character."""
    import re
    for phrase in voice.TEST_PHRASES['ar']:
        heard = re.sub('[\u064b-\u0652\u0670]', '', phrase)
        assert heard != phrase and voice.wer(phrase, heard) == 0
    zh = voice.TEST_PHRASES['zh'][0]
    assert voice.wer(zh, zh.replace('十', '四')) == pytest.approx(1 / len(voice._words(zh)))


def test_too_long_is_trimmed_with_warning(tmp_path):
    x = speech_like(75)
    r = voice.check(x, 16_000)
    assert r.ok and r.warnings
    v = voice.save(x, 16_000, 'me', 'ru', 'text', tmp_path)
    import soundfile as sf
    y, sr = sf.read(v.path / 'reference.wav')
    assert len(y) / sr == pytest.approx(60, abs=0.01)
    assert json.loads((v.path / 'voice.json').read_text())['quality']['seconds'] == pytest.approx(60)  # of what is kept


def test_save_normalises_and_loads(tmp_path):
    x = speech_like(40, level=0.02, noise=0.0002)
    v = voice.save(x, 16_000, 'me', 'ru', 'привет', tmp_path)
    (v.path / 'xtts.pt').write_bytes(b'old cache')
    v = voice.save(x, 16_000, 'me', 'ru', 'привет', tmp_path)
    assert not (v.path / 'xtts.pt').exists(), 'engine cache of the old recording must go'
    import soundfile as sf
    y, sr = sf.read(v.path / 'reference.wav', dtype='float32')
    assert voice.check(y, sr).level_db == pytest.approx(voice.TARGET_DB, abs=1.5)
    w = voice.load('me', tmp_path)
    assert w.meta['transcript'] == 'привет' and w.meta['quality']['seconds'] == pytest.approx(40)


@pytest.mark.skipif(not REF_RU.exists(), reason='bench data not here')
def test_real_reference_passes():
    r = voice.check(*voice.read_audio(REF_RU))
    assert r.ok, r.errors


def test_cli_import_and_list(tmp_path, capsys):
    import soundfile as sf
    f = tmp_path / 'me.wav'
    sf.write(f, speech_like(35), 16_000)
    assert voice.main(['enroll', '--import', str(f), '--voices', str(tmp_path / 'v')]) == 0
    sf.write(f, speech_like(10), 16_000)
    assert voice.main(['enroll', '--import', str(f), '--name', 'short', '--voices', str(tmp_path / 'v')]) == 1
    assert voice.main(['list', '--voices', str(tmp_path / 'v')]) == 0
    out = capsys.readouterr().out
    assert 'saved voice profile' in out and 'Not saved' in out and out.strip().splitlines()[-1].startswith('me\tru')


def test_record_needs_a_prompt_in_that_language(monkeypatch, capsys):
    import peresmeshnik.audio.mic as mic
    monkeypatch.setattr(mic, 'record', lambda *a, **k: pytest.fail('must not record'))
    with pytest.raises(SystemExit):
        voice.main(['enroll', '--record', '30', '--lang', 'nl'])  # none of the 14: no prompt text
    assert '--transcript' in capsys.readouterr().err


@dataclass(frozen=True)
class FakeLatent:
    n: int

    def to(self, device):
        return self


def test_reenroll_replaces_the_voice(tmp_path, monkeypatch):
    """A second recording of the same profile in the same process must not keep the old conditioning."""
    fake_torch = types.SimpleNamespace(save=lambda obj, f: Path(f).write_bytes(pickle.dumps(obj)),
                                       load=lambda f, map_location=None: pickle.loads(Path(f).read_bytes()))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    eng = XttsClone.__new__(XttsClone)  # no model: conditioning stubbed
    eng.device, eng._latents, calls = 'cpu', {}, []

    def condition(pcm, sr):
        calls.append(sr)
        return {'gpt': FakeLatent(len(calls)), 'spk': FakeLatent(len(calls))}
    eng._condition = condition
    first = eng.latents(eng.enroll(speech_like(35), 16_000, root=tmp_path))
    v = eng.enroll(speech_like(35, seed=1), 16_000, root=tmp_path)
    assert len(calls) == 2 and eng.latents(v) != first == (FakeLatent(1), FakeLatent(1))
    assert pickle.loads((v.path / 'xtts.pt').read_bytes())['gpt'] == FakeLatent(2)


def test_split_text():
    assert split_text('Hello, world.', 250) == ['Hello, world.']
    assert split_text(' ... ', 250) == []
    long = ('Это длинное предложение без конца, которое тянется и тянется, ' * 8).strip()
    parts = split_text(long, 182)
    assert len(parts) > 1 and all(len(p) <= 182 for p in parts)
    assert ' '.join(parts) == ' '.join(long.split())
    assert all(len(p) <= 50 for p in split_text('слово ' * 100, 50))
    # zh / ja: cut after their own punctuation (no space follows it), joined back without a space
    assert split_text('你好，世界！今天很好。我们走吧。', 10) == ['你好，世界！', '今天很好。', '我们走吧。']
    assert split_text('警方说，尸体看上去在那里放了一天左右。', 82) == ['警方说，尸体看上去在那里放了一天左右。']
    assert split_text('مرحبا، كيف حالك؟ أنا بخير.', 12) == ['مرحبا،', 'كيف حالك؟', 'أنا بخير.']  # Arabic comma, question mark


@pytest.mark.parametrize('lang, text, said', [
    ('de', 'Um 15:30 Uhr zahlte er 250,50 €, um 9:00.', 'Um 15 Uhr 30 zahlte er 250,50€, um 9 Uhr,'),
    ('de', 'Ab 9.30 Uhr, nicht 25:70. Usw.', 'Ab 9 Uhr 30, nicht 25:70. Usw.'),  # the period of an abbreviation stays
    ('fr', 'À 15h30 et 9h00, 20 €.', 'À 15 heures 30 et 9 heures, 20€,'),
    ('fr', 'De 9h à 1h30 et 21h.', 'De 9 heures à une heure 30 et vingt et une heures,'),  # "heure": feminine
    ('pt', 'Às 15h30 pagou 250,50 € e $20.', 'Às 15 horas e 30 pagou 250,50€ e 20 dólares,'),  # cents: XTTS's own
    ('pt', 'Custa R$ 250, US$ 20 e R$ 1.250,50 às 2:00.', 'Custa 250 reais, 20 dólares e 1.250 reais e 50 centavos às duas horas,'),
    ('en', 'At 15:30 she paid 250 $, score 3:2, 1:30:00, 9:75.', 'At 15 30 she paid 250$, score 3:2, 1:30:00, 9:75.'),
    ('it', 'Alle 3 p.m.', 'Alle 3 p.m.'),
    ('ko', '15:00에 $250를 냈다. 9:05.', '15시에 250 달러를 냈다. 9시 05분,'),
    ('zh', '15:30，250.50€', '15点30分，250.50 欧元'),
    ('it', 'Secondo la polizia, il corpo era lì da un giorno.', 'Secondo la polizia, il corpo era lì da un giorno,'),
    ('es', 'Bueno... Llegó tarde.»', 'Bueno... Llegó tarde,»'),
    ('it', 'Aspetta...', 'Aspetta...'),
])
def test_normalize_money_and_clock(lang, text, said):
    """What XTTS's own cleaning misreads (BENCH stage 6): a sign apart from its number, a currency it names wrong or
    drops, clock times, the final period where XTTS goes on after it (de fr pt ko it es). No num2words needed here."""
    assert ending(normalize(text, lang), lang) == said


class Chunk:  # what XTTS streams: a tensor
    def __init__(self, x):
        self.x = x

    def float(self):
        return self

    cpu = float

    def numpy(self):
        return self.x


def test_every_part_of_a_long_text_ends_with_a_comma():
    """A text over XTTS's limit (it: 213 chars) is said in parts: the final period of each becomes a comma."""
    said = []

    def stream(part, lang, *a, **k):
        said.append(part)
        yield Chunk(np.full(SR // 10, 0.1, np.float32))
    eng = XttsClone.__new__(XttsClone)  # no model
    eng.latents, eng.temperature, eng.top_p = lambda v: (None, None), 0.75, 0.85
    eng.model = types.SimpleNamespace(tokenizer=types.SimpleNamespace(char_limits={'it': 213}), inference_stream=stream)
    list(eng.synth('Il corpo era lì da un giorno e nessuno se ne era accorto prima di stamattina. ' * 6, 'it', 'me'))
    assert len(said) == 3 and all(p.endswith('stamattina,') for p in said), said


@pytest.mark.parametrize('lang, text, xtts_gets, paused', [
    ('it', 'Ciao.', 'Ciao,', True),  # the comma for XTTS, the pause after the sentence all the same
    ('en', 'Hello.', 'Hello.', True),
    ('it', 'perché il', 'perché il,', False),  # a clause cut mid-sentence (stage 5): a comma, no pause
])
def test_the_pause_after_a_sentence_survives_the_comma_ending(lang, text, xtts_gets, paused):
    said = []

    def stream(part, lang, *a, **k):
        said.append(part)
        yield Chunk(np.full(SR // 2, 0.1, np.float32))
    eng = XttsClone.__new__(XttsClone)  # no model
    eng.latents, eng.temperature, eng.top_p = lambda v: (None, None), 0.75, 0.85
    eng.model = types.SimpleNamespace(tokenizer=types.SimpleNamespace(char_limits={}), inference_stream=stream)
    x = np.concatenate(list(eng.synth(text, lang, 'me')))
    quiet = len(x) - 1 - np.nonzero(np.abs(x) > 1e-4)[0][-1]
    assert said == [xtts_gets] and (quiet >= SENTENCE_PAUSE_S * SR - 1) == paused, (said, quiet)


@pytest.mark.parametrize('lang, text, said', [
    ('ja', '2024年3月3日15:30、250.50ドル、12.5%、1,000人。',
     '二千二十四年三月三日十五時三十分、二百五十点五ドル、十二点五パーセント、千人。'),
    ('ja', '1,250,000人、12.05ドル。', '百二十五万人、十二点零五ドル。'),
    ('tr', 'Saat 15:30, 1.250,5 TL; 20€.', 'Saat 15 30, binikiyüzelli virgül beş TL; 20 avro.'),
    ('tr', 'Saat 15.30, 1.250.000 kişi, 3,05 TL, 2,3,4.', 'Saat 15 30, 1250000 kişi, üç virgül sıfır beş TL, 2,3,4.'),
    ('ar', 'دفع 1,250.5 دولار', 'دفع ألف و مئتان و خمسون فاصلة خمسة دولار'),
    ('ar', '12.05 $ و 1,250,000 و E£ 50 و £20 و 1,2,3', 'اثنا عشر فاصلة صفر خمسة دولار و 1250000 و 50 جنيه و 20 جنيه و 1,2,3'),
])
def test_normalize_numbers_xtts_misreads(lang, text, said):
    """Japanese digits (XTTS leaves them to the model), Turkish and Arabic decimals (num2words reads 12,5 as 12,50
    and 3,05 as 3,5), thousands, the Egyptian pound. num2words comes with coqui-tts: without it (the main CI) these run
    in the voice pack's job."""
    pytest.importorskip('num2words')
    assert normalize(text, lang) == said


def test_guard_cuts_runaway_and_stops_generation():
    pulled = []

    def chunks():
        for i in range(100):
            pulled.append(i)
            yield np.ones(1000, np.float32)
    out = list(guard(chunks(), 3500))
    assert sum(map(len, out)) == 3500 and out[-1][-1] == 0.0
    assert len(pulled) == 4  # stopped pulling right after the cut
    assert sum(map(len, guard(iter([np.ones(10, np.float32)] * 3), 3500))) == 30  # short: untouched


def test_budget_allows_slow_speech_only():
    text = "It is Martelly's fifth CEP in four years."  # spoken in 2.5-5.1 s; 8 s when XTTS babbled
    assert 5.1 * SR < budget_samples(text) < 8 * SR
    assert budget_samples('In 1999.') > budget_samples('In this.') + SR  # digits read long
    # Han / kana / Hangul: a syllable or two each; the shortest FLEURS sentences took XTTS 4.0-5.8 s (BENCH stage 6)
    assert budget_samples('警方说，尸体看上去在那里放了一天左右。') > 5.8 * SR


def test_cap_pauses_streams_and_keeps_speech():
    rng = np.random.default_rng(0)
    talk, gap = 0.3 * rng.standard_normal(SR).astype(np.float32), np.zeros(2 * SR, np.float32)
    x = np.concatenate([talk, gap, talk, np.zeros(SR // 5, np.float32), talk])
    out = np.concatenate(list(cap_pauses(iter(np.array_split(x, 37)))))  # odd chunk edges
    assert len(out) == pytest.approx(3 * SR + 0.4 * SR + 0.2 * SR, abs=SR // 50)  # 2 s gap -> 0.4 s, 0.2 s kept
    assert np.sum(out ** 2) == pytest.approx(np.sum(x ** 2))  # no speech lost


@pytest.mark.parametrize('sentence_end', [True, False])
def test_shape_fades_in_and_pauses_after_a_sentence(sentence_end):
    """XTTS starts mid-waveform (a click when the stream steps from silence into it) and ends a sentence with ~0.16 s of
    silence (back to back with the next clause: no pause, BENCH stage 5)."""
    talk = (0.5 + 0.3 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)).astype(np.float32)  # starts at full level
    tail = np.zeros(int(0.16 * SR), np.float32)
    chunks = [talk[:SR // 3], talk[SR // 3:], tail]
    out = list(shape(iter([c.copy() for c in chunks]), sentence_end))
    x = np.concatenate(out)
    assert abs(x[0]) < 0.5 / FADE_IN * 2 and np.max(np.abs(np.diff(x[:FADE_IN]))) < 0.05  # no step at the start
    assert np.array_equal(x[FADE_IN:len(talk)], talk[FADE_IN:])  # the rest untouched
    assert len(out) == 3 + sentence_end  # streams: the chunks as they came, then the pause
    silence = len(x) - len(talk)
    assert silence == (round(SENTENCE_PAUSE_S * SR) if sentence_end else len(tail))
    long = list(shape(iter([talk, np.zeros(SR, np.float32)]), True))  # a long enough silence already: nothing added
    assert len(long) == 2
    short = list(shape(iter([talk, talk[:SR // 200]]), True))  # a loud chunk under 10 ms ends it: not silence
    assert len(short[-1]) == round(SENTENCE_PAUSE_S * SR)


def test_a_fragment_ends_with_a_comma():
    assert closed('number of those opposed to the') == 'number of those opposed to the,'
    assert closed('  education   and ') == 'education and,'
    for done in ('Hello.', 'Really?', 'Wait!', 'and so,', 'as follows:', 'He said "yes."', 'Он сказал «да».', '',
                 '你好。', '真的吗？', '他说「好。」', '然后，', 'هل أنت هنا؟', 'ثم،'):
        assert closed(done) == done
    assert closed('我们') == '我们,'
    assert [bool(SENTENCE_END.search(t)) for t in ('Hello.', '你好。', '真的吗？', 'هل أنت هنا؟', '他说「好。」', 'and so,', '然后，')] \
        == [True] * 5 + [False] * 2


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX modes (Windows: the profile folder under %LOCALAPPDATA%)')
def test_the_voice_is_the_users_only(tmp_path):
    """L-09: the recording (enough to clone the voice) and its latents: folders 0700 whatever the umask."""
    old = os.umask(0o022)
    try:
        v = voice.save(speech_like(20, level=0.02, noise=0.0002), 16_000, 'me', 'ru', 'привет', tmp_path / 'voices')
    finally:
        os.umask(old)
    assert [oct(p.stat().st_mode & 0o777) for p in (tmp_path / 'voices', v.path)] == ['0o700', '0o700']


def test_a_number_num2words_cannot_read_stays_digits():
    """L-43: a translation with a number of 70+ digits (MT stuck on a digit) or a long Turkish fraction: num2words
    raises OverflowError; the clause is said with the digits, not lost (nor the clone with it)."""
    if os.environ.get('PERESMESHNIK_REQUIRE_CLONE') == '1':  # the voice-pack workflow: there, not a skip
        import num2words  # noqa: F401
    else:
        pytest.importorskip('num2words')
    from peresmeshnik.local.xtts import normalize
    long = '1' * 80
    assert long in normalize(f'番号は{long}です', 'ja') and '9' * 45 in normalize('12,' + '9' * 45 + ' TL', 'tr')

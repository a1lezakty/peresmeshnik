"""voice.json v2: kinds, consent, the one "my confirmed voice" check, v1 profiles upgraded, a contact's voice kept."""
import json
import os
import sys

import numpy as np
import pytest
from test_crash import app  # noqa: F401  (the fixture: crash reports into tmp_path)

from peresmeshnik import voice


def speech(seconds=40, sr=16_000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 2 * t) > -0.3)).astype(np.float32), sr


def test_mine_only_confirmed_me(tmp_path):
    v = voice.save(*speech(), 'me', 'ru', 'text', tmp_path)
    assert v.meta['version'] == 2 and v.meta['kind'] == 'me' and v.meta['consent'] is None
    with pytest.raises(voice.NotMine, match='not confirmed'):
        voice.mine('me', tmp_path)
    voice.save(*speech(), 'me', 'ru', 'text', tmp_path, {'how': 'mic-live', 'at': 'now', 'wer': 0.1, 'similarity': None})
    assert voice.mine('me', tmp_path).meta['consent']['how'] == 'mic-live'
    meta = json.loads((tmp_path / 'me' / 'voice.json').read_text(encoding='utf-8'))
    for how, ok in [('import+mic-match', True), ('call-saved', False), ('', False)]:
        assert (voice.why_not_mine(meta | {'consent': {'how': how}}) == '') is ok, how


@pytest.mark.parametrize('read_prompt', [True, False])
def test_v1_upgraded_once(tmp_path, read_prompt):
    voice.save(*speech(), 'me', 'ru', voice.PROMPTS['ru'] if read_prompt else '', tmp_path)
    f = tmp_path / 'me' / 'voice.json'
    v1 = {k: v for k, v in json.loads(f.read_text(encoding='utf-8')).items()
          if k in ('name', 'lang', 'transcript', 'sample_rate', 'created', 'gain_db', 'quality')}
    f.write_text(json.dumps(v1))
    m = voice.load('me', tmp_path).meta
    assert json.loads(f.read_text(encoding='utf-8')) == m and m['version'] == 2 and m['id'] == 'me' and m['kind'] == 'me'
    assert all(m[k] == v1[k] for k in v1)  # an older app still reads it
    if read_prompt:  # recorded from the screen text: live, the words not checked then
        assert m['consent'] == {'how': 'mic-live', 'at': v1['created'], 'wer': None, 'similarity': None}
        assert voice.mine('me', tmp_path)
    else:  # an imported file: confirmed at the microphone first (M-20)
        assert m['consent'] is None
        with pytest.raises(voice.NotMine):
            voice.mine('me', tmp_path)


def test_contact_kept_without_recording(tmp_path):
    state = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = voice.save_them('Анна', 'en', 7.5, {'pocket-tts-en': (state, 'sha1')}, tmp_path)
    assert v.name.startswith('them-') and 'Анна' not in str(v.path)  # no name in a path (logs, crash reports)
    assert sorted(p.name for p in v.path.iterdir()) == ['pocket-tts-en.npy', 'voice.json']
    assert sys.platform == 'win32' or v.path.stat().st_mode & 0o777 == 0o700  # Windows: the user's profile folder
    w = voice.load(v.name, tmp_path)
    assert w.meta['kind'] == 'them' and w.meta['name'] == 'Анна' and w.meta['consent']['how'] == 'call-saved'
    np.testing.assert_array_equal(voice.engine_state(w, 'pocket-tts-en', 'sha1'), state)
    assert voice.engine_state(w, 'pocket-tts-en', 'sha2') is None and voice.engine_state(w, 'qwen3-tts', 'sha1') is None
    with pytest.raises(voice.NotMine, match='headphones') as e:
        voice.mine(v.name, tmp_path)
    assert 'Анна' not in str(e.value)


@pytest.mark.parametrize('meta, why', [({'kind': 'them', 'consent': {'how': 'call-saved'}}, 'headphones'),
                                       ({'kind': 'me', 'consent': None}, 'not confirmed')])
def test_me_side_speaks_only_my_confirmed_voice(monkeypatch, tmp_path, meta, why):
    """RESEARCH-clone 6.1-6.2: a contact's voice, or mine not confirmed at the microphone, never goes into the call:
    the clone is not even made, the stock voice speaks and the side says why."""
    from test_voice_pack import me_side
    side, s, made, _ = me_side(monkeypatch, tmp_path, key=False, meta=meta)
    side._local(s)
    assert made == [None] and len(side.errors) == 1, side.errors
    assert why in side.errors[0] and 'a stock voice speaks instead of your cloned voice' in side.errors[0]


def test_cloud_fallback_never_speaks_a_contact_as_me(monkeypatch, tmp_path):
    from peresmeshnik import config, languages, models
    from peresmeshnik.cloud import fallback
    from peresmeshnik.local import clone
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)  # the clone keeps up here
    monkeypatch.setattr(models, 'path', lambda i: tmp_path / i)  # every model downloaded
    monkeypatch.setattr(clone, 'voice_cmd', lambda: ['voice-runner'])  # the voice pack is installed
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path)
    s = config.Settings()
    s.me_to_them, s.my_lang, s.their_lang = 'cloud', 'ru', 'en'
    s.local.voice = voice.save_them('Анна', 'ru', 5.0, {}).name
    assert fallback.stock('me', s) and 'headphones' in fallback._clone_not_ready(s, 'en')  # the stock voice, and why
    s.local.voice = voice.save(*speech(), 'me', 'ru', 'text', consent={'how': 'mic-live', 'at': '', 'wer': 0.0}).name
    assert not fallback.stock('me', s) and fallback._clone_not_ready(s, 'en') == ''  # my confirmed voice: the clone


class Asr:
    def __init__(self, text):
        self.text, self.langs = text, []

    def recognize(self, pcm, lead=True, lang=None):
        self.langs.append(lang)
        return type('R', (), {'text': self.text})()


class Speaker:
    """The recording's vector, then the live reading's: cosine `sim`."""
    def __init__(self, sim):
        self.vecs = [np.array([1.0, 0.0]), np.array([sim, (1 - sim ** 2) ** 0.5])]

    def embed(self, pcm, sr):
        return self.vecs.pop(0)


@pytest.mark.parametrize('heard, sim, why', [
    (None, 0.8, ''), ('совсем другие слова тут', 0.8, 'not the ones on the screen'),
    (None, 0.1, 'not the voice of the recording'),
    ('', 0.1, 'not the ones on the screen (100% differ): read them as written; this is not')])
def test_confirm_live(tmp_path, heard, sim, why):
    said = voice.live_phrases('ru')
    assert len(said) == 2 and set(said) <= set(voice.LIVE_PHRASES['ru'])
    v = voice.save(*speech(), 'me.new', 'ru', 'text', tmp_path)
    v.meta['quality']['band_db'] = -70.0  # a phone-band recording: said with a refusal
    asr = Asr(' '.join(said) if heard is None else heard)
    consent, text = voice.confirm_live(v, *speech(6, 24_000), said, asr, Speaker(sim), 'import+mic-match')
    assert asr.langs == ['ru'] and why in text and (consent is None) == bool(why), text
    assert ('phone call' in text) == (sim < voice.MATCH) and 'background is loud' not in text  # no noise floor here
    if consent:
        assert consent['how'] == 'import+mic-match' and consent['wer'] == 0 and consent['similarity'] == pytest.approx(0.8)
        assert voice.mine(voice.set_consent('me.new', consent, tmp_path).name, tmp_path)
        with pytest.raises(ValueError):  # a contact's consent never confirms my voice
            voice.set_consent('me.new', consent | {'how': 'call-saved'}, tmp_path)


def test_live_phrases_every_language():
    assert set(voice.LIVE_PHRASES) == set(voice.PROMPTS) and all(len(p) == 8 for p in voice.LIVE_PHRASES.values())
    assert all(4 <= len(voice._words(x)) <= 26 for p in voice.LIVE_PHRASES.values() for x in p)


@pytest.mark.skipif(os.environ.get('PERESMESHNIK_WORK_GATE') != '1', reason='needs the models: PERESMESHNIK_WORK_GATE=1')
def test_speaker_tells_the_reader_from_other_people():
    """The real speaker model: the reader's last 6 s match her first 30 s; FLEURS people (another language, other voices)
    do not."""
    from pathlib import Path

    from peresmeshnik import models
    data = Path(__file__).parent / 'data'
    spk = voice.Speaker(models.path(voice.SPEAKER))
    x, sr = voice.read_audio(data / 'reader-ru.ogg')
    ref = spk.embed(x[:30 * sr], sr)
    assert ref.shape == (256,) and float(ref @ ref) == pytest.approx(1, abs=1e-5)
    same = float(ref @ spk.embed(x[-6 * sr:], sr))
    others = [float(ref @ spk.embed(y[:6 * r], r)) for y, r in (voice.read_audio(data / f) for f in ('call-en.ogg', 'ours-en.ogg'))]
    print(f'\nsame reader {same:.2f}, other people {others}')
    assert same >= voice.MATCH > max(others)



def test_crash_report_names_no_voice(app, tmp_path):  # noqa: F811 (the fixture from test_crash)
    """A contact's and my own voice's names never reach a report; the app's folder ids name nobody and stay: 'me' is not
    cut out of "them -> me"."""
    import threading

    from peresmeshnik import crash
    root = tmp_path / 'data' / 'voices'  # the fixture's data folder
    them = voice.save_them('Анна Смирнова', 'en', 6.0, {}, root)
    voice.save(*speech(), 'Petrov', 'ru', 'text', root)  # a v1-era profile the user named (--name)
    voice.save(*speech(), 'me', 'ru', 'text', root)
    assert voice.names(root) == {'Анна Смирнова', 'Petrov', 'Ivan Petrov'}  # and the fixture's settings' voice
    crash._known_private(None)  # as at a start with these profiles

    def worker():
        raise ValueError(f'voice Анна Смирнова ({them.name}) and Petrov at {them.path}: them -> me')
    t = threading.Thread(target=worker, name='speak')
    t.start()
    t.join()
    text = next(crash.reports_dir().glob('crash-*.txt')).read_text('utf-8')
    assert 'Анна' not in text and 'Смирнова' not in text and 'Petrov' not in text, text[-800:]
    assert f'({them.name})' in text and ': them -> me' in text


def test_a_refusal_in_noise_asks_for_a_quieter_place(tmp_path):
    """BENCH stage 8: the same person is refused 3x as often with a noisy laptop mic: the refusal says so."""
    v = voice.save(*speech(), 'me', 'ru', 'text', tmp_path)
    x, sr = speech(6, 24_000)
    noisy = x + np.random.default_rng(0).normal(0, 0.05, len(x)).astype(np.float32)  # ~10 dB below the speech
    said = voice.LIVE_PHRASES['ru'][:2]
    _, text = voice.confirm_live(v, noisy, sr, said, Asr(' '.join(said)), Speaker(0.3))
    assert 'the background is loud' in text and 'somewhere quiet' in text, text

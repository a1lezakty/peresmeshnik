"""voice.json v2: kinds, consent, the one "my confirmed voice" check, v1 profiles upgraded, a contact's voice kept,
the live check (every window of the recording, the recording it checked, a pause after failed tries)."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest
from test_crash import app  # noqa: F401  (the fixture: crash reports into tmp_path)

from peresmeshnik import voice


def speech(seconds=40, sr=16_000, f=180.0):
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * f * t) * (np.sin(2 * np.pi * 2 * t) > -0.3)).astype(np.float32), sr


def confirmed(name, root):
    """What a passed live check writes: the consent of this very recording."""
    sha = hashlib.sha256((root / name / 'reference.wav').read_bytes()).hexdigest()
    return voice.set_consent(name, {'how': 'mic-live', 'at': '', 'wer': 0.0, 'similarity': 0.9, 'windows': 5,
                                    'ref_sha256': sha}, root)


def test_mine_only_confirmed_me(tmp_path):
    root = tmp_path / 'v'
    v = voice.save(*speech(), 'me', 'ru', 'text', root)
    assert v.meta['version'] == 2 and v.meta['kind'] == 'me' and v.meta['consent'] is None and v.meta['source'] == 'file'
    with pytest.raises(voice.NotMine, match='not confirmed'):
        voice.mine('me', root)
    assert voice.mine(confirmed('me', root).name, root).meta['consent']['how'] == 'mic-live'
    v = voice.load('me', root)
    for how, ok in [('import+mic-match', True), ('call-saved', False), ('', False)]:
        v.meta['consent']['how'] = how
        assert (voice.why_not_mine(v) == '') is ok, how


@pytest.mark.parametrize('bad', ['[]', '"x"', 'null'])
def test_load_refuses_json_that_is_not_an_object(tmp_path, bad):
    """JSON, but not a profile: ValueError as for broken JSON (the callers catch it), not an AttributeError."""
    (tmp_path / 'v').mkdir()
    (tmp_path / 'v' / 'voice.json').write_text(bad)
    with pytest.raises(ValueError, match='not an object'):
        voice.load('v', tmp_path)
    with pytest.raises(ValueError):
        voice.mine('v', tmp_path)


@pytest.mark.parametrize('read_prompt', [True, False])
def test_v1_upgraded_once(tmp_path, monkeypatch, read_prompt):
    """FIXES-s3 2: 0.3.0 checked nothing (its Record put the screen text in `transcript` whatever was heard): every v1
    profile is unconfirmed until read at the microphone once. Upgraded on disk once; v1's keys stay."""
    root = tmp_path / 'v'
    voice.save(*speech(), 'me', 'ru', voice.PROMPTS['ru'] if read_prompt else '', root)
    f = root / 'me' / 'voice.json'
    v1 = {k: v for k, v in json.loads(f.read_text(encoding='utf-8')).items()
          if k in ('name', 'lang', 'transcript', 'sample_rate', 'created', 'gain_db', 'quality')}
    f.write_text(json.dumps(v1))
    m = voice.load('me', root).meta
    assert json.loads(f.read_text(encoding='utf-8')) == m and m['version'] == 2 and m['id'] == 'me' and m['kind'] == 'me'
    assert all(m[k] == v1[k] for k in v1) and m['consent'] is None  # an older app still reads it
    with pytest.raises(voice.NotMine, match='not confirmed'):
        voice.mine('me', root)
    monkeypatch.setattr(voice, '_write', lambda *a: pytest.fail('a v2 profile is not written again on load'))
    assert voice.load('me', root).meta == m


def test_contact_kept_without_recording(tmp_path):
    root = tmp_path / 'v'
    state = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = voice.save_them('Анна', 'en', 7.5, {'pocket-tts-en': (state, 'sha1')}, root)
    assert v.name.startswith('them-') and 'Анна' not in str(v.path)  # no name in a path (logs, crash reports)
    assert sorted(p.name for p in v.path.iterdir()) == ['pocket-tts-en.npy', 'voice.json'] and voice.damaged(root) == []
    assert sys.platform == 'win32' or v.path.stat().st_mode & 0o777 == 0o700  # Windows: the user's profile folder
    w = voice.load(v.name, root)
    assert w.meta['kind'] == 'them' and w.meta['name'] == 'Анна' and w.meta['consent']['how'] == 'call-saved'
    np.testing.assert_array_equal(voice.engine_state(w, 'pocket-tts-en', 'sha1'), state)
    assert voice.engine_state(w, 'pocket-tts-en', 'sha2') is None and voice.engine_state(w, 'qwen3-tts', 'sha1') is None
    with pytest.raises(voice.NotMine, match='headphones') as e:
        voice.mine(v.name, root)
    assert 'Анна' not in str(e.value)


def test_a_failed_save_leaves_no_folder_and_no_temporary_file(tmp_path):
    """FIXES-s3 6: a contact's voice is made aside and renamed whole; voice.json's temporary file goes on a failure."""
    root = tmp_path / 'v'
    with pytest.raises(ValueError):  # an object array: np.save without pickle refuses
        voice.save_them('Bob', 'en', 3.0, {'pocket-tts-en': (np.array([object()]), 'sha')}, root)
    assert list(root.iterdir()) == []
    (root / 'x').mkdir()
    with pytest.raises(TypeError):
        voice._write(root / 'x', {'not json': object()})
    assert list((root / 'x').iterdir()) == []


def test_damaged_folders(tmp_path):
    """FIXES-s3 6, 10: a broken or missing voice.json is no voice: not listed, not a name, offered for deletion;
    mine() says why (an error, not a voice)."""
    root = tmp_path / 'v'
    voice.save(*speech(), 'me', 'ru', 'text', root)
    voice.save(*speech(), 'me.new', 'ru', 'text', root)  # the wizard's take in progress: not damaged, not listed
    (root / 'Broken').mkdir()
    (root / 'Broken' / 'voice.json').write_text('{"name": "Ivan Petrov", ')
    (root / 'them-0a1b2c3d.tmp').mkdir()  # a contact's voice whose save stopped
    assert [v.name for v in voice.profiles(root)] == ['me']
    assert voice.damaged(root) == ['Broken', 'them-0a1b2c3d.tmp']
    assert voice.names(root) == {'Broken'}  # its folder named it; 'me', 'them-…' name nobody
    with pytest.raises(ValueError):
        voice.mine('Broken', root)
    with pytest.raises(OSError):
        voice.mine('missing', root)


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
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'v')
    s = config.Settings()
    s.me_to_them, s.my_lang, s.their_lang = 'cloud', 'ru', 'en'
    s.local.voice = voice.save_them('Анна', 'ru', 5.0, {}).name
    assert fallback.stock('me', s) and 'headphones' in fallback._clone_not_ready(s, 'en')  # the stock voice, and why
    voice.save(*speech(), 'me', 'ru', 'text')
    s.local.voice = confirmed('me', tmp_path / 'v').name
    assert not fallback.stock('me', s) and fallback._clone_not_ready(s, 'en') == ''  # my confirmed voice: the clone


class Asr:
    def __init__(self, text):
        self.text, self.langs = text, []

    def recognize(self, pcm, lead=True, lang=None):
        self.langs.append(lang)
        return type('R', (), {'text': self.text})()


class Speaker:
    """The live reading's vector, then every window of the recording's: cosine `sim`."""
    def __init__(self, sim):
        self.sim, self.calls = sim, 0

    def embed(self, pcm, sr):
        self.calls += 1
        return np.array([1.0, 0.0]) if self.calls == 1 else np.array([self.sim, (1 - self.sim ** 2) ** 0.5])


class Pitch:
    """A speaker vector from the strongest frequency: 180 Hz (me) and 300 Hz (someone else) are two people."""
    def embed(self, pcm, sr):
        f = np.argmax(np.abs(np.fft.rfft(np.asarray(pcm, np.float64)))) * sr / len(pcm)
        return np.array([1.0, 0.0]) if abs(f - 180) < 20 else np.array([0.0, 1.0])


@pytest.mark.parametrize('heard, sim, why', [
    (None, 0.8, ''), ('совсем другие слова тут', 0.8, 'not the ones on the screen'),
    (None, 0.1, 'not the voice of the recording'),
    ('', 0.1, 'not the ones on the screen (100% differ): read them as written; this is not')])
def test_confirm_live(tmp_path, heard, sim, why):
    root = tmp_path / 'v'
    said = voice.live_phrases('ru')
    assert len(said) == 2 and set(said) <= set(voice.LIVE_PHRASES['ru'])
    v = voice.save(*speech(), 'me.new', 'ru', 'text', root)
    v.meta['quality']['band_db'] = -70.0  # a phone-band recording: said with a refusal
    asr = Asr(' '.join(said) if heard is None else heard)
    consent, text = voice.confirm_live(v, *speech(6, 24_000), said, asr, Speaker(sim))
    assert asr.langs == ['ru'] and why in text and (consent is None) == bool(why), text
    assert ('phone call' in text) == (sim < voice.MATCH) and 'background is loud' not in text  # no noise floor here
    if consent:
        assert consent['how'] == 'import+mic-match' and consent['wer'] == 0 and consent['similarity'] == pytest.approx(0.8)
        assert consent['windows'] >= 3 and consent['ref_sha256'] == voice._sha(v.path / 'reference.wav')
        assert voice.mine(voice.set_consent('me.new', consent, root).name, root)
        with pytest.raises(ValueError):  # a contact's consent never confirms my voice
            voice.set_consent('me.new', consent | {'how': 'call-saved'}, root)


def test_a_recording_spliced_from_someone_else_and_me_is_refused(tmp_path):
    """FIXES-s3 1: 30 s of someone else, then 30 s of me, and my live reading: one vector over it all looks like me,
    every window does not; the clone would take the start (someone else). 60 s of me alone passes."""
    root = tmp_path / 'v'
    me, sr = speech(30, 24_000)
    v = voice.save(np.concatenate([speech(30, 24_000, f=300.0)[0], me]), sr, 'me.new', 'ru', 'text', root)
    said = voice.LIVE_PHRASES['ru'][:2]
    consent, text = voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Pitch())
    assert consent is None and 'not the voice of the recording (similarity 0.00)' in text, text
    v = voice.save(np.concatenate([me, me]), sr, 'me.new', 'ru', 'text', root)
    consent, _ = voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Pitch())
    assert consent and consent['similarity'] == 1.0 and consent['windows'] >= 4, consent


def test_consent_holds_only_for_the_recording_checked(tmp_path):
    """FIXES-s3 3: a new take or import while the check ran gets no consent; a recording replaced after its consent
    (a new save, or a crash between reference.wav and voice.json) is not mine."""
    import soundfile as sf
    root = tmp_path / 'v'
    v = voice.save(*speech(), 'me.new', 'ru', 'text', root)
    said = voice.LIVE_PHRASES['ru'][:2]
    consent, _ = voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Speaker(0.9))
    voice.save(*speech(45), 'me.new', 'ru', 'text', root)  # imported meanwhile
    with pytest.raises(ValueError, match='changed while it was checked'):
        voice.set_consent('me.new', consent, root)
    with pytest.raises(voice.NotMine):
        voice.mine('me.new', root)
    voice.mine(confirmed('me.new', root).name, root)
    sf.write(root / 'me.new' / 'reference.wav', speech(35)[0], 16_000)  # only the recording replaced
    with pytest.raises(voice.NotMine, match='changed after it was confirmed'):
        voice.mine('me.new', root)
    confirmed('me.new', root)
    voice.save(*speech(50), 'me.new', 'ru', 'text', root)  # CLI enroll --name over a confirmed voice
    with pytest.raises(voice.NotMine, match='not confirmed'):
        voice.mine('me.new', root)


def test_a_pause_after_failed_tries(tmp_path):
    """FIXES-s3 4: the threshold holds for one try; after TRIES failures in a row a pause, growing; not even heard
    meanwhile; a success starts the count again."""
    root = tmp_path / 'v'
    v = voice.save(*speech(), 'me', 'ru', 'text', root)
    said = voice.LIVE_PHRASES['ru'][:2]
    for _ in range(voice.TRIES):
        assert voice.wait_s(root) == 0
        assert voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Speaker(0.1))[0] is None
    assert voice.PAUSES_S[0] - 5 < voice.wait_s(root) <= voice.PAUSES_S[0]
    asr = Asr(' '.join(said))
    consent, text = voice.confirm_live(v, *speech(6, 24_000), said, asr, Speaker(0.9))
    assert consent is None and text.startswith('Too many tries') and asr.langs == []
    checks = tmp_path / 'voice-checks.json'
    checks.write_text(json.dumps(json.loads(checks.read_text()) | {'until': 0}))  # the pause is over
    voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Speaker(0.1))  # the 4th failure: a longer one
    assert voice.wait_s(root) > voice.PAUSES_S[0]
    checks.write_text(json.dumps({'fails': 4, 'until': 0}))
    assert voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Speaker(0.9))[0]
    assert json.loads(checks.read_text())['fails'] == 0


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
    from peresmeshnik import config
    for name in ('Sergey Ivanov', 'me'):  # FIXES-s3 5: the settings' voice, a v1 name without its folder; 'me' is no name
        s = config.Settings()
        s.local.voice = name
        crash._known_private(s)
    token = 'hf_' + 'Ab3' * 12

    def worker():
        raise ValueError(f'voice Анна Смирнова ({them.name}) and Petrov at {them.path}: them -> me; Sergey Ivanov {token}')
    t = threading.Thread(target=worker, name='speak')
    t.start()
    t.join()
    text = next(crash.reports_dir().glob('crash-*.txt')).read_text('utf-8')
    assert 'Анна' not in text and 'Смирнова' not in text and 'Petrov' not in text, text[-800:]
    assert 'Sergey' not in text and token not in text and '<key>' in text
    assert f'({them.name})' in text and ': them -> me' in text


def test_a_refusal_in_noise_asks_for_a_quieter_place(tmp_path):
    """BENCH stage 8: the same person is refused 3x as often with a noisy laptop mic: the refusal says so."""
    v = voice.save(*speech(), 'me', 'ru', 'text', tmp_path / 'v')
    x, sr = speech(6, 24_000)
    noisy = x + np.random.default_rng(0).normal(0, 0.05, len(x)).astype(np.float32)  # ~10 dB below the speech
    said = voice.LIVE_PHRASES['ru'][:2]
    _, text = voice.confirm_live(v, noisy, sr, said, Asr(' '.join(said)), Speaker(0.3))
    assert 'the background is loud' in text and 'somewhere quiet' in text, text


def test_the_live_reading_loses_only_the_silence_at_its_edges():
    """BENCH (FIXES-s3): the live reading keeps its pauses (cutting every quiet frame cut quiet speech in noise);
    the recording loses every quiet frame."""
    sr = 16_000
    tone = 0.2 * np.sin(2 * np.pi * 180 * np.arange(2 * sr) / sr).astype(np.float32)
    room = np.random.default_rng(0).normal(0, 1e-3, (5, sr // 2)).astype(np.float32)
    x = np.concatenate([room[0], room[1], room[2], tone, room[3], tone, room[4]])  # 1.5 s, 2 s, 0.5 s, 2 s, 0.5 s
    assert len(voice._speech(x, sr, edges=True)) / sr == pytest.approx(4.5, abs=0.1)
    assert len(voice._speech(x, sr)) / sr == pytest.approx(4.0, abs=0.1)

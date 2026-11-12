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
                                    'ref_sha256': sha, 'live': [1.0, 0.0]}, root)


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


def test_a_contacts_voice_of_a_voice_pack_engine_is_not_kept(tmp_path):
    """FIXES r2 2: the voice pack's engines (needs = "gpu": XTTS, Qwen3) keep no voice for the next call (COMMON 23:49):
    save_them refuses their states whoever calls it, and leaves nothing on disk."""
    root = tmp_path / 'v'
    for model in ('xtts-v2', 'qwen3-tts'):
        with pytest.raises(ValueError, match=model):
            voice.save_them('Bob', 'en', 9.0, {'pocket-tts-en': (np.ones(3, np.float32), 'a'),
                                               model: (np.ones(3, np.float32), 'b')}, root)
    assert not root.exists() or list(root.iterdir()) == []
    assert voice.save_them('Bob', 'en', 9.0, {'pocket-tts-en': (np.ones(3, np.float32), 'a')}, root).meta['engines']


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


class Blend:
    """A speaker vector from how much of a window is 180 Hz (me) and how much 300 Hz (someone else): a window of half and
    half scores 0.71 against my live reading."""
    def embed(self, pcm, sr):
        x = np.abs(np.fft.rfft(np.asarray(pcm, np.float64)))
        band = lambda f: x[int((f - 20) * len(pcm) / sr):int((f + 20) * len(pcm) / sr)].sum()  # noqa: E731
        v = np.array([band(180), band(300)])
        return v / np.linalg.norm(v)


def test_a_short_piece_of_someone_else_at_the_start_is_refused(tmp_path):
    """Stage 8 tails: ~4 s of the speech of someone else, then me: an 8 s window of half and half scored as me, and the
    clone takes its prompt from the start (XTTS 6 s, Pocket 10 s). The start's 4 s windows (HEAD_MATCH) refuse it; me alone
    passes."""
    root = tmp_path / 'v'
    sr, said = 24_000, voice.LIVE_PHRASES['ru'][:2]
    x = np.concatenate([speech(7, sr, f=300.0)[0], speech(40, sr)[0]])  # 7 s in syllables: ~4.2 s of their voice
    live = Blend().embed(speech(6, sr)[0], sr)
    assert Blend().embed(voice._start(x, sr)[:8 * sr], sr) @ live > voice.MATCH  # one 8 s window would let it through
    v = voice.save(x, sr, 'me.new', 'ru', 'text', root)
    consent, text = voice.confirm_live(v, *speech(6, sr), said, Asr(' '.join(said)), Blend())
    assert consent is None and 'not the voice of the recording' in text, text
    v = voice.save(speech(44, sr)[0], sr, 'me.new', 'ru', 'text', root)
    consent, _ = voice.confirm_live(v, *speech(6, sr), said, Asr(' '.join(said)), Blend())
    assert consent and consent['windows'] >= 5, consent  # 5 start windows of 4 s every 2 s, then 8 s ones


def noisy(seconds, sr, level=0.005):
    """Me in a room: steady noise in the pauses at -46 dBFS (about 30 dB under the voice)."""
    x, _ = speech(seconds, sr)
    return x + level * np.random.default_rng(7).standard_normal(len(x)).astype(np.float32)


def test_a_voice_turned_down_to_the_noise_of_the_pauses_is_refused(tmp_path):
    """FIXES-s3-tails r2 1: someone else first, denoised (digital silence in the pauses) and turned down 24 dB to just
    above the noise of my pauses, then me: the relative trim of _speech took them for background, every window was mine
    and the engines' prompt was theirs. The raw start with an absolute floor checks them: refused. Me in the same room,
    noise in the pauses and all, passes."""
    root = tmp_path / 'v'
    sr, said = 24_000, voice.LIVE_PHRASES['ru'][:2]
    t = np.arange(8 * sr) / sr
    them = (0.2 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 2 * t) > -0.3)).astype(np.float32) * 10 ** (-24 / 20)
    x = np.concatenate([them, noisy(40, sr)])
    assert Blend().embed(voice._speech(x, sr)[:12 * sr], sr)[1] < 0.05  # the old trim: none of them left
    v = voice.save(x, sr, 'me.new', 'ru', 'text', root)
    consent, text = voice.confirm_live(v, *speech(6, sr), said, Asr(' '.join(said)), Blend())
    assert consent is None and 'not the voice of the recording' in text, text
    v = voice.save(noisy(48, sr), sr, 'me.new', 'ru', 'text', root)
    consent, text = voice.confirm_live(v, *speech(6, sr), said, Asr(' '.join(said)), Blend())
    assert consent, text


def at_db(x, db):
    """`x` with its sound (not its digital silence) at `db` dBFS RMS."""
    return (x * 10 ** (db / 20) / np.sqrt(np.mean(x[x != 0] ** 2))).astype(np.float32)


def test_what_the_check_does_not_hear_the_engines_do_not_get(tmp_path):
    """FIXES-s3-tails r3 1: someone else's 10 s at -62 dBFS (pauses digital silence), then me: under the floor, the check
    did not hear them, yet the engines took the raw start (Qwen3's vector hardly minds the level). save() writes it as
    digital silence: the file's start is what was checked, and a start that silent is refused (the engines would speak
    from it: PROMPT_MIN_S). A file put in the folder by hand, the quiet voice in it, is checked whole: refused."""
    import soundfile as sf
    root = tmp_path / 'v'
    sr, said = 24_000, voice.LIVE_PHRASES['ru'][:2]
    x = np.concatenate([at_db(speech(10, sr, f=300.0)[0], -62), speech(40, sr)[0]])
    v = voice.save(x, sr, 'me.new', 'ru', 'text', root)
    y, _ = sf.read(v.path / 'reference.wav')
    assert not y[:int(9.97 * sr)].any() and y[10 * sr:].any()  # all but the frame where my voice begins
    consent, text = voice.confirm_live(v, *speech(6, sr), said, Asr(' '.join(said)), Blend())
    assert consent is None and 'are almost silent (0.0 s of sound): start reading right away' in text, text
    sf.write(v.path / 'reference.wav', x * 0.5, sr, subtype='FLOAT')  # by hand, not through save()
    consent, text = voice.confirm_live(v, *speech(6, sr), said, Asr(' '.join(said)), Blend())
    assert consent is None and 'not the voice of the recording' in text, text


def test_a_few_loud_seconds_of_mine_do_not_lift_the_floor(tmp_path):
    """FIXES-s3-tails r3 1: 4 s of me 20 dB louder lifted a floor taken from the loudest frames (p95 - 40 dB) and someone
    else 25 dB under my speech fell under it, out of the check but in the file. The floor is under the median speech
    frame: they stay in the file and in the check, refused."""
    import soundfile as sf
    root = tmp_path / 'v'
    sr, said = 24_000, voice.LIVE_PHRASES['ru'][:2]
    me = speech(40, sr)[0]
    level = 10 * np.log10(np.mean(me[me != 0] ** 2))
    x = np.concatenate([at_db(speech(4, sr)[0], level + 20), at_db(speech(12, sr, f=300.0)[0], level - 25), me])
    v = voice.save(x, sr, 'me.new', 'ru', 'text', root)
    assert sf.read(v.path / 'reference.wav')[0][5 * sr:15 * sr].any()  # in the file: so in the check
    consent, text = voice.confirm_live(v, *speech(6, sr), said, Asr(' '.join(said)), Blend())
    assert consent is None and 'not the voice of the recording' in text, text


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


# --- the start of the recording like its rest; the clone made from it like the live reading (BRIEF-s3-clone-output-check)

class Turned:
    """The live reading [1, 0]; a 4 s window of the recording's head at `head` degrees off it, an 8 s window of the rest
    at `rest` degrees the other way: each passes against the live reading, the head and the rest differ by both."""
    def __init__(self, head, rest):
        self.head, self.rest, self.calls = np.radians(head), -np.radians(rest), 0

    def embed(self, pcm, sr):
        self.calls += 1
        a = 0.0 if self.calls == 1 else self.head if len(pcm) <= 4.5 * sr else self.rest
        return np.array([np.cos(a), np.sin(a)])


@pytest.mark.parametrize('head, rest, ok', [(30, 0, True), (60, 60, False)])
def test_a_start_unlike_the_rest_of_the_recording_is_refused(tmp_path, head, rest, ok):
    """Someone else's 3 s cut into the start can pass the live reading's 4 s window (0.40); the start against the rest of
    the same recording (one session) tells it: refused, not counted as a failed try (another reading cannot help)."""
    root = tmp_path / 'v'
    v = voice.save(*speech(), 'me', 'ru', 'text', root)
    said = voice.live_phrases('ru')
    consent, why = voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Turned(head, rest))
    assert (consent is not None) == ok, why
    if not ok:
        assert 'the start of the recording does not sound like the rest of it (similarity -0.50)' in why
        assert voice.RECORD_AGAIN in why and voice.START_NOW in why and 'not the voice of the recording' not in why
        assert not (tmp_path / 'voice-checks.json').exists()  # no try: neither a failure nor a success
    else:
        assert consent['live'] == [1.0, 0.0] and consent['windows'] >= 6


class Tone:
    """A clone that says a tone of `f` Hz for `sec` s (Pitch: 180 Hz is me, anything else someone else)."""
    sample_rate = 24_000

    def __init__(self, f=180.0, sec=2.0):
        self.f, self.sec, self.said = f, sec, []

    def synth(self, text, lang, voice=None, speed=1.0):
        self.said.append((text, lang))
        t = np.arange(int(self.sec * self.sample_rate)) / self.sample_rate
        yield (0.2 * np.sin(2 * np.pi * self.f * t)).astype(np.float32)

    def close(self):
        pass


@pytest.mark.parametrize('f, sec, ok', [(180.0, 2.0, True), (300.0, 2.0, False), (180.0, 0.2, False)])
def test_the_clone_of_the_recording_is_heard_against_the_live_reading(tmp_path, f, sec, ok):
    """After the live check: the clone says the test phrases in their language; a voice not mine (or nothing said) is
    refused, a failed try (it samples: another try is another draw), and the reading's success does not reset the count
    before it (FIXES-s3-40 1): TRIES such refusals in a row, then the pause. Mine keeps its answer in the consent."""
    root = tmp_path / 'v'
    v = voice.save(*speech(), 'me', 'ru', 'text', root)
    said = voice.live_phrases('ru')
    for k in range(1 if ok else voice.TRIES):
        assert voice.wait_s(root) == 0, k
        tts = Tone(f, sec)
        consent, why = voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Pitch(),
                                          clone=lambda c, tts=tts: voice.confirm_clone(c, tts, 'pocket-tts-en', v, 'en', Pitch()))
        assert tts.said == [(p, 'en') for p in voice.TEST_PHRASES['en']] and (consent is not None) == ok, why
    fails = json.loads((tmp_path / 'voice-checks.json').read_text())['fails']
    if ok:
        assert consent['clones']['pocket-tts-en']['similarity'] == 1.0 and fails == 0
        assert voice.why_not_mine(voice.set_consent('me', consent, root)) == ''
    else:
        assert (voice.CLONE_WHY.format(sim=0.0) if sec > 1 else voice.SAID_NOTHING) in why
        assert fails == voice.TRIES and voice.wait_s(root) > 0


def test_a_clone_is_heard_once_per_engine_and_the_answer_kept(tmp_path, monkeypatch):
    """The me side makes an engine not heard at confirmation (installed later, my server's): vet_clone hears it once,
    keeps the answer on this recording, then only reads it; one unlike me is refused there and in _clone_not_ready."""
    from peresmeshnik import config
    from peresmeshnik.cloud import fallback
    root = tmp_path / 'v'
    monkeypatch.setattr(voice, 'voices_dir', lambda: root)
    monkeypatch.setattr(voice, '_speaker', Pitch)
    voice.save(*speech(), 'me', 'ru', 'text', root)
    v = confirmed('me', root)
    mine, other = Tone(), Tone(300.0)
    assert voice.vet_clone(mine, 'xtts-v2', v, 'en') == '' and len(mine.said) == 2
    v = voice.load('me', root)
    assert voice.vet_clone(other, 'xtts-v2', v, 'en') == '' and not other.said  # kept: not heard again
    assert 'does not sound like you' in voice.vet_clone(other, 'pocket-tts-en', v, 'en') and len(other.said) == 2
    v = voice.load('me', root)
    assert v.meta['consent']['clones']['pocket-tts-en']['similarity'] == 0.0
    assert 'does not sound like you' in voice.why_not_mine(v) and 'confirm it again' in voice.why_not_mine(v)
    s = config.Settings()
    s.local.voice = 'me'
    monkeypatch.setattr('peresmeshnik.languages.clone_why', lambda engine, lang='': '')
    assert 'does not sound like you' in fallback._clone_not_ready(s, 'en', 'pocket-tts-en')  # the stock voice instead
    assert fallback.my_voice(s)[0] is None and 'does not sound like you' in fallback.my_voice(s)[1]  # not to my server
    old = dict(v.meta['consent'])
    old.pop('live')
    voice.set_consent('me', old | {'clones': {}}, root)  # confirmed by a build before this check
    assert 'confirm it again' in voice.vet_clone(mine, 'qwen3-tts', voice.load('me', root), 'en')
    sf_ = __import__('soundfile')
    sf_.write(root / 'me' / 'reference.wav', speech(35)[0], 16_000)  # another take: nothing kept for it
    assert voice.why_not_mine(voice.load('me', root))


def test_an_engine_that_says_nothing_once_is_not_kept_as_a_refusal(tmp_path, monkeypatch):
    """FIXES-s3-40 3: a clone that says next to nothing at the side's start (an engine's hiccup) gives the stock voice for
    that session only: no similarity is kept, the next start hears it again."""
    root = tmp_path / 'v'
    monkeypatch.setattr(voice, '_speaker', Pitch)
    voice.save(*speech(), 'me', 'ru', 'text', root)
    v = confirmed('me', root)
    assert voice.vet_clone(Tone(sec=0.2), 'xtts-v2', v, 'en') == voice.SAID_NOTHING
    v = voice.load('me', root)
    assert not v.meta['consent'].get('clones') and voice.why_not_mine(v) == ''
    assert voice.vet_clone(Tone(), 'xtts-v2', v, 'en') == '' and voice.load('me', root).meta['consent']['clones']


def test_a_voice_confirmed_before_the_clone_check_is_confirmed_again(tmp_path, monkeypatch):
    """FIXES-s3-40 2: a consent without the live reading's vector (a build before this check) has nothing to hear the clone
    against: not mine until confirmed again, said in the list (Voices… then offers Confirm), not sent to my server."""
    from peresmeshnik import config
    from peresmeshnik.cloud import fallback
    from peresmeshnik.ui.voices import described
    root = tmp_path / 'v'
    monkeypatch.setattr(voice, 'voices_dir', lambda: root)
    voice.save(*speech(), 'me', 'ru', 'text', root)
    c = dict(confirmed('me', root).meta['consent'])
    c.pop('live')
    v = voice.set_consent('me', c, root)
    assert 'confirmed before the app checked the voice made from it: confirm it again' in voice.why_not_mine(v)
    assert 'confirm it again' in described(v)
    s = config.Settings()
    s.local.voice = 'me'
    assert fallback.my_voice(s)[0] is None and 'confirm it again' in fallback.my_voice(s)[1]


def test_my_server_hears_the_clone_it_makes(monkeypatch, tmp_path):
    """FIXES-s3-40 4: my server's side is app.Side's (remote.server.ServerSide): the clone it makes from the profile sent
    is heard against my live reading (vet_clone) before it speaks; unlike me: its stock voice, said."""
    from test_voice_pack import me_side

    from peresmeshnik.remote import server
    monkeypatch.setattr(voice, '_speaker', Pitch)
    side, s, made, _ = me_side(monkeypatch, tmp_path, Tone(300.0), key=False)
    side.__class__ = server.ServerSide
    side._local(s)
    assert side.stock and made[-1] is None and 'does not sound like you' in side.errors[0], side.errors


def test_the_me_side_falls_back_when_its_clone_does_not_sound_like_me(monkeypatch, tmp_path):
    """app.Side._local: the clone made, its test phrase not my voice: the stock voice speaks, the tray says why."""
    from test_voice_pack import me_side
    monkeypatch.setattr(voice, '_speaker', Pitch)
    side, s, made, _ = me_side(monkeypatch, tmp_path, Tone(300.0), key=False)
    d = side._local(s)
    assert made[-1] is None and d.warm == 1 and len(side.errors) == 1, side.errors
    assert 'a stock voice speaks instead of your cloned voice (the voice made from your recording does not sound like you' \
        in side.errors[0]


def test_the_room_before_the_first_word_is_cut(tmp_path):
    """FIXES-s3-40 9: Record pressed, then 3 s of the room before the first word: the 4 s head windows held mostly it
    (BENCH: the owner refused 63 / 78 % at SNR 30 / 20). save() cuts it, all but LEAD_S, when it is the room alone
    (steady); anything else before the first word (here a tone over the room, someone's quiet voice) stays and is checked
    as it is: a cut stopping inside it would leave the rest beside my speech."""
    sr, root = 24_000, tmp_path / 'v'
    room = 0.005 * np.random.default_rng(1).standard_normal(3 * sr).astype(np.float32)  # the noise of noisy()'s pauses
    alone = len(voice.read_audio(voice.save(noisy(40, sr), sr, 'a', 'ru', 'text', root).path / 'reference.wav')[0])
    y = voice.read_audio(voice.save(np.concatenate([room, noisy(40, sr)]), sr, 'b', 'ru', 'text', root).path
                         / 'reference.wav')[0]
    assert 0 <= len(y) - alone <= (voice.LEAD_S + 0.03) * sr, (len(y) - alone) / sr
    t = np.arange(sr) / sr
    room[sr:2 * sr] += (0.01 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # 6 dB over the room: not steady
    y = voice.read_audio(voice.save(np.concatenate([room, noisy(40, sr)]), sr, 'c', 'ru', 'text', root).path
                         / 'reference.wav')[0]
    assert len(y) >= alone + 3 * sr - 0.03 * sr, (len(y) - alone) / sr


def test_a_recording_with_little_speech_after_its_start_is_refused(tmp_path):
    """FIXES-s3-40 6: under ~16 s of it there is nothing after the start to compare it with (its windows were compared with
    themselves): refused, record a longer one; not a failed try."""
    root = tmp_path / 'v'
    v = voice.save(*speech(15), 'me', 'ru', 'text', root)
    head, rest = voice._windows(voice._start(voice.read_audio(v.path / 'reference.wav')[0], 16_000), 16_000, split=True)
    assert head and rest == []
    said = voice.live_phrases('ru')
    consent, why = voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Pitch())
    assert consent is None and 'too little speech after its first 12 s' in why, why
    assert not (tmp_path / 'voice-checks.json').exists()


def test_what_the_engines_take_of_the_start_is_heard_whole(tmp_path):
    """Someone else 30 dB under me, 3.6 s alone in the first 10 s (the rest silence): in the 4 s windows my speech after
    it outweighs them, yet Qwen3 and Chatterbox take their voice from just that sound. The engines' prompt (_prompt) is a
    window of its own: refused."""
    sr, root = 16_000, tmp_path / 'v'
    t = np.arange(int(1.8 * sr)) / sr
    other, gap = (0.2 * 10 ** (-30 / 20) * np.sin(2 * np.pi * 300 * t)).astype(np.float32), np.zeros(3 * sr, np.float32)
    x = np.concatenate([other, gap, other, gap, gap[:int(0.4 * sr)], speech(40, sr)[0]])  # the first 10 s: 3.6 s of them
    v = voice.save(x, sr, 'me', 'ru', 'text', root)
    y = voice.read_audio(v.path / 'reference.wav')[0]
    assert all((Pitch().embed(w, sr) == [1.0, 0.0]).all() for w, _ in voice._windows(voice._start(y, sr), sr))  # unseen
    assert (Pitch().embed(voice._prompt(y, sr), sr) == [0.0, 1.0]).all()  # what the engines hear of it
    said = voice.live_phrases('ru')
    consent, why = voice.confirm_live(v, *speech(6, 24_000), said, Asr(' '.join(said)), Pitch())
    assert consent is None and 'the start of the recording does not sound like the rest of it' in why


def test_the_cloud_fallback_does_not_speak_a_clone_unlike_me(tmp_path, monkeypatch):
    """Me -> them in the cloud, my clone its fallback (Pocket here): made at the first fallback, heard there (vet_clone);
    unlike my live voice: the language's stock voice speaks, said why, as on the local path (FIXES-s3-40 5); the answer
    is kept, so the profile is not mine until confirmed again."""
    from pathlib import Path

    from test_languages_ui import Direction

    from peresmeshnik import config, languages, models, pipeline
    from peresmeshnik.cloud import fallback, gemini
    from peresmeshnik.local import clone, tts
    root = tmp_path / 'v'
    monkeypatch.setattr(voice, 'voices_dir', lambda: root)
    monkeypatch.setattr(voice, '_speaker', Pitch)
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    monkeypatch.setattr(models, 'path', lambda i: Path('/models') / i)
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')
    monkeypatch.setattr(clone, 'voice_cmd', lambda: None)
    monkeypatch.setattr(clone, 'make_engine', lambda i, d, threads=2: Tone(300.0))
    stock = []
    monkeypatch.setattr(tts, 'PiperTts', lambda vid, threads=None: stock.append(vid) or Tone())
    monkeypatch.setattr(gemini, 'GeminiDirection', lambda src, tgt, *a, **k: Direction(src, tgt))
    monkeypatch.setattr(pipeline, 'make_local', lambda src, tgt, **k: Direction(src, tgt))
    voice.save(*speech(), 'me', 'ru', 'text', root)
    confirmed('me', root)
    s = config.Settings()
    s.me_to_them, s.my_lang, s.their_lang, s.local.voice = 'cloud', 'ru', 'en', 'me'
    assert fallback.local_voice('me', s) == ('pocket-tts-en', '')
    states = []
    fb = fallback.make('me', s, 'a-key', on_state=states.append)
    try:
        fb.muted.load()
        assert fb.muted.error is None and stock == [languages.pick(s.local.tts, 'en')], fb.muted.error
        assert any('a stock voice speaks instead of your cloned voice (the voice made from your recording does not sound '
                   'like you' in x for x in states), states
    finally:
        fb.close()
    assert 'does not sound like you' in fallback._clone_not_ready(s, 'en', 'pocket-tts-en') and fallback.stock('me', s)

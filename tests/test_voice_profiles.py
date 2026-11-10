"""voice.json v2: kinds, consent, the one "my confirmed voice" check, v1 profiles upgraded, a contact's voice kept."""
import json

import numpy as np
import pytest

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
    meta = json.loads((tmp_path / 'me' / 'voice.json').read_text())
    for how, ok in [('import+mic-match', True), ('call-saved', False), ('', False)]:
        assert (voice.why_not_mine(meta | {'consent': {'how': how}}) == '') is ok, how


@pytest.mark.parametrize('read_prompt', [True, False])
def test_v1_upgraded_once(tmp_path, read_prompt):
    voice.save(*speech(), 'me', 'ru', voice.PROMPTS['ru'] if read_prompt else '', tmp_path)
    f = tmp_path / 'me' / 'voice.json'
    v1 = {k: v for k, v in json.loads(f.read_text()).items()
          if k in ('name', 'lang', 'transcript', 'sample_rate', 'created', 'gain_db', 'quality')}
    f.write_text(json.dumps(v1))
    m = voice.load('me', tmp_path).meta
    assert json.loads(f.read_text()) == m and m['version'] == 2 and m['id'] == 'me' and m['kind'] == 'me'
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
    assert v.path.stat().st_mode & 0o777 == 0o700
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

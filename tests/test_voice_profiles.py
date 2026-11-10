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

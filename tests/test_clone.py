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
        voice.main(['enroll', '--record', '30', '--lang', 'de'])
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


def test_cap_pauses_streams_and_keeps_speech():
    rng = np.random.default_rng(0)
    talk, gap = 0.3 * rng.standard_normal(SR).astype(np.float32), np.zeros(2 * SR, np.float32)
    x = np.concatenate([talk, gap, talk, np.zeros(SR // 5, np.float32), talk])
    out = np.concatenate(list(cap_pauses(iter(np.array_split(x, 37)))))  # odd chunk edges
    assert len(out) == pytest.approx(3 * SR + 0.4 * SR + 0.2 * SR, abs=SR // 50)  # 2 s gap -> 0.4 s, 0.2 s kept
    assert np.sum(out ** 2) == pytest.approx(np.sum(x ** 2))  # no speech lost

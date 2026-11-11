"""The clone in the new languages with the real XTTS-v2 and its text tools, in the clone's own environment: the Windows
voice pack's Python, or on Linux a venv from voice-runner/requirements-linux.in (both in the voice-pack workflow).

    PERESMESHNIK_CLONE_PY=<that python> PERESMESHNIK_CLONE_MODEL=<xtts-v2 folder> pytest tests/test_clone_langs.py

Without them it skips; PERESMESHNIK_REQUIRE_CLONE=1 (CI) makes that a failure. The runners have no GPU: the clone runs on
their CPU, slower than real time, so only a few short phrases are said.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from peresmeshnik import config, languages, voice
from peresmeshnik.local import clone, xtts

PY, MODEL = os.environ.get('PERESMESHNIK_CLONE_PY'), os.environ.get('PERESMESHNIK_CLONE_MODEL')
DATA = Path(__file__).parent / 'data'
# money, a clock time, a decimal and a percentage in each language XTTS speaks (its own cleaning misreads some: BENCH)
NUMBERS = {
    'en': 'At 15:30 it costs 250.50 € (12.5% more).',
    'ru': 'В 15:30 это стоит 250,50 € (на 12,5% больше).',
    'de': 'Um 15:30 Uhr kostet es 250,50 € (12,5 % mehr).',
    'fr': 'À 15h30, cela coûte 250,50 € (12,5 % de plus).',
    'es': 'A las 15:30 cuesta 250,50 € (un 12,5 % más).',
    'it': "Alle 15:30 costa 250,50 € (il 12,5% in più).",
    'pt': 'Às 15h30 custa 250,50 € (12,5% a mais).',
    'pl': 'O 15:30 kosztuje 250,50 € (o 12,5% więcej).',
    'tr': "Saat 15:30'da 250,50 € tutuyor (%12,5 daha fazla).",
    'zh': '15:30的时候要250.50€（多了12.5%）。',
    'ja': '15:30に250.50€かかります（12.5%多い）。1,000人。',
    'ko': '15:30에 250.50€입니다 (12.5% 더 많음).',
    'ar': 'في الساعة 15:30 يكلف 250.50 € (أكثر بنسبة 12.5%).',
}
TOOLS = '''import importlib.util, json, re, sys
spec = importlib.util.spec_from_file_location('xtts', sys.argv[1])
xtts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xtts)
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
t = VoiceBpeTokenizer(sys.argv[2])
unk = t.tokenizer.token_to_id('[UNK]')
out = {}
for lang, text in json.loads(sys.argv[3]).items():
    said = xtts.normalize(text, lang)
    out[lang] = {'unk': t.encode(said, lang).count(unk), 'raw_unk': t.encode(text, lang).count(unk),
                 'tokens': t.preprocess_text(said, lang)}
print(json.dumps(out))
'''


@pytest.fixture(scope='module')
def env():
    if not (PY and MODEL):
        (pytest.fail if os.environ.get('PERESMESHNIK_REQUIRE_CLONE') == '1' else pytest.skip)(
            'no clone environment: PERESMESHNIK_CLONE_PY and PERESMESHNIK_CLONE_MODEL')
    return PY, Path(MODEL)


def test_text_tools_of_every_language(env):
    """Each language's text cleaning runs in the clone's environment (pypinyin for zh, cutlet + fugashi + unidic-lite
    for ja, ko-speech-tools for ko: an ImportError otherwise) and, after normalize(), leaves no [UNK] token and no digit
    for Japanese (its cleaning reads none). Languages XTTS cannot speak: languages.py says so (uk)."""
    py, model = env
    r = subprocess.run([py, '-c', TOOLS, str(Path(xtts.__file__)), str(model / 'vocab.json'), json.dumps(NUMBERS)],
                       capture_output=True, text=True, encoding='utf-8', timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    for lang, g in got.items():
        print(f'{lang}: {g["raw_unk"]} -> {g["unk"]} [UNK]: {g["tokens"]}')
    assert {lang for lang, g in got.items() if g['unk']} == set()
    assert not any(c.isdigit() for c in got['ja']['tokens']), got['ja']
    assert sum(g['raw_unk'] for g in got.values()) > 0  # what normalize() is for: the € sign in tr / ko as XTTS has it


def test_a_pair_without_ru_or_en_speaks_in_the_clone(env, tmp_path, monkeypatch):
    """German -> Japanese as the app builds it: languages.direction picks the clone for Japanese, make_clone starts the
    runner of this environment, a voice is conditioned from a recording and says the wizard's test phrase and a
    sentence with numbers in Japanese, and a phrase in German; each is speech of a plausible length."""
    py, model = env
    monkeypatch.setattr(languages, 'addon_here', lambda: False)  # XTTS itself: with the addon Qwen3-TTS takes Japanese
    s = config.Settings(my_lang='de', their_lang='ja', me_to_them='local')
    d = languages.direction(s, 'me_to_them', has_gpu=True)
    assert (d.voice, d.tts) == ('clone', 'xtts-v2'), d
    monkeypatch.setenv('PERESMESHNIK_VOICE_CMD', f'"{py}" "{clone.RUNNER}" --device cpu')
    eng = clone.make_clone(model, threads=os.cpu_count() or 2)
    try:
        assert isinstance(eng, clone.VoicePackClone) and eng.info['device'] == 'cpu', eng.info
        x, sr = voice.read_audio(DATA / 'call-ru.ogg')
        v = eng.enroll(x, sr, name='me', lang='de', root=tmp_path)
        for lang, text in (('ja', voice.TEST_PHRASES['ja'][0]), ('ja', NUMBERS['ja']), ('de', voice.TEST_PHRASES['de'][0])):
            pcm = np.concatenate(list(eng.synth(text, lang, v)))
            sec, rms = len(pcm) / eng.sample_rate, float(np.sqrt(np.mean(pcm ** 2)))
            limit = sum(xtts.budget_samples(p) for p in xtts.split_text(xtts.normalize(text, lang), 250)) / xtts.SR
            print(f'{lang}: {sec:.2f} s (limit {limit:.1f} s), RMS {rms:.3f}: {text}')
            assert 1.5 < sec < limit and rms > 0.01, (lang, text, sec, rms)
    finally:
        eng.close()


CHECKPOINT = '''import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location('xtts', sys.argv[1])
xtts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xtts)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models import xtts as X
p = sys.argv[2]
with X._legacy_safe_globals():  # a legacy checkpoint, as the registry's
    torch.save({'model': {'w': torch.arange(4.0)}, 'config': XttsConfig()}, p)
before = open(p, 'rb').read()
assert torch.equal(xtts._read_checkpoint(p)['w'], torch.arange(4.0)) and open(p, 'rb').read() == before, 'written'
X.Xtts._load_checkpoint(None, p)
print('coqui writes it' if open(p, 'rb').read() != before else 'coqui leaves it')
'''


def test_loading_the_clone_does_not_write_its_checkpoint(env, tmp_path):
    """M-19: the registry's xtts-v2 model.pth keeps its config as an XttsConfig object, and coqui-tts's own loader writes
    the file again in place on the first load: a write cut short (the runner killed, the power gone) leaves a broken
    file that `.verified` still vouches for. The app's engine (xtts.Engine) reads it and writes nothing."""
    py, _ = env
    r = subprocess.run([py, '-c', CHECKPOINT, str(Path(xtts.__file__)), str(tmp_path / 'model.pth')],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and 'coqui writes it' in r.stdout, r.stderr[-2000:]  # the reason for _read_checkpoint

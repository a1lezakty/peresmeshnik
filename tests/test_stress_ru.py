"""local/stress_ru.py: silero-stress's n-gram accentor on numpy. The logic on a tiny synthetic model; the real data
file (packaging/stress/export.py) only where it is, via PERESMESHNIK_STRESS_FILE or stress_ru.FILE."""
import os
from pathlib import Path

import numpy as np
import pytest

from peresmeshnik.local import stress_ru as sr

VOCAB = ['UNK', 'а', 'о', '<д', 'да', 'а>', '<да>']


def model(stress: int, yo: int, sure: bool = True, exceptions: str = '') -> sr.Accentor:
    """A synthetic accentor: random embeddings and hidden layers; the last layers' biases pick `stress` / `yo`
    (with a probability near 1, or 0.1-ish when not `sure`)."""
    rng = np.random.default_rng(0)
    z = {'emb': rng.integers(-128, 128, (len(VOCAB), 16), dtype=np.int8), 'scale': np.float32(0.5),
         'zero_point': np.float32(-3), 'ngrams': np.frombuffer('\n'.join(VOCAB).encode(), np.uint8),
         'exceptions': np.frombuffer(exceptions.encode(), np.uint8)}
    for h, n, pick in (('stress', 10, stress), ('yo', 7, yo)):
        for i, (o, k) in enumerate([(64, 16), (128, 64), (64, 128), (n, 64)]):
            z[f'{h}{i}.w'] = rng.normal(size=(o, k)).astype(np.float32) * (i < 3)
            z[f'{h}{i}.b'] = np.zeros(o, np.float32)
        z[f'{h}3.b'][pick] = 20 if sure else 0.1
    return sr.Accentor(z)


def test_ngrams_mean_and_mlp():
    assert sr.ngrams('да') == ['<', 'д', 'а', '>', '<д', 'да', 'а>', '<да', 'да>', '<да>']
    acc = model(stress=1, yo=0)
    q = np.random.default_rng(0).integers(-128, 128, (len(VOCAB), 16), dtype=np.int8)  # model()'s first draw
    emb = 0.5 * (q.astype(np.float32) + 3)  # restored as silero does: scale * (q - zero_point)
    np.testing.assert_array_equal(acc.emb, emb)
    ids = [VOCAB.index(g) for g in ('а', '<д', 'да', 'а>', '<да>')]  # '<', 'д', '>', '<да', 'да>': not in VOCAB
    np.testing.assert_allclose(acc.embed(['да'])[0], emb[ids].mean(axis=0), rtol=1e-6)
    np.testing.assert_array_equal(acc.embed(['щщ'])[0], emb[0])  # nothing known: 'UNK'
    x = acc.embed(['да'])
    h = x
    for i, (w, b) in enumerate(acc.heads['stress']):
        h = h @ w.T + b
        h = np.maximum(h, 0) if i < 3 else h
    np.testing.assert_allclose(sr.mlp(x, acc.heads['stress']), h, rtol=1e-6)
    assert acc.predict(['да', 'молоко']) == [(1, True, 0, True)] * 2


def test_stress_position_yo_and_rules():
    acc = model(stress=1, yo=0)  # the second vowel
    assert acc('Молоко, да!') == f'Моло{sr.MARK}ко, да{sr.MARK}!'  # a single vowel is always stressed
    assert acc('что-то') == f'что{sr.MARK}-то'  # '-то' is left alone
    assert acc('ёлка Ёж') == f'ё{sr.MARK}лка Ё{sr.MARK}ж'  # the user's 'ё' is stressed, no model
    assert acc(f'мо{sr.MARK}локо') == f'мо{sr.MARK}локо'  # already stressed: as is
    assert acc('берег') == f'бере{sr.MARK}г'
    assert model(stress=1, yo=2)('Берег') == f'Берё{sr.MARK}г'  # 'ё' only where the stress is
    assert model(stress=0, yo=2)('берег') == f'бе{sr.MARK}рег'
    assert model(stress=1, yo=0, sure=False)('молоко да') == f'молоко да{sr.MARK}'  # unsure: no stress
    assert model(stress=7, yo=0)('молоко') == 'молоко'  # past the last vowel: none
    assert acc('…  - 42 brr') == '…  - 42 brr'


def test_exceptions():
    acc = model(stress=0, yo=0, exceptions='замок 3 -1\nелка 0 0\n')
    assert acc('Замок, «замок» и Елка') == f'Замо{sr.MARK}к, «замо{sr.MARK}к» и{sr.MARK} Ё{sr.MARK}лка'


FILE = Path(os.environ.get('PERESMESHNIK_STRESS_FILE') or sr.FILE)


@pytest.mark.skipif(not FILE.exists(), reason=f'no stress data file {FILE} (export: packaging/stress/export.py; '
                    'PERESMESHNIK_STRESS_FILE=path)')
def test_real_data_like_silero():
    a = sr.MARK
    # silero-stress 1.5 without its homograph BERT ('"большим': its exception index lands on the quote, fixed here)
    assert sr.stress('Замок на двери, «привет» мир! Ёлка и елка, что-то.', FILE) == (
        f'Замо{a}к на{a} две{a}ри, «приве{a}т» ми{a}р! Ё{a}лка и{a} ё{a}лка, что{a}-то.')
    assert sr.stress('"большим', FILE) == f'"бо{a}льшим'

"""Export silero-stress 1.5's Russian n-gram accentor (MIT, https://github.com/snakers4/silero-stress) to the numpy data
file of peresmeshnik/local/stress_ru.py. Its homograph BERT is not exported.

Run by us, not by users, where `silero_stress` (torch) is installed, from the repo root:

    python packaging/stress/export.py --out stress_ru.npz [--check FLEURS_TSV ...]

The file (np.savez_compressed): `emb` the n-gram embeddings as silero ships them (int8 levels, `scale`, `zero_point`;
the app restores them as silero does), `ngrams` the vocabulary in index order and `exceptions` ('word stress yo' lines)
as UTF-8 text, `stress{i}.w/b` and `yo{i}.w/b` the two MLP heads (float32).
--check compares stress_ru.stress with silero (without and with its homograph BERT) on the sentences of FLEURS .tsv
files (column 3), on their unique words one by one and on every exception word; prints agreement, mismatches, speed.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from silero_stress import load_accentor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from peresmeshnik.local import stress_ru  # noqa: E402


def export(acc, out: Path) -> None:
    m = acc.model
    assert (acc.vowels, acc.re_cond, acc.stress_token) == (stress_ru.VOWELS, stress_ru._NOT_RU.pattern, '+')
    w = m.embedding.weight.detach().numpy()
    scale, zp = np.float32(m.scale.item()), np.float32(m.zero_point.item())
    q = np.rint(w / scale + zp)
    assert q.min() >= -128 and q.max() <= 127, (q.min(), q.max())
    q = q.astype(np.int8)
    assert np.array_equal(scale * (q.astype(np.float32) - zp), w), 'embeddings are not int8 levels'
    vocab = sorted(m.embedding.ngram_dict.items(), key=lambda kv: kv[1])
    assert [i for _, i in vocab] == list(range(len(w))) and not any(re.search(r'\s', g) for g, _ in vocab)
    exc = ''.join(f'{k} {s} {y}\n' for k, (s, y) in acc.exceptions.items())
    arrays = {'emb': q, 'scale': scale, 'zero_point': zp,
              'ngrams': np.frombuffer('\n'.join(g for g, _ in vocab).encode(), np.uint8),
              'exceptions': np.frombuffer(exc.encode(), np.uint8)}
    for h in ('stress', 'yo'):
        linears = [x for x in getattr(m, h + '_clf').children() if x.original_name == 'Linear']
        acts = [x.original_name for x in getattr(m, h + '_clf').children() if x.original_name != 'Linear']
        assert len(linears) == stress_ru.LAYERS and acts == ['ReLU'] * (stress_ru.LAYERS - 1), acts
        for i, lin in enumerate(linears):
            arrays[f'{h}{i}.w'] = lin.weight.detach().numpy()
            arrays[f'{h}{i}.b'] = lin.bias.detach().numpy()
    np.savez_compressed(out, **arrays)
    print(f'{out}: {out.stat().st_size} bytes, {len(vocab)} n-grams, {len(acc.exceptions)} exceptions')


def acute(text: str) -> str:
    """silero's '+' before the vowel as the app's U+0301 after it."""
    return re.sub(r'\+(.)', '\\1' + stress_ru.MARK, text)


def compare(name: str, texts: list[str], ours, theirs) -> None:
    same_t = same_w = words = 0
    bad = []
    for t in texts:
        a, b = ours(t), acute(theirs(t))
        same_t += a == b
        wa, wb = a.split(), b.split()
        words += len(wb)
        same_w += sum(x == y for x, y in zip(wa, wb)) if len(wa) == len(wb) else 0
        bad += [(x, y) for x, y in zip(wa, wb) if x != y]
    print(f'{name}: {same_t}/{len(texts)} texts, {same_w}/{words} words ({100 * same_w / max(words, 1):.2f} %) same')
    for x, y in bad[:40]:
        print(f'   ours {x!r}  silero {y!r}')


def check(silero, out: Path, tsvs: list[Path]) -> None:
    acc = stress_ru.load(out)
    sentences = list(dict.fromkeys(line.split('\t')[2] for p in tsvs for line in p.read_text().splitlines()))
    words = sorted({w for s in sentences for w in re.findall(r'[А-Яа-яЁё-]+', s)})
    exc = sorted(silero.accentor.exceptions)
    print(f'{len(sentences)} sentences, {len(words)} unique words, {len(exc)} exception words')
    for name, texts in (('sentences', sentences), ('words', words), ('exception words', exc)):
        compare(f'{name} vs silero without BERT', texts, acc, silero.accentor)
        compare(f'{name} vs silero with BERT', texts, acc, silero)
    n = sum(len(s.split()) for s in sentences)
    for f, label in ((acc, 'numpy'), (silero.accentor, 'silero n-gram, torch'), (silero, 'silero with BERT, torch')):
        t = time.perf_counter()
        for s in sentences:
            f(s)
        print(f'{label}: {1000 * (time.perf_counter() - t) / n:.3f} ms/word ({n} words)')
    t = time.perf_counter()
    stress_ru.load.cache_clear()
    stress_ru.load(out)
    print(f'load: {1000 * (time.perf_counter() - t):.0f} ms')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=Path('stress_ru.npz'))
    ap.add_argument('--check', type=Path, nargs='*', default=[], help='FLEURS .tsv files to compare on')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    silero = load_accentor('ru')
    export(silero.accentor, a.out)
    if a.check:
        check(silero, a.out, a.check)


if __name__ == '__main__':
    main()

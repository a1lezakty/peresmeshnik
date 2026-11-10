"""Russian stress marks for the Russian Pocket TTS model (Shtorm-PocketTTS-RU reads U+0301 after the stressed vowel):
a numpy port of silero-stress 1.5's n-gram accentor (MIT, https://github.com/snakers4/silero-stress, `AccentorNgram`),
its stress and 'ё' rules unchanged. Not ported: its homograph BERT ('замок' gets the n-gram model's guess) and the
options (stress_single_vowel, skip lists, words_to_ignore: silero's defaults are fixed here).

The model: a word's character n-grams (`<word>`, lengths 1..len+2) looked up in a vocabulary, their 16-dim embeddings
averaged (fastText-like), then two MLPs (16-64-128-64, ReLU): the stressed vowel's ordinal (10 classes) and which 'е'
is 'ё' (7 classes, 0: none). A prediction counts if its softmax probability is over 0.5. Words in the exception list
take stress and 'ё' from it. The data file (packaging/stress/export.py): `PERESMESHNIK_STRESS_FILE` or FILE."""
from __future__ import annotations

import functools
import os
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np

FILE = Path(__file__).with_name('stress_ru.npz')
MARK = '́'  # combining acute, after the stressed vowel
VOWELS = 'аоуыэиеяёю'
_SPLIT = re.compile(r'([\s.,!?;:<>=()/\\]+)')
_NOT_RU = re.compile(r'[^А-Яа-яёЁ]')
LAYERS = 4  # Linear layers per head, ReLU between


def ngrams(word: str) -> list[str]:
    """silero's word_ngrams(word, 1, len(word) + 3): every substring of '<word>'; '' also gets itself."""
    w = f'<{word}>'
    grams = [w[i:i + n] for n in range(1, len(w) + 1) for i in range(len(w) - n + 1)]
    return grams if word else grams + ['']


def mlp(x: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """torch Linear (x @ W.T + b) layers with ReLU between."""
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def best(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """argmax per row and whether its softmax probability is over 0.5."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return p.argmax(axis=1), p.max(axis=1) > 0.5


def _mark(word: str, positions: list[int]) -> str:
    for p in sorted(positions, reverse=True):
        word = word[:p + 1] + MARK + word[p + 1:]
    return word


def _yo(word: str, pos: int) -> str:
    return word[:pos] + ('ё' if word[pos].islower() else 'Ё') + word[pos + 1:]


def tokenize(text: str) -> list[tuple[str, str, bool]]:
    """silero's _tokenize: (raw token, Cyrillic-only lowercase, whether to accentuate); 'что-то' -> 'что-', 'то'
    (the '-то' tail is left alone); separators are tokens too, kept as they are."""
    out = []
    for word in _SPLIT.split(text):
        parts = word.split('-')
        tokens = [p + '-' for p in parts[:-1]] + [parts[-1]]
        tail = len(parts) == 1 or parts[-1] != 'то'
        for i, t in enumerate(tokens):
            clean = _NOT_RU.sub('', t.lower())
            out.append((t, clean, bool(clean) and (i < len(tokens) - 1 or tail)))
    return out


class Accentor:
    """The accentor over the exported arrays (export.py: `emb` int8 with `scale`/`zero_point`, `ngrams` and
    `exceptions` as UTF-8 text, `stress{i}.w/b`, `yo{i}.w/b`)."""

    def __init__(self, z: Mapping[str, np.ndarray]):
        self.emb = np.float32(z['scale']) * (z['emb'].astype(np.float32) - np.float32(z['zero_point']))
        self.ids = {g: i for i, g in enumerate(bytes(z['ngrams']).decode().split('\n'))}
        self.exceptions = {}  # word -> (stress char index, 'ё' char index or -1)
        for line in bytes(z['exceptions']).decode().splitlines():
            w, s, y = line.split()
            self.exceptions[w] = (int(s), int(y))
        self.heads = {h: [(np.asarray(z[f'{h}{i}.w'], np.float32), np.asarray(z[f'{h}{i}.b'], np.float32))
                          for i in range(LAYERS)] for h in ('stress', 'yo')}

    def embed(self, words: list[str]) -> np.ndarray:
        """Mean of each word's known n-gram embeddings ('UNK' if none)."""
        out = np.empty((len(words), self.emb.shape[1]), np.float32)
        for k, w in enumerate(words):
            ids = [self.ids[g] for g in ngrams(w) if g in self.ids] or [self.ids['UNK']]
            out[k] = self.emb[ids].mean(axis=0)
        return out

    def predict(self, words: list[str]) -> list[tuple[int, bool, int, bool]]:
        """Per word: stressed vowel ordinal, its confidence over 0.5, 'ё' ordinal among the 'е' letters (1-based, 0 =
        none), its confidence."""
        if not words:
            return []
        x = self.embed(words)
        s, s_ok = best(mlp(x, self.heads['stress']))
        y, y_ok = best(mlp(x, self.heads['yo']))
        return list(zip(s.tolist(), s_ok.tolist(), y.tolist(), y_ok.tolist()))

    def _exception(self, raw: str, clean: str) -> str:
        stress, yo = self.exceptions[clean]
        # the list indexes the clean word; silero applies it to the raw one ('«привет' would mark 'в'): mapped back
        kept = [i for i, c in enumerate(raw) if not _NOT_RU.match(c)]
        if yo != -1:
            raw = _yo(raw, kept[yo])
        return _mark(raw, [kept[stress]])

    def __call__(self, text: str) -> str:
        tokens = tokenize(text)
        model = [i for i, (raw, clean, todo) in enumerate(tokens)
                 if todo and clean not in self.exceptions and not self._as_is(raw) and 'ё' not in raw.lower()]
        preds = dict(zip(model, self.predict([tokens[i][1] for i in model])))
        out = []
        for i, (raw, clean, todo) in enumerate(tokens):
            low = raw.lower()
            if not todo or self._as_is(raw):
                out.append(raw)
            elif 'ё' in low:  # the user's 'ё' is stressed
                out.append(_mark(raw, [k for k, c in enumerate(low) if c == 'ё']))
            elif clean in self.exceptions:
                out.append(self._exception(raw, clean))
            else:
                out.append(self._model(raw, low, *preds[i]))
        return ''.join(out)

    @staticmethod
    def _as_is(raw: str) -> bool:
        # ponytail: a word the user already stressed stays as it is (silero would still fit its 'ё' to that stress)
        return MARK in raw or '+' in raw

    @staticmethod
    def _model(raw: str, low: str, s: int, s_ok: bool, y: int, y_ok: bool) -> str:
        vowels = [k for k, c in enumerate(low) if c in VOWELS]
        if not vowels:
            return raw
        ye = [k for k, c in enumerate(low) if c == 'е']
        stressed = [vowels[s]] if s < len(vowels) else []
        if y_ok and 0 < y <= len(ye) and ye[y - 1] in stressed:
            raw = _yo(raw, ye[y - 1])
        if len(vowels) == 1:
            return _mark(raw, vowels)
        return _mark(raw, stressed) if s_ok else raw


@functools.cache
def load(path: Path) -> Accentor:
    with np.load(path) as z:
        return Accentor(z)


def stress(text: str, data: Path | None = None) -> str:
    """Russian text with U+0301 after each stressed vowel and the 'ё' letters restored ('все елки' -> 'все́ ёлки')."""
    return load(Path(data or os.environ.get('PERESMESHNIK_STRESS_FILE') or FILE))(text)

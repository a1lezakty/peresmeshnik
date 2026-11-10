"""Text as the Pocket TTS models read it (local/pocket.py). The Russian model (Shtorm-PocketTTS-RU) knows no digits and
no Latin letters (its SentencePiece vocabulary): numbers become words, Latin words Cyrillic. Other languages: as is."""
from __future__ import annotations

import re

_UNITS = 'ноль один два три четыре пять шесть семь восемь девять'.split()
_TEENS = ('десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать '
          'девятнадцать').split()
_TENS = ['', ''] + 'двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто'.split()
_HUNDREDS = [''] + 'сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот'.split()
_SCALES = [(('тысяча', 'тысячи', 'тысяч'), True), (('миллион', 'миллиона', 'миллионов'), False),
           (('миллиард', 'миллиарда', 'миллиардов'), False), (('триллион', 'триллиона', 'триллионов'), False)]
_FRACTIONS = [('десятая', 'десятых'), ('сотая', 'сотых'), ('тысячная', 'тысячных')]


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """'1 тысяча', '2 тысячи', '5 тысяч'."""
    return forms[2] if n % 100 in range(11, 15) else forms[0] if n % 10 == 1 else forms[1] if n % 10 in (2, 3, 4) else forms[2]


def _triad(n: int, female: bool) -> list[str]:
    words = [_HUNDREDS[n // 100]]
    n %= 100
    if 10 <= n < 20:
        return [w for w in words + [_TEENS[n - 10]] if w]
    words.append(_TENS[n // 10])
    if n % 10:
        words.append({1: 'одна', 2: 'две'}.get(n % 10, _UNITS[n % 10]) if female else _UNITS[n % 10])
    return [w for w in words if w]


def number(n: int, female: bool = False) -> str:
    """A cardinal in words, nominative: 2026 -> 'две тысячи двадцать шесть'."""
    if n < 0:
        return 'минус ' + number(-n, female)
    if n == 0:
        return _UNITS[0]
    words, scale = [], 0
    while n:
        n, part = divmod(n, 1000)
        if part:
            if scale:
                forms, fem = _SCALES[scale - 1]
                words[:0] = _triad(part, fem) + [plural(part, forms)]
            else:
                words[:0] = _triad(part, female)
        scale += 1
        if scale > len(_SCALES) and n:  # beyond trillions: digit by digit
            return ' '.join(_UNITS[int(d)] for d in str(n)) + ' ' + ' '.join(words)
    return ' '.join(words)


def _decimal(whole: str, frac: str) -> str:
    """'3,5' -> 'три целых пять десятых'; past three decimals: digit by digit."""
    w = int(whole)
    if len(frac) > 3:
        return f'{number(w)} запятая ' + ' '.join(_UNITS[int(d)] for d in frac)
    f = int(frac)
    return (f'{number(w, True)} {plural(w, ("целая", "целых", "целых"))} {number(f, True)} '
            f'{_FRACTIONS[len(frac) - 1][0 if f % 10 == 1 and f % 100 != 11 else 1]}')


# Latin in a Russian sentence: names and brands, read the English way (approximately); short capitals as letters
_LETTERS = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', ('эй би си ди и эф джи эйч ай джей кей эл эм эн оу пи кью ар эс ти ю ви '
                                                    'дабл-ю экс уай зед').split()))
_PAIRS = {'sch': 'ш', 'sh': 'ш', 'ch': 'ч', 'zh': 'ж', 'th': 'т', 'ph': 'ф', 'kh': 'х', 'ts': 'ц', 'oo': 'у', 'ee': 'и',
          'ea': 'и', 'ou': 'ау', 'ck': 'к', 'qu': 'кв', 'wh': 'у', 'ya': 'я', 'yo': 'йо', 'yu': 'ю', 'ai': 'ей', 'ay': 'ей',
          'ey': 'ей', 'oy': 'ой', 'oi': 'ой'}
_ONE = dict(zip('abcdefghijklmnopqrstuvwxyz', 'а б к д е ф г х и дж к л м н о п к р с т у в у кс и з'.split()))


def latin(word: str) -> str:
    """'Microsoft' -> 'микрософт', 'NASA' -> 'эн эй эс эй' (up to 4 capitals: letter by letter)."""
    # ponytail: rules, not a dictionary; an English name reads close enough, other languages' names less so
    if word.isupper() and len(word) <= 4:
        return ' '.join(_LETTERS[c] for c in word)
    w, out, i = word.lower(), [], 0
    while i < len(w):
        for n in (3, 2):
            if w[i:i + n] in _PAIRS:
                out.append(_PAIRS[w[i:i + n]])
                i += n
                break
        else:
            c = w[i]
            out.append('с' if c == 'c' and w[i + 1:i + 2] in ('e', 'i', 'y') else _ONE.get(c, c))
            i += 1
    return ''.join(out)


_NUM = re.compile(r'(?<![\d.,])(-|−)?(\d{1,3}(?:\s\d{3})+|\d+)(?:[.,](\d+))?(?:-[а-яё]{1,3}\b)?(\s*%)?')


def _say_number(m: re.Match) -> str:
    sign, whole, frac, pct = m[1], re.sub(r'\D', '', m[2]), m[3], m[4]
    # ponytail: an ordinal ('5-й', '1-го') is read as the cardinal; cases and genders of ordinals are not built
    words = _decimal(whole, frac) if frac else number(int(whole))
    if pct:
        words += ' ' + ('процента' if frac else plural(int(whole), ('процент', 'процента', 'процентов')))
    return ('минус ' if sign else '') + words


def russian(text: str) -> str:
    """Digits and Latin letters as words the Russian Pocket model reads; clock times '9:05' -> 'девять ноль пять'."""
    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', lambda m: f'{number(int(m[1]))} '
                  f'{"ноль " if m[2][0] == "0" and m[2] != "00" else ""}{number(int(m[2])) if m[2] != "00" else "ноль ноль"}',
                  text)
    text = re.sub(r'(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])', ' ', text)  # 'MP3', '5кг': apart
    text = _NUM.sub(_say_number, text)
    return re.sub(r'[A-Za-z]+', lambda m: latin(m[0]), text)


def prepare(text: str, lang: str) -> str:
    return russian(text) if lang == 'ru' else text

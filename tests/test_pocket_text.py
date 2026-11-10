"""local/pocket_text.py: the Russian Pocket model reads no digits and no Latin letters."""
import pytest

from peresmeshnik.local import pocket_text as pt


@pytest.mark.parametrize('n, words', [(0, 'ноль'), (1, 'один'), (12, 'двенадцать'), (21, 'двадцать один'),
                                      (101, 'сто один'), (1000, 'одна тысяча'), (2026, 'две тысячи двадцать шесть'),
                                      (11000, 'одиннадцать тысяч'), (21000, 'двадцать одна тысяча'),
                                      (1_000_000, 'один миллион'), (2_500_000, 'два миллиона пятьсот тысяч'),
                                      (-5, 'минус пять')])
def test_numbers_in_words(n, words):
    assert pt.number(n) == words


def test_russian_text_has_no_digits_or_latin_left():
    said = pt.prepare('В 2026 году Microsoft и NASA подняли цену на 3,5% до 1 000 000 рублей в 9:05, MP3 и 5-й раз.', 'ru')
    assert said == ('В две тысячи двадцать шесть году микрософт и эн эй эс эй подняли цену на три целых пять десятых '
                    'процента до один миллион рублей в девять ноль пять, эм пи три и пять раз.')
    assert pt.prepare('Hello 2026', 'en') == 'Hello 2026'  # other languages: as they are
    assert pt.prepare('21 процент, 1,25 и 0,1', 'ru') == ('двадцать один процент, одна целая двадцать пять сотых и ноль '
                                                         'целых одна десятая')

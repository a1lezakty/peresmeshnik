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
    said = pt.russian('В 2026 году Microsoft и NASA подняли цену на 3,5% до 1 000 000 рублей в 9:05, MP3 и 5-й раз.')
    assert said == ('В две тысячи двадцать шесть году микрософт и эн эй эс эй подняли цену на три целых пять десятых '
                    'процента до один миллион рублей в девять ноль пять, эм пи три и пять раз.')
    assert pt.prepare('Hello 2026', 'en') == 'Hello 2026'  # other languages: as they are
    assert pt.russian('21 процент, 1,25 и 0,1') == ('двадцать один процент, одна целая двадцать пять сотых и ноль '
                                                  'целых одна десятая')


def test_russian_for_the_model_has_stress_marks():
    """prepare(): the words, then a stress mark (U+0301) after each stressed vowel, as Shtorm-PocketTTS-RU was trained."""
    said = pt.prepare('Привет, в 2026 году.', 'ru')
    assert said.replace('\u0301', '') == pt.russian('Привет, в 2026 году.') and said.count('\u0301') >= 4


@pytest.mark.parametrize('text, said', [
    ('10.11.2026', 'десять, одиннадцать, две тысячи двадцать шесть'),
    ('1,2,3', 'один, два, три'), ('версия 1.2.3.', 'версия один, два, три.'),
    ('+7 (999) 123-45-67', 'плюс семь, девятьсот девяносто девять, сто двадцать три, сорок пять, шестьдесят семь'),
    ('8-800-555-35-35', 'восемь, восемьсот, пятьсот пятьдесят пять, тридцать пять, тридцать пять'),
    ('звони 123-05-00', 'звони сто двадцать три, ноль пять, ноль ноль'),
    ('3,5 и 2.75', 'три целых пять десятых и две целых семьдесят пять сотых'),
    ('1 000 000', 'один миллион'), ('1:2:3', 'один:два:три')])
def test_groups_phones_and_no_digit_left(text, said):
    """Dates, versions and phones group by group, not as a decimal fraction; whatever is left, digit by digit."""
    assert pt.russian(text) == said
    assert not any(c.isdigit() for c in pt.prepare(text, 'ru'))

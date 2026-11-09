"""languages.py: names, what a pair runs on locally and why not, the cloud's language codes, the models a pair needs."""
import pytest

from peresmeshnik import config, languages, models


def fake_registry(monkeypatch, *extra: models.Model) -> None:
    """The real registry plus stand-in models (no files): pairs whose real models are not in the registry yet."""
    reg = {**models.registry(), **{m.id: m for m in extra}}
    monkeypatch.setattr(models, 'registry', lambda: reg)


def stub(mid: str, role: str, *langs: str) -> models.Model:
    return models.Model(mid, role, langs, 'MIT', 'https://example.invalid', 'https://example.invalid/', {})


def test_fourteen_languages_with_names_and_cloud_codes():
    assert languages.LANGS == ('en', 'ru', 'uk', 'de', 'fr', 'es', 'it', 'pt', 'pl', 'tr', 'zh', 'ja', 'ko', 'ar')
    assert languages.label('de') == 'Deutsch (German)' and languages.label('en') == 'English'
    assert languages.label('ar') == 'العربية (Arabic)' and languages.RTL == {'ar'}
    assert languages.CLOUD == set(languages.LANGS)  # both Gemini Live models list all 14 (links in the module)
    assert languages.CLOUD_TRANSLATE['zh'] == 'zh-Hans' and languages.CLOUD_TRANSCRIBE['zh'] == 'cmn-Hans-CN'
    assert set(languages.WRITING) == set(languages.LANGS)


def test_ru_en_is_local_both_ways():
    s = config.Settings()
    them, me = (languages.direction(s, side, has_gpu=True) for side in languages.SIDES)
    assert (them.src, them.tgt, me.src, me.tgt) == ('en', 'ru', 'ru', 'en')
    assert them.local and me.local and them.subtitles
    assert them.models == (s.local.asr['en'], s.local.mt['en-ru'], s.local.tts['ru'])
    assert (me.tts, me.voice, me.notes) == (s.local.clone, 'clone', ())
    assert languages.direction(s, 'me_to_them', has_gpu=False).notes  # the clone falls behind without an NVIDIA GPU
    s.local.clone = 'piper'
    me = languages.direction(s, 'me_to_them', has_gpu=False)
    assert (me.tts, me.voice, me.notes) == (s.local.tts['en'], 'stock', ())


def test_a_pair_without_local_models_says_why_and_goes_to_the_cloud():
    s = config.Settings(my_lang='de', their_lang='ja')
    s.local.asr, s.local.mt, s.local.tts = {}, {}, {}  # as before stage 6: nothing for German or Japanese
    them = languages.direction(s, 'them_to_me')
    assert not them.local and them.cloud and not them.subtitles
    assert them.why == ('no local speech recognition for Japanese', 'no local translation from Japanese to German',
                        'no local voice for German')
    s.them_to_me = s.me_to_them = 'cloud'
    assert languages.cloud_subtitles(s, 'them_to_me')  # no local ASR + MT: the cloud makes the subtitles too
    assert languages.needed_models(s) == ['silero-vad']


def test_a_pair_on_stand_in_models_needs_only_its_own(monkeypatch):
    fake_registry(monkeypatch, stub('asr-de', 'asr', 'de'), stub('asr-ja', 'asr', 'ja'), stub('mt-many', 'mt', 'de', 'ja'),
                  stub('tts-de', 'tts', 'de'), stub('tts-ja', 'tts', 'ja'), stub('clone-ja', 'clone', 'ja'))
    s = config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local')
    s.local.asr.update(de='asr-de', ja='asr-ja')
    s.local.mt.update({'de-ja': 'mt-many', 'ja-de': 'mt-many'})  # a multilingual model: both languages in its langs
    s.local.tts.update(de='tts-de', ja='tts-ja')
    s.local.clone = 'clone-ja'
    assert all(languages.direction(s, side, has_gpu=True).local for side in languages.SIDES)
    assert languages.needed_models(s) == ['silero-vad', 'asr-ja', 'mt-many', 'tts-de', 'asr-de', 'clone-ja']
    s.local.clone = 'xtts-v2'  # a clone without Japanese: the stock voice, said so
    me = languages.direction(s, 'me_to_them', has_gpu=True)
    assert me.local and me.voice == 'stock' and me.tts == 'tts-ja' and 'cannot speak Japanese' in me.notes[0]


@pytest.mark.parametrize('mid, key, ok', [('asr-en', 'en', True), ('asr-en', 'de', False), ('mt-en-ru', 'en-ru', True),
                                          ('mt-en-ru', 'ru-en', False), ('mt-many', 'ja-de', True), ('mt-many', 'ja-en', False),
                                          ('vad', 'ko', True), ('no-such-model', 'en', False)])
def test_covers_reads_the_registry_langs(monkeypatch, mid, key, ok):
    fake_registry(monkeypatch, stub('asr-en', 'asr', 'en'), stub('mt-en-ru', 'mt', 'en-ru'), stub('mt-many', 'mt', 'de', 'ja'),
                  stub('vad', 'vad', '*'))
    assert languages.covers(mid, key) is ok


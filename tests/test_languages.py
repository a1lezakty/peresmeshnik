"""languages.py: names, what a pair runs on locally and why not, the cloud's language codes, the models a pair needs."""
import pytest

from peresmeshnik import config, languages, models


def fake_registry(monkeypatch, *extra: models.Model) -> None:
    """The real registry plus stand-in models (no files): pairs whose real models are not in the registry yet."""
    reg = {**models.registry(), **{m.id: m for m in extra}}
    monkeypatch.setattr(models, 'registry', lambda: reg)


def stub(mid: str, role: str, *langs: str, needs: str = '') -> models.Model:
    return models.Model(mid, role, langs, 'MIT', 'https://example.invalid', 'https://example.invalid/', {}, needs=needs)


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
    assert languages.needed_models(s) == ['silero-vad', 'asr-ja', 'mt-many', 'tts-de', 'asr-de', 'clone-ja',
                                          'tts-ja']  # the stock voice: the clone's fallback (S2)
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


def test_readme_table_is_the_generated_one():
    """README's "what runs locally, what in the cloud" is languages.table(): a registry change shows up there."""
    from pathlib import Path
    readme = (Path(__file__).parents[1] / 'README.md').read_text('utf-8')
    block = readme.split('<!-- languages:begin', 1)[1].split('\n', 1)[1].split('<!-- languages:end -->')[0].strip()
    assert block == languages.table(), 'README is stale: python -m peresmeshnik.languages, into README.md'


def test_a_model_that_needs_a_gpu_is_not_offered_without_one(monkeypatch):
    """`needs = "gpu"` (models.toml): without an NVIDIA GPU the ASR / MT / stock voice make the side cloud-only, said why;
    the clone gives way to the stock voice, said; with a GPU all of it is local. The README table marks them."""
    import tomllib
    row = tomllib.loads('role = "asr"\nlangs = ["ja"]\nlicense = "MIT"\nlicense_url = "u"\nurl = "u"\nneeds = "gpu"\n[files]\n')
    assert models.Model(id='asr-ja', langs=tuple(row.pop('langs')), **row).needs == 'gpu'  # as registry() reads a row
    fake_registry(monkeypatch, stub('asr-ja', 'asr', 'ja', needs='gpu'), stub('asr-de', 'asr', 'de'),
                  stub('mt-many', 'mt', 'de', 'ja'), stub('tts-de', 'tts', 'de'), stub('tts-ja', 'tts', 'ja'),
                  stub('clone-big', 'clone', 'de', 'ja', needs='gpu'))
    s = config.Settings(my_lang='de', their_lang='ja')
    s.local.asr, s.local.mt, s.local.tts = {'ja': 'asr-ja', 'de': 'asr-de'}, {'*': 'mt-many'}, {'de': 'tts-de', 'ja': 'tts-ja'}
    s.local.clone = 'clone-big'
    them = languages.direction(s, 'them_to_me', has_gpu=False)
    assert not them.local and them.why == ('speech recognition for Japanese needs an NVIDIA GPU (asr-ja)',)
    assert not them.subtitles and languages.cloud_subtitles(s, 'them_to_me')  # no local ASR here: the cloud's subtitles
    assert languages.direction(s, 'them_to_me', has_gpu=True).local
    me = languages.direction(s, 'me_to_them', has_gpu=False)
    assert me.local and (me.tts, me.voice) == ('tts-ja', 'stock')
    assert me.notes == ('your cloned voice needs an NVIDIA GPU: a stock voice speaks instead',)
    assert languages.direction(s, 'me_to_them', has_gpu=True).voice == 'clone'
    s.local.tts = {'de': 'tts-de'}  # and no stock voice for Japanese: no local voice, and why
    assert languages.direction(s, 'me_to_them', has_gpu=False).why == ('your cloned voice needs an NVIDIA GPU',)
    assert '| 日本語 (Japanese) | asr-ja (NVIDIA GPU) |' in languages.table(s)
    fake_registry(monkeypatch, stub('asr-de', 'asr', 'de'), stub('mt-many', 'mt', 'de', 'ja'), stub('tts-ja', 'tts', 'ja', needs='gpu'))
    s.local.clone, s.local.tts = 'piper', {'ja': 'tts-ja'}  # a stock voice that needs one: said so, not "no voice"
    assert languages.direction(s, 'me_to_them', has_gpu=False).why == ('the voice for Japanese needs an NVIDIA GPU (tts-ja)',)
    assert languages.direction(s, 'me_to_them', has_gpu=True).tts == 'tts-ja'


def test_a_gpu_model_is_not_offered_where_its_engine_has_no_cuda(monkeypatch):
    """An NVIDIA GPU, but onnxruntime without CUDA (the built .exe, a from-source install): Whisper turbo for Turkish
    is not offered (on the CPU 4.6 s per 1 s window, BENCH stage 6), said why; built anyway, OnnxAsr refuses it."""
    from peresmeshnik.local import asr, device
    monkeypatch.setattr(languages.shutil, 'which', lambda name: '/usr/bin/nvidia-smi')
    monkeypatch.setattr(device, 'onnx_cuda', lambda: False)
    s = config.Settings(my_lang='ru', their_lang='tr')
    them = languages.direction(s, 'them_to_me')
    assert not them.local and len(them.why) == 1
    assert them.why[0].endswith('needs CUDA, and onnxruntime in this app runs on the CPU only (whisper-large-v3-turbo)')
    monkeypatch.setattr(device, 'onnx_cuda', lambda: True)
    assert languages.direction(s, 'them_to_me').asr == 'whisper-large-v3-turbo'
    monkeypatch.setattr(device, 'onnx_providers', lambda: ['CPUExecutionProvider'])
    with pytest.raises(RuntimeError, match='only on CUDA'):
        asr.OnnxAsr('whisper-large-v3-turbo')


def test_star_is_one_model_for_every_pair_it_has(monkeypatch):
    """`'*'` in a [local] table: the model for every language or pair its registry langs have; an own key wins."""
    fake_registry(monkeypatch, stub('mt-many', 'mt', 'de', 'ja', 'en'), stub('mt-de-ja', 'mt', 'de-ja'))
    s = config.Settings(my_lang='de', their_lang='ja')
    s.local.mt = {'*': 'mt-many', 'de-ja': 'mt-de-ja'}
    assert languages.direction(s, 'them_to_me').mt == 'mt-many' and languages.direction(s, 'me_to_them').mt == 'mt-de-ja'
    s.their_lang = 'ko'  # not among its languages
    assert languages.direction(s, 'them_to_me').mt == '' and 'no local translation from Korean to German' in \
        languages.direction(s, 'them_to_me').why
    assert 'de → ja, any other pair of en, de, ja' in languages.table(s)


def test_one_download_of_a_model_at_a_time(server, tmp_path, monkeypatch):
    """The wizard and the tray (or two pair switches) fetching one model together: one download, both get it."""
    import hashlib
    import threading
    httpd, url = server
    httpd.slow = 0.3
    blob = b'x' * 100_000
    (httpd.root / 'a.bin').write_bytes(blob)
    m = models.Model('m', 'mt', ('de-ja',), 'MIT', 'u', url, {'a.bin': {'sha256': hashlib.sha256(blob).hexdigest(), 'size': len(blob)}})
    monkeypatch.setattr(models, 'registry', lambda: {'m': m})
    got = []
    ts = [threading.Thread(target=lambda: got.append(models.fetch('m', tmp_path / 'models'))) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert len(got) == 2 and models.is_fetched(m, tmp_path / 'models')
    assert [p for p, _ in httpd.ranges] == ['/a.bin']  # the second waited and found it there


def test_the_cloud_gets_the_mapped_codes():
    """GeminiDirection's connections: the translator's target and the transcriber's language in Gemini's own codes."""
    from types import SimpleNamespace

    from peresmeshnik.cloud.gemini import _Link
    for src in languages.LANGS:
        d = SimpleNamespace(src=src, tgt=src, cfg=config.Cloud())
        assert _Link(d, 'translate').config().translation_config.target_language_code == languages.CLOUD_TRANSLATE[src]
        assert _Link(d, 'transcribe').config().input_audio_transcription.language_codes == [languages.CLOUD_TRANSCRIBE[src]]
    d = SimpleNamespace(src='ar', tgt='pt', cfg=config.Cloud())
    assert _Link(d, 'translate').config().translation_config.target_language_code == 'pt-BR'
    assert _Link(d, 'transcribe').config().input_audio_transcription.language_codes == ['ar-EG']
    assert (languages.CLOUD_TRANSLATE['zh'], languages.CLOUD_TRANSCRIBE['zh'], languages.CLOUD_TRANSCRIBE['es']) == \
        ('zh-Hans', 'cmn-Hans-CN', 'es-419')

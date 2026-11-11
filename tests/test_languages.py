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


def test_ru_en_is_local_both_ways(monkeypatch):
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')  # Kyutai's gated weights: the user gave a token
    monkeypatch.setattr(languages, 'pack_here', lambda: True)  # and the voice pack (or torch) for an NVIDIA engine
    s = config.Settings()
    them, me = (languages.direction(s, side, has_gpu=True) for side in languages.SIDES)
    assert (them.src, them.tgt, me.src, me.tgt) == ('en', 'ru', 'ru', 'en')
    assert them.local and me.local and them.subtitles
    assert them.models == (s.local.asr['en'], s.local.mt['en-ru'], s.local.tts['ru'])
    assert (me.tts, me.voice, me.notes) == ('xtts-v2', 'clone', ())  # an NVIDIA GPU: its clone engine
    me = languages.direction(s, 'me_to_them', has_gpu=False)  # none: Pocket TTS on the CPU (stage 8)
    assert (me.tts, me.voice, me.notes) == ('pocket-tts-en', 'clone', ())
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


def no_clones(monkeypatch, *extra: models.Model) -> None:
    """fake_registry without the real clone engines: `extra` the only ones."""
    reg = {**{k: m for k, m in models.registry().items() if m.role != 'clone'}, **{m.id: m for m in extra}}
    monkeypatch.setattr(models, 'registry', lambda: reg)


def test_a_pair_on_stand_in_models_needs_only_its_own(monkeypatch):
    stand_ins = (stub('asr-de', 'asr', 'de'), stub('asr-ja', 'asr', 'ja'), stub('mt-many', 'mt', 'de', 'ja'),
                 stub('tts-de', 'tts', 'de'), stub('tts-ja', 'tts', 'ja'))
    no_clones(monkeypatch, *stand_ins, stub('clone-ja', 'clone', 'ja'), stub('clone-de', 'clone', 'de'))
    s = config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local')
    s.local.asr.update(de='asr-de', ja='asr-ja')
    s.local.mt.update({'de-ja': 'mt-many', 'ja-de': 'mt-many'})  # a multilingual model: both languages in its langs
    s.local.tts.update(de='tts-de', ja='tts-ja')
    assert all(languages.direction(s, side, has_gpu=True).local for side in languages.SIDES)
    assert languages.needed_models(s) == ['silero-vad', 'asr-ja', 'mt-many', 'tts-de', 'asr-de', 'clone-ja',
                                          'tts-ja']  # the stock voice: the clone's fallback (S2)
    no_clones(monkeypatch, *stand_ins, stub('clone-de', 'clone', 'de'))  # no clone engine has Japanese: the stock voice
    me = languages.direction(s, 'me_to_them', has_gpu=True)
    assert me.local and me.voice == 'stock' and me.tts == 'tts-ja' and 'cannot speak Japanese' in me.notes[0]


@pytest.mark.parametrize('mid, key, ok', [('asr-en', 'en', True), ('asr-en', 'de', False), ('mt-en-ru', 'en-ru', True),
                                          ('mt-en-ru', 'ru-en', False), ('mt-many', 'ja-de', True), ('mt-many', 'ja-en', False),
                                          ('vad', 'ko', True), ('no-such-model', 'en', False)])
def test_covers_reads_the_registry_langs(monkeypatch, mid, key, ok):
    fake_registry(monkeypatch, stub('asr-en', 'asr', 'en'), stub('mt-en-ru', 'mt', 'en-ru'), stub('mt-many', 'mt', 'de', 'ja'),
                  stub('vad', 'vad', '*'))
    assert languages.covers(mid, key) is ok



def test_the_gpu_is_asked_once_unless_its_driver_did_not_answer(monkeypatch):
    """FIXES-b r2 1: nvidia-smi once per process, the wizard's tier and direction() alike (a hung driver costs its
    timeouts once, not per call); a driver that did not answer is asked again later ("update it, then open this page
    again"), not at once."""
    from types import SimpleNamespace

    from peresmeshnik.ui import wizard
    answers, now = [languages.Gpu(why='timed out'), languages.Gpu(8.0, (572, 83), 8.6)], [100.0]
    calls = []
    monkeypatch.setattr(languages, '_PROBED', {})
    monkeypatch.setattr(languages.shutil, 'which', lambda name: '/usr/bin/nvidia-smi')
    monkeypatch.setattr(languages, 'nvidia', lambda: calls.append(1) or answers[min(len(calls), 2) - 1])
    monkeypatch.setattr(languages, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    assert 'NVIDIA driver does not answer' in languages.clone_why('xtts-v2') and wizard.tier()[1] == 'cloud'
    assert len(calls) == 1
    now[0] += languages.FAILED_TTL_S + 1
    assert languages.clone_why('xtts-v2') == '' and wizard.tier()[1] == 'local' and len(calls) == 2
    now[0] += 3600
    assert languages.nvidia_once().vram == 8.0 and len(calls) == 2  # a card that answered: asked once


def test_readme_pair_sizes_are_the_registrys(monkeypatch):
    """H-08, FIXES-b 8: README's download per pair (no NVIDIA GPU, them -> me local, a side that cannot run here in the
    cloud) is the registry's: the smallest and largest, and the quartiles as "typically"."""
    from pathlib import Path
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    xs = []
    for a in languages.LANGS:
        for b in set(languages.LANGS) - {a}:
            s = config.Settings(my_lang=a, their_lang=b, them_to_me='local', me_to_them='cloud')
            for k in languages.SIDES:
                if getattr(s, k) == 'local' and not languages.direction(s, k).local:
                    setattr(s, k, 'cloud')
            xs.append(sum(models.size(models.get(i)) for i in languages.needed_models(s)) / 1e9)
    xs = sorted(x for x in xs if x > 0.01)
    q = lambda f: xs[int(f * (len(xs) - 1))]  # noqa: E731
    want = f'{xs[0]:.1f}–{xs[-1]:.1f} GB for a pair, typically {q(.25):.1f}–{q(.75):.1f} GB'
    assert (Path(__file__).parents[1] / 'README.md').read_text('utf-8').count(want) == 2, want

def test_readme_table_is_the_generated_one():
    """README's "what runs locally, what in the cloud" is languages.table(): a registry change shows up there."""
    from pathlib import Path
    readme = (Path(__file__).parents[1] / 'README.md').read_text('utf-8')
    block = readme.split('<!-- languages:begin', 1)[1].split('\n', 1)[1].split('<!-- languages:end -->')[0].strip()
    assert block == languages.table(), 'README is stale: python -m peresmeshnik.languages, into README.md'
    # H-07: the builds' onnxruntime has no CUDA: speech recognition that needs it is the cloud's there, said so
    assert '| Türkçe (Turkish) | — (cloud; whisper-large-v3-turbo from source with onnxruntime-gpu on an NVIDIA GPU) |' in block


def test_a_model_that_needs_a_gpu_is_not_offered_without_one(monkeypatch):
    """`needs = "gpu"` (models.toml): without an NVIDIA GPU the ASR / MT / stock voice make the side cloud-only, said why;
    the clone gives way to the stock voice, said; with a GPU all of it is local. The README table marks them."""
    import tomllib
    row = tomllib.loads('role = "asr"\nlangs = ["ja"]\nlicense = "MIT"\nlicense_url = "u"\nurl = "u"\nneeds = "gpu"\n[files]\n')
    assert models.Model(id='asr-ja', langs=tuple(row.pop('langs')), **row).needs == 'gpu'  # as registry() reads a row
    no_clones(monkeypatch, stub('asr-ja', 'asr', 'ja', needs='gpu'), stub('asr-de', 'asr', 'de'),
              stub('mt-many', 'mt', 'de', 'ja'), stub('tts-de', 'tts', 'de'), stub('tts-ja', 'tts', 'ja'),
              stub('clone-big', 'clone', 'de', 'ja', needs='gpu'))
    s = config.Settings(my_lang='de', their_lang='ja')
    s.local.asr, s.local.mt, s.local.tts = {'ja': 'asr-ja', 'de': 'asr-de'}, {'*': 'mt-many'}, {'de': 'tts-de', 'ja': 'tts-ja'}
    them = languages.direction(s, 'them_to_me', has_gpu=False)
    assert not them.local and them.why == ('speech recognition for Japanese needs an NVIDIA GPU (asr-ja)',)
    assert not them.subtitles and languages.cloud_subtitles(s, 'them_to_me')  # no local ASR here: the cloud's subtitles
    assert languages.direction(s, 'them_to_me', has_gpu=True).local
    me = languages.direction(s, 'me_to_them', has_gpu=False)
    assert me.local and (me.tts, me.voice) == ('tts-ja', 'stock')
    assert me.notes == (f'{languages.clone_why("clone-big", has_gpu=False)}: a stock voice speaks instead',)
    assert languages.direction(s, 'me_to_them', has_gpu=True).voice == 'clone'
    s.local.tts = {'de': 'tts-de'}  # and no stock voice for Japanese: no local voice, and why
    assert languages.direction(s, 'me_to_them', has_gpu=False).why == (
        f'{languages.clone_why("clone-big", has_gpu=False)}; no local voice for Japanese',)
    assert '| 日本語 (Japanese) | — (cloud; asr-ja from source with onnxruntime-gpu on an NVIDIA GPU) |' in languages.table(s)
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

    def loaded(*a, **k):
        raise LookupError('past the CUDA check')
    monkeypatch.setattr(models, 'path', loaded)
    with pytest.raises(LookupError):  # an offline check (the wizard's clone check): slow on the CPU is fine
        asr.OnnxAsr('whisper-large-v3-turbo', live=False)


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


def test_the_clone_is_offered_only_where_it_can_speak(monkeypatch):
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')
    """M-23: `clone_why` is the one answer to "can the clone speak here": XTTS-v2 keeps up only on an NVIDIA GPU (its
    registry row says `needs = "gpu"`), so without one me -> them local speaks in the stock voice where no CPU engine has
    the language (Polish), nothing of the clone is downloaded and the why is said; a language it has no stock voice for is
    then not local at all (M-24), said why. Stage 8: English without a GPU — Pocket TTS on the CPU."""
    s = config.Settings(me_to_them='local', their_lang='pl')
    assert languages.clone_why(s.local.clone, 'en', has_gpu=True) == ''
    assert languages.clone_why(s.local.clone, 'en', has_gpu=False) in ('your cloned voice needs an NVIDIA GPU',
                                                                       'no clone of your voice keeps up on a Mac')
    assert languages.clone_why(s.local.clone, 'uk', has_gpu=True) == 'your cloned voice cannot speak Ukrainian'
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    me = languages.direction(s, 'me_to_them')
    assert me.local and me.voice == 'stock' and me.tts == s.local.tts['pl']
    assert me.notes == (f'{languages.clone_why(s.local.clone, "pl")}: a stock voice speaks instead',)
    assert 'xtts-v2' not in languages.needed_models(s)
    s.their_lang = 'en'
    assert languages.direction(s, 'me_to_them').tts == 'pocket-tts-en' and 'xtts-v2' not in languages.needed_models(s)
    s.their_lang = 'zh'  # no stock voice for Chinese
    me = languages.direction(s, 'me_to_them')
    assert not me.local and me.why == (f'{languages.clone_why(s.local.clone, "zh")}; no local voice for Chinese',)
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    me = languages.direction(s, 'me_to_them')
    assert me.voice == 'clone' and 'no stock voice for Chinese takes over if your cloned voice fails' in me.notes


def test_a_cloud_side_downloads_its_fallback_voice():
    """H-09: a cloud side with local subtitles can go local when the cloud fails: its stock voice is downloaded too; with
    the cloud's subtitles (no local ASR + MT) there is nothing local to fall back on, so nothing is."""
    s = config.Settings(my_lang='ru', their_lang='de', them_to_me='cloud', me_to_them='cloud')
    s.local.clone = 'piper'
    ids = languages.needed_models(s)
    assert s.local.tts['ru'] in ids and s.local.tts['de'] in ids  # them -> me speaks Russian, me -> them German
    s.cloud.subtitles = True
    assert languages.needed_models(s) == ['silero-vad']


def test_a_fallback_voice_under_a_license_to_accept_is_optional():
    """FIXES-b 2: a cloud side's fallback voice that needs consent (tr, ko, ja: noncommercial) is not needed: the setup
    runs without it (that side stops while the cloud is down), it is downloaded only when its license is accepted."""
    s = config.Settings(my_lang='ru', their_lang='ja', them_to_me='local', me_to_them='cloud')  # the default engines
    ja = s.local.tts['ja']
    assert models.get(ja).consent and ja not in languages.needed_models(s) and languages.optional_models(s) == [ja]
    s.their_lang = 'de'  # a free fallback voice: needed, nothing optional
    assert s.local.tts['de'] in languages.needed_models(s) and languages.optional_models(s) == []
    s.their_lang, s.me_to_them = 'ja', 'local'  # chosen local: the voice is the side's own, needed (asked as before)
    assert ja in languages.needed_models(s) and languages.optional_models(s) == []


def test_the_clone_needs_a_card_it_runs_on(monkeypatch):
    """FIXES-b 4: clone_why also looks at the card (once per run): under 6 GB of memory — and on Windows an old driver or
    GPU for the voice pack's torch — the clone is not offered, said why; the stock voice speaks and nothing of the clone
    is downloaded. A card that answers nothing (a stand-in nvidia-smi) is taken as able."""
    big, small = languages.Gpu(8.0, (572, 83), 8.6), languages.Gpu(4.0, (572, 83), 8.6)
    assert languages.clone_why('xtts-v2', 'en', has_gpu=big) == ''
    assert 'at least 6 GB' in languages.clone_why('xtts-v2', 'en', has_gpu=small)
    old = languages.Gpu(8.0, (472, 12), 8.6)
    assert 'driver 570.65' in languages.gpu_why(old) and languages.gpu_why(old, windows=False) == ''
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: small)
    s = config.Settings(me_to_them='local', their_lang='pl')  # no CPU clone engine for Polish
    me = languages.direction(s, 'me_to_them')
    assert me.voice == 'stock' and 'at least 6 GB' in me.notes[0] and 'xtts-v2' not in languages.needed_models(s)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu())
    assert languages.direction(s, 'me_to_them').voice == 'clone'


def test_clone_engine_is_the_best_one_the_platform_runs(monkeypatch):
    """Stage 8 S1: `clone_engine(lang, side, platform=)` picks the clone engine per platform and language (RESEARCH-clone
    §7): Qwen3-TTS on an NVIDIA GPU, XTTS-v2 for what Qwen3 lacks, Pocket TTS on a CPU or a Mac (a model per language,
    Russian too); a card the voice pack cannot use falls to the CPU engine, not to the stock voice; else (None, why)."""
    reg = {'qwen3-tts': stub('qwen3-tts', 'clone', 'en', 'ru', 'de', 'fr', 'es', 'it', 'pt', 'zh', 'ja', 'ko', needs='gpu'),
           **models.registry(), 'pocket-tts-en': stub('pocket-tts-en', 'clone', 'en'),
           'pocket-tts-fr': stub('pocket-tts-fr', 'clone', 'fr'), 'pocket-tts-ru': stub('pocket-tts-ru', 'clone', 'ru')}
    monkeypatch.setattr(models, 'registry', lambda: reg)
    cpu, mac = languages.Platform('linux', False), languages.Platform('darwin', False)
    card, old = languages.Gpu(8.0, (572, 83), 8.6), languages.Gpu(8.0, (472, 12), 8.6)
    nvidia, old_win = languages.Platform('linux', card), languages.Platform('win32', old)
    engine = languages.clone_engine
    assert engine('en', platform=cpu) == engine('en', platform=mac) == ('pocket-tts-en', '')
    assert engine('fr', platform=old_win) == ('pocket-tts-fr', '')
    assert engine('ru', 'them_to_me', platform=cpu) == ('pocket-tts-ru', '')
    assert engine('en', platform=nvidia) == engine('ru', platform=nvidia) == ('qwen3-tts', '')
    assert engine('pl', platform=nvidia) == ('xtts-v2', '')
    assert engine('zh', platform=old_win) == (None, languages.gpu_why(old))
    assert engine('pl', platform=cpu) == (None, 'your cloned voice needs an NVIDIA GPU')
    assert engine('pl', 'them_to_me', platform=mac) == (None, 'no clone of their voice keeps up on a Mac')
    assert engine('uk', platform=nvidia) == (None, 'your cloned voice cannot speak Ukrainian')
    with pytest.raises(ValueError):
        engine('en', 'me')


def test_clone_engine_on_this_computer_asks_the_card_once(monkeypatch):
    """platform=None: this computer — sys.platform and its card (nvidia-smi once per run, as clone_why)."""
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(8.0, (572, 83), 8.6))
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')
    monkeypatch.setattr(languages, 'pack_here', lambda: True)
    monkeypatch.setattr(languages, 'addon_here', lambda: False)
    assert languages.clone_engine('en') == ('xtts-v2', '')  # Qwen3-TTS only with the voice pack's addon
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(4.0, (572, 83), 8.6))
    assert languages.clone_engine('en') == ('pocket-tts-en', '')  # too small a card: Pocket TTS on the CPU
    assert 'at least 6 GB' in languages.clone_engine('pl')[1]


def test_qwen3_brings_its_fallback_engine():
    """Qwen3-TTS on an NVIDIA GPU (with the voice pack's addon): Chatterbox Multilingual (its `fallback`) is
    downloaded with it, it takes over a voice whose clause breaks down (voice pack, local/qwen3.py); XTTS-v2 stays for
    Polish, Turkish, Arabic."""
    s = config.Settings(me_to_them='local')
    assert models.get('qwen3-tts').fallback == 'chatterbox-mtl'
    import unittest.mock as um
    with um.patch.object(languages, 'gpu', lambda engine='': True), \
            um.patch.object(models, 'is_fetched', lambda m, base=None: True), \
            um.patch.object(languages, 'nvidia_once', lambda: languages.Gpu(8.0, (572, 83), 8.6)), \
            um.patch.object(languages, 'pack_here', lambda: True), um.patch.object(languages, 'addon_here', lambda: True):
        ids = languages.needed_models(s)
        assert {'qwen3-tts', 'chatterbox-mtl'} <= set(ids) and 'xtts-v2' not in ids
        s.their_lang = 'pl'
        assert 'xtts-v2' in languages.needed_models(s) and 'chatterbox-mtl' not in languages.needed_models(s)


def test_qwen3_speaks_where_the_voice_packs_addon_is_and_the_card_has_8_gb(monkeypatch):
    """Stage 8: Qwen3-TTS (before XTTS-v2 in the registry) only with the pack's addon "Qwen3 + Chatterbox" and a card of
    8 GB (its `vram`: with Chatterbox after a breakdown 5.6-5.9 GB); without the addon, or on 6 GB, XTTS-v2 as before;
    without the pack the engine that needs only the pack (the wizard offers the pack, not the addon)."""
    card = {'8': languages.Gpu(8.0, (572, 83), 8.6), '6': languages.Gpu(6.0, (572, 83), 8.6)}
    has = {'gpu': '8', 'pack': True, 'addon': True}
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(models, 'is_fetched', lambda m, base=None: True)  # the addon's models are there
    monkeypatch.setattr(languages, 'nvidia_once', lambda: card[has['gpu']])
    monkeypatch.setattr(languages, 'pack_here', lambda: has['pack'])
    monkeypatch.setattr(languages, 'addon_here', lambda: has['addon'])
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')
    engine = languages.clone_engine
    assert engine('en') == engine('ru') == engine('zh') == ('qwen3-tts', '') and engine('pl') == ('xtts-v2', '')
    s = config.Settings(me_to_them='local')
    assert {'qwen3-tts', 'chatterbox-mtl'} <= set(languages.needed_models(s)) and 'xtts-v2' not in languages.needed_models(s)
    has['gpu'] = '6'
    assert engine('en') == ('xtts-v2', '')
    assert 'at least 8 GB' in languages.clone_why('qwen3-tts', 'en', card['6'])
    assert languages.clone_why('xtts-v2', 'en', card['6']) == ''
    has.update(gpu='8', addon=False)
    assert engine('en') == ('xtts-v2', '')
    assert languages.not_here('qwen3-tts') == "the voice pack's addon Qwen3 + Chatterbox is not installed (Components…)"
    assert languages.not_here('xtts-v2') == ''
    has['pack'] = False
    assert engine('en') == ('pocket-tts-en', '')  # FIXES 9: the CPU engine where it can
    assert engine('zh') == ('xtts-v2', '')  # none can: the one that needs only the pack, not the addon's
    assert languages.not_here('qwen3-tts') == languages.not_here('xtts-v2') == \
        'the voice pack is not installed (setup wizard, "Your voice")'


def test_the_addon_is_here_by_the_pack_or_by_the_runners_own_word(monkeypatch):
    """Windows: the pack's addon state; a runner of one's own ($PERESMESHNIK_VOICE_CMD: Linux from source, the server):
    $PERESMESHNIK_VOICE_ADDON=1 says its venv has requirements-linux-addon*.txt."""
    from peresmeshnik.local import voicepack
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    monkeypatch.setenv('PERESMESHNIK_VOICE_ADDON', '1')
    st = {'addon': 'none'}
    monkeypatch.setattr(voicepack, 'state', lambda lock=None, where=None, addon=False: st['addon'] if addon else 'ready')
    assert not languages.addon_here()
    st['addon'] = 'ready'
    assert languages.addon_here()
    monkeypatch.setenv('PERESMESHNIK_VOICE_CMD', '/opt/clone/bin/python voice_runner.py')
    st['addon'] = 'none'
    assert languages.addon_here()
    monkeypatch.delenv('PERESMESHNIK_VOICE_ADDON')
    assert not languages.addon_here()


def test_gated_weights_without_a_token_give_the_stock_voice(monkeypatch):
    """FIXES 1: Kyutai's gated Pocket TTS without the user's token (and not downloaded) is not chosen: the stock voice
    speaks, said why, and it is not among the models to download; with a token it is chosen again."""
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    monkeypatch.setattr(models, 'hf_token', lambda: '')
    monkeypatch.setattr(models, 'is_fetched', lambda m, base=None: False)
    s = config.Settings(me_to_them='local')  # ru -> en
    me = languages.direction(s, 'me_to_them')
    assert me.voice == 'stock' and me.notes == ('your cloned voice needs a Hugging Face token (Voices…, setup wizard): a '
                                                'stock voice speaks instead',)
    assert 'pocket-tts-en' not in languages.needed_models(s)
    s.their_lang = 'ru'  # Shtorm's Russian weights are not gated
    assert languages.direction(s, 'me_to_them').tts == 'pocket-tts-ru'
    s.their_lang = 'en'
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')
    assert languages.direction(s, 'me_to_them').tts == 'pocket-tts-en' and 'pocket-tts-en' in languages.needed_models(s)


def test_an_nvidia_card_without_the_voice_pack_takes_pocket(monkeypatch):
    """FIXES 9: Windows with an NVIDIA card of 6 GB or more but no voice pack (and no torch): the next engine that runs
    here (Pocket TTS), not the stock voice; without one (Polish) the NVIDIA engine stays: its pack is the wizard's offer.
    FIXES 10: Chatterbox (Qwen3's fallback) is never a clone engine of its own."""
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(8.0, (572, 83), 8.6))
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')
    monkeypatch.setattr(languages, 'pack_here', lambda: False)
    assert languages.clone_engine('en') == ('pocket-tts-en', '')
    assert languages.clone_engine('pl') == ('xtts-v2', '')
    monkeypatch.setattr(languages, 'pack_here', lambda: True)
    assert languages.clone_engine('en') == ('xtts-v2', '')
    assert 'chatterbox-mtl' not in languages.clone_engines() and 'qwen3-tts' in languages.clone_engines()
    from peresmeshnik.local.clone import make_engine
    with pytest.raises(ValueError, match='not a clone engine'):
        make_engine('chatterbox-mtl', '/nowhere')



def test_the_addon_without_its_models_leaves_xtts(monkeypatch):
    """FIXES addon 3: the addon's packages are there but Qwen3's (or Chatterbox's) weights are not: XTTS-v2 speaks, ready
    as it is, not the stock voice; with both downloaded Qwen3-TTS does."""
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(12.0, (572, 83), 8.6))
    monkeypatch.setattr(languages, 'pack_here', lambda: True)
    monkeypatch.setattr(languages, 'addon_here', lambda: True)
    have = {'qwen3-tts'}
    monkeypatch.setattr(models, 'is_fetched', lambda m, base=None: m.id in have)
    assert languages.clone_engine('ru') == ('xtts-v2', '')  # Chatterbox missing
    have.add('chatterbox-mtl')
    assert languages.clone_engine('ru') == ('qwen3-tts', '')


def test_both_directions_cloned_share_the_card(monkeypatch):
    """FIXES addon 5: them -> me's clone passes me -> them's engine (`beside`): on an 8 GB card Qwen3 (with Chatterbox)
    and XTTS-v2 fit together, two Qwen3 do not, so their voice takes XTTS-v2; where nothing fits beside, the next engine
    (Pocket on the CPU) or none, said why; on a 24 GB card both are Qwen3."""
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'pack_here', lambda: True)
    monkeypatch.setattr(languages, 'addon_here', lambda: True)
    monkeypatch.setattr(models, 'is_fetched', lambda m, base=None: True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(8188 / 1024, (572, 83), 8.6))  # 8 GB as reported
    me, _ = languages.clone_engine('en')
    assert me == 'qwen3-tts' and languages.clone_engine('ru', 'them_to_me', beside=me) == ('xtts-v2', '')
    assert languages.clone_engine('ru', 'them_to_me') == ('qwen3-tts', '')  # nothing beside it: the best one
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(6.0, (572, 83), 8.6))
    assert languages.clone_engine('en') == ('xtts-v2', '')  # Qwen3 needs 8 GB
    assert languages.clone_engine('pl', 'them_to_me', beside='xtts-v2') == ('xtts-v2', '')  # 2 + 2 GB of 6
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(7.5, (572, 83), 8.6))  # Qwen3 just fits alone
    assert languages.clone_engine('en') == ('qwen3-tts', '')
    assert languages.clone_engine('de', 'them_to_me', beside='qwen3-tts')[0] == 'pocket-tts-de'  # XTTS beside: 7.9 GB
    engine, why = languages.clone_engine('pl', 'them_to_me', beside='qwen3-tts')
    assert engine is None and 'does not fit on the NVIDIA GPU beside' in why
    monkeypatch.setattr(languages, 'nvidia_once', lambda: languages.Gpu(24.0, (572, 83), 8.6))
    assert languages.clone_engine('ru', 'them_to_me', beside='qwen3-tts') == ('qwen3-tts', '')

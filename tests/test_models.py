"""Registry pins, download with resume and SHA-256 check, license screen, settings round trip."""
import hashlib

import pytest

from peresmeshnik import config, models, pipeline


def test_registry_pins():
    reg = models.registry()
    assert {'silero-vad', 'parakeet-tdt-0.6b-v3', 'opus-mt-en-ru', 'opus-mt-ru-en'} <= set(reg)
    for m in reg.values():
        if m.via:  # a pair through another language: its models exist, chain up and carry its license
            parts = [reg[i] for i in m.via]
            assert m.role == 'mt' and not m.files and all(p.role == 'mt' and not p.via for p in parts), m.id
            (pair,) = m.langs
            hops = [p.langs[0].split('-') for p in parts]
            assert pair == f'{hops[0][0]}-{hops[-1][1]}' and all(a[1] == b[0] for a, b in zip(hops, hops[1:])), m.id
            assert m.consent == any(p.consent for p in parts) and m.size == sum(p.size for p in parts), m.id
            continue
        assert m.role in ('vad', 'asr', 'mt', 'tts', 'clone') and m.files and m.license and m.license_url
        assert m.url.startswith('https://') and ('/resolve/' in m.url and len(m.url.split('/resolve/')[1].split('/')[0]) == 40
                                                 or '/releases/download/' in m.url), m.id  # revision pinned
        for f in m.files.values():
            assert len(f['sha256']) == 64 and f['size'] > 0
    assert reg['xtts-v2'].consent and not reg['opus-mt-en-ru'].consent
    # every direction of the 14 languages has an MT row: its own, or through English by the `pivot` rows (registry())
    from peresmeshnik.languages import LANGS
    pairs = {p for m in reg.values() if m.role == 'mt' for p in m.langs}
    assert {f'{a}-{b}' for a in LANGS for b in LANGS if a != b} <= pairs
    assert reg['mt-tr-ja-via-en'].via == ('opus-mt-tr-en', 'quickmt-en-ja') and 'mt-de-ru-via-en' not in reg  # de-ru: direct
    assert reg['mt-tr-ja-via-en'].license == 'Apache-2.0 + CC-BY-4.0'  # both bind: CC-BY's credit too
    assert [m.license for m in models.parts(reg['mt-tr-ja-via-en'])] == ['Apache-2.0', 'CC-BY-4.0']
    # one pivot per language and way; X->en: OPUS-MT where quickmt makes up a continuation of a one-word reply (config)
    to_en = sorted(m.langs[0][:2] for m in reg.values() if m.pivot and m.langs[0].endswith('-en'))
    from_en = sorted(m.langs[0][3:] for m in reg.values() if m.pivot and m.langs[0].startswith('en-'))
    assert to_en == from_en == sorted(set(LANGS) - {'en'})
    assert {m.langs[0][:2] for m in reg.values() if m.pivot and m.id.startswith('opus-mt-') and m.langs[0].endswith('-en')} \
        == set(config.OPUS_TO_EN)


def fake(monkeypatch, url, files):
    m = models.Model(id='fake', role='mt', langs=('en-ru',), license='MIT', license_url='https://x', url=url,
                     files={n: {'sha256': hashlib.sha256(b).hexdigest(), 'size': len(b)} for n, b in files.items()})
    monkeypatch.setattr(models, 'registry', lambda: {'fake': m})
    return m


def test_fetch_resume_and_verify(server, tmp_path, monkeypatch):
    srv, url = server
    blob = bytes(range(256)) * 4000
    (srv.root / 'm-model.bin').write_bytes(blob)
    (srv.root / 'm-config.json').write_bytes(b'{}')
    fake(monkeypatch, url + 'm-', {'model.bin': blob, 'config.json': b'{}'})
    cache = tmp_path / 'cache'
    with pytest.raises(models.ModelMissing):
        models.path('fake', cache)
    (cache / 'fake').mkdir(parents=True)
    (cache / 'fake' / 'model.bin.part').write_bytes(blob[:1000])  # interrupted download
    got = []
    folder = models.fetch('fake', cache, got.append)
    assert (folder / 'model.bin').read_bytes() == blob and not (folder / 'model.bin.part').exists()
    assert sum(got) == len(blob) + 2  # resumed part counted once, the rest downloaded
    assert ('/m-model.bin', 'bytes=1000-') in srv.ranges and ('/m-model.bin', None) not in srv.ranges
    assert models.path('fake', cache) == folder


def test_a_pair_through_english_is_its_two_models(server, tmp_path, monkeypatch):
    srv, url = server
    pins = {}
    for name in ('de-en', 'en-ru'):
        (srv.root / f'{name}-model.bin').write_bytes(name.encode() * 100)
        pins[name] = {'model.bin': {'sha256': hashlib.sha256(name.encode() * 100).hexdigest(), 'size': 500}}
    reg = {f'opus-mt-{n}': models.Model(f'opus-mt-{n}', 'mt', (n,), 'CC-BY-4.0', 'https://x', f'{url}{n}-', pins[n])
           for n in pins}
    reg['via'] = models.Model('via', 'mt', ('de-ru',), 'CC-BY-4.0', 'https://x', '', {}, via=('opus-mt-de-en', 'opus-mt-en-ru'))
    monkeypatch.setattr(models, 'registry', lambda: reg)
    assert reg['via'].size == 1000 and not models.is_fetched(reg['via'], tmp_path)
    with pytest.raises(models.ModelMissing):
        models.path('via', tmp_path)
    models.fetch('opus-mt-de-en', tmp_path)
    assert not models.is_fetched(reg['via'], tmp_path)  # one of the two is not a pair
    got = []
    assert models.fetch('via', tmp_path, got.append) == tmp_path / 'opus-mt-de-en'
    assert models.is_fetched(reg['via'], tmp_path) and sum(got) == 1000  # the one already there counted, not fetched
    assert [r[0] for r in srv.ranges] == ['/de-en-model.bin', '/en-ru-model.bin']


def test_fetch_resumes_after_cut(server, tmp_path, monkeypatch):
    srv, url = server
    blob = bytes(range(256)) * 4000
    (srv.root / 'model.bin').write_bytes(blob)
    srv.cut.add('/model.bin')  # the server closes in the middle of the body
    fake(monkeypatch, url, {'model.bin': blob})
    folder = models.fetch('fake', tmp_path)
    assert (folder / 'model.bin').read_bytes() == blob
    assert srv.ranges == [('/model.bin', None), ('/model.bin', f'bytes={len(blob) // 2}-')]  # resumed, not restarted


def test_fetch_rejects_bad_hash(server, tmp_path, monkeypatch):
    srv, url = server
    (srv.root / 'model.bin').write_bytes(b'tampered')
    m = fake(monkeypatch, url, {'model.bin': b'original'})
    with pytest.raises(RuntimeError, match='mismatch'):
        models.fetch('fake', tmp_path)
    assert not (tmp_path / 'fake' / 'model.bin').exists() and not models.is_fetched(m, tmp_path)


def test_license_screen_needs_yes():
    m = models.registry()['xtts-v2']
    quiet = lambda *a: None  # noqa: E731
    assert not models.license_screen([m], ask=lambda _: 'no', out=quiet)
    assert models.license_screen([m], ask=lambda _: 'yes', out=quiet)
    assert models.license_screen([m], accept=True, out=quiet)
    assert models.license_screen([models.registry()['silero-vad']], ask=lambda _: 1 / 0, out=quiet)  # no question


def test_settings_round_trip(tmp_path):
    s = config.Settings()
    s.me_to_them, s.local.asr['ru'], s.cloud.reconnect_s, s.local.voice = 'local', 'gigaam-v3-e2e-ctc', 300.0, 'my "voice"'
    p = config.save(s, tmp_path / 'settings.toml')
    assert config.load(p) == s
    p.write_text('them_to_me = "both"\n')
    with pytest.raises(ValueError, match='them_to_me'):
        config.load(p)
    p.write_text('[local]\nthreads = "two"\nfuture_key = 1\n')
    with pytest.raises(ValueError, match='local.threads'):
        config.load(p)


def test_saved_settings_take_a_new_default(tmp_path, monkeypatch):
    """Saved [local] tables keep only what the user changed: a default changed later (fr's ASR, a new MT) reaches them."""
    s = config.Settings()
    s.local.mt['de-ru'] = 'mt-de-ru-via-en'
    p = config.save(s, tmp_path / 'settings.toml')
    assert 'fr-ru' not in p.read_text('utf-8') and config.load(p) == s
    new = {**config._mt(), 'fr-ru': 'mt-fr-ru-via-en', 'de-ru': 'opus-mt-de-ru'}
    monkeypatch.setattr(config, '_mt', lambda: new)
    loc = config.load(p).local
    assert (loc.mt['fr-ru'], loc.mt['de-ru']) == ('mt-fr-ru-via-en', 'mt-de-ru-via-en')


def test_settings_of_0_2_0_get_the_new_en_ru(tmp_path):
    """0.2.0 saved the [local] tables whole: its en-ru (OPUS-MT) gives way to today's default (quickmt); the same value
    chosen since (a table of changed keys only) stays."""
    p = tmp_path / 'settings.toml'
    p.write_text('[local]\nasr = { "en" = "parakeet-tdt-0.6b-v3", "ru" = "gigaam-v3-e2e-ctc" }\n'
                 'mt = { "en-ru" = "opus-mt-en-ru", "ru-en" = "opus-mt-ru-en" }\n', 'utf-8')
    assert config.load(p).local.mt['en-ru'] == config.Local().mt['en-ru'] == 'quickmt-en-ru'
    s = config.Settings()
    s.local.mt['en-ru'] = 'opus-mt-en-ru'  # chosen again: an 8 GB Mac's smaller MT
    assert config.load(config.save(s, p)).local.mt['en-ru'] == 'opus-mt-en-ru'


def test_the_default_engines_cover_the_14_languages():
    """Settings' defaults: MT for every direction, ASR and a stock voice where the registry has one (BENCH stage 6)."""
    from peresmeshnik.languages import LANGS
    loc, reg = config.Local(), models.registry()
    assert config.LANGS == LANGS and set(pipeline.HELLO) == set(LANGS)  # every language has its warm-up phrase
    assert set(loc.mt) == {f'{a}-{b}' for a in LANGS for b in LANGS if a != b}
    assert all(k in reg[i].langs for k, i in loc.mt.items()) and set(loc.asr) == set(LANGS)
    assert all(k in reg[i].langs for d in (loc.asr, loc.tts) for k, i in d.items())
    assert set(LANGS) - set(loc.tts) == {'zh', 'ar'}  # no stock voice with a usable license (BENCH stage 6)
    # X<->ru direct where through English loses the speaker's gender and ты / вы; X->en, en->X: the pivots
    assert [loc.mt[f'{x}-ru'] for x in ('de', 'tr')] == ['opus-mt-de-ru', 'mt-tr-ru-via-en'] and loc.mt['ru-uk'] == 'opus-mt-ru-uk'
    assert all(reg[loc.mt[f'{x}-en']].pivot and reg[loc.mt[f'en-{x}']].pivot for x in set(LANGS) - {'en'})

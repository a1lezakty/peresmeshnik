"""Registry pins, download with resume and SHA-256 check, license screen, settings round trip."""
import hashlib

import pytest

from peresmeshnik import config, models


def test_registry_pins():
    reg = models.registry()
    assert {'silero-vad', 'parakeet-tdt-0.6b-v3', 'opus-mt-en-ru', 'opus-mt-ru-en'} <= set(reg)
    for m in reg.values():
        assert m.role in ('vad', 'asr', 'mt', 'tts', 'clone') and m.files and m.license and m.license_url
        assert m.url.startswith('https://') and ('/resolve/' in m.url and len(m.url.split('/resolve/')[1].split('/')[0]) == 40
                                                 or '/releases/download/' in m.url), m.id  # revision pinned
        for f in m.files.values():
            assert len(f['sha256']) == 64 and f['size'] > 0
    assert reg['xtts-v2'].consent and not reg['opus-mt-en-ru'].consent


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

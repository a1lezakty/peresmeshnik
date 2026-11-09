"""The voice pack installer (local/voicepack.py) against a local HTTP server (conftest `server`, with Range): a tiny zip
stands in for the embeddable Python, tiny wheels for torch and the rest, a fake pip unpacks them (or the real pip,
where a pip wheel is on this machine). SHA-256 mismatch, cancel and resume, cleanup of a half-installed pack,
.complete last, another identity is outdated; make_clone's answers to each state. Windows itself:
test_voice_pack_windows.py."""
import glob
import hashlib
import io
import json
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

from peresmeshnik.local import clone, voicepack
from peresmeshnik.local.voicepack import Cancelled, PackError

FAKE_PIP = '''import sys, time, zipfile
a = sys.argv[1:]
target, reqs = a[a.index('--target') + 1], a[a.index('-r') + 1]
assert '--require-hashes' in a and '--no-index' in a and '--no-deps' in a
for line in open(reqs):
    whl = line.split(' --hash=')[0]
    print('Processing', whl, flush=True)
    if 'slow' in whl:
        time.sleep(60)
    zipfile.ZipFile(whl).extractall(target)
print('Successfully installed', flush=True)
'''


def zipped(files: dict) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        for name, body in files.items():
            z.writestr(name, body)
    return b.getvalue()


def wheel(name: str, body: bytes = b'') -> tuple[str, bytes]:
    """A minimal valid wheel (real pip installs it) with `body` as data."""
    di = f'{name}-1.0.dist-info'
    files = {f'{name}/__init__.py': 'X = 1\n', f'{name}/data.bin': body,
             f'{di}/METADATA': f'Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n',
             f'{di}/WHEEL': 'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
    files[f'{di}/RECORD'] = ''.join(f'{n},,\n' for n in [*files, f'{di}/RECORD'])
    return f'{name}-1.0-py3-none-any.whl', zipped(files)


@pytest.fixture
def pack(server, tmp_path):
    """(lock path, pack folder, server, publish(name, bytes) -> lock entry)."""
    httpd, url = server

    def publish(name: str, body: bytes) -> dict:
        (httpd.root / name).write_bytes(body)
        return {'name': name.split('-')[0], 'url': url + name, 'sha256': hashlib.sha256(body).hexdigest(), 'size': len(body)}
    spec = {'python': publish('python-3.12.10-embed-amd64.zip', zipped({'python.exe': 'MZ', voicepack.PTH: 'python312.zip\n.\n'})),
            'pip': publish('pip-0-py3-none-any.whl', zipped({'pip/__main__.py': FAKE_PIP})),
            'wheels': [publish(*wheel('torch', bytes(range(256)) * 20_000)), publish(*wheel('tts'))],  # torch: 5 MB, 5 reads
            'installed_bytes': 4 << 20}
    lock = tmp_path / 'pack-windows.json'
    lock.write_text(json.dumps(spec))
    return lock, tmp_path / 'data' / 'voice-pack', httpd, publish


def run(lock, where, **kw):
    return voicepack.install(lock=lock, where=where, run_python=[sys.executable], **kw)


def test_installs_with_complete_last(pack):
    lock, where, httpd, _ = pack
    seen = []
    assert voicepack.state(lock, where) == 'none'
    assert run(lock, where, progress=lambda *a: seen.append(a)) == where
    assert (where / 'Lib' / 'site-packages' / 'torch' / 'data.bin').stat().st_size == 5_120_000
    assert (where / '.complete').read_text().strip() == voicepack.identity(lock) and voicepack.state(lock, where) == 'ready'
    assert (where / voicepack.PTH).read_text() == 'python312.zip\n.\nLib\\site-packages\nimport site\n'
    assert sorted(p.name for p in where.parent.iterdir()) == ['voice-pack']  # downloads, temp folders: gone
    stages = [a[0] for a in seen]
    assert stages[0] == 'cleaning' and stages.index('downloading') < stages.index('unpacking') < stages.index('installing')
    spec = json.loads(lock.read_text())
    total = sum(f['size'] for f in [spec['python'], spec['pip'], *spec['wheels']])
    assert [a for a in seen if a[0] == 'downloading'][-1] == ('downloading', total, total, 'tts-1.0-py3-none-any.whl')
    assert ('installing', 0, 0, 'Successfully installed') in seen
    n = len(httpd.ranges)
    assert run(lock, where) == where and len(httpd.ranges) == n  # ready: nothing again


def test_sha_mismatch_is_refused(pack):
    lock, where, _, _ = pack
    spec = json.loads(lock.read_text())
    spec['wheels'][1]['sha256'] = '0' * 64
    lock.write_text(json.dumps(spec))
    with pytest.raises(PackError, match='tts-1.0-py3-none-any.whl: size or SHA-256 mismatch'):
        run(lock, where)
    dl = where.with_name('voice-pack.download')
    assert not where.exists() and not list(dl.glob('tts*'))  # the bad download is gone; the good ones stay
    assert (dl / 'torch-1.0-py3-none-any.whl').exists() and voicepack.state(lock, where) == 'partial'


def test_cancel_then_resume(pack):
    """Cancelled mid-download: the .part stays and the next install goes on from there (HTTP Range)."""
    lock, where, httpd, _ = pack
    cancel = threading.Event()

    def progress(stage, done, total, text=''):
        if stage == 'downloading' and text.startswith('torch') and done > 1 << 20:  # after its first MB
            cancel.set()
    with pytest.raises(Cancelled):
        run(lock, where, progress=progress, cancel=cancel)
    part = where.with_name('voice-pack.download') / 'torch-1.0-py3-none-any.whl.part'
    assert part.stat().st_size == 2 << 20 and voicepack.state(lock, where) == 'partial'
    run(lock, where)
    assert ('/torch-1.0-py3-none-any.whl', f'bytes={2 << 20}-') in httpd.ranges and voicepack.state(lock, where) == 'ready'


def test_cancel_while_pip_installs(pack):
    lock, where, _, publish = pack
    spec = json.loads(lock.read_text())
    spec['wheels'].append(publish(*wheel('slow')))
    lock.write_text(json.dumps(spec))
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    t = time.perf_counter()
    with pytest.raises(Cancelled):
        run(lock, where, cancel=cancel)
    assert time.perf_counter() - t < 10 and not where.exists() and voicepack.state(lock, where) == 'partial'


def test_a_half_installed_pack_is_cleaned_up(pack, monkeypatch):
    lock, where, _, _ = pack
    monkeypatch.setattr(voicepack, '_pip', lambda *a: (_ for _ in ()).throw(PackError('pip failed (1): no space')))
    with pytest.raises(PackError, match='no space'):
        run(lock, where)
    assert not (where / '.complete').exists()
    where.mkdir()
    (where / 'junk.dll').write_text('from a crash')  # a pack folder without .complete
    assert voicepack.state(lock, where) == 'partial'
    monkeypatch.undo()
    run(lock, where)
    assert not (where / 'junk.dll').exists() and voicepack.state(lock, where) == 'ready'
    assert sorted(p.name for p in where.parent.iterdir()) == ['voice-pack']


def test_another_identity_is_outdated(pack, monkeypatch):
    lock, where, _, publish = pack
    run(lock, where)
    spec = json.loads(lock.read_text())
    spec['wheels'][1] = publish(*wheel('tts', b'new'))  # an app update with another lock
    lock.write_text(json.dumps(spec))
    assert voicepack.state(lock, where) == 'outdated'
    monkeypatch.setattr(voicepack, 'lock_file', lambda: lock)
    monkeypatch.setattr(voicepack, 'folder', lambda: where)
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    assert clone.voice_cmd() is None
    with pytest.raises(clone.CloneUnavailable, match='installed for another version of the app: install it again'):
        clone.make_clone(where)
    run(lock, where)
    assert (where / 'Lib' / 'site-packages' / 'tts' / 'data.bin').read_bytes() == b'new'
    assert clone.voice_cmd() == [str(where / 'python.exe'), str(clone.RUNNER)]


def test_not_enough_disk_space(pack, monkeypatch):
    lock, where, httpd, _ = pack
    monkeypatch.setattr(voicepack.shutil, 'disk_usage', lambda p: shutil._ntuple_diskusage(100 << 30, 99 << 30, 1 << 30))
    with pytest.raises(PackError, match=r'needs 1\.0 GB free .* there are 1\.0 GB'):
        run(lock, where)
    assert httpd.ranges == []  # refused before any download


def test_need_bytes_of_the_real_lock():
    spec = json.loads(voicepack.lock_file().read_text())
    names = {w['name'] for w in spec['wheels']}
    assert {'torch', 'torchaudio', 'coqui-tts', 'transformers', 'six', 'soundfile'} <= names
    assert all('win_amd64' in w['url'] or w['url'].endswith('-any.whl') for w in spec['wheels'])
    assert 3e9 < sum(w['size'] for w in spec['wheels']) < 5e9 and 7e9 < spec['installed_bytes'] < 10e9


PIP = sorted(glob.glob('/usr/share/python-wheels/pip-*.whl'))


@pytest.mark.skipif(not PIP, reason='no pip wheel on this machine')
def test_real_pip_installs_the_verified_wheels(pack):
    """The pip step for real: no index, no deps, every wheel by its hash, into Lib/site-packages."""
    lock, where, _, publish = pack
    spec = json.loads(lock.read_text())
    spec['pip'] = publish(Path(PIP[-1]).name, Path(PIP[-1]).read_bytes())
    lock.write_text(json.dumps(spec))
    run(lock, where)
    site = where / 'Lib' / 'site-packages'
    assert (site / 'torch' / '__init__.py').exists() and (site / 'tts-1.0.dist-info' / 'RECORD').exists()


def test_the_frozen_app_has_its_runner_in_its_data(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path / '_internal'), raising=False)
    assert voicepack.runner_dir() == tmp_path / '_internal' / 'voice-runner'
    assert voicepack.lock_file() == tmp_path / '_internal' / 'voice-runner' / 'pack-windows.json'

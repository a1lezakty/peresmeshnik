"""The voice pack on Windows: its own Python with torch CUDA and coqui-tts in <data dir>/voice-pack, for the clone's
runner (voice-runner/voice_runner.py, local/clone.py VoicePackClone).

`install()` downloads what voice-runner/pack-windows.json pins (voice-runner/lock.py: the embeddable CPython from
python.org, a pip wheel, ~100 wheels, each by URL, SHA-256 and size; ~3.7 GB, ~8 GB installed), resuming a
download that stopped, then unpacks the Python, lets it see Lib/site-packages and has pip install the verified wheels
from disk (no index, no network). `.complete` is written last and holds the lock's identity: a pack without it is a
half-installed one (cleaned up by the next install), one with another identity was made for another app version and
is outdated (installed again). The runner itself ships with the app (beside the executable when frozen), so it always
matches the app.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from peresmeshnik import config, models

MARGIN = 1 << 30  # free space left after the install
PTH = 'python312._pth'


class PackError(RuntimeError):
    pass


class Cancelled(PackError):
    def __init__(self):
        super().__init__('the voice pack install was cancelled')


def runner_dir() -> Path:
    """voice_runner.py, xtts.py and pack-windows.json: in the frozen app's data (PyInstaller's _internal beside the
    executable), else in the checkout."""
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)) / 'voice-runner'
    return Path(__file__).resolve().parents[2] / 'voice-runner'


def folder() -> Path:
    return config.data_dir() / 'voice-pack'


def lock_file() -> Path:
    return runner_dir() / 'pack-windows.json'


def identity(lock: Path | None = None) -> str:
    return models.sha256(lock or lock_file())


def state(lock: Path | None = None, where: Path | None = None) -> str:
    """'ready', 'outdated' (another app version's pack), 'partial' (an install that did not finish) or 'none'."""
    where = where or folder()
    done = where / '.complete'
    if done.is_file():  # no lock: an app without the pack's files (not the Windows build)
        lock = lock or lock_file()
        return 'ready' if lock.is_file() and done.read_text().strip() == identity(lock) else 'outdated'
    return 'partial' if where.exists() or where.with_name('voice-pack.download').exists() else 'none'


def python(where: Path | None = None) -> Path:
    return (where or folder()) / 'python.exe'


def need_bytes(lock: Path | None = None, where: Path | None = None) -> int:
    """Free space the install needs: what is not downloaded yet, the installed size and a margin."""
    spec = json.loads((lock or lock_file()).read_text())
    dl = (where or folder()).with_name('voice-pack.download')
    files = [spec['python'], spec['pip'], *spec['wheels']]
    return sum(f['size'] - _have(dl, f) for f in files) + spec['installed_bytes'] + MARGIN


def _name(f: dict) -> str:
    return urllib.parse.unquote(f['url'].rsplit('/', 1)[1])


def _have(dl: Path, f: dict) -> int:
    for p in (dl / _name(f), dl / (_name(f) + '.part')):
        if p.exists():
            return min(p.stat().st_size, f['size'])
    return 0


def _discard(p: Path) -> None:
    """Out of the way at once (a rename), then deleted: a half-deleted folder never looks like a pack."""
    if p.exists():
        aside = p.with_name(f'{p.name}.old-{time.time_ns()}')
        p.rename(aside)
        shutil.rmtree(aside, ignore_errors=True)


def install(progress=lambda stage, done, total, text='': None, cancel: threading.Event | None = None,
            lock: Path | None = None, where: Path | None = None, run_python: list[str] | None = None,
            retries: int = 3) -> Path:
    """Install the pack; `progress(stage, done_bytes, total_bytes, text)` with stage 'cleaning', 'downloading',
    'unpacking' or 'installing' (then `text` is pip's last line); `cancel` set: stops soon with Cancelled, the
    downloads so far stay for the next attempt. `run_python`: how to run the pack's Python (tests)."""
    lock, where = lock or lock_file(), where or folder()
    cancel = cancel or threading.Event()
    spec = json.loads(lock.read_text())
    if state(lock, where) == 'ready':
        return where
    progress('cleaning', 0, 0)
    try:
        _discard(where)  # half installed or outdated: from scratch
    except OSError as e:  # Windows: its runner still has files open
        raise PackError(f'cannot move the old voice pack away ({e}): turn translation off first') from e
    tmp, pip_tmp = where.with_name('voice-pack.tmp'), where.with_name('voice-pack.pip-tmp')
    for p in (tmp, pip_tmp):  # an earlier attempt's, up to ~8 GB: gone before the free space is counted
        _discard(p)
    for old in where.parent.glob(f'{where.name}.old-*'):  # what an earlier cleanup did not finish
        shutil.rmtree(old, ignore_errors=True)
    dl = where.with_name('voice-pack.download')
    dl.mkdir(parents=True, exist_ok=True)
    free, need = shutil.disk_usage(dl).free, need_bytes(lock, where)
    if free < need:
        raise PackError(f'the voice pack needs {need / 2 ** 30:.1f} GB free on {dl.anchor or dl.parent}, '
                        f'there are {free / 2 ** 30:.1f} GB')
    files = [spec['python'], spec['pip'], *spec['wheels']]
    total, done = sum(f['size'] for f in files), 0
    for f in files:
        name = _name(f)
        done += _fetch(f, dl, retries, cancel, lambda n, base=done, name=name: progress('downloading', base + n, total, name))
        progress('downloading', done, total, name)
    progress('unpacking', 0, 0)
    shutil.unpack_archive(dl / _name(spec['python']), tmp, 'zip')
    site = tmp / 'Lib' / 'site-packages'
    site.mkdir(parents=True)
    (tmp / PTH).write_text('python312.zip\n.\nLib\\site-packages\nimport site\n')  # the runner's packages; .pth files
    reqs = tmp / 'pack-requirements.txt'
    reqs.write_text(''.join(f'{dl / _name(w)} --hash=sha256:{w["sha256"]}\n' for w in spec['wheels']), 'utf-8')
    cmd = [*(run_python or [str(python(tmp))]), str(dl / _name(spec['pip']) / 'pip'), 'install', '--no-index', '--no-deps',
           '--require-hashes', '--only-binary', ':all:', '--no-cache-dir', '--disable-pip-version-check',
           '--no-warn-script-location', '--target', str(site), '-r', str(reqs)]
    pip_tmp.mkdir()  # pip's temp on the drive checked above: its moves are renames
    _pip(cmd, cancel, lambda line: progress('installing', 0, 0, line), pip_tmp)
    tmp.rename(where)
    (where / '.complete').write_text(identity(lock) + '\n')  # last: until here the pack is not there
    for p in (dl, pip_tmp):
        shutil.rmtree(p, ignore_errors=True)
    return where


def _fetch(f: dict, dl: Path, retries: int, cancel: threading.Event, progress) -> int:
    """One file into `dl`, resumed from its .part, checked against its size and SHA-256."""
    dest = dl / _name(f)
    if dest.exists() and dest.stat().st_size == f['size']:
        return f['size']  # checked when it was downloaded
    part = dest.with_name(dest.name + '.part')
    got = [models._size(part)]

    def step(n):
        if cancel.is_set():
            raise Cancelled
        got[0] += n
        progress(got[0])
    err = ''
    for attempt in range(retries):
        try:
            models._download(f['url'], part, f['size'], step)
        except Cancelled:
            raise
        except Exception as e:  # URLError, timeouts, resets, IncompleteRead
            err = str(e)
        if models._size(part) >= f['size']:
            break
        if attempt == retries - 1:
            raise PackError(f'{_name(f)}: download incomplete ({models._size(part)} of {f["size"]} bytes, {err}); '
                            'install again to resume')
        time.sleep(models.RETRY_PAUSE_S * (attempt + 1))
    if models._size(part) != f['size'] or models.sha256(part) != f['sha256']:
        part.unlink()
        raise PackError(f'{_name(f)}: size or SHA-256 mismatch, download discarded')
    part.replace(dest)
    return f['size']


def _pip(cmd: list[str], cancel: threading.Event, line, tmp: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIP_')}  # no user pip settings (PIP_USER, ...)
    env |= {'PIP_CONFIG_FILE': os.devnull, 'PYTHONNOUSERSITE': '1', 'TMP': str(tmp), 'TEMP': str(tmp), 'TMPDIR': str(tmp)}
    # a user folder with letters outside the ANSI code page (Windows): pip 26 reads the requirements as UTF-8 (then the
    # locale's), but prints their paths in the locale's encoding, where a Cyrillic one fails; UTF-8 mode: both UTF-8
    env['PYTHONUTF8'] = '1'
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                         encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=env)
    tail: list[str] = []

    def read():
        for s in p.stdout:
            if s.strip():
                tail[:] = (tail + [s.rstrip()])[-20:]
                line(s.rstrip())
    t = threading.Thread(target=read, daemon=True)
    t.start()
    while p.poll() is None:  # only while installing: the app is busy anyway
        if cancel.wait(0.2):
            p.kill()
            p.wait()
            raise Cancelled
    t.join(5)
    if p.returncode:
        raise PackError(f'pip failed ({p.returncode}): ' + ' / '.join(tail[-3:]))

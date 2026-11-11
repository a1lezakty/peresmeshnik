"""The voice pack on Windows: its own Python with torch CUDA and coqui-tts in <data dir>/voice-pack, for the clone's
runner (voice-runner/voice_runner.py, local/clone.py VoicePackClone).

`install()` downloads what voice-runner/pack-windows.json pins (voice-runner/lock.py: the embeddable CPython from
python.org, a pip wheel, ~100 wheels, each by URL, SHA-256 and size; ~3.7 GB, ~8 GB installed), resuming a
download that stopped, then unpacks the Python, lets it see Lib/site-packages and has pip install the verified wheels
from disk (no index, no network). A package that has no wheel, only its data (unidic-lite, the Japanese dictionary of
XTTS's text cleaning), is an `archives` entry: its folder is unpacked from the verified source archive.
`.complete` is written last and holds the lock's identity: a pack without it is a
half-installed one (cleaned up by the next install), one with another identity was made for another app version and
is outdated (installed again). The runner itself ships with the app (beside the executable when frozen), so it always
matches the app.

The addon "Qwen3 + Chatterbox" (the lock's section `addon`, ~80 MB of wheels; its models are the registry's) goes into
the same pack only when asked: `install(addon=True)` after the pack, into Lib/addon, which the pack's Python reads
before Lib/site-packages (its transformers 5.16.1 over the pack's 5.22.2), with `.addon-complete` last;
`uninstall(addon=True)` takes it away and leaves the pack as it was. One install or removal at a time (the wizard and
the tray's Components… share the pack): another one meanwhile is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
from pathlib import Path

from peresmeshnik import config, models

MARGIN = 1 << 30  # free space left after the install
PTH = 'python312._pth'
ADDON = 'addon'  # <pack>/Lib/addon
ADDON_DONE = '.addon-complete'
_BUSY = threading.Lock()


class PackError(RuntimeError):
    pass


class Cancelled(PackError):
    def __init__(self):
        super().__init__('the voice pack install was cancelled')


def runner_dir() -> Path:
    """voice_runner.py, xtts.py, qwen3.py and pack-windows.json: in the frozen app's data (PyInstaller's _internal beside the
    executable), else in the checkout."""
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)) / 'voice-runner'
    return Path(__file__).resolve().parents[2] / 'voice-runner'


def folder() -> Path:
    return config.data_dir() / 'voice-pack'


def lock_file() -> Path:
    return runner_dir() / 'pack-windows.json'


def identity(lock: Path | None = None, addon: bool = False) -> str:
    """The pack's identity: the SHA-256 of its lock as lock.py writes it without the addon (the bytes of a lock that has
    none: a new addon outdates no pack); the addon's: of all of it (a new pack outdates its addon too)."""
    spec = json.loads((lock or lock_file()).read_text())
    if not addon:
        spec.pop('addon', None)
    return hashlib.sha256((json.dumps(spec, indent=1) + '\n').encode()).hexdigest()


def addon_section(lock: Path | None = None) -> dict:
    """The lock's section `addon` ({} : this app has none): wheels, archives, installed_bytes, models, vram_gb."""
    lock = lock or lock_file()
    return json.loads(lock.read_text()).get('addon', {}) if lock.is_file() else {}


def state(lock: Path | None = None, where: Path | None = None, addon: bool = False) -> str:
    """'ready', 'outdated' (another app version's pack), 'partial' (an install that did not finish) or 'none'; `addon`:
    the addon's, in this pack ('none' too where the lock has no addon)."""
    where, lock = where or folder(), lock or lock_file()
    if addon:
        if not addon_section(lock):
            return 'none'
        done = where / ADDON_DONE
        if done.is_file():
            return 'ready' if state(lock, where) == 'ready' and done.read_text().strip() == identity(lock, True) else 'outdated'
        return 'partial' if (where / 'Lib' / ADDON).exists() or _dl(where, True).exists() else 'none'
    done = where / '.complete'
    if done.is_file():  # no lock: an app without the pack's files (not the Windows build)
        return 'ready' if lock.is_file() and done.read_text().strip() == identity(lock) else 'outdated'
    return 'partial' if where.exists() or _dl(where).exists() else 'none'


def python(where: Path | None = None) -> Path:
    return (where or folder()) / 'python.exe'


def _dl(where: Path, addon: bool = False) -> Path:
    return where.with_name('voice-pack.addon-download' if addon else 'voice-pack.download')


def _files(spec: dict, addon: bool = False) -> list[dict]:
    """What the install downloads; the addon's without the pip wheel (fetched first, not counted in its progress)."""
    part = spec['addon'] if addon else spec
    return [*([] if addon else [spec['python'], spec['pip']]), *part['wheels'], *part.get('archives', [])]


def need_bytes(lock: Path | None = None, where: Path | None = None, addon: bool = False) -> int:
    """Free space the install (of the addon) needs: what is not downloaded yet, the installed size and a margin."""
    spec = json.loads((lock or lock_file()).read_text())
    dl = _dl(where or folder(), addon)
    files = _files(spec, addon) + [spec['pip']] * addon
    return sum(f['size'] - _have(dl, f) for f in files) + (spec['addon'] if addon else spec)['installed_bytes'] + MARGIN


def _name(f: dict) -> str:
    """The file name in voice-pack.download: the URL's last part, never a path (an encoded '../', L-44)."""
    name = urllib.parse.unquote(f['url'].rsplit('/', 1)[-1] if '/' in f['url'] else '')
    if name in ('', '.', '..') or any(c in name for c in '/\\:'):
        raise PackError(f'not a file name the pack can use: {f["url"]!r}')
    return name


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
            retries: int = 3, addon: bool = False) -> Path:
    """Install the pack, or with `addon` its addon into the installed pack; `progress(stage, done_bytes, total_bytes,
    text)` with stage 'cleaning', 'downloading', 'unpacking' (the pack's Python) or 'installing' (then `text` is pip's
    last line); `cancel` set: stops soon with Cancelled, the downloads so far stay for the next attempt. `run_python`: how
    to run the pack's Python (tests). Another install or removal still going: PackError."""
    lock, where = lock or lock_file(), where or folder()
    if not _BUSY.acquire(blocking=False):
        raise PackError('the voice pack is being installed or removed already (setup wizard or Components…): '
                        'wait until that ends')
    try:
        return (_install_addon if addon else _install)(progress, cancel or threading.Event(), lock, where, run_python,
                                                        retries)
    finally:
        _BUSY.release()


def _install(progress, cancel: threading.Event, lock: Path, where: Path, run_python, retries: int) -> Path:
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
    dl = _dl(where)
    dl.mkdir(parents=True, exist_ok=True)
    _room(dl, need_bytes(lock, where), 'the voice pack')
    _fetch_all(_files(spec), dl, retries, cancel, progress)
    progress('unpacking', 0, 0)
    shutil.unpack_archive(dl / _name(spec['python']), tmp, 'zip')
    site = tmp / 'Lib' / 'site-packages'
    site.mkdir(parents=True)
    (tmp / PTH).write_text('python312.zip\n.\nLib\\site-packages\nimport site\n')  # the runner's packages; .pth files
    _pip_install(spec, dl, site, tmp / 'pack-requirements.txt', run_python or [str(python(tmp))], cancel, progress, pip_tmp)
    _unpack(spec.get('archives', []), dl, site, progress)
    tmp.rename(where)
    (where / '.complete').write_text(identity(lock) + '\n')  # last: until here the pack is not there
    for p in (dl, pip_tmp):
        shutil.rmtree(p, ignore_errors=True)
    return where


def _install_addon(progress, cancel: threading.Event, lock: Path, where: Path, run_python, retries: int) -> Path:
    spec = json.loads(lock.read_text())
    if 'addon' not in spec:
        raise PackError('this version of the app has no addon for the voice pack')
    if state(lock, where) != 'ready':
        raise PackError('the addon goes into the voice pack: install the voice pack first')
    if state(lock, where, addon=True) == 'ready':
        return where
    progress('cleaning', 0, 0)
    lib, pip_tmp, dl = where / 'Lib', where.with_name('voice-pack.pip-tmp'), _dl(where, True)
    tmp = lib / f'{ADDON}.tmp'
    (where / ADDON_DONE).unlink(missing_ok=True)
    _addon_path(where, False)
    try:
        for p in (lib / ADDON, tmp, pip_tmp):  # half installed or outdated: from scratch
            _discard(p)
    except OSError as e:
        raise PackError(f'cannot move the old addon away ({e}): turn translation off first') from e
    for old in lib.glob(f'{ADDON}.old-*'):
        shutil.rmtree(old, ignore_errors=True)
    dl.mkdir(exist_ok=True)
    _room(dl, need_bytes(lock, where, addon=True), 'the addon')
    _fetch(spec['pip'], dl, retries, cancel, lambda n: None)
    _fetch_all(_files(spec, addon=True), dl, retries, cancel, progress)
    tmp.mkdir()
    pip_tmp.mkdir()
    _pip_install(spec['addon'], dl, tmp, pip_tmp / 'addon-requirements.txt', run_python or [str(python(where))], cancel,
                 progress, pip_tmp, spec['pip'])
    _unpack(spec['addon'].get('archives', []), dl, tmp, progress)
    tmp.rename(lib / ADDON)
    _addon_path(where, True)
    (where / ADDON_DONE).write_text(identity(lock, addon=True) + '\n')  # last
    for p in (dl, pip_tmp):
        shutil.rmtree(p, ignore_errors=True)
    return where


def uninstall(addon: bool = False, lock: Path | None = None, where: Path | None = None) -> None:
    """Remove the pack (its addon with it), or with `addon` only the addon: the marker goes first, so a removal that
    stops halfway (Windows: a file the runner has open) never leaves a 'ready' one. PackError: it could not."""
    where = where or folder()
    if not _BUSY.acquire(blocking=False):
        raise PackError('the voice pack is being installed or removed already: wait until that ends')
    try:
        (where / (ADDON_DONE if addon else '.complete')).unlink(missing_ok=True)
        if addon:
            _addon_path(where, False)
        names = ('addon-download', 'pip-tmp') if addon else ('download', 'addon-download', 'tmp', 'pip-tmp')
        try:
            for p in [*([where / 'Lib' / ADDON, where / 'Lib' / f'{ADDON}.tmp'] if addon else [where]),
                      *(where.with_name(f'voice-pack.{n}') for n in names)]:
                _discard(p)
        except OSError as e:
            raise PackError(f'cannot remove {"the addon" if addon else "the voice pack"} ({e}): turn translation off '
                            'first') from e
    finally:
        _BUSY.release()


def _addon_path(where: Path, on: bool) -> None:
    """Lib\\addon on the pack's Python path before Lib\\site-packages (`on`), or off it again."""
    if (pth := where / PTH).is_file():
        text = pth.read_text().replace('Lib\\addon\n', '')
        pth.write_text(text.replace('Lib\\site-packages\n', 'Lib\\addon\nLib\\site-packages\n') if on else text)


def _room(dl: Path, need: int, what: str) -> None:
    if (free := shutil.disk_usage(dl).free) < need:
        raise PackError(f'{what} needs {need / 2 ** 30:.1f} GB free on {dl.anchor or dl.parent}, '
                        f'there are {free / 2 ** 30:.1f} GB')


def _fetch_all(files: list[dict], dl: Path, retries: int, cancel: threading.Event, progress) -> None:
    total, done = sum(f['size'] for f in files), 0
    for f in files:
        name = _name(f)
        done += _fetch(f, dl, retries, cancel, lambda n, base=done, name=name: progress('downloading', base + n, total, name))
        progress('downloading', done, total, name)


def _pip_install(part: dict, dl: Path, target: Path, reqs: Path, run_python: list[str], cancel: threading.Event, progress,
                 pip_tmp: Path, pip: dict | None = None) -> None:
    """pip from its verified wheel installs `part`'s verified wheels from disk into `target`: no index, no deps."""
    reqs.write_text(''.join(f'{dl / _name(w)} --hash=sha256:{w["sha256"]}\n' for w in part['wheels']), 'utf-8')
    cmd = [*run_python, str(dl / _name(pip or part['pip']) / 'pip'), 'install', '--no-index', '--no-deps',
           '--require-hashes', '--only-binary', ':all:', '--no-cache-dir', '--disable-pip-version-check',
           '--no-warn-script-location', '--target', str(target), '-r', str(reqs)]
    pip_tmp.mkdir(exist_ok=True)  # pip's temp on the drive checked above: its moves are renames
    _pip(cmd, cancel, lambda line: progress('installing', 0, 0, line), pip_tmp)


def _unpack(archives: list[dict], dl: Path, site: Path, progress) -> None:
    for a in archives:  # 'folder': the package's folder in the archive, e.g. unidic-lite-1.0.8/unidic_lite
        progress('installing', 0, 0, f'unpacking {a["folder"]}')
        root, name = a['folder'].rsplit('/', 1)
        with tarfile.open(dl / _name(a)) as t:
            members = []
            for m in t.getmembers():
                if (m.name + '/').startswith(a['folder'] + '/'):
                    m.name = m.name[len(root) + 1:]
                elif m.isfile() and m.name.startswith(f'{root}/LICENSE') and '/' not in m.name[len(root) + 1:]:
                    m.name = f'{name}/{m.name[len(root) + 1:]}'  # the archive's own license files go with its folder
                else:
                    continue
                members.append(m)
            t.extractall(site, members, filter='data')


def _fetch(f: dict, dl: Path, retries: int, cancel: threading.Event, progress) -> int:
    """One file into `dl`, resumed from its .part, checked against its size and SHA-256."""
    dest = dl / _name(f)
    if dest.exists() and dest.stat().st_size == f['size'] and models.sha256(dest) == f['sha256']:
        return f['size']
    dest.unlink(missing_ok=True)  # its bytes are not the pinned ones (cut before they reached the disk, or put there: L-07)
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

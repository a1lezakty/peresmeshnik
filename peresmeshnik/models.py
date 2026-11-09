"""Model registry (models.toml) and downloads: pinned URLs, resume, SHA-256 check, license screen.

Engines ask for a model's local folder and never download anything themselves:

    from peresmeshnik import models
    folder = models.path('parakeet-tdt-0.6b-v3')   # raises ModelMissing until it is fetched

    peresmeshnik models list
    peresmeshnik models fetch silero-vad parakeet-tdt-0.6b-v3 opus-mt-en-ru [--accept-licenses]

Downloads are plain HTTPS without credentials (public Hugging Face files and this project's
GitHub releases). A folder counts as fetched once every file matched its hash; the marker
holds a fingerprint of the pins, so a registry update makes the model "missing" again.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import sys
import threading
import time
import tomllib
import urllib.request
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

from peresmeshnik import __version__, config

MARKER = '.verified'
RETRY_PAUSE_S = 2.0


class ModelMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Model:
    id: str
    role: str  # vad | asr | mt | tts | clone
    langs: tuple[str, ...]
    license: str
    license_url: str
    url: str  # prefix; a file's URL is url + its name
    files: dict  # name -> {'sha256': str, 'size': int}
    consent: bool = False  # noncommercial / territory-limited: explicit acceptance before download
    note: str = ''
    needs: str = ''  # 'gpu': runs live only with an NVIDIA GPU; without one languages.direction() does not offer it

    @property
    def size(self) -> int:
        return sum(f['size'] for f in self.files.values())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(''.join(f'{n}:{f["sha256"]}\n' for n, f in sorted(self.files.items())).encode()).hexdigest()


@cache
def registry() -> dict[str, Model]:
    data = tomllib.loads(resources.files('peresmeshnik').joinpath('models.toml').read_text('utf-8'))
    return {k: Model(id=k, langs=tuple(v.pop('langs')), **v) for k, v in data.items()}


def root() -> Path:
    return Path(os.environ.get('PERESMESHNIK_MODELS') or config.load().models_dir or config.cache_dir() / 'models')


def get(model_id: str) -> Model:
    try:
        return registry()[model_id]
    except KeyError:
        raise ModelMissing(f'unknown model {model_id!r}; known: {", ".join(registry())}') from None


def size(m: Model) -> int:
    """Bytes to download for the model: what the wizard and the tray show for a pair."""
    return m.size


def is_fetched(m: Model, base: Path | None = None) -> bool:
    marker = (base or root()) / m.id / MARKER
    return marker.exists() and marker.read_text().strip() == m.fingerprint


def path(model_id: str, base: Path | None = None) -> Path:
    """Local folder of a fetched and verified model. Offline: never downloads."""
    m = get(model_id)
    if not is_fetched(m, base):
        raise ModelMissing(f'model {model_id} is not downloaded: peresmeshnik models fetch {model_id}')
    return (base or root()) / model_id


def sha256(file: Path) -> str:
    h = hashlib.sha256()
    with open(file, 'rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _size(file: Path) -> int:
    return file.stat().st_size if file.exists() else 0


def _download(url: str, part: Path, size: int, progress):
    """Append to `part` from where it stopped (HTTP Range); restart when the server ignores the range.
    A connection closed mid-body just ends the read (no exception): the caller sees a short part."""
    have = _size(part)
    if have >= size:
        return
    headers = {'User-Agent': f'peresmeshnik/{__version__}'}
    if have:
        headers['Range'] = f'bytes={have}-'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        if r.status != 206 and have:
            progress(-have)  # the server sent the whole file again
            have = 0
        with open(part, 'ab' if have else 'wb') as f:
            while b := r.read(1 << 20):
                f.write(b)
                have += len(b)
                progress(len(b))


_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def fetch(model_id: str, base: Path | None = None, progress=lambda n: None, retries: int = 3) -> Path:
    """Download what is missing, check every SHA-256, then mark the folder as verified. One download of a model at a
    time in this process (the wizard and the tray may ask together): the second waits and finds it there."""
    with _locks_lock:
        lock = _locks.setdefault(model_id, threading.Lock())
    with lock:
        return _fetch(model_id, base, progress, retries)


def _fetch(model_id: str, base: Path | None, progress, retries: int) -> Path:
    m = get(model_id)
    folder = (base or root()) / m.id
    if is_fetched(m, base):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER).unlink(missing_ok=True)
    for name, f in m.files.items():
        dest = folder / name
        if dest.exists() and dest.stat().st_size == f['size'] and sha256(dest) == f['sha256']:
            progress(f['size'])
            continue
        part = dest.with_name(dest.name + '.part')
        progress(min(_size(part), f['size']))
        for attempt in range(retries):
            try:
                _download(m.url + name, part, f['size'], progress)
                err = 'connection closed early'
            except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, resets, IncompleteRead
                err = str(e)
            if _size(part) >= f['size']:
                break
            if attempt == retries - 1:  # the .part stays: the next fetch resumes it
                raise RuntimeError(f'{m.id}/{name}: download incomplete ({_size(part)} of {f["size"]} bytes, {err}); '
                                   'run fetch again to resume')
            print(f'{m.id}/{name}: {err}, resuming at {_size(part)} bytes', file=sys.stderr)
            time.sleep(RETRY_PAUSE_S * (attempt + 1))
        if _size(part) != f['size'] or sha256(part) != f['sha256']:  # too long or wrong at full size: start over
            part.unlink()
            raise RuntimeError(f'{m.id}/{name}: size or SHA-256 mismatch, download discarded')
        part.replace(dest)
    (folder / MARKER).write_text(m.fingerprint + '\n')
    return folder


def license_screen(ms: list[Model], accept: bool = False, ask=input, out=print) -> bool:
    """Show every license before a download; restricted ones need an explicit 'yes'."""
    for m in ms:
        out(f'  {m.id:28} {size(m) / 1e6:8.1f} MB  {m.license}  {m.license_url}' + (f'\n  {"":28} {m.note}' if m.note else ''))
    need = [m for m in ms if m.consent]
    if not need or accept:
        return True
    out('Restricted licenses (noncommercial or territory-limited) bind you as the user: '
        + ', '.join(f'{m.id} ({m.license})' for m in need))
    try:
        return ask('Type "yes" to accept them: ').strip().lower() == 'yes'
    except EOFError:  # no terminal: only --accept-licenses can accept
        return False


def cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog='peresmeshnik models', description='Model registry: list and download models.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='every model with size, license and status')
    f = sub.add_parser('fetch', help='download and verify models (shows the licenses first)')
    f.add_argument('ids', nargs='+')
    f.add_argument('--accept-licenses', action='store_true', help='accept restricted licenses without asking')
    a = ap.parse_args(argv)
    base = root()
    if a.cmd == 'list':
        print(f'models folder: {base}')
        for m in registry().values():
            state = 'ok' if is_fetched(m, base) else '-'
            print(f'{state:3} {m.id:28} {m.role:6} {",".join(m.langs):10} {size(m) / 1e6:8.1f} MB  {m.license}')
        return 0
    try:
        ms = [get(i) for i in dict.fromkeys(a.ids)]
    except ModelMissing as e:
        print(e, file=sys.stderr)
        return 2
    todo = [m for m in ms if not is_fetched(m, base)]
    if not todo:
        print('all present')
        return 0
    print(f'Downloading into {base}:', flush=True)  # before the progress lines on stderr
    if not license_screen(todo, a.accept_licenses, out=lambda t: print(t, flush=True)):
        print('not accepted, nothing downloaded', file=sys.stderr)
        return 1
    for m in todo:
        done, last = [0], [0.0]

        def progress(n, m=m, done=done, last=last):
            done[0] += n
            if time.monotonic() - last[0] > 2 or done[0] == size(m):
                last[0] = time.monotonic()
                print(f'\r{m.id}: {done[0] * 100 // max(size(m), 1):3d} %', end='', file=sys.stderr, flush=True)
        fetch(m.id, base, progress)
        print(f'\r{m.id}: ok      ', file=sys.stderr)
    return 0

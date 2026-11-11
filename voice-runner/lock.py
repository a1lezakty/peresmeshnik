"""Pin the Windows voice pack: requirements-windows.in -> requirements-windows.txt (uv, hashes) -> pack-windows.json.

    python voice-runner/lock.py [--python 3.12.10] [--pip 26.3] [--dir DIR NAME]

NAME (default windows) reads DIR/requirements-NAME.in and writes DIR/requirements-NAME.txt and DIR/pack-NAME.json;
tests/data/pack-tiny.json (numpy and soundfile) is how the Windows CI test installs a real pack in a minute.
A package in DATA has no wheel, only a source archive with its data (unidic-lite: the Japanese dictionary of XTTS's
text cleaning, through cutlet and fugashi): it goes to `archives` with the folder the installer unpacks from it.

pack-windows.json is what the app's installer (peresmeshnik/local/voicepack.py) reads: the embeddable CPython zip from
python.org, a pip wheel and one wheel per package for cp312 win_amd64, each with its URL, SHA-256 and size, and the
bytes they take installed (read from each wheel's zip directory with HTTP Range, nothing big is downloaded). Every
wheel's SHA-256 must be one of uv's hashes. Its own SHA-256 is the pack's identity: a new lock means a new pack.
docopt is left out: no wheel, and only num2words' command-line script imports it.

    python voice-runner/lock.py --addon windows|linux|linux-cuda

The pack's addon "Qwen3 + Chatterbox": requirements-PLATFORM-addon.in -> requirements-PLATFORM-addon[-cuda].txt, every
pin of the pack's own lock kept but those requirements-addon-overrides.txt replaces (its packages' own pins: torch 2.6,
numpy<2, ...; transformers 5.16.1 for Qwen3), and what the overrides leave out never installed (gradio, GPL pykakasi).
windows also writes the section `addon` of pack-windows.json: the wheels (and source archives) of what the addon adds
or changes, `installed_bytes`, `models` (the registry's `addon = true` rows) and `vram_gb` (their largest `vram`); the
pack's own part stays as it is (its identity too). Each .txt says in its first lines how it is installed.
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TORCH_INDEX = 'https://download.pytorch.org/whl/cu128'
SKIP = ('docopt',)
DATA = {'unidic-lite': 'unidic_lite', 'sox': 'sox'}  # package -> its folder in the source archive (sox: pure Python)
OVERRIDES = HERE / 'requirements-addon-overrides.txt'
PLATFORM = {'windows': ['--python-platform', 'x86_64-pc-windows-msvc', '--extra-index-url', TORCH_INDEX,
                        '--index-strategy', 'unsafe-best-match', *[x for s in SKIP for x in ('--no-emit-package', s)]],
            'linux': ['--python-platform', 'x86_64-manylinux_2_28', '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                      '--index-strategy', 'unsafe-best-match', '--emit-index-url'],
            'linux-cuda': ['--python-platform', 'x86_64-manylinux_2_28', '--emit-index-url']}
INSTALL = {'windows': 'the app installs it (local/voicepack.py install(addon=True))',
           'linux': 'uv pip install --no-deps --require-hashes --index-strategy unsafe-best-match -r {txt} --build-constraints {txt}',
           'linux-cuda': 'uv pip install --no-deps --require-hashes -r {txt} --build-constraints {txt}'}
PIN = r'^([A-Za-z0-9_.\-]+)==(\S+)((?:\s+--hash=sha256:[0-9a-f]{64})+)'


class Remote(io.RawIOBase):
    """A remote file read with HTTP Range: zipfile only needs its end (the directory)."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = (off, self.pos + off, self.size + off)[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        d = urllib.request.urlopen(req, timeout=60).read()
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def unpacked(url: str, size: int) -> int:
    with zipfile.ZipFile(io.BufferedReader(Remote(url, size), 1 << 16)) as z:
        return sum(i.file_size for i in z.infolist())


def fits(filename: str) -> bool:
    py, abi, plat = filename[:-4].split('-')[-3:]
    return plat in ('any', 'win_amd64') and abi in ('none', 'abi3', 'cp312') and \
        any(t == 'cp312' or t.startswith('py3') or (abi == 'abi3' and t.startswith('cp3')) for t in py.split('.'))


def wheel(name: str, version: str) -> dict:
    if '+' in version:  # the CUDA builds: PyTorch's own index, the hash in the link
        page = urllib.request.urlopen(f'{TORCH_INDEX}/{name}/', timeout=60).read().decode()
        fn = f'{name}-{version}-cp312-cp312-win_amd64.whl'.replace('+', '%2B')
        sha = re.search(re.escape(fn) + r'#sha256=([0-9a-f]{64})', page).group(1)
        url = f'{TORCH_INDEX}/{fn}'
        size = int(urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60).headers['Content-Length'])
        return {'name': name, 'url': url, 'sha256': sha, 'size': size}
    d = json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=60))
    ws = sorted((u for u in d['urls'] if u['filename'].endswith('.whl') and fits(u['filename'])),
                key=lambda u: (u['filename'].endswith('-any.whl'), 'cp312' not in u['filename'], 'abi3' not in u['filename']))
    if not ws:
        raise SystemExit(f'{name}=={version}: no wheel for cp312 win_amd64')
    return {'name': name, 'url': ws[0]['url'], 'sha256': ws[0]['digests']['sha256'], 'size': ws[0]['size']}


def archive(name: str, version: str, hashes: str) -> tuple[dict, int]:
    """The source archive of a data-only package and the bytes its folder takes unpacked (it is downloaded: ~50 MB)."""
    d = json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=60))
    u = next(u for u in d['urls'] if u['packagetype'] == 'sdist')
    if u['digests']['sha256'] not in hashes:
        raise SystemExit(f'{name}=={version}: {u["url"]} is not one of the locked hashes')
    body = urllib.request.urlopen(u['url'], timeout=300).read()
    if hashlib.sha256(body).hexdigest() != u['digests']['sha256']:
        raise SystemExit(f'{name}=={version}: the download does not match its SHA-256')
    with tarfile.open(fileobj=io.BytesIO(body)) as t:
        top = t.getmembers()[0].name.split('/')[0] + '/' + DATA[name]
        size = sum(m.size for m in t.getmembers() if (m.name + '/').startswith(top + '/'))
    return {'name': name, 'url': u['url'], 'sha256': u['digests']['sha256'], 'size': u['size'], 'folder': top}, size


def pinned(pins: list[tuple[str, str, str]]) -> tuple[list[dict], list[dict], int]:
    """The wheels and source archives of `pins` (name, version, its hashes) and the bytes they take installed."""
    wheels, archives, total = [], [], 0
    for name, version, hashes in pins:
        if name in DATA:
            arc, size = archive(name, version, hashes)
            archives.append(arc)
            total += size
            print(f'{name}=={version}: {arc["size"] / 1e6:.1f} MB (source archive, {DATA[name]}/ unpacked)', flush=True)
            continue
        w = wheel(name, version)
        if w['sha256'] not in hashes:
            raise SystemExit(f'{name}=={version}: {w["url"]} is not one of the locked hashes')
        total += unpacked(w['url'], w['size'])
        wheels.append(w)
        print(f'{name}=={version}: {w["size"] / 1e6:.1f} MB', flush=True)
    return wheels, archives, total


def pins_of(txt: Path) -> list[tuple[str, str, str]]:
    return re.findall(PIN, txt.read_text().replace('\\\n', ''), re.M)


def addon(d: Path, name: str) -> None:
    """The addon's lock for `name` (windows, linux, linux-cuda) over the pack's: see the module's docstring."""
    plat = name.split('-')[0]
    base, txt = d / f'requirements-{name}.txt', d / f'requirements-{plat}-addon{name[len(plat):]}.txt'
    # an override that caps the version (==, <) replaces the pack's pin; a floor alone (>=) lifts only the pin of the
    # package that asks for it, and the pack's pin stays
    replaced = {m.group(1).lower() for m in re.finditer(r'^([A-Za-z0-9_.\-]+)([^;#\n]*)', OVERRIDES.read_text(), re.M)
                if '==' in m.group(2) or '<' in m.group(2)}
    with tempfile.TemporaryDirectory() as tmp:  # the pack's pins as constraints, but those the overrides replace
        keep = Path(tmp) / 'constraints.txt'
        keep.write_text(''.join(f'{n}=={v}\n' for n, v, _ in pins_of(base) if n.lower() not in replaced))
        subprocess.run([shutil.which('uv') or 'uv', 'pip', 'compile', '-q', f'requirements-{plat}-addon.in', '-o', txt.name,
                        '-c', str(keep), '--overrides', OVERRIDES.name, '--python-version', '3.12', '--generate-hashes',
                        '--custom-compile-command', f'python voice-runner/lock.py --addon {name}', *PLATFORM[name]],
                       check=True, cwd=d)  # relative paths in its annotations
        lines = txt.read_text().replace(str(keep), base.name).split('\n')  # the pack's lock, not a temp file, in the notes
    lines.insert(2, f'# Installed with: {INSTALL[name].format(txt=f"voice-runner/{txt.name}")}')
    txt.write_text('\n'.join(lines))
    if name != 'windows':
        return
    have = {(n, v) for n, v, _ in pins_of(base)}
    wheels, archives, total = pinned([p for p in pins_of(txt) if p[:2] not in have])  # what it adds or changes
    reg = tomllib.loads((d.parent / 'peresmeshnik' / 'models.toml').read_text('utf-8'))
    ms = [k for k, v in reg.items() if v.get('addon')]
    pack = d / 'pack-windows.json'
    lock = json.loads(pack.read_text())
    lock['addon'] = {'wheels': wheels, 'archives': archives, 'installed_bytes': total, 'models': ms,
                     'vram_gb': max(reg[m].get('vram', 0) for m in ms)}
    pack.write_text(json.dumps(lock, indent=1) + '\n')
    dl = sum(w['size'] for w in wheels + archives)
    print(f'addon: {len(wheels)} wheels, {len(archives)} archives: download {dl / 1e9:.2f} GB, installed {total / 1e9:.2f} GB')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--python', default='3.12.10', help='the last 3.12 with Windows binaries')
    ap.add_argument('--pip', default='26.3')
    ap.add_argument('--dir', type=Path, default=HERE)
    ap.add_argument('--addon', choices=sorted(PLATFORM), help='the addon\'s lock over the pack\'s')
    ap.add_argument('name', nargs='?', default='windows')
    a = ap.parse_args()
    if a.addon:
        return addon(a.dir, a.addon)
    txt = a.dir / f'requirements-{a.name}.txt'
    subprocess.run([shutil.which('uv') or 'uv', 'pip', 'compile', str(a.dir / f'requirements-{a.name}.in'), '-o', str(txt),
                    '--python-version', '3.12', '--generate-hashes', '--no-header', *PLATFORM['windows']], check=True)
    url = f'https://www.python.org/ftp/python/{a.python}/python-{a.python}-embed-amd64.zip'
    body = urllib.request.urlopen(url, timeout=120).read()
    lock = {'python': {'name': 'python', 'url': url, 'sha256': hashlib.sha256(body).hexdigest(), 'size': len(body)},
            'pip': wheel('pip', a.pip)}
    lock['wheels'], lock['archives'], total = pinned(pins_of(txt))
    lock['installed_bytes'] = total + sum(i.file_size for i in zipfile.ZipFile(io.BytesIO(body)).infolist())
    if (old := a.dir / f'pack-{a.name}.json').is_file() and 'addon' in (spec := json.loads(old.read_text())):
        lock['addon'] = spec['addon']  # last, outside the pack's identity
        print('the addon is kept as it was: lock it again over the new pins (--addon windows)')
    (a.dir / f'pack-{a.name}.json').write_text(json.dumps(lock, indent=1) + '\n')
    dl = lock['python']['size'] + lock['pip']['size'] + sum(w['size'] for w in lock['wheels'] + lock['archives'])
    print(f'{len(lock["wheels"])} wheels: download {dl / 1e9:.2f} GB, installed {lock["installed_bytes"] / 1e9:.2f} GB')


if __name__ == '__main__':
    main()

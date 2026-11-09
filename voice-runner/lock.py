"""Pin the Windows voice pack: requirements-windows.in -> requirements-windows.txt (uv, hashes) -> pack-windows.json.

    python voice-runner/lock.py [--python 3.12.10] [--pip 26.3] [--dir DIR NAME]

NAME (default windows) reads DIR/requirements-NAME.in and writes DIR/requirements-NAME.txt and DIR/pack-NAME.json;
tests/data/pack-tiny.json (numpy and soundfile) is how the Windows CI test installs a real pack in a minute.

pack-windows.json is what the app's installer (peresmeshnik/local/voicepack.py) reads: the embeddable CPython zip from
python.org, a pip wheel and one wheel per package for cp312 win_amd64, each with its URL, SHA-256 and size, and the
bytes they take installed (read from each wheel's zip directory with HTTP Range, nothing big is downloaded). Every
wheel's SHA-256 must be one of uv's hashes. Its own SHA-256 is the pack's identity: a new lock means a new pack.
docopt is left out: no wheel, and only num2words' command-line script imports it.
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TORCH_INDEX = 'https://download.pytorch.org/whl/cu128'
SKIP = ('docopt',)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--python', default='3.12.10', help='the last 3.12 with Windows binaries')
    ap.add_argument('--pip', default='26.3')
    ap.add_argument('--dir', type=Path, default=HERE)
    ap.add_argument('name', nargs='?', default='windows')
    a = ap.parse_args()
    txt = a.dir / f'requirements-{a.name}.txt'
    subprocess.run([shutil.which('uv') or 'uv', 'pip', 'compile', str(a.dir / f'requirements-{a.name}.in'), '-o', str(txt),
                    '--python-platform', 'x86_64-pc-windows-msvc', '--python-version', '3.12', '--generate-hashes',
                    '--extra-index-url', TORCH_INDEX, '--index-strategy', 'unsafe-best-match', '--no-header',
                    *[x for s in SKIP for x in ('--no-emit-package', s)]], check=True)
    pins = re.findall(r'^([A-Za-z0-9_.\-]+)==(\S+)((?:\s+--hash=sha256:[0-9a-f]{64})+)', txt.read_text().replace('\\\n', ''), re.M)
    url = f'https://www.python.org/ftp/python/{a.python}/python-{a.python}-embed-amd64.zip'
    body = urllib.request.urlopen(url, timeout=120).read()
    lock = {'python': {'name': 'python', 'url': url, 'sha256': hashlib.sha256(body).hexdigest(), 'size': len(body)},
            'pip': wheel('pip', a.pip), 'wheels': []}
    total = sum(i.file_size for i in zipfile.ZipFile(io.BytesIO(body)).infolist())
    for name, version, hashes in pins:
        w = wheel(name, version)
        if w['sha256'] not in hashes:
            raise SystemExit(f'{name}=={version}: {w["url"]} is not one of the locked hashes')
        total += unpacked(w['url'], w['size'])
        lock['wheels'].append(w)
        print(f'{name}=={version}: {w["size"] / 1e6:.1f} MB', flush=True)
    lock['installed_bytes'] = total
    (a.dir / f'pack-{a.name}.json').write_text(json.dumps(lock, indent=1) + '\n')
    dl = lock['python']['size'] + lock['pip']['size'] + sum(w['size'] for w in lock['wheels'])
    print(f'{len(lock["wheels"])} wheels: download {dl / 1e9:.2f} GB, installed {total / 1e9:.2f} GB')


if __name__ == '__main__':
    main()

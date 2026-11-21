"""Updates: the newest published release of this repository on GitHub, its file for this computer downloaded, checked
and installed over the running app, which then starts again.

- Which release: the highest vX.Y.Z among the published ones (drafts and prereleases do not count, nor other tags such as
  the models' releases).
- Its file: the .dmg on a Mac (Apple Silicon), the -setup.exe on Windows (x64); none for Linux (from source) or the .zip
  (unpacked by hand): there the release page is opened instead (`installable`).
- Checked: its SHA-256 against the release's .sha256 file, and GitHub's build attestation for that digest made by this
  repository's release.yml from the release's tag (the attestations API; the statement is read, its Sigstore signature
  is not verified: the answer is trusted by TLS to GitHub, as the download itself). A mismatch removes the file.
- Installed: macOS copies the new .app out of the mounted .dmg next to the running one, and after the app quits a small
  script swaps the two (the old one back when the swap fails) and opens it again. Windows runs the installer with no
  questions
  (it closes the app, keeps the settings; the old install folder is kept aside and put back when the install fails)
  and starts the app again, old or new; it is started only with room for it on the disk.
  Settings, voices, the voice pack and the models live in the user's folders, never in the app: an update touches none.

$PERESMESHNIK_UPDATE_API points the whole check at another server (tests/fake_github.py in the tests of the built app).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from peresmeshnik import __version__, config

REPO = 'a1lezakty/peresmeshnik'
WORKFLOW = '.github/workflows/release.yml'
DAY_S = 24 * 3600
FILES = {('darwin', 'arm64'): '-macos-arm64.dmg',  # the names release.yml gives them
         ('win32', 'amd64'): '-windows-x64-setup.exe', ('win32', 'arm64'): '-windows-x64-setup.exe'}  # x64compatible


class Failed(Exception):
    """Said to the user as it is."""


@dataclass
class Release:
    version: str
    page: str  # its GitHub page
    notes: str = ''
    files: dict[str, str] = field(default_factory=dict)  # name -> download URL
    sizes: dict[str, int] = field(default_factory=dict)


def api() -> str:
    return (os.environ.get('PERESMESHNIK_UPDATE_API') or 'https://api.github.com').rstrip('/')


def version(tag: str) -> tuple[int, ...] | None:
    """'v0.3.10' -> (0, 3, 10); None for anything else ('models-2', 'v0.4.0-rc1')."""
    m = re.fullmatch(r'v?(\d+(?:\.\d+)*)', tag.strip())
    return tuple(map(int, m[1].split('.'))) if m else None


def newer(rel: Release | None, current: str = '') -> bool:
    return bool(rel) and (version(rel.version) or ()) > (version(current or __version__) or ())


def _secure(url: str) -> str:
    local = 'PERESMESHNIK_UPDATE_API' in os.environ and urllib.parse.urlsplit(url).hostname in ('127.0.0.1', 'localhost')
    if not url.startswith('https://') and not local:  # plain http only to a fake on this computer
        raise Failed(f'not over HTTPS: {url}')
    return url


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # a release file: on to GitHub's file server
        return super().redirect_request(req, fp, code, msg, headers, _secure(newurl))


_opener = urllib.request.build_opener(_Redirect)


def _get(url: str, accept: str = 'application/vnd.github+json', timeout: float = 20):
    req = urllib.request.Request(_secure(url), headers={'Accept': accept, 'User-Agent': f'peresmeshnik/{__version__}'})
    return _opener.open(req, timeout=timeout)


def _json(url: str):
    with _get(url) as r:
        return json.loads(r.read())


def latest() -> Release | None:
    """The newest published release, or None. Raises on a network error (the caller decides whether to say it)."""
    rels = [r for r in _json(f'{api()}/repos/{REPO}/releases?per_page=100')
            if not r.get('draft') and not r.get('prerelease') and version(r.get('tag_name', ''))]
    if not rels:
        return None
    r = max(rels, key=lambda r: version(r['tag_name']))
    assets = r.get('assets') or []
    return Release('.'.join(map(str, version(r['tag_name']))), r.get('html_url') or f'https://github.com/{REPO}/releases',
                   r.get('body') or '', {a['name']: a['browser_download_url'] for a in assets},
                   {a['name']: a.get('size', 0) for a in assets})


def file_for(rel: Release, system: str = sys.platform, machine: str = platform.machine()) -> str | None:
    """The release's file for this OS and architecture, when it has one with its .sha256."""
    suffix = FILES.get((system, machine.lower()))
    name = f'Peresmeshnik-{rel.version}{suffix}' if suffix else ''
    return name if name in rel.files and f'{name}.sha256' in rel.files else None


def app_path() -> Path | None:
    """What an update replaces: the .app (macOS) or the install folder (Windows) of this frozen app."""
    if not getattr(sys, 'frozen', False):
        return None
    exe = Path(sys.executable).resolve()
    return exe.parents[2] if sys.platform == 'darwin' and exe.parents[2].suffix == '.app' else exe.parent


def installable() -> str:
    """'' when this copy of the app can be updated in place, else why not (the release page is opened instead)."""
    if sys.platform.startswith('linux'):
        return 'there is no Linux build: update the source (git pull)'
    if not (app := app_path()):
        return 'it runs from source: update the source (git pull)'
    if sys.platform == 'darwin':
        if app.suffix != '.app':
            return f'it does not run from an .app ({app})'
        if not os.access(app.parent, os.W_OK):  # the disk image, App Translocation, /Applications of a standard user
            return f'{app.parent} cannot be written by you: drag the new version to Applications by hand'
    elif not (app / 'unins000.exe').exists():
        return 'it runs from the .zip, not from the installer: unpack the new .zip by hand'
    return ''


def folder() -> Path:
    return config.cache_dir() / 'updates'


def _state_file() -> Path:
    return config.cache_dir() / 'update.json'


def state() -> dict:
    try:
        return json.loads(_state_file().read_text('utf-8'))
    except (OSError, ValueError):
        return {}


def remember(rel: Release | None) -> None:
    """The time of this check and what it found (an offer that survives a restart without asking GitHub again)."""
    data = {'checked': time.time(), **({'release': asdict(rel)} if rel else {})}
    try:
        _state_file().parent.mkdir(parents=True, exist_ok=True)
        _state_file().write_text(json.dumps(data), 'utf-8')
    except OSError:
        pass


def known() -> Release | None:
    """The release the last check found, if it is newer than this app."""
    try:
        rel = Release(**state()['release'])
    except (KeyError, TypeError):
        return None
    return rel if newer(rel) else None


def due_in(now: float | None = None) -> float:
    """Seconds until the next daily check."""
    return max(0.0, state().get('checked', 0) + DAY_S - (time.time() if now is None else now))


def _sha_of(text: str, name: str) -> str:
    for line in text.splitlines():  # "<hex>  <name>", as shasum / sha256sum write it
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip('*') == name and re.fullmatch(r'[0-9a-f]{64}', parts[0].lower()):
            return parts[0].lower()
    raise Failed(f'{name}.sha256 of the release does not give its SHA-256')


def _unsnappy(b: bytes) -> bytes:
    """Snappy's raw block format, as the attestations' `bundle_url` serves a bundle that the API does not give inline:
    the length as a varint, then literals and copies of what came before. Raises ValueError / IndexError on garbage."""
    n = shift = i = 0
    while True:
        n, shift, i = n | (b[i] & 0x7f) << shift, shift + 7, i + 1
        if b[i - 1] < 0x80:
            break
    out = bytearray()
    while i < len(b):
        tag, i = b[i], i + 1
        if tag & 3 == 0:  # a literal: its length - 1 in the tag, or in the 1-4 bytes after it
            ln = tag >> 2
            if ln >= 60:
                ln, i = int.from_bytes(b[i:i + ln - 59], 'little'), i + ln - 59
            out += b[i:i + ln + 1]
            i += ln + 1
            continue
        if tag & 3 == 1:
            ln, off, i = 4 + (tag >> 2 & 7), (tag >> 5) << 8 | b[i], i + 1
        else:
            k = 2 if tag & 3 == 2 else 4
            ln, off, i = (tag >> 2) + 1, int.from_bytes(b[i:i + k], 'little'), i + k
        if not 0 < off <= len(out):
            raise ValueError('a Snappy copy from before the start')
        start = len(out) - off
        for j in range(ln):  # may overlap what it writes
            out.append(out[start + j])
    if len(out) != n:
        raise ValueError(f'Snappy: {len(out)} bytes, not {n}')
    return bytes(out)


def attested(digest: str, ver: str) -> None:
    """GitHub's build attestation of this digest: made by this repository's release.yml from the tag v<ver>. Raises Failed."""
    try:
        found = _json(f'{api()}/repos/{REPO}/attestations/sha256:{digest}').get('attestations') or []
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        found = []
    for a in found:
        try:
            if not (bundle := a.get('bundle')):
                with _get(a['bundle_url'], 'application/octet-stream') as r:
                    bundle = json.loads(_unsnappy(r.read(1 << 20)))  # a bundle is a few KB
            st = json.loads(base64.b64decode((bundle.get('dsseEnvelope') or {}).get('payload', '')))
        except (KeyError, ValueError, IndexError, AttributeError, TypeError, OSError):  # the next one may do
            continue
        wf = st.get('predicate', {}).get('buildDefinition', {}).get('externalParameters', {}).get('workflow', {})
        if (st.get('predicateType') == 'https://slsa.dev/provenance/v1'
                and any(s.get('digest', {}).get('sha256') == digest for s in st.get('subject', []))
                and wf.get('repository') == f'https://github.com/{REPO}' and wf.get('path') == WORKFLOW
                and wf.get('ref') == f'refs/tags/v{ver}'):
            return
    raise Failed(f'GitHub has no build attestation of it by {WORKFLOW} of {REPO} from v{ver}: not installed')


def download(rel: Release, name: str, progress=lambda done, total: None) -> Path:
    """The release's file, checked, in `folder()` (earlier downloads there are removed first). Raises Failed (the file
    removed) on a SHA-256 or attestation mismatch, OSError / URLError on the network."""
    with _get(rel.files[f'{name}.sha256'], 'application/octet-stream') as r:
        want = _sha_of(r.read(4096).decode('utf-8', 'replace'), name)
    d = folder()
    d.mkdir(parents=True, exist_ok=True)
    for old in d.iterdir():
        if old.is_file():
            old.unlink(missing_ok=True)
    part, h, done = d / f'{name}.part', hashlib.sha256(), 0
    try:
        with _get(rel.files[name], 'application/octet-stream', timeout=60) as r, open(part, 'wb') as f:
            total = int(r.headers.get('Content-Length') or rel.sizes.get(name) or 0)
            while chunk := r.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                progress(done, total)
        if h.hexdigest() != want:
            raise Failed(f'{name}: its SHA-256 is not the one the release gives (a damaged or replaced file): not installed')
        attested(want, rel.version)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return part.replace(d / name)


# macOS, after the app has quit: the new .app in, the old one out (back when that fails), the result for the next start,
# then it starts again. $1 pid, $2 the .app, $3 the new .app beside it, $4 the result file, the rest: `open`'s options
SWAP = r'''pid=$1 app=$2 new=$3 res=$4
shift 4
i=0
while kill -0 "$pid" 2>/dev/null; do
  i=$((i + 1))
  if [ $i -gt 600 ]; then echo 'the app did not quit within 2 minutes' > "$res"; rm -rf "$new"; exit 1; fi
  sleep 0.2
done
old="$app.old-$$"
if mv "$app" "$old" 2>> "$res"; then
  if mv "$new" "$app" 2>> "$res"; then rm -rf "$old"; echo ok > "$res"; else mv "$old" "$app"; fi
fi
rm -rf "$new"
open "$@" "$app"
'''


def pending() -> Path:
    return folder() / 'pending.json'


def prepare(file: Path) -> Path:
    """What `hand_over` starts from: on macOS the new .app copied out of the .dmg next to the running one (the same
    volume: the swap is a rename), the old one untouched; the installer itself on Windows. Raises Failed."""
    if sys.platform != 'darwin':
        return file
    new = staged()
    if new.exists():
        shutil.rmtree(new)
    try:
        with tempfile.TemporaryDirectory(prefix='peresmeshnik-update-') as mnt:
            r = subprocess.run(['hdiutil', 'attach', str(file), '-nobrowse', '-readonly', '-noautoopen', '-mountpoint', mnt],
                               capture_output=True, text=True, timeout=300)
            if r.returncode:
                raise Failed(f'the .dmg did not open: {r.stderr.strip()[-300:]}')
            try:
                r = subprocess.run(['ditto', f'{mnt}/Peresmeshnik.app', str(new)], capture_output=True, text=True,
                                   timeout=900)
            finally:
                subprocess.run(['hdiutil', 'detach', mnt, '-force'], capture_output=True, timeout=120)
        if r.returncode or not (new / 'Contents' / 'MacOS' / 'Peresmeshnik').exists():
            raise Failed(f'the new app could not be copied next to {app_path()}: {r.stderr.strip()[-300:]}')
    except BaseException:  # a timeout too: no half copy left next to the app
        shutil.rmtree(new, ignore_errors=True)
        raise
    return new


def staged() -> Path | None:
    """The new .app `prepare` leaves next to the running one until the swap (macOS)."""
    app = app_path()
    return app.with_name(f'.{app.stem}-update.app') if sys.platform == 'darwin' and app else None


def hand_over(rel: Release, ready: Path) -> subprocess.Popen:
    """Starts what replaces the app once it quits (`prepare`'s result): on macOS the caller quits the app right after, on
    Windows the installer closes it. Raises Failed / OSError when it cannot start: nothing changed then."""
    # Windows: the installer keeps the old folder aside until the new one is in (Peresmeshnik.iss): room for both;
    # installed it is ~4.5 times its installer
    if sys.platform != 'darwin' and shutil.disk_usage(app_path()).free < 6 * ready.stat().st_size:
        raise Failed('the disk has too little room for it')
    res = folder() / 'result.txt'
    res.unlink(missing_ok=True)
    pending().write_text(json.dumps({'version': rel.version, 'result': str(res)}), 'utf-8')
    try:
        return _start(ready, res)
    except BaseException:
        pending().unlink(missing_ok=True)  # nothing started: nothing to say at the next start
        raise


def _start(ready: Path, res: Path) -> subprocess.Popen:
    if sys.platform == 'darwin':  # `open` gets the app's own environment ($PERESMESHNIK_CONFIG, ...) as LaunchServices does not
        env = [a for k, v in os.environ.items() if k.startswith('PERESMESHNIK_') for a in ('--env', f'{k}={v}')]
        return subprocess.Popen(['/bin/sh', '-c', SWAP, 'peresmeshnik-update', str(os.getpid()), str(app_path()), str(ready),
                          str(res), *env], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True)
    else:  # the installer closes the app (WM_CLOSE: Peresmeshnik.iss), and with /Relaunch=1 starts it again, old or new;
        # /SILENT: its progress shows, it asks nothing
        return subprocess.Popen([str(ready), '/SILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/NOCANCEL', '/Relaunch=1',
                          f'/LOG={res}'], close_fds=True,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)


def outcome() -> str:
    """After an update was handed over, once at the next start: what came of it ('' when there was none). A new .app left
    staged by an update that did not happen (translation went on, the app was quit) goes too."""
    if (new := staged()) and new.exists():
        shutil.rmtree(new, ignore_errors=True)
    try:
        p = json.loads(pending().read_text('utf-8'))
        pending().unlink()
    except (OSError, ValueError):
        return ''
    if p.get('version') == __version__:
        for f in folder().iterdir():  # the installer or the .dmg (on Windows the installer may still hold its file)
            try:
                f.unlink()
            except OSError:
                pass
        return f'Updated to {__version__}.'
    try:
        why = Path(p.get('result', '')).read_text('utf-8', 'replace').strip()
    except OSError:
        why = ''
    why = '' if why == 'ok' else why.splitlines()[-1][:200] if why and sys.platform == 'darwin' else \
        f'its log: {p.get("result")}' if why else ''
    hint = (' If macOS said Peresmeshnik was prevented from modifying apps, allow it in System Settings → Privacy & '
            'Security → App Management and update again.' if 'not permitted' in why else '')
    return f'The update to {p.get("version")} did not install{f" ({why})" if why else ""}: {__version__} goes on working.{hint}'

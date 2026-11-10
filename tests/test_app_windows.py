"""The built Windows installer and .zip (release.yml), on the runner:
- the installer runs silently for the current user: the files, a Start menu shortcut, no start at sign-in unless
  ticked, an uninstall entry; the .zip holds the same folder;
- the installed app starts (first run: the wizard, a sample subtitle with --show-overlay), its tray icon is in the
  notification area, it idles 60 s under the same 0.13 s CPU limit as tests/test_idle_cpu.py, and taskkill (WM_CLOSE)
  ends it with exit code 0; started from its Start menu shortcut as a user does, the same;
- an update over the running app and the uninstaller both close it cleanly first; the uninstaller removes it all;
- the frozen CLI (from the .zip unpacked to a folder with spaces) fetches a model over HTTPS and translates a
  recording with the bundled engines and the bundled Piper runner (needs the work-gate models in
  $PERESMESHNIK_MODELS); the frozen cloud direction gets as far as Google refusing a fake key.

    PERESMESHNIK_SETUP=dist\\Peresmeshnik-0.2.0-windows-x64-setup.exe PERESMESHNIK_ZIP=dist\\...zip pytest -s tests/test_app_windows.py
"""
import fnmatch
import json
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import psutil
import pytest

SETUP, ZIP = os.environ.get('PERESMESHNIK_SETUP'), os.environ.get('PERESMESHNIK_ZIP')
pytestmark = pytest.mark.skipif(not (SETUP and ZIP and sys.platform == 'win32'),
                                reason='needs the built installer and .zip: PERESMESHNIK_SETUP, PERESMESHNIK_ZIP')
if sys.platform == 'win32':
    import winreg

    from test_windows import describe, pid_of, toplevel, tray_icon_rect  # noqa: E402

    from peresmeshnik.ui import windows
DATA = Path(__file__).parent / 'data'
WINDOW_S = float(os.environ.get('IDLE_WINDOW_S', 60))
APP_ID = '{F900FAE6-BF0A-4AA3-B66D-0CE8539E4E55}_is1'
UNINSTALL = rf'Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_ID}'
PROGRAMS = Path(os.environ.get('APPDATA', '')) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'
SHORTCUT, STARTUP = PROGRAMS / 'Peresmeshnik.lnk', PROGRAMS / 'Startup' / 'Peresmeshnik.lnk'
DIR = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Peresmeshnik'  # per-user default: {autopf}


def env(tmp_path=None) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'VIRTUAL_ENV'))}
    if tmp_path:
        e['PERESMESHNIK_CONFIG'] = str(tmp_path / 'settings.toml')  # no file: a first run, the wizard opens
    return e


def cpu_s(proc) -> float:
    return sum(sum(p.cpu_times()[:2]) for p in [proc, *proc.children(recursive=True)])


def setup(*args: str) -> None:
    r = subprocess.run([SETUP, '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', *args], timeout=300)
    assert r.returncode == 0, f'setup exited {r.returncode}'


def uninstall_key():
    try:
        return winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL)
    except OSError:
        return None


@pytest.fixture(scope='module')
def installed(tmp_path_factory):
    log = tmp_path_factory.mktemp('setup') / 'setup.log'
    t0 = time.perf_counter()
    setup(f'/LOG={log}')
    print(f'silent install: {time.perf_counter() - t0:.1f} s')
    assert (DIR / 'Peresmeshnik.exe').exists(), log.read_text('utf-8', 'replace')[-3000:]
    return DIR


def app_windows(pid: int) -> list[tuple[int, str]]:
    return [(h, describe(h)) for h in toplevel() if pid_of(h) == pid]


def tkill(pid: int) -> None:
    r = subprocess.run(['taskkill', '/PID', str(pid)], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_installed_files_shortcut_and_uninstall_entry(installed):
    import peresmeshnik
    for f in ('Peresmeshnik.exe', 'piper-runner.exe', 'unins000.exe', 'licenses/Peresmeshnik-LICENSE.txt',
              'licenses/piper-runner-LICENSE.txt',
              'licenses/Python-LICENSE.txt',  # Python's, with OpenSSL, libffi, sqlite of its build
              'licenses/PySide6_Essentials/LGPL-3.0.txt', 'licenses/PySide6_Essentials/GPL-3.0.txt',  # Qt: its wheel has
              'licenses/shiboken6/LGPL-3.0.txt',  # only the commercial option's text
              'licenses/ctranslate2/libiomp5md-LICENSE.txt', 'licenses/sounddevice/PortAudio-LICENSE.txt',
              *(f'_internal/voice-runner/{f}' for f in ('voice_runner.py', 'pack-windows.json', 'xtts.py'))):  # voicepack.runner_dir
        assert (installed / f).exists(), f
    if os.environ.get('REQUIRE_LOOPBACK') == '1':  # a release: the sound helpers (test_zip_holds_the_same_folder: the .zip too)
        for f in ('peresmeshnik-loopback.exe', 'peresmeshnik-audio.exe'):
            assert (installed / f).exists(), f'no {f}'
    assert 'espeak-ng' in (installed / 'licenses' / 'piper-runner-SOURCE.txt').read_text()
    assert not list(installed.rglob('torch')), 'torch in the base app'
    # stage 7 (C3): the LGPL source offer, the notices inside Qt, hts_engine, the Unicode data, the Required Notice
    for f in ('licenses/LGPL-SOURCE.txt', 'licenses/PySide6_Essentials/Qt-third-party.txt',
              'licenses/pyopenjtalk-plus/hts_engine-COPYING.txt', 'licenses/piper-tts/espeak-ng-COPYING.UCD.txt',
              'licenses/pyopenjtalk-plus/OpenJTalk-COPYING.txt',  # FIXES 12: Open JTalk with its MeCab
              'licenses/protobuf/mingw-w64-runtime-COPYING.txt', 'licenses/numpy/mingw-w64-runtime-COPYING.txt'):  # MinGW builds
        assert (installed / f).exists(), f
    lgpl = (installed / 'licenses' / 'LGPL-SOURCE.txt').read_text()
    assert 'Written offer' in lgpl and 'reverse' in lgpl  # FIXES 15: the LGPL libraries may be changed and debugged
    assert re.search(r'libmpg123 \d+\.\d+\.\d+ ', lgpl) and 'libmp3lame 3.' in lgpl, lgpl  # the exact versions of vcpkg's
    runner_src = (installed / 'licenses' / 'piper-runner-SOURCE.txt').read_text()  # FIXES 16: the runner's Corresponding Source
    assert re.search(r'onnxruntime \d', runner_src) and 'CPython 3.' in runner_src and 'Written offer' in runner_src
    assert 'sources.tar' in runner_src and 'sources.tar' in lgpl  # attached to the release (packaging/sources.sh)
    # the build's CPython by name (FIXES C r2 10): built from this interpreter (release.yml tests in the build's env), so
    # uv's python-build-standalone with its release (uv's BUILD file), else python.org's; its source tarball either way
    py, pbs = platform.python_version(), Path(sys.base_prefix) / 'BUILD'
    if pbs.is_file():
        tag = pbs.read_text().strip()
        assert f'python-build-standalone, release {tag}' in runner_src, runner_src
        assert f'https://github.com/astral-sh/python-build-standalone/releases/tag/{tag}' in runner_src
    else:
        assert 'python-build-standalone' not in runner_src, runner_src
    assert f'https://www.python.org/ftp/python/{py}/Python-{py}.tar.xz' in runner_src, runner_src
    # no MinGW runtime DLLs (Git's, from the runner's PATH): no notice of theirs is shipped
    mingw = ('libgcc*', 'libstdc++*', 'libwinpthread*', 'libgfortran*', 'libquadmath*', 'libgomp*', 'libiconv*', 'libintl*',
             'zlib1.dll')  # by pattern: libgcc_s_seh-1, libgcc_s_dw2-1, libgfortran-5... whatever the GCC
    assert not [f for f in installed.rglob('*.dll') if any(fnmatch.fnmatch(f.name.lower(), m) for m in mingw)]
    assert 'Required Notice: Copyright' in (installed / 'licenses' / 'Peresmeshnik-LICENSE.txt').read_text()
    ossl = (installed / 'licenses' / 'OpenSSL-LICENSE.txt').read_text()
    assert 'Apache License' in ossl
    if [f for f in installed.rglob('_rust*.pyd') if f.parent.name == 'bindings']:  # cryptography's own OpenSSL, built in
        assert re.search(r'OpenSSL \d+\.\d+\.\d+ .*, built statically into the _rust module of cryptography \d', ossl), ossl[:300]
    left_out = ('qjpeg.dll', 'qtiff.dll', 'qwebp.dll', 'nakdimon.onnx',  # unused (Peresmeshnik.spec drops them)
                'qopensslbackend.dll', 'libssl-3-x64.dll', 'libcrypto-3-x64.dll')
    assert not [f for f in installed.rglob('*') if f.name in left_out]
    for exe in ('Peresmeshnik.exe', 'piper-runner.exe'):  # a version resource (Properties -> Details)
        v = subprocess.run(['powershell', '-NoProfile', '-Command', f"(Get-Item '{installed / exe}').VersionInfo.FileVersion"],
                           capture_output=True, text=True).stdout.strip()
        assert v.startswith(peresmeshnik.__version__), (exe, v)
    dropped = [f.name for f in installed.rglob('*.dll') if f.name.lower() == 'opengl32sw.dll' or '-asio' in f.name.lower()]
    assert not dropped, dropped  # Mesa's software OpenGL, PortAudio with ASIO: not shipped (Peresmeshnik.spec)
    assert SHORTCUT.exists(), 'no Start menu shortcut'
    assert not STARTUP.exists(), 'starts at sign-in without being asked'
    k = uninstall_key()
    assert k, 'no uninstall entry'
    with k:
        assert winreg.QueryValueEx(k, 'DisplayVersion')[0] == peresmeshnik.__version__
    size = sum(f.stat().st_size for f in installed.rglob('*') if f.is_file())
    print(f'installed: {size / 1e6:.0f} MB in {installed}; installer {Path(SETUP).stat().st_size / 1e6:.0f} MB, '
          f'.zip {Path(ZIP).stat().st_size / 1e6:.0f} MB')


def _modules(executable: Path) -> set[str]:
    """The Python modules in a frozen program's own archive (PYZ inside its executable)."""
    from PyInstaller.archive.readers import CArchiveReader
    pkg = CArchiveReader(str(executable))
    return set(pkg.open_embedded_archive(next(n for n in pkg.toc if n.endswith('.pyz'))).toc)


def test_piper_stays_out_of_the_app(installed):
    """GPL separation: piper (piper-tts) is only in piper-runner's archive, never in the app's."""
    ours, runner = _modules(installed / 'Peresmeshnik.exe'), _modules(installed / 'piper-runner.exe')
    assert 'peresmeshnik.app' in ours and 'piper' in runner
    assert not {m for m in ours if m == 'piper' or m.startswith('piper.')}, 'piper in the app'
    assert not {m for m in runner if m.startswith('peresmeshnik')}, 'the app in piper-runner'


def test_piper_runner_error_exits_without_a_dialog(installed, tmp_path):
    """The windowed runner: an error (here a missing voice) ends it with its traceback and exit code 1, not the
    bootloader's dialog that would leave it, and the app reading its pipe, waiting."""
    r = subprocess.run([str(installed / 'piper-runner.exe'), str(tmp_path / 'missing.onnx')], capture_output=True, text=True,
                       timeout=60, env=env())
    assert r.returncode == 1 and 'FileNotFoundError' in r.stderr, (r.returncode, r.stderr[-1000:])


def test_zip_holds_the_same_folder(installed):
    with zipfile.ZipFile(ZIP) as z:
        zipped = {n.removeprefix('Peresmeshnik/') for n in z.namelist() if not n.endswith('/')}
    files = {f.relative_to(installed).as_posix() for f in installed.rglob('*') if f.is_file()}
    assert zipped == {f for f in files if not f.startswith('unins000.')}


def test_installed_app_starts_shows_tray_and_overlay_idles_and_closes(installed, tmp_path):
    t0 = time.perf_counter()
    p = subprocess.Popen([str(installed / 'Peresmeshnik.exe'), 'app', '--show-overlay'], stdout=subprocess.PIPE, text=True,
                         env=env(tmp_path))
    try:
        line = p.stdout.readline()
        started = time.perf_counter() - t0
        assert 'idle' in line, line
        proc = psutil.Process(p.pid)
        print(f'start: {started:.2f} s to "{line.strip()}", RSS {proc.memory_info().rss / 1e6:.0f} MB')
        time.sleep(1)
        wins = app_windows(p.pid)
        print('windows:', [w for _, w in wins])
        where = tray_icon_rect(p.pid)
        print('tray icon:', where)
        assert where, 'no tray icon'
        visible = [h for h, _ in wins if windows.user32.IsWindowVisible(h)]
        assert any(windows.ex_style(h) & windows.OVERLAY_STYLE == windows.OVERLAY_STYLE for h in visible), 'no overlay'
        assert any('setup' in w for h, w in wins if h in visible), 'no setup wizard on the first run'
        if shots := os.environ.get('SCREENSHOTS'):  # the tray, the overlay, the wizard: for a look
            subprocess.run(['powershell', '-NoProfile', '-Command',
                            'Add-Type -AssemblyName System.Windows.Forms,System.Drawing; '
                            '$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; '
                            '$i = New-Object System.Drawing.Bitmap $b.Width, $b.Height; '
                            '[System.Drawing.Graphics]::FromImage($i).CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size); '
                            f'$i.Save("{Path(shots) / "app-started.png"}")'], check=False, timeout=60)
        before = cpu_s(proc)
        time.sleep(WINDOW_S)
        used = cpu_s(proc) - before
        assert p.poll() is None, 'the app exited while idle'
        print(f'idle CPU of the installed app: {used:.3f} s in {WINDOW_S:.0f} s')
        assert used < 0.13 * WINDOW_S / 60
        tkill(p.pid)
        assert p.wait(20) == 0
    finally:
        if p.poll() is None:
            p.kill()


def _running(exe: Path, timeout: float = 60):
    end, want = time.monotonic() + timeout, str(exe).lower()
    while time.monotonic() < end:
        for q in psutil.process_iter(['exe']):
            if (q.info['exe'] or '').lower() == want:
                return q
        time.sleep(0.5)
    raise AssertionError(f'{exe} did not start')


def test_started_from_the_start_menu_shortcut(installed, tmp_path, monkeypatch):
    """As a user starts it: the shortcut, no arguments, no console; its tray icon comes up, taskkill ends it cleanly."""
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    os.startfile(SHORTCUT)
    proc = _running(installed / 'Peresmeshnik.exe')
    try:
        end, where = time.monotonic() + 60, None
        while not where and time.monotonic() < end:
            time.sleep(0.5)
            where = tray_icon_rect(proc.pid)
        print('started from the shortcut: tray icon', where)
        assert where, 'no tray icon'
        tkill(proc.pid)
        assert proc.wait(20) == 0
    finally:
        if proc.is_running():
            proc.kill()


def test_update_over_the_running_app_closes_it_first(installed, tmp_path):
    p = subprocess.Popen([str(installed / 'Peresmeshnik.exe'), 'app', '--no-wizard'], stdout=subprocess.PIPE, text=True,
                         env=env(tmp_path))
    try:
        assert 'idle' in p.stdout.readline()
        t0 = time.perf_counter()
        setup()
        print(f'update over the running app: {time.perf_counter() - t0:.1f} s')
        assert p.wait(5) == 0, 'the app did not quit cleanly'
        assert (installed / 'Peresmeshnik.exe').exists()
    finally:
        if p.poll() is None:
            p.kill()


def test_frozen_cloud_direction_reaches_google(installed, tmp_path):
    e = env()
    e['GEMINI_API_KEY'] = 'not-a-key'
    log = tmp_path / 'cloud.log'
    with open(log, 'wb') as f:  # stdin stays open: the direction waits for audio while it connects
        p = subprocess.Popen([str(installed / 'Peresmeshnik.exe'), 'cloud', '--in', '-'], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=f, env=e)
        try:
            code = p.wait(180)  # six connect attempts with growing pauses, then it gives up
        except subprocess.TimeoutExpired:
            code = 'still running after 180 s'
        finally:
            p.stdin.close()
            if p.poll() is None:
                p.kill()
    err = log.read_text('utf-8', 'replace')
    assert code != 'still running after 180 s', err[-3000:]
    print('frozen cloud with a fake key:', code, err.strip().splitlines()[-1][:200] if err.strip() else '')
    assert code != 0
    assert not any(w in err for w in ('No module named', 'PackageNotFoundError', 'ImportError')), err[-2000:]
    assert 'API key not valid' in err, err[-2000:]  # Google's answer


def test_frozen_cli_fetches_and_translates(tmp_path):
    """From the .zip, unpacked to a folder with spaces in its path (the app starts piper-runner.exe beside it)."""
    models = os.environ.get('PERESMESHNIK_MODELS')
    if not models:
        pytest.skip('needs the work-gate models: PERESMESHNIK_MODELS')
    where = tmp_path / 'Program Files test'
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(where)
    exe = str(where / 'Peresmeshnik' / 'Peresmeshnik.exe')
    e = env()
    e['PATH'] = os.pathsep.join([os.environ['SystemRoot'] + r'\System32', os.environ['SystemRoot']])  # no uv, no Python
    r = subprocess.run([exe, 'models', 'fetch', 'silero-vad'], capture_output=True, text=True, timeout=300,
                       env={**e, 'PERESMESHNIK_MODELS': str(tmp_path / 'fresh')})  # over HTTPS from the bundle
    assert r.returncode == 0, r.stderr[-2000:]
    out = tmp_path / 'ru.wav'
    r = subprocess.run([exe, 'run', '--in', str(DATA / 'call-en.ogg'), '--from', 'en', '--out', str(out)],
                       capture_output=True, text=True, encoding='utf-8', timeout=900, env={**e, 'PERESMESHNIK_MODELS': models})
    assert r.returncode == 0, r.stderr[-3000:]
    ru = [ln for ln in r.stdout.splitlines() if ln.startswith('ru: ')]
    assert ru and all(re.search('[а-яА-Я]', ln) for ln in ru), r.stdout[-2000:]
    assert out.stat().st_size > 100_000, 'no translated speech from the bundled Piper runner'
    frozen = json.loads(r.stderr.strip().splitlines()[-1])
    r = subprocess.run([sys.executable, '-m', 'peresmeshnik', 'run', '--in', str(DATA / 'call-en.ogg'), '--from', 'en'],
                       capture_output=True, text=True, encoding='utf-8', timeout=900, env={**os.environ, 'PERESMESHNIK_MODELS': models})
    assert r.returncode == 0, r.stderr[-3000:]
    source = json.loads(r.stderr.strip().splitlines()[-1])  # the same run from Python on the same machine
    keys = ('rtf', 'cpu_s_per_audio_s', 'asr_partial_rtf', 'tts_rtf')
    print('frozen run:', json.dumps({k: frozen.get(k) for k in keys}), '| from Python:', json.dumps({k: source.get(k) for k in keys}))


def test_uninstaller_closes_the_app_and_removes_it_all(installed, tmp_path):
    """Last in this file: the app is gone after it. A silent uninstall leaves the voice pack (~8 GB; asked about only
    when the user uninstalls by hand, "no" by default: USER-CHECKS)."""
    from peresmeshnik import config
    pack = config.data_dir() / 'voice-pack'  # the uninstaller's {localappdata}\Peresmeshnik\voice-pack
    assert not pack.exists(), f'{pack} is there already'
    (pack / 'runtime').mkdir(parents=True)
    (pack / '.complete').write_text('')
    p = subprocess.Popen([str(installed / 'Peresmeshnik.exe'), 'app', '--no-wizard'], stdout=subprocess.PIPE, text=True,
                         env=env(tmp_path))
    try:
        assert 'idle' in p.stdout.readline()
        r = subprocess.run([str(installed / 'unins000.exe'), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'], timeout=120)
        assert r.returncode == 0
        assert p.wait(20) == 0, 'the app did not quit cleanly'
        end = time.monotonic() + 60  # the uninstaller runs on from a copy of itself in %TEMP%
        while installed.exists() and time.monotonic() < end:
            time.sleep(0.5)
        left = [str(f) for f in installed.rglob('*')] if installed.exists() else []
        assert not installed.exists(), f'left behind: {left[:20]}'
        assert not SHORTCUT.exists() and uninstall_key() is None
        assert (pack / '.complete').exists(), 'a silent uninstall removed the voice pack'
    finally:
        if p.poll() is None:
            p.kill()
        if pack.exists():  # moved away, not removed
            pack.rename(tmp_path / 'voice-pack')

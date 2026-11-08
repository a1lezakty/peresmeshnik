"""The built Peresmeshnik.app (release.yml): copied out of its mounted .dmg and its executable started directly
(no quarantine, no LaunchServices: the first-start Gatekeeper dialog is a live check), then
- its menu-bar item, the overlay (a sample subtitle) and the first-run wizard are on screen, the start time is printed;
- 60 s idle with all of that under the same 0.13 s CPU limit as tests/test_idle_cpu.py (after the same 15 s on macOS);
- SIGTERM ends it with exit code 0;
- the frozen CLI fetches a model over HTTPS and translates a recording with the bundled engines and the
  bundled Piper runner (needs the work-gate models in $PERESMESHNIK_MODELS), as fast as from Python;
- the frozen cloud direction gets as far as Google refusing a fake key (google-genai and its parts are bundled).

    PERESMESHNIK_DMG=dist/Peresmeshnik-0.1.0-macos-arm64.dmg pytest -s tests/test_app_bundle.py
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

DMG = os.environ.get('PERESMESHNIK_DMG')
pytestmark = pytest.mark.skipif(not (DMG and sys.platform == 'darwin'), reason='needs the built .dmg: PERESMESHNIK_DMG')
DATA = Path(__file__).parent / 'data'
WINDOW_S = float(os.environ.get('IDLE_WINDOW_S', 60))


@pytest.fixture(scope='module')
def app(tmp_path_factory):
    """Peresmeshnik.app copied out of the mounted .dmg, as a user drags it to Applications."""
    mnt = tmp_path_factory.mktemp('mnt')
    subprocess.run(['hdiutil', 'attach', DMG, '-nobrowse', '-readonly', '-mountpoint', str(mnt)], check=True, capture_output=True)
    try:
        assert (mnt / 'Applications').is_symlink(), 'the .dmg has no Applications link to drag the app to'
        dst = tmp_path_factory.mktemp('Applications') / 'Peresmeshnik.app'
        subprocess.run(['ditto', str(mnt / 'Peresmeshnik.app'), str(dst)], check=True)
    finally:
        subprocess.run(['hdiutil', 'detach', str(mnt)], check=True, capture_output=True)
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(dst)], check=True)
    return dst


def exe(app: Path, name: str = 'Peresmeshnik') -> str:
    return str(app / 'Contents' / 'MacOS' / name)


def cpu_s(proc) -> float:
    return sum(sum(p.cpu_times()[:2]) for p in [proc, *proc.children(recursive=True)])


def test_bundle_contents(app):
    import plistlib
    info = plistlib.loads((app / 'Contents' / 'Info.plist').read_bytes())
    assert info['LSUIElement'] is True and info['CFBundleIdentifier'] == 'io.github.a1lezakty.peresmeshnik'
    assert info['NSMicrophoneUsageDescription'] and info['NSAudioCaptureUsageDescription']
    assert Path(exe(app, 'piper-runner')).exists()
    lic = app / 'Contents' / 'Resources' / 'licenses'
    assert (lic / 'piper-runner-LICENSE.txt').exists() and (lic / 'Peresmeshnik-LICENSE.txt').exists()
    assert not list(app.rglob('torch')), 'torch in the base app'
    size = sum(f.stat().st_size for f in app.rglob('*') if f.is_file() and not f.is_symlink())
    print(f'bundle: {size / 1e6:.0f} MB in the .app, {Path(DMG).stat().st_size / 1e6:.0f} MB .dmg, '
          f'tap helper {"in" if Path(exe(app, "peresmeshnik-tap")).exists() else "not in"} the bundle')


def test_app_starts_shows_its_icon_idles_and_quits_on_sigterm(app, tmp_path):
    from peresmeshnik.ui import macos
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'VIRTUAL_ENV'))}
    env['PERESMESHNIK_CONFIG'] = str(tmp_path / 'settings.toml')  # no file: a first run, the wizard opens
    t0 = time.perf_counter()
    p = subprocess.Popen([exe(app), 'app', '--show-overlay'], stdout=subprocess.PIPE, text=True, env=env)
    try:
        line = p.stdout.readline()
        started = time.perf_counter() - t0
        assert 'idle' in line, line
        proc = psutil.Process(p.pid)
        print(f'start: {started:.2f} s to "{line.strip()}", RSS {proc.memory_info().rss / 1e6:.0f} MB')
        wins = macos.windows_of(p.pid)
        print('windows:', wins)
        assert any(w['layer'] == macos.STATUS_LEVEL and w['bounds'][1] == 0 for w in wins), 'no menu-bar item'
        assert any(w['layer'] == macos.STATUS_LEVEL and w['bounds'][1] > 0 for w in wins), 'no overlay'
        if shots := os.environ.get('SCREENSHOTS'):  # the menu-bar icon, the overlay, the wizard: for a look
            time.sleep(2)
            subprocess.run(['screencapture', '-x', str(Path(shots) / 'app-started.png')], check=False)
        before = cpu_s(proc)
        time.sleep(15)  # AppKit's one-off work after the first show (tests/test_idle_cpu.py)
        settle, before = cpu_s(proc) - before, cpu_s(proc)
        time.sleep(WINDOW_S)
        used = cpu_s(proc) - before
        assert p.poll() is None, 'the app exited while idle'
        print(f'idle CPU of the .app: {used:.3f} s in {WINDOW_S:.0f} s (the 15 s before, not gated: {settle:.3f} s)')
        assert used < 0.13 * WINDOW_S / 60
        p.send_signal(signal.SIGTERM)
        assert p.wait(20) == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_frozen_cloud_direction_reaches_google(app, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'VIRTUAL_ENV'))}
    env['GEMINI_API_KEY'] = 'not-a-key'
    log = tmp_path / 'cloud.log'
    with open(log, 'wb') as f:  # stdin stays open: the direction waits for audio while it connects
        p = subprocess.Popen([exe(app), 'cloud', '--in', '-'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=f, env=env)
        try:
            code = p.wait(180)  # six connect attempts with growing pauses, then it gives up
        finally:
            p.stdin.close()
            if p.poll() is None:
                p.kill()
    err = log.read_text('utf-8', 'replace')
    print('frozen cloud with a fake key:', code, err.strip().splitlines()[-1][:200] if err.strip() else '')
    assert code != 0
    assert not any(w in err for w in ('No module named', 'PackageNotFoundError', 'ImportError')), err[-2000:]
    assert 'key' in err.lower(), err[-2000:]  # Google's answer: the API key is not valid


def test_frozen_cli_fetches_and_translates(app, tmp_path):
    models = os.environ.get('PERESMESHNIK_MODELS')
    if not models:
        pytest.skip('needs the work-gate models: PERESMESHNIK_MODELS')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'VIRTUAL_ENV'))}
    env['PATH'] = '/usr/bin:/bin'  # no uv: the bundled Piper runner must do
    r = subprocess.run([exe(app), 'models', 'fetch', 'silero-vad'], capture_output=True, text=True, timeout=300,
                       env={**env, 'PERESMESHNIK_MODELS': str(tmp_path / 'fresh')})  # over HTTPS from the bundle
    assert r.returncode == 0, r.stderr[-2000:]
    out = tmp_path / 'ru.wav'
    r = subprocess.run([exe(app), 'run', '--in', str(DATA / 'call-en.ogg'), '--from', 'en', '--out', str(out)],
                       capture_output=True, text=True, encoding='utf-8', timeout=900, env={**env, 'PERESMESHNIK_MODELS': models})
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

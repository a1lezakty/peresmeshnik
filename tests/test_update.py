"""The updater (peresmeshnik/update.py, ui/tray.py Updates) against a fake GitHub (tests/fake_github.py): versions,
the file per OS, drafts and prereleases, the SHA-256 and the attestation, no restart during a translation, no request
with the setting off, the macOS swap script. The real replace and restart: tests/test_app_bundle.py, test_app_windows.py."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from fake_github import FakeGitHub, statement  # noqa: E402

from peresmeshnik import config, update  # noqa: E402

DMG = 'Peresmeshnik-0.3.4-macos-arm64.dmg'


@pytest.fixture
def gh(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'cache_dir', lambda: tmp_path / 'cache')
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(update, '__version__', '0.3.3')
    with FakeGitHub() as g:
        monkeypatch.setenv('PERESMESHNIK_UPDATE_API', g.url)
        yield g


def test_versions_compare_as_numbers():
    assert update.version('v0.3.10') == (0, 3, 10) > update.version('0.3.9')
    assert update.version('models-2') is None and update.version('v0.4.0-rc1') is None
    assert update.newer(update.Release('0.3.10', ''), '0.3.9') and not update.newer(update.Release('0.3.9', ''), '0.3.10')
    assert not update.newer(update.Release('0.3.3', ''), '0.3.3') and not update.newer(None, '0.3.3')


def test_the_file_for_each_os_and_architecture():
    files = {n: '' for n in ('Peresmeshnik-0.3.4-macos-arm64.dmg', 'Peresmeshnik-0.3.4-macos-arm64.dmg.sha256',
                             'Peresmeshnik-0.3.4-windows-x64-setup.exe', 'Peresmeshnik-0.3.4-windows-x64-setup.exe.sha256',
                             'Peresmeshnik-0.3.4-windows-x64.zip', 'Peresmeshnik-0.3.4-windows-x64.zip.sha256')}
    rel = update.Release('0.3.4', '', files=files)
    assert update.file_for(rel, 'darwin', 'arm64') == 'Peresmeshnik-0.3.4-macos-arm64.dmg'
    assert update.file_for(rel, 'win32', 'AMD64') == update.file_for(rel, 'win32', 'ARM64') == \
        'Peresmeshnik-0.3.4-windows-x64-setup.exe'  # the installer, never the .zip; x64 runs on Windows on Arm
    assert update.file_for(rel, 'darwin', 'x86_64') is None  # no Intel build
    assert update.file_for(rel, 'linux', 'x86_64') is None
    del files['Peresmeshnik-0.3.4-macos-arm64.dmg.sha256']
    assert update.file_for(rel, 'darwin', 'arm64') is None  # nothing to check it against: not offered


def test_drafts_prereleases_and_other_tags_do_not_count(gh):
    assert update.latest() is None
    gh.publish('0.3.9', {DMG: b'a'})
    gh.publish('0.3.10', {DMG: b'b'})
    gh.publish('0.4.0', {DMG: b'c'}, draft=True)
    gh.publish('0.5.0', {DMG: b'd'}, prerelease=True)
    gh.publish('9', {}, tag='models-3')  # the models' releases share the repository
    rel = update.latest()
    assert rel.version == '0.3.10' and rel.page.endswith('/v0.3.10') and 'What is new' in rel.notes
    assert set(rel.files) == {DMG, f'{DMG}.sha256'}


def test_download_checks_sha256_and_attestation(gh):
    data = os.urandom(3 << 20)
    gh.publish('0.3.4', {DMG: data})
    seen = []
    path = update.download(update.latest(), DMG, lambda done, total: seen.append((done, total)))
    assert path == update.folder() / DMG and path.read_bytes() == data
    assert seen[-1] == (len(data), len(data)) and len(seen) >= 3  # progress per MB


def test_a_sha256_mismatch_is_refused_and_the_file_removed(gh):
    gh.publish('0.3.4', {DMG: b'the real one'}, sha='0' * 64)
    with pytest.raises(update.Failed, match='SHA-256'):
        update.download(update.latest(), DMG)
    assert not list(update.folder().iterdir())


def test_an_attestation_given_inline_counts_too(gh):
    import hashlib
    data = b'attested inline, as the API gave it before'
    gh.publish('0.3.4', {DMG: data}, attest=False)
    d = hashlib.sha256(data).hexdigest()
    gh.attest(d, statement(DMG, d, '0.3.4'), inline=True)
    assert update.download(update.latest(), DMG).read_bytes() == data


def test_snappy_copies_and_garbage():
    assert update._unsnappy(b'\x0c\x0cabcd\x11\x04') == b'abcdabcdabcd'  # a literal, then an overlapping copy
    for bad in (b'\x05\x0cabcd', b'\x08\x11\x04', b''):  # too short; a copy from before the start; nothing
        with pytest.raises((ValueError, IndexError)):
            update._unsnappy(bad)


@pytest.mark.parametrize('how', ['none', 'another workflow', 'another tag', 'another repository', 'another digest',
                                 'not a bundle'])
def test_without_the_attestation_of_release_yml_nothing_is_installed(gh, how):
    import hashlib
    data = b'built elsewhere'
    gh.publish('0.3.4', {DMG: data}, attest=False)
    d = hashlib.sha256(data).hexdigest()
    bundle = {
        'none': None,
        'another workflow': statement(DMG, d, '0.3.4', workflow='.github/workflows/ci.yml'),
        'another tag': statement(DMG, d, '0.3.3'),
        'another repository': statement(DMG, d, '0.3.4', repo='someone/peresmeshnik'),
        'another digest': statement(DMG, '1' * 64, '0.3.4'),
        'not a bundle': {'dsseEnvelope': {'payload': 'bm90IGpzb24='}}}[how]
    if bundle:
        gh.attest(d, bundle)
    with pytest.raises(update.Failed, match='attestation'):
        update.download(update.latest(), DMG)
    assert not list(update.folder().iterdir())


def test_where_it_cannot_install_it_says_why(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert 'Linux' in update.installable()
    monkeypatch.setattr(sys, 'platform', 'win32')
    assert 'source' in update.installable()  # not frozen
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'Peresmeshnik.exe'))
    assert '.zip' in update.installable()
    (tmp_path / 'unins000.exe').write_bytes(b'')
    assert update.installable() == ''


def test_the_result_is_said_once_at_the_next_start(gh):
    update.folder().mkdir(parents=True)
    update.pending().write_text('{"version": "0.3.3", "result": ""}')
    assert update.outcome() == 'Updated to 0.3.3.' and update.outcome() == ''
    (update.folder() / 'result.txt').write_text('mv: Permission denied\n')
    update.pending().write_text(f'{{"version": "0.3.4", "result": "{(update.folder() / "result.txt").as_posix()}"}}')
    said = update.outcome()
    assert said.startswith('The update to 0.3.4 did not install') and said.endswith('0.3.3 goes on working.')


@pytest.mark.skipif(sys.platform == 'win32', reason='the macOS swap script is POSIX sh')
@pytest.mark.parametrize('fails', [False, True])
def test_the_swap_waits_for_the_app_then_replaces_it_or_keeps_the_old_one(tmp_path, fails):
    app, new, res = tmp_path / 'Peresmeshnik.app', tmp_path / '.Peresmeshnik-update.app', tmp_path / 'result.txt'
    (app / 'Contents').mkdir(parents=True)
    (app / 'Contents' / 'old').write_text('')
    if not fails:  # a new app that cannot be moved in: the old one stays and starts
        (new / 'Contents').mkdir(parents=True)
        (new / 'Contents' / 'new').write_text('')
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    (bin_ / 'open').write_text(f'#!/bin/sh\necho "$@" > {tmp_path / "opened"}\n')
    (bin_ / 'open').chmod(0o755)
    running = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1)'])
    threading.Thread(target=running.wait, daemon=True).start()  # reaped at once, as launchd does: no zombie for kill -0
    t0 = time.monotonic()
    r = subprocess.run(['/bin/sh', '-c', update.SWAP, 'x', str(running.pid), str(app), str(new), str(res),
                        '--env', 'PERESMESHNIK_CONFIG=/x'], env={**os.environ, 'PATH': f'{bin_}:/usr/bin:/bin'}, timeout=30)
    assert r.returncode == 0 and time.monotonic() - t0 > 0.5  # waited for the app to quit
    assert (app / 'Contents' / ('old' if fails else 'new')).exists() and not new.exists()
    assert [p.name for p in tmp_path.iterdir() if '.old-' in p.name] == []
    assert (res.read_text().strip() == 'ok') != fails
    assert (tmp_path / 'opened').read_text().split() == ['--env', 'PERESMESHNIK_CONFIG=/x', str(app)]


# the tray: no restart while translating, no request with the setting off, the daily timer

def _tray(on=False):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from peresmeshnik.ui import tray as tray_mod

    class Ctl:
        live = False
    Ctl.on = on
    t = tray_mod.Tray(Ctl())
    t.said = []
    t.message = t.said.append
    return t


def _until(cond, timeout=10):
    from PySide6.QtCore import QCoreApplication
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.01)
    assert cond()


def test_switch_off_means_no_request_but_the_menu_still_checks(gh):
    gh.publish('0.3.4', {DMG: b'x'})
    s = config.load()
    s.ui.check_updates = False
    config.save(s)
    t = _tray()
    t.updates.start()
    assert not t.updates.timer.isActive()
    t.updates.check(asked=False)  # the timer's call: nothing at all
    time.sleep(0.3)
    assert gh.requests == []
    t.updates.ask = lambda box: None  # the offer: closed
    t.updates.check()  # Check for updates…
    _until(lambda: t.update_offer.isVisible())
    assert gh.requests == [f'/repos/{update.REPO}/releases'] and t.update_offer.text() == 'Update to 0.3.4…'


def test_once_a_day_one_timer_and_silent_network_errors(gh, monkeypatch):
    t = _tray()
    t.updates.start()
    assert t.updates.timer.isSingleShot() and t.updates.timer.interval() == t.updates.START_S * 1000  # never checked
    update.remember(None)  # its folder
    update._state_file().write_text(json.dumps({'checked': time.time() + 90 * 86400}))  # checked "later": a clock set back
    t.updates.start()
    assert t.updates.timer.interval() == update.DAY_S * 1000
    asked = []
    t.updates.check = lambda asked_=True, **kw: asked.append(kw)
    t.updates.due()  # the menu opens: not due
    update._state_file().write_text('{"checked": 0}')
    t.updates.due()  # due: checked (a Mac's timer may not count its sleep)
    assert asked == [{'asked': False}]
    del t.updates.check
    monkeypatch.setenv('PERESMESHNIK_UPDATE_API', 'http://127.0.0.1:9')  # nothing listens: refused
    t.updates.check(asked=False)
    _until(lambda: abs(t.updates.timer.interval() - update.DAY_S * 1000) < 60_000)  # failed: tried again in a day
    assert t.said == []
    t.updates.check()  # asked: said
    _until(lambda: t.said)
    assert t.said[0].startswith('Could not check for updates')


def test_no_restart_during_a_translation(gh, monkeypatch):
    gh.publish('0.3.4', {DMG: b'x'})
    monkeypatch.setattr(update, 'installable', lambda: '')
    monkeypatch.setattr(update, 'file_for', lambda rel: DMG)
    calls = []
    monkeypatch.setattr(update, 'prepare', lambda f: calls.append(('prepare', f.name)) or f)
    monkeypatch.setattr(update, 'hand_over', lambda rel, ready: calls.append(('hand over', rel.version)))
    from PySide6.QtWidgets import QApplication
    monkeypatch.setattr(QApplication, 'quit', lambda: calls.append(('quit',)))
    t = _tray(on=True)
    boxes = []
    t.updates.ask = lambda box: boxes.append(box) or None
    t.updates.check()
    _until(lambda: boxes)
    buttons = [b.text() for b in boxes[0].buttons()]
    assert 'Update and restart' not in buttons and 'Release page' in buttons  # shown, not offered
    assert 'Translation is on' in boxes[0].informativeText()
    t.updates.install()
    assert calls == [] and 'waits until you turn it off' in t.said[-1]
    t.ctl.on = False  # off: it downloads ...
    t.updates.install()
    t.ctl.on = True  # ... and translation goes on again meanwhile: held, nothing restarts
    _until(lambda: not t.updates.busy)
    assert calls == [('prepare', DMG)] and 'downloaded and checked' in t.said[-1]
    t.ctl.on = False
    t.ctl.wanted = True  # turned on, still starting (its models load): translating already
    t.updates.install()
    assert calls[1:] == []
    t.ctl.wanted = False
    t.updates.install()  # chosen again with translation off: from what was downloaded
    assert calls[1:] == [('hand over', '0.3.4')] + ([('quit',)] if sys.platform == 'darwin' else [])
    if sys.platform != 'darwin':  # the installer closes it
        assert 'closes and starts again by itself' in t.said[-1] and not t.update_offer.isEnabled()


def test_the_offer_says_what_stays_and_the_permissions_on_macos(gh, monkeypatch):
    gh.publish('0.3.4', {DMG: b'x'})
    monkeypatch.setattr(update, 'installable', lambda: '')
    monkeypatch.setattr(update, 'file_for', lambda rel: DMG)
    t = _tray()
    boxes = []
    t.updates.ask = lambda box: boxes.append(box) or None
    t.updates.check()
    _until(lambda: boxes)
    info = boxes[0].informativeText()
    assert 'settings, voices and downloaded models stay' in info
    assert ('microphone' in info) == (sys.platform == 'darwin')
    assert 'Update and restart' in [b.text() for b in boxes[0].buttons()]
    assert 'What is new' in boxes[0].detailedText()


def test_a_known_release_is_offered_after_a_restart_without_a_request(gh):
    rel = update.Release('0.3.4', 'https://example/', files={DMG: 'x'})
    update.remember(rel)
    t = _tray()
    assert t.update_offer.isVisible() and gh.requests == []
    update.remember(update.Release('0.3.3', ''))  # this version: installed by hand since, no offer
    assert not _tray().update_offer.isVisible()

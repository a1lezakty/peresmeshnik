"""The setup wizard on macOS: the devices page (BlackHole 2ch there, or how to install it and "Check again"), the
self-test page (a chirp through BlackHole 2ch) and the voice page recording from a Core Audio device. The first one
runs anywhere (the devices are faked); the others on the macOS runner (BlackHole 16ch stands in for the microphone)."""
import os
import sys
import time

import pytest

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from peresmeshnik import config, voice  # noqa: E402
from peresmeshnik.audio import macos  # noqa: E402

app = QApplication.instance() or QApplication([])
MIC = {'uid': 'BuiltInMicrophoneDevice', 'name': 'MacBook Pro Microphone', 'inputs': 1, 'outputs': 0, 'rate': 48000.0,
       'transport': 'bltn', 'default_input': True, 'default_output': False, 'id': 1}
SPEAKERS = {**MIC, 'uid': 'BuiltInSpeakerDevice', 'name': 'MacBook Pro Speakers', 'inputs': 0, 'outputs': 2,
            'default_input': False, 'default_output': True, 'id': 2}
BLACKHOLE = {**MIC, 'uid': 'BlackHole2ch_UID', 'name': 'BlackHole 2ch', 'inputs': 2, 'outputs': 2, 'transport': 'virt',
             'default_input': False, 'id': 3}


def pump(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, 'timed out'
        QCoreApplication.processEvents()
        time.sleep(0.01)


def test_devices_page_without_blackhole_says_how_to_install(monkeypatch):
    from peresmeshnik.ui import selftest, wizard
    devs = [MIC, SPEAKERS]
    monkeypatch.setattr(wizard, 'MACOS', True)
    monkeypatch.setattr(wizard, 'WINDOWS', False)
    monkeypatch.setattr(selftest, 'HELPER', True)
    monkeypatch.setattr(selftest, 'WINDOWS', False)
    monkeypatch.setattr(sys, 'platform', 'darwin')  # mic.native() picks audio/macos.py (on the Windows runner too)
    monkeypatch.setattr(macos, 'devices', lambda: list(devs))
    monkeypatch.setattr(macos, 'permission', lambda: 'authorized')
    page = wizard.DevicesPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    text = page.vmic.text()
    assert 'not installed' in text and 'brew install blackhole-2ch' in text and 'existential.audio/blackhole' in text
    assert not page.recheck.isHidden()
    assert [page.mic.itemData(i) for i in range(page.mic.count())] == ['', 'BuiltInMicrophoneDevice']
    devs.append(BLACKHOLE)  # installed meanwhile
    page.recheck.click()
    assert page.vmic.text().startswith('✔ BlackHole 2ch is installed') and page.recheck.isHidden()
    assert 'BlackHole2ch_UID' not in [page.mic.itemData(i) for i in range(page.mic.count())]  # never the microphone
    monkeypatch.setattr(macos, 'permission', lambda: 'denied')
    page.check_blackhole()
    assert 'Privacy & Security → Microphone' in page.vmic.text()


@pytest.mark.parametrize('granted', [True, False])
def test_call_sound_page_tap(monkeypatch, granted):
    """macOS 14.4+: the tap needs "System Audio Recording": the page says so, "Allow" asks off the GUI thread."""
    import threading
    import types

    from peresmeshnik.ui import wizard
    gui, asked = threading.current_thread(), []
    why = 'no permission to record the call app\'s sound: allow Peresmeshnik in System Settings > Privacy & Security'

    def request():
        asked.append(threading.current_thread() is not gui)
        return (0, '') if granted else (4, why)
    cm = types.SimpleNamespace(path=lambda: 'tap', check=lambda: (4, why), request=request, CALL_DEVICE='BlackHole 16ch')
    monkeypatch.setitem(sys.modules, 'peresmeshnik.audio.capture_macos', cm)
    import peresmeshnik.audio
    monkeypatch.setattr(peresmeshnik.audio, 'capture_macos', cm, raising=False)
    monkeypatch.setattr(wizard, 'WINDOWS', False)  # the macOS page on the Windows runner too
    page = wizard.CallSoundPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    assert 'System Audio Recording' in page.text.text() and page.state.text() == f'✘ {why}'
    assert page.button.text() == 'Allow' and not page.button.isHidden() and page.isComplete()
    page.button.click()
    assert not page.isComplete()  # waiting for the dialog
    pump(lambda: page.isComplete())
    assert asked == [True]  # request() blocks until the user answers: never on the GUI thread
    assert page.state.text().startswith('✔ Allowed') if granted else page.state.text() == f'✘ {why}'
    assert page.button.isHidden() == granted


def test_call_sound_page_blackhole(monkeypatch):
    """Before macOS 14.4: the call app's speaker is BlackHole 16ch; missing, how to install it and "Check again"."""
    import types

    from peresmeshnik.ui import wizard
    cm = types.SimpleNamespace(path=lambda: 'blackhole', CALL_DEVICE='BlackHole 16ch')
    monkeypatch.setitem(sys.modules, 'peresmeshnik.audio.capture_macos', cm)
    import peresmeshnik.audio
    monkeypatch.setattr(peresmeshnik.audio, 'capture_macos', cm, raising=False)
    devs = [MIC, SPEAKERS, BLACKHOLE]
    monkeypatch.setattr(macos, 'devices', lambda: list(devs))
    monkeypatch.setattr(wizard, 'WINDOWS', False)  # the macOS page on the Windows runner too
    page = wizard.CallSoundPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    assert 'choose the speaker <b>BlackHole 16ch</b>' in page.text.text()
    assert 'brew install --cask blackhole-16ch' in page.state.text() and page.button.text() == 'Check again'
    devs.append({**BLACKHOLE, 'uid': 'BlackHole16ch_UID', 'name': 'BlackHole 16ch', 'inputs': 16, 'outputs': 16, 'id': 4})
    page.button.click()
    assert page.state.text() == '✔ BlackHole 16ch is installed.' and page.button.isHidden()


MAC = sys.platform == 'darwin'


def _ready():
    try:
        return MAC and bool(macos.blackhole())
    except Exception:
        if MAC and os.environ.get('CI'):
            raise
        return False


@pytest.mark.skipif(not _ready(), reason='macOS with BlackHole 2ch and the helper built')
def test_self_test_page(caplog):
    """The app runs, translation off: its loop (microphone -> BlackHole 2ch) waits for a call app to record BlackHole 2ch.
    The self-test records it like one: the loop must stay closed meanwhile (else the real microphone opens: the orange
    dot, AirPods' call profile, and its sound over the chirp). Here a loud tone plays into the stand-in microphone."""
    import logging

    from test_virtual_mic_macos import MIC16, Feed, after_sound

    from peresmeshnik import app as app_
    from peresmeshnik.ui import wizard
    caplog.set_level(logging.INFO, logger='peresmeshnik.app')
    s = config.Settings()
    s.ui.mic = MIC16
    states = []
    overlay = type('O', (), {'side': lambda self, n: self, 'hide': lambda self: None, 'show': lambda self, c: None})()
    ctl = app_.Controller(overlay, states.append, settings=s)
    print('\nbefore:', after_sound())
    feed = Feed(MIC16, f=300.0)
    try:
        ctl.startup()
        assert ctl._loop is not None and ctl._loop.poll() is None, (ctl.startup_error, states)
        page = wizard.SelfTestPage(type('W', (), {'s': s, 'tray': type('T', (), {'ctl': ctl})()})())
        page.run()
        pump(lambda: page.ran, 30)
        loop_back = ctl._loop is not None and ctl._loop.poll() is None
    finally:
        ctl.quit()
        feed.stop()
    said = [r.getMessage() for r in caplog.records if r.getMessage().startswith('microphone loop:')]
    print('\n' + page.out.text(), said)
    assert '✔ virtual mic (BlackHole 2ch): chirp found' in page.out.text()
    assert 'microphone loop: call' not in said, said  # the loop never opened the microphone during the test
    assert loop_back  # and runs again after it (translation is still off)


@pytest.mark.skipif(not _ready(), reason='macOS with BlackHole 2ch and the helper built')
def test_voice_page_records_from_a_core_audio_device(tmp_path, monkeypatch):
    """A steady tone from the stand-in microphone: recorded, not saved, and the page says why."""
    from test_virtual_mic_macos import Feed

    from peresmeshnik.ui.wizard import VoicePage
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    s = config.Settings()
    s.me_to_them, s.ui.mic = 'local', 'BlackHole16ch_UID'
    page = VoicePage(type('W', (), {'s': s})())
    page.initializePage()
    feed = Feed(f=300.0)
    try:
        page.seconds = 3.0
        page.record()
        pump(lambda: not page.running, 15)
    finally:
        feed.stop()
    text = page.out.text()
    assert text.startswith('<b>Record again: 3 s recorded, need 30-60 s; only 0 s of speech'), text
    assert 'Not saved' in text and page.voice is None

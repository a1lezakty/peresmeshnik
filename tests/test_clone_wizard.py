"""The wizard's voice page: an imported or recorded voice gets a verdict (good / record again, because ...), a bad one is
not saved; the clone check reports what the app's ASR heard. Qt offscreen unless a display (Xvfb) is set."""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from peresmeshnik import config, voice  # noqa: E402

app = QApplication.instance() or QApplication([])
SOUND = (sys.platform.startswith('linux') and shutil.which('parec')
         and subprocess.run(['pactl', 'info'], capture_output=True).returncode == 0)


def pump(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, 'timed out'
        QCoreApplication.processEvents()
        time.sleep(0.01)


def speech_like(seconds, sr=24_000, seed=0):
    from test_clone import speech_like
    return speech_like(seconds, sr=sr, seed=seed)


@pytest.fixture
def page(tmp_path, monkeypatch):
    from peresmeshnik.ui.wizard import VoicePage
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    s = config.Settings()
    s.me_to_them = 'local'
    p = VoicePage(type('W', (), {'s': s})())
    p.initializePage()
    return p


def test_imported_voice_good_and_bad(page, tmp_path):
    assert not page.isComplete()  # local me -> them needs a voice
    bad = tmp_path / 'short.wav'
    sf.write(bad, speech_like(10), 24_000)
    page.import_file(str(bad))
    pump(lambda: not page.running)
    assert page.out.text().startswith('<b>Record again: 10 s recorded, need 30-60 s')
    assert 'Not saved' in page.out.text() and not page.isComplete() and page.voice is None
    good = tmp_path / 'me.wav'
    sf.write(good, speech_like(45), 24_000)
    page.import_file(str(good))
    pump(lambda: not page.running)
    assert page.out.text().startswith('<b>Good: ') and 'Saved as your voice "me"' in page.out.text()
    assert page.isComplete() and page.w.s.local.voice == 'me' and (tmp_path / 'voices' / 'me' / 'reference.wav').exists()
    page.import_file(str(bad))  # a worse take later does not replace the saved voice
    pump(lambda: not page.running)
    assert page.voice.name == 'me' and page.isComplete()


def test_cloud_needs_no_voice(page):
    page.w.s.me_to_them = 'cloud'
    page.initializePage()
    assert page.isComplete() and page.check_button.isHidden()


class FakeClone:
    sample_rate = 24_000

    def __init__(self, sim):
        self.sim = sim

    def synth(self, text, lang, voice):
        yield np.zeros(24_000, np.float32)

    def similarity(self, voice, pcm, sr):
        return self.sim


class FakeAsr:
    def __init__(self, hears):
        self.hears = hears

    def recognize(self, pcm):
        return type('R', (), {'text': self.hears.pop(0)})()


@pytest.mark.parametrize('hears,sim,verdict', [
    (list(voice.TEST_PHRASES['en']), 0.6, 'Good: the clone is understood and sounds like the recording (WER 0%'),
    (['good morning', 'thank you'], 0.6, 'Record again: the clone is hard to understand'),
    (list(voice.TEST_PHRASES['en']), 0.1, 'Record again: the clone does not sound like the recording'),
])
def test_clone_check(page, tmp_path, monkeypatch, hears, sim, verdict):
    import types
    good = tmp_path / 'me.wav'
    sf.write(good, speech_like(45), 24_000)
    page.import_file(str(good))
    pump(lambda: not page.running)
    monkeypatch.setitem(sys.modules, 'peresmeshnik.local.clone', types.SimpleNamespace(XttsClone=lambda *a, **k: FakeClone(sim)))
    from peresmeshnik import models
    from peresmeshnik.local import asr
    monkeypatch.setattr(asr, 'OnnxAsr', lambda *a, **k: FakeAsr(list(hears)))
    monkeypatch.setattr(models, 'path', lambda model_id: tmp_path)
    page.check_clone()
    pump(lambda: not page.running)
    assert page.clone_out.text().startswith(verdict), page.clone_out.text()


@pytest.mark.skipif(not SOUND, reason='needs PipeWire / PulseAudio')
def test_recorded_voice_from_a_bad_microphone_is_rejected_with_the_reason(page):
    """Recorded from a PipeWire source by name (a steady tone, no speech): not saved, and the page says why."""
    from test_virtual_mic_linux import Source
    src = Source('test_voice_mic', 300)
    try:
        page.w.s.ui.mic, page.seconds = 'test_voice_mic', 3.0
        page.record()
        assert not page.bar.isHidden() and not page.record_button.isEnabled()
        pump(lambda: not page.running, 15)
    finally:
        src.remove()
    text = page.out.text()
    assert text.startswith('<b>Record again: 3 s recorded, need 30-60 s; only 0 s of speech'), text
    assert 'Not saved' in text and page.voice is None and not page.isComplete()

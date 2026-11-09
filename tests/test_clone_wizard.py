"""The wizard's voice page: an imported or recorded voice gets a verdict (good / record again, because ...), a bad one is
not saved; the clone check reports what the app's ASR heard. Qt offscreen unless a display (Xvfb) is set."""
import os
import shutil
import subprocess
import sys
import time
import types

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
    from peresmeshnik.ui import wizard
    from peresmeshnik.ui.wizard import VoicePage
    monkeypatch.setattr(wizard, 'WINDOWS', False)  # the page without the voice pack (the Windows runner too): see windows()
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
    assert page.out.text().startswith('<b>Good: ') and 'saved when you finish the wizard' in page.out.text()
    assert page.isComplete() and page.w.s.local.voice == 'me.new'  # the test phrase already speaks with it
    page.import_file(str(bad))  # a worse take later does not replace the good one
    pump(lambda: not page.running)
    assert page.voice.name == 'me.new' and page.isComplete()
    page.commit()  # Finish
    v = tmp_path / 'voices'
    assert page.w.s.local.voice == 'me' and (v / 'me' / 'reference.wav').exists() and not (v / 'me.new').exists()
    assert voice.load('me', v).meta['name'] == 'me'


def test_cancelled_wizard_keeps_the_old_voice(page, tmp_path):
    v = tmp_path / 'voices'
    voice.save(speech_like(40, seed=1), 24_000, 'me', 'ru', root=v)
    page.w.s.local.voice = 'me'
    old = (v / 'me' / 'reference.wav').read_bytes()
    page.voice = None
    page.initializePage()
    assert page.voice.name == 'me' and page.isComplete()  # the saved one is there
    f = tmp_path / 'new.wav'
    sf.write(f, speech_like(45, seed=2), 24_000)
    page.import_file(str(f))
    pump(lambda: not page.running)
    assert (v / 'me.new').exists() and (v / 'me' / 'reference.wav').read_bytes() == old
    page.rollback()  # Cancel
    assert not (v / 'me.new').exists() and (v / 'me' / 'reference.wav').read_bytes() == old


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
    monkeypatch.setitem(sys.modules, 'peresmeshnik.local.clone', types.SimpleNamespace(make_clone=lambda *a, **k: FakeClone(sim)))
    from peresmeshnik import models
    from peresmeshnik.local import asr
    monkeypatch.setattr(asr, 'OnnxAsr', lambda *a, **k: FakeAsr(list(hears)))
    monkeypatch.setattr(models, 'path', lambda model_id: tmp_path)
    page.check_clone()
    pump(lambda: not page.running)
    assert page.clone_out.text().startswith(verdict), page.clone_out.text()


def test_no_clone_check_while_translating(page, tmp_path):
    """A local me -> them already holds the clone: a second one for the check would not fit on 8 GB."""
    good = tmp_path / 'me.wav'
    sf.write(good, speech_like(45), 24_000)
    page.import_file(str(good))
    pump(lambda: not page.running)
    assert page.check_button.isEnabled()
    page.w.tray = type('T', (), {'ctl': type('C', (), {'on': True})()})()
    page._buttons()
    assert not page.check_button.isEnabled() and 'Turn translation off' in page.check_button.toolTip()
    page.check_clone()
    assert page.clone_out.text().startswith('Turn translation off first') and not page.running


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


def test_stock_voice_instead_of_the_clone(page, monkeypatch):
    """Where the clone falls behind (an 8 GB Mac, BENCH): me -> them local in the stock Piper voice, no recording needed;
    the models are that voice's, not the clone's, and the app makes the direction without the clone."""
    from peresmeshnik import app, models, pipeline
    from peresmeshnik.ui.wizard import needed_models
    monkeypatch.setattr(models, 'is_fetched', lambda m, base=None: True)
    assert not page.isComplete() and not page.stock.isHidden()
    page.stock.setChecked(True)
    s = page.w.s
    assert page.isComplete() and page.check_button.isHidden() and s.local.clone == voice.STOCK
    ids = needed_models(s)
    assert s.local.tts[s.their_lang] in ids and 'xtts-v2' not in ids
    made = []
    monkeypatch.setattr(pipeline, 'make_local', lambda *a, **k: made.append((a, k)) or type('D', (), {'warm_up': lambda self: None})())
    side = app.Side.__new__(app.Side)
    side.name, side.src, side.tgt = 'me', s.my_lang, s.their_lang
    side._local(s)
    assert made == [((s.my_lang, s.their_lang), {'settings': s})]  # no clone, no voice profile
    page.stock.setChecked(False)
    assert s.local.clone == 'xtts-v2' and not page.isComplete()


# --- the voice pack (Windows): the GPU decides, the page installs the pack

@pytest.mark.parametrize('smi, vram, rec', [
    ('12288\n', 12.0, 'local'), ('4096\n', 4.0, 'cloud'), ('4096\n8192\n', 8.0, 'local'),
    ('NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.\n', 0.0, 'cloud'), (None, 0.0, 'cloud'),
])
def test_tier_by_vram(tmp_path, monkeypatch, smi, vram, rec):
    from peresmeshnik.ui import wizard
    monkeypatch.setattr(wizard.shutil, 'which', lambda name: 'nvidia-smi' if smi is not None and name == 'nvidia-smi' else None)
    monkeypatch.setattr(wizard.subprocess, 'run', lambda cmd, **k: types.SimpleNamespace(  # a script would not run on Windows
        stdout=smi if cmd[1:] == ['--query-gpu=memory.total', '--format=csv,noheader,nounits'] else ''))
    assert wizard.gpu_vram_gb() == vram
    what, r = wizard.tier()
    assert r == rec and (f'an NVIDIA GPU with {vram:.0f} GB' if vram else 'no NVIDIA GPU') in what


@pytest.fixture
def windows(page, tmp_path, monkeypatch):
    """The voice page as on Windows, with `vram` GB and the voice pack folder in tmp_path."""
    from peresmeshnik.ui import wizard
    monkeypatch.setattr(wizard, 'WINDOWS', True)
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    monkeypatch.setattr(wizard.voicepack, 'folder', lambda: tmp_path / 'voice-pack')

    def at(vram):
        monkeypatch.setattr(wizard, 'gpu_vram_gb', lambda: vram)
        page.initializePage()
        return page
    return at


@pytest.mark.parametrize('vram, says', [(0.0, 'this computer has no NVIDIA GPU'),
                                        (4.0, 'this computer has an NVIDIA GPU with 4 GB')])
def test_no_gpu_for_the_clone_says_why(windows, vram, says):
    page = windows(vram)
    assert page.pack() == 'gpu' and says in page.pack_out.text() and 'stock voice' in page.pack_out.text()
    assert page.pack_button.isHidden() and page.pack_consent.isHidden() and not page.pack_out.isHidden()
    page.stock.setChecked(True)
    assert page.pack() == '' and page.pack_out.isHidden()


def test_install_the_pack_from_the_page(windows, monkeypatch, tmp_path):
    from peresmeshnik.ui import wizard
    page = windows(8.0)
    assert page.pack() == 'none' and not page.pack_button.isHidden() and page.pack_button.text() == 'Install the voice pack'
    assert 'GB to download' in page.pack_out.text() and 'GB on disk' in page.pack_out.text()
    assert 'noncommercial' in page.pack_consent.text() and not page.pack_consent.isHidden()
    assert not page.pack_button.isEnabled()  # the licenses (CPML) first: nothing is downloaded before
    monkeypatch.setattr(wizard.voicepack, 'install', lambda *a: pytest.fail('installed without the consent'))
    page.install_pack()
    assert not page.packing
    page.pack_consent.setChecked(True)
    assert page.pack_button.isEnabled()
    good = tmp_path / 'me.wav'
    sf.write(good, speech_like(45), 24_000)
    page.import_file(str(good))
    pump(lambda: not page.running)
    assert not page.check_button.isEnabled()  # no pack yet: the check would only say so

    def install(progress, cancel):  # downloads 1 GB of 4, then waits for Cancel
        progress('cleaning', 0, 0)
        progress('downloading', 10 ** 9, 4 * 10 ** 9, 'torch.whl')
        assert cancel.wait(20)
        raise wizard.voicepack.Cancelled
    monkeypatch.setattr(wizard.voicepack, 'install', install)
    page.install_pack()
    pump(lambda: 'Downloading' in page.pack_out.text())
    assert page.pack() == 'installing' and not page.isComplete() and not page.stock.isEnabled()
    assert not page.pack_bar.isHidden() and page.pack_bar.value() == 250 and not page.pack_cancel.isHidden()
    assert page.pack_out.text() == 'Downloading the voice pack: 1.00 of 4.00 GB (torch.whl)'
    page.pack_cancel.click()
    pump(lambda: not page.packing)
    (tmp_path / 'voice-pack.download').mkdir()  # what the real install leaves after a cancel
    page._show_pack()
    assert page.pack_out.text().startswith('Cancelled: what is downloaded stays for the next try. The voice pack install '
                                           'did not finish') and page.pack_button.text() == 'Resume the install'
    assert page.isComplete() and page.pack_cancel.isHidden()

    def done(progress, cancel):
        (tmp_path / 'voice-pack').mkdir()
        (tmp_path / 'voice-pack' / '.complete').write_text(wizard.voicepack.identity())
    monkeypatch.setattr(wizard.voicepack, 'install', done)
    page.install_pack()
    pump(lambda: not page.packing)
    assert page.pack() == 'ready' and page.pack_out.text() == 'The voice pack is installed.' and page.pack_button.isHidden()
    assert page.check_button.isEnabled()  # the clone check now goes through the pack's runner


def test_an_error_and_an_outdated_pack(windows, monkeypatch, tmp_path):
    from peresmeshnik.ui import wizard
    page = windows(8.0)
    page.pack_consent.setChecked(True)

    def full(progress, cancel):
        raise wizard.voicepack.PackError('the voice pack needs 12.6 GB free on C:\\, there are 3.1 GB')
    monkeypatch.setattr(wizard.voicepack, 'install', full)
    page.install_pack()
    pump(lambda: not page.packing)
    assert page.pack_out.text().startswith('error: the voice pack needs 12.6 GB free on C:\\, there are 3.1 GB. The clone')
    (tmp_path / 'voice-pack').mkdir()
    (tmp_path / 'voice-pack' / '.complete').write_text('the lock of an older app\n')
    page.pack_note = ''
    page._show_pack()
    assert page.pack() == 'outdated' and 'installed for another version' in page.pack_out.text()
    assert page.pack_button.text() == 'Install the voice pack'

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
    from peresmeshnik import languages
    real = languages.clone_why  # an NVIDIA GPU for the clone: it is offered (the rest of the GPU checks as they are)
    monkeypatch.setattr(languages, 'clone_why', lambda model, lang='', has_gpu=None: real(model, lang, True))
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    s = config.Settings()
    s.me_to_them = 'local'
    p = VoicePage(type('W', (), {'s': s})())
    p.initializePage()
    p.own.setChecked(True)  # "this is my own voice" (M-20)
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
    monkeypatch.setitem(sys.modules, 'peresmeshnik.local.clone',
                        types.SimpleNamespace(make_engine=lambda *a, **k: FakeClone(sim), voice_cmd=lambda: None))
    from peresmeshnik import models
    from peresmeshnik.local import asr
    monkeypatch.setattr(asr, 'OnnxAsr', lambda *a, **k: FakeAsr(list(hears)))
    monkeypatch.setattr(models, 'path', lambda model_id: tmp_path)
    page.check_clone()
    pump(lambda: not page.running)
    assert page.clone_out.text().startswith(verdict), page.clone_out.text()


def test_clone_check_runs_a_gpu_only_asr_offline(page, tmp_path, monkeypatch):
    """Turkish on an NVIDIA GPU without CUDA in onnxruntime (the .exe): Whisper turbo is not offered live, yet the check's
    two phrases run it offline on the CPU (slow is fine there)."""
    import types

    from peresmeshnik import languages, models
    from peresmeshnik.local import asr, device
    good = tmp_path / 'me.wav'
    sf.write(good, speech_like(45), 24_000)
    page.import_file(str(good))
    pump(lambda: not page.running)
    monkeypatch.setattr(languages.shutil, 'which', lambda name: '/usr/bin/nvidia-smi')
    monkeypatch.setattr(device, 'onnx_cuda', lambda: False)
    page.w.s.their_lang = 'tr'
    assert languages.direction(page.w.s, 'them_to_me').asr == ''  # not live
    phrases = ['bir iki üç dört', 'yarın görüşürüz']
    monkeypatch.setitem(voice.TEST_PHRASES, 'tr', phrases)  # voice.py has en / ru only
    made = []
    monkeypatch.setitem(sys.modules, 'peresmeshnik.local.clone',
                        types.SimpleNamespace(make_engine=lambda *a, **k: FakeClone(0.6), voice_cmd=lambda: None))
    monkeypatch.setattr(asr, 'OnnxAsr', lambda *a, **k: made.append((a, k)) or FakeAsr(list(phrases)))
    monkeypatch.setattr(models, 'path', lambda model_id: tmp_path)
    page.check_clone()
    pump(lambda: not page.running)
    assert made[0][0][0] == 'whisper-large-v3-turbo' and made[0][1] == {'live': False}
    assert page.clone_out.text().startswith('Good'), page.clone_out.text()


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


def test_no_clone_check_without_an_asr_model_of_their_language(page, tmp_path):
    """The check hears the clone with an ASR model of their language: without one (none set for German here) the button
    says why instead of failing with "unknown model ''"."""
    good = tmp_path / 'me.wav'
    sf.write(good, speech_like(45), 24_000)
    page.import_file(str(good))
    pump(lambda: not page.running)
    page.w.s.their_lang = 'de'
    assert page.check_button.isEnabled()  # stage 6: Parakeet hears German
    page.w.s.local.asr = {'en': page.w.s.local.asr['en']}
    page._buttons()
    assert not page.check_button.isEnabled() and 'No speech recognition model for German' in page.check_button.toolTip()
    page.check_clone()
    assert page.clone_out.text().startswith('No speech recognition model for German') and not page.running


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
    # no clone, no voice profile: the stock voice and the ids languages.direction picked
    assert made == [((s.my_lang, s.their_lang), {'tts': s.local.tts[s.their_lang], 'asr': s.local.asr[s.my_lang],
                                                  'mt': s.local.mt[f'{s.my_lang}-{s.their_lang}'], 'settings': s})]
    page.stock.setChecked(False)
    assert s.local.clone == 'xtts-v2' and not page.isComplete()


def test_a_clone_that_dies_with_no_stock_voice_restarts_the_side(monkeypatch, tmp_path):
    """Mid-call the clone dies and its stock fallback cannot start (its model is not on disk: the real lookup, no
    models.path stand-in): the side does not go silent, it fails, so the controller restarts it or stops it and says
    why (before: None, the call heard nothing until translation went off)."""
    import threading

    from peresmeshnik import app, config
    from peresmeshnik.audio.mic import LiveSwitch
    from peresmeshnik.engines import Voice
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))  # nothing fetched

    class Dead:  # the clone's direction after its speak worker died
        voice, closed = Voice('me', 'xtts', tmp_path), False

        def __init__(self):
            try:
                self._speak()
            except RuntimeError as e:
                self._error = e

        def _speak(self):
            raise RuntimeError('the voice pack exited (-11)')

        def events(self):
            return iter(())

        def feed(self, pcm):
            pass

        def close(self):
            self.closed = True
    s = config.Settings()
    s.me_to_them, s.local.voice = 'local', 'me'
    side = app.Side.__new__(app.Side)
    side.name, side.src, side.tgt, side.engine, side.settings, side.live = 'me', s.my_lang, s.their_lang, 'local', s, False
    side._lock, side.errors, failures = threading.Lock(), [], []
    side.on_error, side.on_failure = side.errors.append, lambda why, e: failures.append(why)
    dead = Dead()
    side.directions = [(LiveSwitch(dead), True, True)]
    side._consume(side.directions[0][0], True, True)
    assert dead.closed and len(failures) == 1, (failures, side.errors)
    assert 'the voice clone stopped (the voice pack exited (-11)) and the stock voice does not start' in failures[0]
    assert 'is not downloaded' in failures[0]  # models.ModelMissing, said by the controller's restart


# --- the voice pack (Windows): the GPU decides, the page installs the pack

@pytest.mark.parametrize('smi, vram, rec', [
    ('12288, 572.83, 8.6\n', 12.0, 'local'), ('4096, 572.83, 7.5\n', 4.0, 'cloud'),
    ('4096, 572.83, 7.5\n8192, 572.83, 8.9\n', 8.0, 'local'),
    ('NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.\n', 0.0, 'cloud'), (None, 0.0, 'cloud'),
])
def test_tier_by_vram(tmp_path, monkeypatch, smi, vram, rec):
    from peresmeshnik.ui import wizard
    fake_smi(monkeypatch, smi)
    assert wizard.gpu().vram == vram
    what, r = wizard.tier()
    assert r == rec and (f'an NVIDIA GPU with {vram:.0f} GB' if vram else 'no NVIDIA GPU') in what


def fake_smi(monkeypatch, out, old=None):
    """nvidia-smi answering `out` to the full query, `old` to the one without compute_cap (a driver before R510)."""
    from peresmeshnik import languages
    monkeypatch.setattr(languages, '_PROBED', {})  # nvidia_once: this nvidia-smi is asked anew
    monkeypatch.setattr(languages.shutil, 'which', lambda name: 'nvidia-smi' if out is not None and name == 'nvidia-smi' else None)
    answers = {'--query-gpu=memory.total,driver_version,compute_cap': out, '--query-gpu=memory.total,driver_version': old}
    monkeypatch.setattr(languages.subprocess, 'run', lambda cmd, **k: types.SimpleNamespace(  # a script would not run on Windows
        stdout=answers.get(cmd[1], '') or '', returncode=0 if answers.get(cmd[1]) else 6))


@pytest.mark.parametrize('out, old, says', [
    ('8192, 572.83, 8.6\n', None, ''),
    ('8192, 566.36, 8.6\n', None, ''),  # 528.33 .. 570.64: not refused (CUDA 12.x compatibility), a note instead
    ('Field "compute_cap" is not a valid field to query.\n', '8192, 472.12\n',
     'needs NVIDIA driver 570.65 or newer; this computer has 472.12'),
    ('8192, 572.83, 6.1\n', None, ''),  # Pascal (GTX 1070): the Windows torch has sm_61
    ('6144, 572.83, 5.2\n', None, 'too old for the voice pack (compute capability 5.2; it needs 6.1 or newer'),
    ('NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.\n', None,
     'The NVIDIA driver does not answer (NVIDIA-SMI has failed'),
])
def test_the_gpu_is_checked_before_the_download(monkeypatch, out, old, says):
    """Memory, driver and compute capability: an old driver, a Pascal card or a driver that does not answer is said
    before the pack's 3.7 GB, not found after them."""
    from peresmeshnik.ui import wizard
    fake_smi(monkeypatch, out, old)
    g = wizard.gpu()
    why = wizard.gpu_refusal(g)
    assert (says in why) if says else why == '', why
    note = wizard.gpu_note(g)
    assert ('566.36 is older than 570.65' in note and 'not checked, it may work' in note) if '566.36' in out else note == ''


@pytest.fixture
def windows(page, tmp_path, monkeypatch):
    """The voice page as on Windows, with `vram` GB and the voice pack folder in tmp_path."""
    from peresmeshnik.ui import wizard
    monkeypatch.setattr(wizard, 'WINDOWS', True)
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    monkeypatch.setattr(wizard.voicepack, 'folder', lambda: tmp_path / 'voice-pack')

    def at(vram):
        monkeypatch.setattr(wizard, 'gpu', lambda: wizard.Gpu(vram, (572, 83), 8.6) if vram else wizard.Gpu())
        page.gpu, page._asking = None, False  # the page fixture showed it off Windows already
        page.initializePage()
        assert page.pack() == 'checking' and page.pack_out.text() == 'Checking the NVIDIA GPU…'  # nvidia-smi: a worker
        end = time.monotonic() + 5
        while page.gpu is None and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
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
    assert 'NVIDIA CUDA' in page.pack_consent.text()  # 3.5 GB of the pack are NVIDIA's libraries (L-12)
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


def test_the_voice_is_yours_before_it_is_recorded_or_imported(page):
    """M-20: "your own voice only" holds in the wizard: recording and importing wait for the box that says the voice is
    the user's own."""
    page.own.setChecked(False)  # the fixture ticked it
    assert not page.own.isHidden()
    assert not page.record_button.isEnabled() and not page.import_button.isEnabled()
    page.own.setChecked(True)
    assert page.record_button.isEnabled() and page.import_button.isEnabled()


def test_where_the_pack_cannot_run_the_clone_the_page_wants_the_stock_voice(windows, monkeypatch):
    """S3-01 (b): a GPU the voice pack refuses (4 GB): a recording alone does not finish the page, the stock voice does."""
    from peresmeshnik import models
    # the stock voice is here: nothing to download (Pocket TTS is not: with it the CPU would clone, no pack needed)
    monkeypatch.setattr(models, 'is_fetched', lambda m, base=None: not m.id.startswith('pocket-tts'))
    page = windows(4.0)
    page.voice = types.SimpleNamespace(name='me.new')  # a good take
    assert page.pack() == 'gpu' and not page.isComplete()
    page.stock.setChecked(True)
    assert page.isComplete()


def test_no_stock_voice_box_where_there_is_none(page):
    """M-24: Chinese has no stock voice: the box offering one is not shown."""
    page.w.s.their_lang = 'zh'
    page.initializePage()
    assert page.stock.isHidden()
    page.w.s.their_lang = 'en'
    page.initializePage()
    assert not page.stock.isHidden()



def test_no_dead_end_where_there_is_no_stock_voice(page, windows):
    """FIXES-b 3: Chinese (no stock voice) and a GPU the voice pack refuses: the page sends to the first page's cloud, not
    to a stock-voice box that is not there."""
    page.w.s.their_lang = 'zh'
    windows(4.0)
    text = page.pack_out.text()
    assert page.pack() == 'gpu' and page.stock.isHidden() and not page.isComplete()
    assert 'Choose cloud for "me → them" on the first page' in text and 'stock voice above' not in text, text


def test_pocket_on_windows_needs_neither_the_voice_pack_nor_a_gpu(page, tmp_path, monkeypatch):
    """FIXES 2: Windows without a fitting NVIDIA GPU, English: Pocket TTS clones on the CPU in the app itself, so the
    page neither offers the 3.7 GB voice pack nor holds back on the GPU once the recording is there."""
    from peresmeshnik import languages, models
    from peresmeshnik.ui import wizard
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')  # Kyutai's gated weights: with the user's token
    monkeypatch.setattr(languages, 'pack_here', lambda: False)
    monkeypatch.setattr(wizard, 'WINDOWS', True)
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    page.gpu = wizard.Gpu()  # no NVIDIA GPU
    assert languages.direction(page.w.s, 'me_to_them').tts == 'pocket-tts-en'
    good = tmp_path / 'me.wav'
    sf.write(good, speech_like(45), 24_000)
    page.import_file(str(good))
    pump(lambda: not page.running)
    assert page.pack() == '' and page.isComplete()

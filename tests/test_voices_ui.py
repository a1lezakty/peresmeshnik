"""The voices window: the list says whose voice and whether it may speak into the call, delete one or all the contacts',
confirm my voice by reading two phrases (ConfirmBox), keep a contact's voice only with their agreement."""
import os
import sys

import numpy as np
import pytest

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from test_clone_wizard import pump  # noqa: E402
from test_voice_profiles import Asr, Speaker, speech  # noqa: E402

from peresmeshnik import config, voice  # noqa: E402
from peresmeshnik.ui import voices  # noqa: E402

app = QApplication.instance() or QApplication([])


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)
    voice.save(*speech(), 'me', 'ru', '', tmp_path / 'voices')  # imported: not confirmed
    voice.save_them('Анна', 'en', 9.0, {}, tmp_path / 'voices')
    voice.save_them('Bob', 'de', 4.0, {}, tmp_path / 'voices')
    return tmp_path / 'voices'


def test_list_and_delete(root):
    s, changed = config.Settings(), []
    s.local.voice = 'me'
    d = voices.VoicesDialog(s, changed.append)
    rows = [d.list.item(i).text() for i in range(d.list.count())]
    assert len(rows) == 3 and rows[0].startswith('Your voice: ru, 40 s, not confirmed yet'), rows
    assert any(r.startswith('Анна: someone you talk to, en, 9 s') for r in rows)
    d.list.setCurrentRow(0)
    assert d.confirm_button.isEnabled() and d.delete_theirs.isEnabled()
    d.delete_all_theirs()
    assert [v.meta['kind'] for v in voice.profiles()] == ['me'] and changed == [False] and not d.delete_theirs.isEnabled()
    d.list.setCurrentRow(0)
    d.delete()
    assert voice.profiles() == [] and s.local.voice == '' and changed == [False, True]
    assert d.list.item(0).text().startswith('No voices yet')


@pytest.mark.parametrize('sim', [0.8, 0.1])
def test_confirm_my_voice(root, monkeypatch, sim):
    s = config.Settings()
    monkeypatch.setattr('peresmeshnik.audio.mic.record', lambda sec, dev, sr, progress: speech(sec, sr)[0])
    box = voices.ConfirmBox(s, 'me', 'import+mic-match')
    said = box.said
    monkeypatch.setattr(voices, 'check_models', lambda s, lang, say: (Asr(' '.join(said)), Speaker(sim)))
    got = []
    box.confirmed.connect(got.append)
    box.start()
    pump(lambda: not box.running)
    meta = voice.load('me').meta
    if sim > voice.MATCH:
        assert box.out.text().startswith('✔ Confirmed') and got and meta['consent']['how'] == 'import+mic-match'
        assert voice.mine('me')
    else:
        assert 'not the voice of the recording' in box.out.text() and meta['consent'] is None and not got
        assert all(f'<b>{x}</b>' in box.phrases.text() for x in box.said)  # a new pick shown for the next try


def test_keep_a_contacts_voice_only_as_agreed(tmp_path):
    d = voices.RememberDialog()
    assert not d.save.isEnabled()
    d.name.setText('Анна')
    assert not d.save.isEnabled()  # not without their agreement
    d.agreed.setChecked(True)
    assert d.save.isEnabled()
    v = d.keep({'pocket-tts-en': (np.zeros(3, np.float32), 'sha')}, 'en', 5.0, tmp_path)
    assert v.meta['kind'] == 'them' and v.meta['name'] == 'Анна' and voice.why_not_mine(v.meta)


def test_wizard_voice_page_confirms_the_take(tmp_path, monkeypatch):
    """The wizard's voice page: an imported take is not mine until two phrases are read at the microphone (the box under
    it); no Next while that records; Finish keeps the consent with the profile."""
    import soundfile as sf
    from test_clone_wizard import speech_like

    from peresmeshnik import languages
    from peresmeshnik.ui import wizard
    monkeypatch.setattr(wizard, 'WINDOWS', False)
    real = languages.clone_why
    monkeypatch.setattr(languages, 'clone_why', lambda model, lang='', has_gpu=None: real(model, lang, True))
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    s = config.Settings()
    s.me_to_them = 'local'
    page = wizard.VoicePage(type('W', (), {'s': s})())
    page.initializePage()
    page.own.setChecked(True)
    assert page.confirm is None
    sf.write(tmp_path / 'me.wav', speech_like(45), 24_000)
    page.import_file(str(tmp_path / 'me.wav'))
    pump(lambda: not page.running)
    assert page.confirm is not None and page.confirm.how == 'import+mic-match' and page.isComplete()
    monkeypatch.setattr('peresmeshnik.audio.mic.record', lambda sec, dev, sr, progress: speech(sec, sr)[0])
    said = page.confirm.said
    monkeypatch.setattr(voices, 'check_models', lambda s, lang, say: (Asr(' '.join(said)), Speaker(0.9)))
    page.confirm.start()
    assert not page.isComplete()  # it records into the take
    pump(lambda: not page.confirm.running)
    assert page.isComplete() and voice.mine('me.new')
    page.commit()
    assert voice.mine('me').meta['consent']['how'] == 'import+mic-match' and s.local.voice == 'me'


def test_tray_voices_and_keeping_a_contacts_voice(tmp_path, monkeypatch):
    """The tray: "Voices…" opens the window (my voice deleted there: the settings forget it); "Keep the voice of the
    person in this call…" only while translating with S2's them_voice(), and only through the agreement dialog."""
    from peresmeshnik.ui import tray as tray_mod
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)
    s = config.Settings()
    s.local.voice = voice.save(*speech(), 'me', 'ru', 'text').name
    config.save(s)
    live = [None]
    ctl = type('Ctl', (), {'live': False, 'on': True, 'them_voice': lambda self: live[0]})()
    t = tray_mod.Tray(ctl)
    t.restart = lambda: None
    said = []
    t.message = said.append
    assert t.remember.isVisible() and [a.text() for a in t.menu.actions()].count('Voices…') == 1
    t.open_voices()
    d = t.windows[-1]
    d.list.setCurrentRow(0)
    d.delete()
    assert config.load().local.voice == '' and voice.profiles() == []
    t.remember_voice()
    assert said == ['Their voice is not ready yet: it takes a few seconds of their speech.']
    live[0] = ({'pocket-tts-en': (np.ones(4, np.float32), 'sha')}, 'en', 12.0)

    def agree(dialog):
        dialog.name.setText('Bob')
        dialog.agreed.setChecked(True)
        return True
    monkeypatch.setattr(voices.RememberDialog, 'exec', agree)
    t.remember_voice()
    (v,) = voice.profiles()
    assert v.meta['kind'] == 'them' and v.meta['name'] == 'Bob' and said[-1].startswith('Kept')
    t.ctl = type('Ctl', (), {'live': False, 'on': False})()  # no S2 hook, translation off: not offered
    t.sync()
    assert not t.remember.isVisible()

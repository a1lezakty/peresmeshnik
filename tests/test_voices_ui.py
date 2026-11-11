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
    voice.save(*speech(), 'me.new', 'ru', '', root)  # a take the wizard kept aside
    voice.save(*speech(), 'me.old', 'ru', '', root)
    (root / 'them-0a1b2c3d.tmp').mkdir()  # a contact's voice whose save stopped (FIXES-s3 6)
    d = voices.VoicesDialog(s, changed.append)
    rows = [d.list.item(i).text() for i in range(d.list.count())]
    assert len(rows) == 4 and rows[0].startswith('Your voice: ru, 40 s, it is not confirmed as your own voice yet'), rows
    assert any(r.startswith('Анна: someone you talk to, en, 9 s') for r in rows)
    assert rows[3] == 'them-0a1b2c3d.tmp: damaged (no readable voice.json) — delete'
    d.list.setCurrentRow(3)
    assert d.delete_button.isEnabled() and not d.confirm_button.isEnabled()
    d.delete()
    assert not (root / 'them-0a1b2c3d.tmp').exists() and changed == []
    d.list.setCurrentRow(0)
    assert d.confirm_button.isEnabled() and d.delete_theirs.isEnabled()
    d.delete_all_theirs()
    assert [v.meta['kind'] for v in voice.profiles()] == ['me'] and changed == [False] and not d.delete_theirs.isEnabled()
    d.list.setCurrentRow(0)
    d.delete()
    assert voice.profiles() == [] and s.local.voice == '' and changed == [False, True]
    assert sorted(p.name for p in root.iterdir()) == []  # me.new and me.old went with it
    assert d.list.item(0).text().startswith('No voices yet')


@pytest.mark.parametrize('sim', [0.8, 0.1])
def test_confirm_my_voice(root, monkeypatch, sim):
    s = config.Settings()
    monkeypatch.setattr('peresmeshnik.audio.mic.record', lambda sec, dev, sr, progress: speech(sec, sr)[0])
    box = voices.ConfirmBox(s, 'me')
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
    assert v.meta['kind'] == 'them' and v.meta['name'] == 'Анна' and voice.why_not_mine(v)


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
    assert page.confirm is not None and page.isComplete()
    import threading
    go = threading.Event()
    monkeypatch.setattr('peresmeshnik.audio.mic.record', lambda sec, dev, sr, progress: (go.wait(10), speech(sec, sr)[0])[1])
    said = page.confirm.said
    monkeypatch.setattr(voices, 'check_models', lambda s, lang, say: (Asr(' '.join(said)), Speaker(0.9)))
    page.confirm.start()
    assert not page.isComplete()  # it records into the take: no Next, no new take, no clone check (FIXES-s3 3)
    assert not page.record_button.isEnabled() and not page.import_button.isEnabled() and not page.check_button.isEnabled()
    box = page.confirm
    page.import_file(str(tmp_path / 'me.wav'))
    assert page.confirm is box and not page.running
    go.set()
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
    live = [(None, 'not enough of their speech yet: 2 of 10 s')]
    ctl = type('Ctl', (), {'live': False, 'on': True, 'them_voice': lambda self: live.pop(0) if len(live) > 1 else live[0]})()
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
    assert said == ['Their voice cannot be kept now: not enough of their speech yet: 2 of 10 s.']

    def agree(dialog):
        dialog.name.setText('Bob')
        dialog.agreed.setChecked(True)
        return True
    monkeypatch.setattr(voices.RememberDialog, 'exec', agree)
    alone = ({'pocket-tts-en': (np.ones(4, np.float32), 'sha')}, 'en', 12.0)
    live[:] = [(alone, ''), (None, '2 people speak in this call: a voice can be kept in a call with one of them only')]
    t.remember_voice()  # someone else spoke while the dialog was open: the voice of then is not kept silently
    assert voice.profiles() == [] and said[-1] == ('Their voice was not kept: 2 people speak in this call: a voice can '
                                                   'be kept in a call with one of them only.')
    live[:] = [(alone, '')]
    t.remember_voice()
    (v,) = voice.profiles()
    assert v.meta['kind'] == 'them' and v.meta['name'] == 'Bob' and said[-1].startswith('Kept')
    t.ctl = type('Ctl', (), {'live': False, 'on': False})()  # no S2 hook, translation off: not offered
    t.sync()
    assert not t.remember.isVisible()


def test_tray_their_own_voice_switch_and_what_is_not_downloaded(tmp_path, monkeypatch):
    """FIXES-s2 r2 11, 8: "Their own voice (them -> me)" in the tray sets [local] their_voice (a running translation
    restarts, and a missing engine is said); a pair's missing clone engines are said per side: theirs as theirs, the
    speaker model waits with them."""
    from peresmeshnik import languages, models
    from peresmeshnik.ui import tray as tray_mod
    from peresmeshnik.voices import live
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    config.save(config.Settings(my_lang='ru', their_lang='en'))
    t = tray_mod.Tray(type('Ctl', (), {'live': False, 'on': False})())
    said, restarts = [], []
    t.message, t.restart = said.append, lambda: restarts.append(1)
    assert t.their_voice.isChecked()
    t.their_voice.setChecked(False)
    assert config.load().local.their_voice is False and restarts == [1] and said == []
    t.their_voice.setChecked(True)
    assert config.load().local.their_voice is True and 'their voice (pocket-tts-ru' in said[-1]
    s = config.Settings(my_lang='ru', their_lang='en', them_to_me='local', me_to_them='cloud')
    todo = [models.get(i) for i in languages.needed_models(s)]
    assert {'pocket-tts-ru', 'wespeaker-resnet34'} <= {m.id for m in todo}
    rest, notes = tray_mod.clone_notes(s, todo)
    assert {m.id for m in rest} == {m.id for m in todo} - {'pocket-tts-ru', 'wespeaker-resnet34'}
    assert notes == ['Them → me: their voice (pocket-tts-ru, 0.4 GB) is not downloaded: a stock voice speaks for them; '
                     'Components… in this menu downloads it']  # not "Me → them: your cloned voice"
    monkeypatch.setattr(live, 'engine', lambda s: (None, 'off'))
    assert tray_mod.clone_notes(s, [m for m in todo if m.id != 'pocket-tts-ru'])[1] == []


def test_models_page_asks_for_the_hugging_face_token(monkeypatch):
    """A gated model (Kyutai's Pocket TTS, `token = "hf"`) to download: the Components page shows its page and the token
    field; the token goes to models.save_hf_token (S1's: its own file); once fetched, the box goes."""
    from peresmeshnik import components, models
    from peresmeshnik.ui import components as box_mod
    from peresmeshnik.ui import wizard

    def pocket(installed):
        return components.Component('pocket-tts-en', 'Your cloned voice: pocket-tts-en', 0 if installed else 1, 'CC-BY-4.0',
                                    'https://huggingface.co/kyutai/pocket-tts', 'speaks', installed, 0, True, token=True)
    fetched, saved = [False], []
    monkeypatch.setattr(components, 'listing', lambda s: [pocket(fetched[0])])
    monkeypatch.setattr(models, 'hf_token', lambda: saved[-1] if saved else '', raising=False)
    monkeypatch.setattr(models, 'save_hf_token', saved.append, raising=False)
    opened = []
    monkeypatch.setattr(box_mod.QDesktopServices, 'openUrl', lambda u: opened.append(u.toString()))
    page = wizard.ModelsPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    hf = page.box.hf
    assert not hf.isHidden() and hf.token.placeholderText() == 'hf_…'
    hf.open.click()
    assert opened == ['https://huggingface.co/kyutai/pocket-tts']
    hf.token.setText(' hf_secret ')
    hf.save.click()
    assert saved == ['hf_secret'] and hf.token.text() == '' and 'saved' in hf.token.placeholderText()
    fetched[0] = True
    page.box.refresh()
    assert hf.isHidden()


def wizard_page(tmp_path, monkeypatch):
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
    return page, s


def test_wizard_recorded_take_confirmed_after_finish(tmp_path, monkeypatch):
    """FIXES-s3 10: Record (the app's own recording: 'mic'), the box, Finish: the profile is mine, 'mic-live'."""
    from test_clone_wizard import speech_like
    page, s = wizard_page(tmp_path, monkeypatch)
    monkeypatch.setattr('peresmeshnik.audio.mic.record', lambda sec, dev, sr, progress: speech_like(sec, sr))
    page.record()
    pump(lambda: not page.running)
    assert page.voice.meta['source'] == 'mic' and page.confirm is not None
    said = page.confirm.said
    monkeypatch.setattr(voices, 'check_models', lambda s, lang, say: (Asr(' '.join(said)), Speaker(0.9)))
    page.confirm.start()
    pump(lambda: not page.confirm.running)
    page.commit()
    assert voice.mine('me').meta['consent']['how'] == 'mic-live' and s.local.voice == 'me'


def test_wizard_no_box_where_the_clone_does_not_speak(tmp_path, monkeypatch):
    """FIXES-s3 8: the stock voice speaks for this pair (clone_off): nothing to confirm."""
    import soundfile as sf
    from test_clone_wizard import speech_like
    page, _ = wizard_page(tmp_path, monkeypatch)
    monkeypatch.setattr(page, 'clone_off', lambda: 'the clone cannot speak Korean here')
    sf.write(tmp_path / 'me.wav', speech_like(45), 24_000)
    page.import_file(str(tmp_path / 'me.wav'))
    pump(lambda: not page.running)
    assert page.voice is not None and page.confirm is None


def test_voices_window_a_box_per_profile(root, monkeypatch):
    """FIXES-s3 8: a box per profile (a second one can be confirmed too); no Delete of the one being checked; not
    while translating; a language without phrases says so."""
    import threading
    voice.save(*speech(), 'Petrov', 'ru', '', root)  # a v1-era profile named by the user
    go = threading.Event()
    monkeypatch.setattr('peresmeshnik.audio.mic.record', lambda sec, dev, sr, progress: (go.wait(10), speech(sec, sr)[0])[1])
    monkeypatch.setattr(voices, 'check_models', lambda s, lang, say: (Asr('x'), Speaker(0.9)))
    translating = [False]
    d = voices.VoicesDialog(config.Settings(), translating=lambda: translating[0])
    names = [v.name for v in d.voices]
    d.list.setCurrentRow(names.index('me'))
    d.start_confirm()
    me = d.boxes['me']
    translating[0] = True
    me.start()
    assert not me.running and me.out.text().startswith('Turn translation off first')
    translating[0] = False
    me.start()
    assert me.running and not d.delete_button.isEnabled() and not d.confirm_button.isEnabled()
    d.list.setCurrentRow(names.index('Petrov'))
    assert d.delete_button.isEnabled() and d.confirm_button.isEnabled() and me.isHidden()
    d.start_confirm()
    assert set(d.boxes) == {'me', 'Petrov'} and not d.boxes['Petrov'].isHidden()
    go.set()
    pump(lambda: not me.running)
    voice.save(*speech(), 'Xx', 'xx', '', root)  # a language without phrases (CLI --lang xx)
    box = voices.ConfirmBox(config.Settings(), 'Xx')
    assert not box.button.isEnabled() and box.phrases.text().startswith('No phrases to read in "xx"')


def test_settings_lists_only_my_voices(root, tmp_path, monkeypatch):
    """FIXES-s3 7: Settings → "My voice (local clone)": my profiles by their names, never a contact's."""
    from peresmeshnik.ui import tray as tray_mod
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    voice.save(*speech(), 'Petrov', 'ru', '', root)
    d = tray_mod.SettingsDialog(tray_mod.Tray(type('Ctl', (), {'live': False, 'on': False})()))
    items = [(d.voice.itemText(i), d.voice.itemData(i)) for i in range(d.voice.count())]
    assert items == [('none', ''), ('Petrov', 'Petrov'), ('me', 'me')], items


def test_keep_their_voice_only_where_the_engine_keeps_one(tmp_path, monkeypatch):
    """COMMON 23:49: "Keep the voice…" while translating with S2's them_voice(), only where the them side's clone engine
    keeps a voice (Pocket TTS: its state); not with an engine of the NVIDIA voice pack (XTTS, Qwen3), nor with none."""
    from peresmeshnik import languages
    from peresmeshnik.ui import tray as tray_mod
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    engine = ['pocket-tts-de', '']
    asked = []
    monkeypatch.setattr(languages, 'clone_engine', lambda lang, side='me_to_them', **k: asked.append((lang, side)) or engine)
    t = tray_mod.Tray(type('Ctl', (), {'live': False, 'on': True, 'them_voice': lambda self: None})())
    assert t.remember.isVisible() and asked[-1] == (config.load().my_lang, 'them_to_me')
    for engine[:] in (['xtts-v2', ''], ['qwen3-tts', ''], [None, 'no clone of their voice here']):
        t.sync()
        assert not t.remember.isVisible(), engine
    engine[:] = ['pocket-tts-de', '']
    s = config.load()
    s.local.their_voice = False  # S2's live.engine: their voice off, nothing to keep
    config.save(s)
    t.sync()
    assert not t.remember.isVisible()
    t.icon.hide()


def test_my_voice_deleted_while_the_wizard_is_open(tmp_path, monkeypatch):
    """COMMON 21:53 tail: Voices… deletes my voice (and its take) while the wizard holds the take: the page forgets it
    (Next does not take a voice that is gone), Finish promotes nothing and the settings name no voice."""
    from test_clone_wizard import speech_like
    page, s = wizard_page(tmp_path, monkeypatch)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)  # Delete: yes
    monkeypatch.setattr('peresmeshnik.audio.mic.record', lambda sec, dev, sr, progress: speech_like(sec, sr))
    page.record()
    pump(lambda: not page.running)
    assert page.voice.name == 'me.new' and s.local.voice == 'me.new'
    d = voices.VoicesDialog(s)
    voice.save(*speech(), 'me', 'ru', '', voice.voices_dir())  # the saved voice, deleted with its take
    d.refresh()
    d.list.setCurrentRow(0)
    d.delete()
    assert not (voice.voices_dir() / 'me.new').exists()
    page.initializePage()  # Back, then here again
    assert page.voice is None and 'deleted' in page.out.text()
    page.voice = type('V', (), {'name': 'me.new'})()  # or straight to Finish, the page still holding it
    page.commit()
    assert page.voice is None and s.local.voice == '' and not (voice.voices_dir() / 'me').exists()

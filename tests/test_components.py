"""Components: what the app downloads after install — the pair's models ticked by default, what other pairs left on disk to
remove, the clone engines and the Windows voice pack with its addon offered only where they run; the box downloads only
the ticked ones after Download and says what the pair does without an unticked one."""
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from peresmeshnik import components, config, languages, models  # noqa: E402
from peresmeshnik.languages import Gpu  # noqa: E402
from peresmeshnik.local import voicepack  # noqa: E402

app = QApplication.instance() or QApplication([])
CARD = Gpu(12.0, (580, 0), 8.6)


@pytest.fixture(autouse=True)
def base(tmp_path, monkeypatch):
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    return tmp_path / 'models'


def fetched(base, mid):
    (base / mid).mkdir(parents=True, exist_ok=True)
    (base / mid / 'x.bin').write_bytes(b'x' * 1000)
    (base / mid / models.MARKER).write_text(models.get(mid).fingerprint + '\n')


def test_the_pairs_models_ticked_other_pairs_to_remove(base):
    s = config.Settings(my_lang='ru', their_lang='en', them_to_me='local', me_to_them='cloud')
    fetched(base, 'opus-mt-de-ru')  # an earlier pair's
    fetched(base, 'silero-vad')
    rows = {c.id: c for c in components.listing(s)}
    need = {p.id for i in languages.needed_models(s) for p in models.parts(models.get(i))}
    assert {i for i, c in rows.items() if c.recommended} == need
    vad, asr = rows['silero-vad'], rows['parakeet-tdt-0.6b-v3']
    assert vad.installed and vad.size == 0 and vad.on_disk > 1000 and 'every setup' in vad.why
    assert not asr.installed and asr.size == models.get(asr.id).size and asr.license == 'CC-BY-4.0'
    assert 'hears English' in asr.why and asr.required and 'choose cloud for this direction' in asr.instead  # FIXES 2
    assert vad.required and all(c.required for i, c in rows.items() if i in need)  # no clone here: all of it runs the pair
    old = rows['opus-mt-de-ru']
    assert old.installed and not old.recommended and 'not used' in old.why
    clone = rows['xtts-v2']  # me -> them in the cloud: offered, not ticked; no NVIDIA GPU here: why not
    assert not clone.recommended and clone.unavailable == languages.clone_why('xtts-v2', 'en', Gpu()) and clone.consent
    old.remove()
    assert not (base / 'opus-mt-de-ru').exists() and 'opus-mt-de-ru' not in {c.id for c in components.listing(s)}


def test_the_clone_brings_the_voice_check_and_its_stock_voice(monkeypatch):
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: CARD)
    s = config.Settings(my_lang='ru', their_lang='en', them_to_me='cloud', me_to_them='local')
    rows = {c.id: c for c in components.listing(s, CARD)}
    clone, check, stock = rows['xtts-v2'], rows['wespeaker-resnet34'], rows['piper-en_US-john-medium']
    assert clone.recommended and not clone.unavailable and 'your own voice' in clone.why
    assert 'a stock voice speaks instead' == clone.instead
    assert check.recommended and 'stock voice speaks into the call' in check.instead
    assert stock.recommended and 'if your cloned voice fails' in stock.why
    s = config.Settings(my_lang='ru', their_lang='en', them_to_me='local', me_to_them='cloud')  # local subtitles
    voice = {c.id: c for c in components.listing(s, CARD)}['piper-en_US-john-medium']  # the cloud side's fallback voice
    assert voice.required and 'when the cloud fails' in voice.why and voice.instead.startswith('me → them goes on in this voice')


def lock(tmp_path, addon=True):
    f = lambda n: {'url': 'https://x/' + n, 'sha256': '0' * 64, 'size': 1000}  # noqa: E731
    spec = {'python': f('py.zip'), 'pip': f('pip.whl'), 'wheels': [f('torch.whl')], 'archives': [], 'installed_bytes': 8e9}
    if addon:
        spec['addon'] = {'wheels': [f('qwen.whl')], 'archives': [], 'installed_bytes': 2e9, 'models': ['xtts-v2'],
                         'vram_gb': 8}
    (tmp_path / 'pack-windows.json').write_text(json.dumps(spec))
    return tmp_path / 'pack-windows.json'


def test_the_voice_pack_and_its_addon_only_where_they_run(tmp_path, monkeypatch):
    """Windows (the pack's lock in the build): the pack, ticked when the clone needs it; the addon unticked, offered only
    with the GPU memory it needs (S1's `addon` section); no section, no row; no NVIDIA GPU, neither."""
    monkeypatch.setattr(components, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: CARD)
    states = []
    monkeypatch.setattr(voicepack, 'state', lambda addon=False: states.append(addon) or 'none')
    monkeypatch.setattr(voicepack, 'folder', lambda: tmp_path / 'voice-pack')
    monkeypatch.setattr(voicepack, 'lock_file', lambda: lock(tmp_path))
    s = config.Settings(my_lang='ru', their_lang='en', me_to_them='local')
    rows = {c.id: c for c in components.listing(s, CARD)}
    pack, addon = rows[components.VOICE_PACK], rows[components.ADDON]
    assert pack.recommended and pack.size == 3000 and pack.consent and not pack.unavailable and not pack.required
    (tmp_path / 'voice-pack.download').mkdir()
    (tmp_path / 'voice-pack.download' / 'torch.whl.part').write_bytes(b'x' * 400)  # FIXES 7: what is left to download
    assert components.listing(s, CARD)[-2].size == 2600
    assert not addon.recommended and not addon.unavailable and addon.size == 1000 + models.get('xtts-v2').size
    assert True in states and 'xtts-v2' not in rows  # the addon's models come with it, not as rows of their own
    small = Gpu(6.0, (580, 0), 8.6)
    rows = {c.id: c for c in components.listing(s, small)}
    assert '8 GB' in rows[components.ADDON].unavailable and not rows[components.VOICE_PACK].unavailable
    monkeypatch.setattr(voicepack, 'lock_file', lambda: lock(tmp_path, addon=False))
    assert components.ADDON not in {c.id for c in components.listing(s, CARD)}
    rows = {c.id: c for c in components.listing(s, Gpu())}
    assert 'NVIDIA' in rows[components.VOICE_PACK].unavailable and not rows[components.VOICE_PACK].recommended
    from peresmeshnik.ui.components import ComponentsBox
    monkeypatch.setattr(components, 'listing', lambda s: [pack, addon])
    box = ComponentsBox(s)
    box.refresh()  # the box shows them (a field out of place broke it on Windows)
    assert box.tree.topLevelItemCount() == 2 and box.consent.isHidden() and not box.pack_consent.isHidden()  # FIXES 7
    assert 'PyTorch' in box.pack_consent.text()
    box.download()
    assert not box.busy and 'Accept' in box.state.text()
    (tmp_path / 'voice-pack').mkdir()
    (tmp_path / 'voice-pack' / '.complete').write_text('x')
    monkeypatch.setattr(components.shutil, 'rmtree', lambda p, ignore_errors=False: None)  # Windows: files still held
    pack.remove()
    assert not (tmp_path / 'voice-pack' / '.complete').exists()  # FIXES 6: never "ready" half removed


def settle(seconds=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.01)


def item(box, c):
    return next(box.tree.topLevelItem(i) for i in range(box.tree.topLevelItemCount())
                if box.tree.topLevelItem(i).text(0) == c.title)


def test_the_box_downloads_only_the_ticked_and_says_what_goes_without(monkeypatch):
    from peresmeshnik.ui.components import ComponentsBox
    here, got = {'old'}, []

    def row(cid, recommended, consent=False, instead='', required=False):
        def fetch(progress, cancel):
            progress('downloading', 500, 1000, cid)
            got.append(cid)
            here.add(cid)
        return components.Component(cid, f'Row {cid}', 0 if cid in here else 1000, 'MIT', 'https://x', 'for the pair',
                                    cid in here, 2000 if cid in here else 0, recommended, instead, '', consent,
                                    required=required, _fetch=fetch, _remove=lambda: here.discard(cid))
    monkeypatch.setattr(components, 'listing', lambda s: [
        row('asr', True, instead='needed for them → me on this computer', required=True),
        row('clone', True, instead='a stock voice speaks instead'), row('nc', False, True), row('old', False)])
    on, asked = [True], []
    box = ComponentsBox(config.Settings(), translating=lambda: on[0])
    box.confirm = lambda text: asked.append(text) or True
    box.refresh()
    assert [c.id for c in box.missing()] == ['asr', 'clone'] and box.consent.isHidden() and box.without.isHidden()
    asr, clone = item(box, box.rows[0]), item(box, box.rows[1])
    assert not asr.flags() & Qt.ItemFlag.ItemIsUserCheckable and 'Without it: needed for them' in asr.toolTip(0)  # FIXES 2
    asr.setCheckState(0, Qt.CheckState.Unchecked)  # even so: the pair's ASR stays
    assert [c.id for c in box.missing()] == ['asr', 'clone']
    clone.setCheckState(0, Qt.CheckState.Unchecked)  # the clone is a choice: said what speaks instead, Next waits less
    assert [c.id for c in box.missing()] == ['asr'] and 'Without Row clone: a stock voice speaks instead.' in box.without.text()
    item(box, box.rows[2]).setCheckState(0, Qt.CheckState.Checked)  # a restricted license: asked first
    assert not box.consent.isHidden() and 'Row nc (MIT)' in box.consent.text()
    box.download()
    assert got == [] and 'Accept' in box.state.text()
    box.consent.setChecked(True)
    box.download()
    end = time.monotonic() + 5
    while box.busy and time.monotonic() < end:
        settle()
    assert got == ['asr', 'nc'] and not box.missing() and box.state.text().startswith('Everything ticked is here')
    assert 'clone' not in here and 'stock voice' in box.without.text()  # FIXES 4: the user's untick stays
    box.remove(next(c for c in box.rows if c.id == 'old'))  # translating: not now
    assert 'old' in here and 'translation off' in box.state.text() and asked == []
    on[0] = False
    box.remove(next(c for c in box.rows if c.id == 'old'))
    assert 'old' not in here and asked and 'Row old' in asked[0]


def test_a_removal_that_fails_says_why(monkeypatch):
    """FIXES #38 2: Remove's error (a voice pack file still held open, Windows) is shown in the box, not lost in Qt."""
    from peresmeshnik.ui.components import ComponentsBox

    def held():
        raise voicepack.PackError('cannot remove the voice pack (Access is denied): turn translation off first, or try '
                                  'again in a minute')
    monkeypatch.setattr(components, 'listing', lambda s: [components.Component(
        'pack', 'Row pack', 0, 'MIT', 'https://x', 'for the pair', True, 2000, False, _remove=held)])
    box = ComponentsBox(config.Settings())
    box.confirm = lambda text: True
    box.refresh()
    box.remove(box.rows[0])
    assert box.state.text().startswith('Could not remove Row pack: cannot remove the voice pack') and \
        'try again in a minute' in box.state.text()


def test_a_failed_optional_one_does_not_hold_the_pair(monkeypatch):
    from peresmeshnik.ui.components import ComponentsBox
    here = set()

    def row(cid, recommended, fail=False):
        def fetch(progress, cancel):
            if fail:
                raise OSError('network down')
            here.add(cid)
        return components.Component(cid, f'Row {cid}', 1000, 'MIT', 'https://x', '', cid in here, 0, recommended,
                                    _fetch=fetch)
    monkeypatch.setattr(components, 'listing', lambda s: [row('opt', False, fail=True), row('asr', True)])
    box = ComponentsBox(config.Settings())
    box.refresh()
    item(box, box.rows[0]).setCheckState(0, Qt.CheckState.Checked)
    assert [c.id for c in box.todo()] == ['asr', 'opt']  # the pair's own first
    box.download()
    end = time.monotonic() + 5
    while box.busy and time.monotonic() < end:
        settle()
    assert box.missing() == [] and 'Row opt: network down' in box.state.text()


def test_the_tray_opens_components(tmp_path, monkeypatch):
    from peresmeshnik.ui import tray as tray_mod
    from peresmeshnik.ui.components import ComponentsDialog
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    t = tray_mod.Tray(type('Ctl', (), {'live': False, 'on': True})())
    (act,) = [a for a in t.menu.actions() if a.text() == 'Components…']
    act.trigger()
    d = t.windows[-1]
    assert isinstance(d, ComponentsDialog) and d.box.rows and d.box.translating() and not d.busy()
    d.close()


def test_a_tick_while_it_downloads_keeps_it_busy(monkeypatch):
    """FIXES 1: a tick meanwhile does not end `busy`: Download and Remove stay off, no second download starts."""
    from peresmeshnik.ui.components import ComponentsBox
    go, started = threading.Event(), []

    def fetch(progress, cancel):
        started.append(1)
        go.wait(5)
    rows = [components.Component('pack', 'Row pack', 1000, 'MIT', 'https://x', '', False, 0, True, _fetch=fetch),
            components.Component('opt', 'Row opt', 1000, 'MIT', 'https://x', '', False, 0, False, _fetch=fetch)]
    monkeypatch.setattr(components, 'listing', lambda s: rows)
    box = ComponentsBox(config.Settings())
    box.refresh()
    box.download()
    settle(0.1)
    item(box, rows[1]).setCheckState(0, Qt.CheckState.Checked)
    box.consent.setChecked(True)
    assert box.busy and not box.button.isEnabled() and not box.tree.isEnabled()
    box.download()
    settle(0.1)
    assert started == [1]
    go.set()
    end = time.monotonic() + 5
    while box.busy and time.monotonic() < end:
        settle()
    assert not box.busy and box.tree.isEnabled()


def test_cancel_stops_inside_a_model(monkeypatch, base):
    """FIXES 5: Cancel stops within the 1 MB pieces of a big model, not after it; the .part stays to resume."""
    pieces = []

    def fetch(mid, where, progress):
        for _ in range(5):
            progress(1 << 20)
            pieces.append(1)
    monkeypatch.setattr(models, 'fetch', fetch)
    c = components._model(models.get('xtts-v2'), base, True, '')
    stop = threading.Event()
    with pytest.raises(components.Cancelled):
        c.fetch(lambda stage, done, total, text='': stop.set(), stop)
    assert pieces == []  # the first piece's report already stopped it


def test_the_addon_counts_onwards(tmp_path, monkeypatch):
    """FIXES 7: the addon's bar goes on from its wheels into its models, never back."""
    monkeypatch.setattr(components, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(voicepack, 'state', lambda addon=False: 'none')
    monkeypatch.setattr(voicepack, 'folder', lambda: tmp_path / 'voice-pack')
    monkeypatch.setattr(voicepack, 'lock_file', lambda: lock(tmp_path))
    monkeypatch.setattr(voicepack, 'install', lambda progress, cancel, addon=False: progress('downloading', 1000, 1000, 'w'))
    monkeypatch.setattr(models, 'fetch', lambda mid, where, step: [step(1 << 20) for _ in range(3)])
    seen = []
    (addon,) = [c for c in components.listing(config.Settings(), CARD) if c.id == components.ADDON]
    addon.fetch(lambda stage, done, total, text='': seen.append(done))
    assert seen == sorted(seen) and seen[0] == 1000 and seen[-1] == 1000 + 3 * (1 << 20)


def test_another_pairs_default_follows_the_pair(monkeypatch):
    """FIXES 4: only the user's own ticks are kept: me -> them cloud (the clone unticked by default), then local: the clone
    is ticked, as for that setup."""
    from peresmeshnik.ui.components import ComponentsBox
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: CARD)
    s = config.Settings(my_lang='ru', their_lang='en', me_to_them='cloud')
    box = ComponentsBox(s)
    box.refresh()
    assert 'xtts-v2' not in {c.id for c in box.todo()}
    s.me_to_them = 'local'
    box.refresh()
    assert 'xtts-v2' in {c.id for c in box.todo()} and not box.ticked


def test_the_wizard_goes_on_without_the_clone_not_without_the_pair(monkeypatch):
    """FIXES 2: Next waits for what the setup runs on; the clone unticked, Next is free and says the stock voice speaks."""
    from peresmeshnik.ui import wizard
    here = {'vad'}
    rows = lambda s: [components.Component(i, f'Row {i}', 0 if i in here else 9, 'MIT', 'https://x', '', i in here, 0,  # noqa: E731
                                           True, instead, required=req)
                      for i, instead, req in (('vad', 'needed for every setup', True),
                                              ('clone', 'a stock voice speaks instead', False))]
    monkeypatch.setattr(components, 'listing', rows)
    page = wizard.ModelsPage(type('W', (), {'s': config.Settings()})())
    page.initializePage()
    assert not page.isComplete()  # the clone is ticked by default: it waits for it
    item(page.box, page.box.rows[1]).setCheckState(0, Qt.CheckState.Unchecked)
    assert page.isComplete() and 'Without Row clone: a stock voice speaks instead.' in page.box.without.text()
    here.discard('vad')
    page.initializePage()
    item(page.box, page.box.rows[0]).setCheckState(0, Qt.CheckState.Unchecked)  # locked: stays
    assert not page.isComplete() and [c.id for c in page.box.missing()] == ['vad']


@pytest.fixture
def no_token(tmp_path, monkeypatch):
    """The user's real token file (models.hf_token, not a stand-in), in a folder of this test's own."""
    from peresmeshnik import voice
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path / 'config')
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    monkeypatch.delenv('HF_TOKEN', raising=False)
    return models.hf_token_file()


def test_the_hugging_face_token_where_the_clone_waits_for_it(no_token):
    """COMMON 23:38: no NVIDIA GPU, English: Pocket TTS is the clone and waits for the token. The field is in Components,
    on the wizard's voice page and in Voices…; saved, the engine is chosen again (the page records, Components ticks
    Pocket); saved empty, the token is forgotten (its file goes) and the field is back."""
    from peresmeshnik.ui import voices, wizard
    from peresmeshnik.ui.components import ComponentsBox
    s = config.Settings(my_lang='ru', their_lang='en', me_to_them='local')
    assert components.token_page('en') == 'https://huggingface.co/kyutai/pocket-tts' and not models.hf_token()
    box = ComponentsBox(s)
    box.refresh()
    dialog = voices.VoicesDialog(s)
    page = wizard.VoicePage(type('W', (), {'s': s})())
    page.initializePage()
    assert not box.hf.isHidden() and not dialog.hf.isHidden() and not page.hf.isHidden()
    assert page.record_button.isHidden() and 'Hugging Face token' in page.subTitle()  # a stock voice for now, said why
    page.hf.token.setText(' hf_secret ')
    page.hf.save.click()
    assert models.hf_token() == 'hf_secret' and no_token.read_text().strip() == 'hf_secret'
    if sys.platform != 'win32':
        assert no_token.stat().st_mode & 0o777 == 0o600
    assert page.hf.isHidden() and not page.record_button.isHidden() and 'Components page' in page.out.text()
    box.hf.saved.emit()  # the token changed: the list again
    (pocket,) = [c for c in box.rows if c.id == 'pocket-tts-en']
    assert pocket.recommended and pocket.token and box.wanted(pocket) and box.hf.isHidden()
    dialog.hf.token.setText('')
    dialog.hf.save.click()  # empty: forgotten
    assert not no_token.exists() and models.hf_token() == '' and 'forgotten' in dialog.token_note.text()
    assert not dialog.hf.isHidden()


def test_hugging_face_refusing_does_not_hold_the_wizard(no_token, monkeypatch):
    """COMMON 23:38: a 403 (the conditions not accepted) on the clone: Next does not wait for it, the stock voice is said,
    the field and the page to accept the conditions are shown."""
    from peresmeshnik.ui.components import ComponentsBox
    models.save_hf_token('hf_wrong')

    def refused(progress, cancel):
        raise models.Gated('Hugging Face refused pocket-tts-en (403): accept its conditions', 'https://hf.example/pocket')
    rows = [components.Component('pocket-tts-en', 'Your cloned voice: pocket-tts-en', 9, 'CC-BY-4.0', 'https://x', '',
                                 False, 0, True, 'a stock voice speaks instead', token=True, _fetch=refused)]
    monkeypatch.setattr(components, 'listing', lambda s: rows)
    box = ComponentsBox(config.Settings())
    box.refresh()
    assert box.missing() and box.hf.isHidden()  # a token is saved: nothing to ask yet
    box.download()
    end = time.monotonic() + 5
    while box.busy and time.monotonic() < end:
        settle()
    assert box.missing() == [] and 'accept its conditions' in box.state.text()
    assert not box.hf.isHidden() and box.hf.page == 'https://hf.example/pocket'
    assert 'Without Your cloned voice: pocket-tts-en: a stock voice speaks instead.' in box.without.text()


def test_removing_the_pack_goes_through_its_lock(tmp_path, monkeypatch):
    """FIXES addon 4: Remove of the voice pack is voicepack.uninstall: while its addon installs (the pack's lock taken)
    it refuses with the reason and leaves the pack as it is; then it removes the pack, its marker first."""
    monkeypatch.setattr(components, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: CARD)
    monkeypatch.setattr(voicepack, 'folder', lambda: tmp_path / 'voice-pack')
    monkeypatch.setattr(voicepack, 'lock_file', lambda: lock(tmp_path))
    (tmp_path / 'voice-pack').mkdir()
    (tmp_path / 'voice-pack' / '.complete').write_text('x')
    pack = {c.id: c for c in components.listing(config.Settings(me_to_them='local'), CARD)}[components.VOICE_PACK]
    assert voicepack._BUSY.acquire(blocking=False)  # the addon's install runs
    try:
        with pytest.raises(voicepack.PackError, match='being installed or removed'):
            pack.remove()
        assert (tmp_path / 'voice-pack' / '.complete').exists()
    finally:
        voicepack._BUSY.release()
    pack.remove()
    assert not (tmp_path / 'voice-pack').exists()


def test_no_addon_row_without_its_models_in_the_registry(tmp_path, monkeypatch):
    """The pack's addon section names registry models (qwen3-tts, chatterbox-mtl); a registry without them (another
    build's, a test's stand-ins) gives no addon row instead of failing the whole list."""
    monkeypatch.setattr(components, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.delenv('PERESMESHNIK_VOICE_CMD', raising=False)
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)
    monkeypatch.setattr(languages, 'nvidia_once', lambda: CARD)
    monkeypatch.setattr(voicepack, 'folder', lambda: tmp_path / 'voice-pack')
    f = lambda n: {'url': 'https://x/' + n, 'sha256': '0' * 64, 'size': 1000}  # noqa: E731
    spec = {'python': f('py.zip'), 'pip': f('pip.whl'), 'wheels': [f('torch.whl')], 'archives': [], 'installed_bytes': 8e9,
            'addon': {'wheels': [f('qwen.whl')], 'archives': [], 'installed_bytes': 2e9,
                      'models': ['qwen3-tts', 'chatterbox-mtl'], 'vram_gb': 7.5}}
    (tmp_path / 'pack-windows.json').write_text(json.dumps(spec))
    monkeypatch.setattr(voicepack, 'lock_file', lambda: tmp_path / 'pack-windows.json')
    s = config.Settings(me_to_them='local')
    assert components.ADDON in {c.id for c in components.listing(s, CARD)}
    reg = {k: v for k, v in models.registry().items() if k != 'qwen3-tts'}
    monkeypatch.setattr(models, 'registry', lambda: reg)
    ids = {c.id for c in components.listing(s, CARD)}
    assert components.ADDON not in ids and components.VOICE_PACK in ids

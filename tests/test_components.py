"""Components: what the app downloads after install — the pair's models ticked by default, what other pairs left on disk to
remove, the clone engines and the Windows voice pack with its addon offered only where they run; the box downloads only
the ticked ones after Download and says what the pair does without an unticked one."""
import json
import os
import sys
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
    assert 'hears English' in asr.why and 'the cloud can' in asr.instead
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
    assert pack.recommended and pack.size == 3000 and pack.consent and not pack.unavailable
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
    assert box.tree.topLevelItemCount() == 2 and not box.consent.isHidden()


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

    def row(cid, recommended, consent=False, instead=''):
        def fetch(progress, cancel):
            progress('downloading', 500, 1000, cid)
            got.append(cid)
            here.add(cid)
        return components.Component(cid, f'Row {cid}', 0 if cid in here else 1000, 'MIT', 'https://x', 'for the pair',
                                    cid in here, 2000 if cid in here else 0, recommended, instead, '', consent,
                                    _fetch=fetch, _remove=lambda: here.discard(cid))
    monkeypatch.setattr(components, 'listing', lambda s: [row('asr', True, instead='the cloud can'), row('nc', False, True),
                                                          row('old', False)])
    on, asked = [True], []
    box = ComponentsBox(config.Settings(), translating=lambda: on[0])
    box.confirm = lambda text: asked.append(text) or True
    box.refresh()
    assert [c.id for c in box.missing()] == ['asr'] and box.consent.isHidden() and box.without.isHidden()
    item(box, box.rows[0]).setCheckState(0, Qt.CheckState.Unchecked)  # the pair's ASR unticked: said, Next is free
    assert box.missing() == [] and 'Without Row asr: the cloud can.' in box.without.text()
    item(box, box.rows[0]).setCheckState(0, Qt.CheckState.Checked)
    item(box, box.rows[1]).setCheckState(0, Qt.CheckState.Checked)  # a restricted license: asked first
    assert not box.consent.isHidden() and 'Row nc (MIT)' in box.consent.text()
    box.download()
    assert got == [] and 'Accept' in box.state.text()
    box.consent.setChecked(True)
    box.download()
    end = time.monotonic() + 5
    while box.busy and time.monotonic() < end:
        settle()
    assert got == ['asr', 'nc'] and not box.missing() and box.state.text().startswith('Everything ticked is here')
    box.remove(next(c for c in box.rows if c.id == 'old'))  # translating: not now
    assert 'old' in here and 'translation off' in box.state.text() and asked == []
    on[0] = False
    box.remove(next(c for c in box.rows if c.id == 'old'))
    assert 'old' not in here and asked and 'Row old' in asked[0]


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

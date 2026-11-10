"""Languages in the shell: the wizard's languages page says what each side of a pair runs on before any download, the tray
switches the pair without restarting the app (its models fetched first, then the sides made again), the overlay draws
Chinese, Japanese, Korean and Arabic (right to left) — a snapshot per OS for CI. Pairs whose real models are not in the
registry yet run on stand-ins (`test_languages.stub`)."""
import dataclasses
import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from peresmeshnik import config, languages, models  # noqa: E402
from peresmeshnik.engines import Caption, Speech  # noqa: E402

app = QApplication.instance() or QApplication([])
REGISTRY = models.registry()  # the real one, before a test puts its own in
STAND_INS = {'asr': ('asr-de', 'asr-ja'), 'mt': ('mt-de-ja', 'mt-ja-de'), 'tts': ('tts-de', 'tts-ja')}


def settle(seconds=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def before_stage6(reg: dict) -> dict:
    """The registry with real models only for en and ru, as before stage 6 (S1 put models for the other 12 languages
    into it): a language here has none but its stand-ins."""
    out = {}
    for k, m in reg.items():
        langs = m.langs if m.role in ('vad', 'clone') else tuple(p for p in m.langs if set(p.split('-')) <= {'en', 'ru'})
        if langs:
            out[k] = dataclasses.replace(m, langs=langs)
    return out


def stand_ins(monkeypatch, url='https://example.invalid/', files=None, consent=()):
    """German and Japanese on stand-in models (served by `url`); `wire(s)` puts them into settings, with a stand-in clone
    that speaks English only (XTTS speaks both: a test must not fetch its 1.9 GB). Silero VAD counts as downloaded: a test
    never fetches a real model."""
    reg = before_stage6(REGISTRY)
    for role, ids in STAND_INS.items():
        for mid in ids:
            reg[mid] = models.Model(mid, role, (mid.split('-', 1)[1],), 'CPML' if mid in consent else 'MIT',
                                    'https://example.invalid/lic', url, dict(files or {}), consent=mid in consent)
    reg['clone-en'] = models.Model('clone-en', 'clone', ('en',), 'CPML', 'https://example.invalid/lic', url, {}, consent=True)
    monkeypatch.setattr(models, 'registry', lambda: reg)
    vad = models.root() / 'silero-vad'
    vad.mkdir(parents=True, exist_ok=True)
    (vad / models.MARKER).write_text(reg['silero-vad'].fingerprint + '\n')


def wire(s: config.Settings) -> config.Settings:
    s.local.asr.update(de='asr-de', ja='asr-ja')
    s.local.mt.update({'de-ja': 'mt-de-ja', 'ja-de': 'mt-ja-de'})
    s.local.tts.update(de='tts-de', ja='tts-ja')
    s.local.clone = 'clone-en'
    return s


def test_languages_page_says_before_the_download_what_runs_where(tmp_path, monkeypatch):
    from peresmeshnik.ui import wizard
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    monkeypatch.setenv('GEMINI_API_KEY', 'a-key')
    monkeypatch.setattr(models, 'registry', lambda: before_stage6(REGISTRY))
    page = wizard.LanguagesPage(type('W', (), {'s': config.Settings(them_to_me='local', me_to_them='local')})())
    assert page.my_lang.count() == 14 and page.my_lang.itemText(page.my_lang.findData('ja')) == '日本語 (Japanese)'
    page.my_lang.setCurrentIndex(page.my_lang.findData('de'))
    page.their_lang.setCurrentIndex(page.their_lang.findData('ja'))
    local = config.ENGINES.index('local')
    for key in languages.SIDES:  # no German / Japanese models in the registry: local is not offered, and it says why
        assert not page.engine[key].model().item(local).isEnabled() and page.engine[key].currentText() == 'cloud'
        assert 'Not on this computer: no local speech recognition for' in page.why[key].text()
        assert 'subtitles come from the cloud too' in page.why[key].text()
    assert page.download.text().startswith('1 MB for this pair') or page.download.text().startswith('2 MB')  # Silero VAD
    assert page.validatePage() and (page.w.s.my_lang, page.w.s.their_lang, page.w.s.them_to_me) == ('de', 'ja', 'cloud')
    assert 'On a Mac your voice goes' not in page.why['me_to_them'].text()  # FIXES-b 4: clone_why's note says it

    stand_ins(monkeypatch, files={'w.bin': {'sha256': '0' * 64, 'size': 50_000_000}}, consent=('tts-ja',))
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local'))
    config.save(s)  # saved settings: the page starts from their engines
    page = wizard.LanguagesPage(type('W', (), {'s': s})())
    for key in languages.SIDES:
        assert page.engine[key].model().item(local).isEnabled() and page.engine[key].currentText() == 'local'
    assert 'cannot speak Japanese: a stock voice speaks instead' in page.why['me_to_them'].text()  # the clone: en only
    text = page.download.text()
    assert text.startswith('300 MB for this pair (6 of 7 models')  # only the pair's: 6 stand-ins + Silero VAD
    assert 'CPML</a> (noncommercial' in text and 'MIT' in text


def test_tray_switches_the_pair_fetching_only_its_models(tmp_path, monkeypatch, server):
    """Another pair from the tray: only its missing models are downloaded (restricted licenses asked first), then it is
    saved and the running controller makes its sides again — the app is not restarted."""
    from peresmeshnik.ui import tray as tray_mod
    httpd, url = server
    blob = b'w' * 70_000
    (httpd.root / 'w.bin').write_bytes(blob)
    files = {'w.bin': {'sha256': hashlib.sha256(blob).hexdigest(), 'size': len(blob)}}
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    stand_ins(monkeypatch, url, files, consent=('mt-de-ja',))
    monkeypatch.setenv('GEMINI_API_KEY', 'a-key')
    s = wire(config.Settings(them_to_me='local', me_to_them='local'))
    s.local.clone = 'piper'  # the stock voice: no clone for these stand-ins
    config.save(s)
    calls = []

    class Ctl:
        live, on = False, True

        def set_languages(self):
            calls.append(config.load())
    t = tray_mod.Tray(Ctl())
    asked = []
    t.confirm = lambda text: asked.append(text) or False
    pick = {k: {a.data()[1]: a for a in g.actions()} for k, g in t.langs.items()}
    assert len(pick['my_lang']) == 14 and pick['my_lang']['ru'].isChecked() and pick['their_lang']['en'].isChecked()
    pick['my_lang']['de'].trigger()  # de <-> en: no German models -> both sides to the cloud, asked first (H-02)
    settle(0.1)
    assert 'Google' in asked[-1] and config.load().my_lang == 'ru' and not calls  # declined: nothing changes
    t.confirm = lambda text: asked.append(text) or 'Google' in text  # yes to the cloud, no to restricted licenses
    pick['my_lang']['de'].trigger()
    settle(0.1)
    assert (config.load().my_lang, config.load().them_to_me, config.load().me_to_them) == ('de', 'cloud', 'cloud')
    assert len(calls) == 1 and pick['my_lang']['de'].isChecked()
    config.save(s)  # back to ru <-> en, both local
    t.sync()
    pick['my_lang']['de'].setChecked(True)
    t.switch_pair('de', 'ja')
    assert 'mt-de-ja (CPML' in asked[-1] and config.load().my_lang == 'ru' and len(calls) == 1  # declined: nothing changes
    t.confirm = lambda text: True
    t.switch_pair('de', 'ja')
    end = time.monotonic() + 30
    while len(calls) < 2 and time.monotonic() < end:
        settle(0.05)
    got = config.load()
    assert len(calls) == 2 and (got.my_lang, got.their_lang, got.them_to_me, got.me_to_them) == ('de', 'ja', 'local', 'local')
    fetched = sorted(p.name for p in (tmp_path / 'models').iterdir() if (p / models.MARKER).exists())
    assert fetched == ['asr-de', 'asr-ja', 'mt-de-ja', 'mt-ja-de', 'silero-vad', 'tts-de', 'tts-ja']  # nothing else
    assert pick['their_lang']['ja'].isChecked()
    pick['their_lang']['de'].trigger()  # my language as theirs: the pair turns round
    settle(0.1)
    assert (config.load().my_lang, config.load().their_lang) == ('ja', 'de')


def tray_with(monkeypatch, tmp_path, s, on=True):
    from peresmeshnik.ui import tray as tray_mod
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setattr(config, 'config_dir', lambda: tmp_path)
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    monkeypatch.setenv('GEMINI_API_KEY', 'a-key')
    stand_ins(monkeypatch)  # stand-ins without files: "downloaded" at once
    config.save(s)
    calls = []

    class Ctl:
        live = False

        def set_languages(self):
            calls.append(config.load())
    Ctl.on = on
    t = tray_mod.Tray(Ctl())
    t.said = []
    t.message = t.said.append
    t.confirm = lambda text: True  # a real question box would wait for a click
    return t, calls


def wait_for(cond, s=10.0):
    end = time.monotonic() + s
    while not cond() and time.monotonic() < end:
        settle(0.02)
    return cond()


def test_tray_pair_switch_details(tmp_path, monkeypatch):
    """The language it has: nothing; a choice while a pair downloads builds on that pair; when it lands only the sides it
    moved to the cloud are written (a change of the other meanwhile stays); the controller is told even while a start is
    under way (off: its control thread says idle); a clone that cannot speak the language is said."""
    import threading as th

    from peresmeshnik import models as models_mod
    s = wire(config.Settings(them_to_me='local', me_to_them='local'))
    t, calls = tray_with(monkeypatch, tmp_path, s, on=False)
    switched = []
    real_switch, t.switch_pair = t.switch_pair, lambda my, their, engines=None: switched.append((my, their))
    pick = {k: {a.data()[1]: a for a in g.actions()} for k, g in t.langs.items()}
    pick['my_lang']['ru'].trigger()  # already mine
    assert switched == []
    t._pair = (config.Settings(my_lang='de', their_lang='ja'), [], [], {})  # de -> ja downloading
    pick['their_lang']['ko'].trigger()
    assert switched == [('de', 'ko')]  # from the pair on its way, not from the saved ru / en
    t._pair, t.switch_pair = None, real_switch

    gate = th.Event()
    real_fetch = models_mod.fetch
    monkeypatch.setattr(models_mod, 'fetch', lambda mid, *a, **k: gate.wait(10))  # lands nothing: the next part fetches
    t.set_state('translating (them: local, me: local; languages ru ⇄ en)')
    t.switch_pair('de', 'ja')
    settle(0.3)
    assert t._pair is not None and t.status.text().startswith('downloading')
    t.switch_pair('ru', 'en')  # back to the saved pair while de -> ja is on its way
    assert t._pair is None and t.status.text().startswith('translating') and calls == []  # nothing made again
    gate.set()
    settle(0.5)
    assert calls == [] and config.load().my_lang == 'ru'  # the download let go: it ends unused

    gate = th.Event()
    monkeypatch.setattr(models_mod, 'fetch', lambda mid, *a, **k: gate.wait(10) and real_fetch(mid, *a, **k))
    t.switch_pair('ja', 'de')  # both sides local in ja <-> de: its stand-ins are fetched (the gate holds them)
    settle(0.3)
    assert t._pair is not None and calls == [] and any('downloading' in m for m in t.said)
    saved = config.load()
    saved.cloud.subtitles, saved.me_to_them = True, 'cloud'  # changes elsewhere while it downloads (a side it left alone)
    config.save(saved)
    gate.set()
    assert wait_for(lambda: calls)
    got = config.load()
    assert (got.my_lang, got.their_lang, got.cloud.subtitles, got.me_to_them) == ('ja', 'de', True, 'cloud')  # all kept
    assert len(calls) == 1  # translation off: the controller still hears of it (its thread knows when it is on)

    s = wire(config.Settings(my_lang='de', their_lang='en', them_to_me='local', me_to_them='cloud'))
    config.save(s)
    t.switch_pair('de', 'fr')  # them -> me: no French models -> to the cloud; me -> them was cloud
    assert wait_for(lambda: len(calls) == 2)
    saved = config.load()
    assert (saved.them_to_me, saved.me_to_them) == ('cloud', 'cloud')
    s = wire(config.Settings(my_lang='ru', their_lang='en', them_to_me='cloud', me_to_them='local'))
    s.local.clone = 'clone-en'  # the clone, which has no Japanese
    config.save(s)
    t.switch_pair('de', 'ja')
    assert wait_for(lambda: len(calls) == 3)
    assert any('Me → them: your cloned voice cannot speak Japanese: a stock voice speaks instead' in m for m in t.said), t.said


def test_two_pair_switches_download_each_model_once(tmp_path, monkeypatch, server):
    """The pair chosen again while it downloads (the tray, then the settings window; the server is slow): every model
    is downloaded once, the pair made once."""
    httpd, url = server
    httpd.slow = 0.2
    blob = b'x' * 10_000
    (httpd.root / 'w.bin').write_bytes(blob)
    t, calls = tray_with(monkeypatch, tmp_path, wire(config.Settings(them_to_me='local', me_to_them='local')))
    stand_ins(monkeypatch, url, files={'w.bin': {'sha256': hashlib.sha256(blob).hexdigest(), 'size': len(blob)}})
    t.switch_pair('de', 'ja')
    t.switch_pair('de', 'ja')
    assert wait_for(lambda: calls, 30)
    settle(0.5)
    assert len(httpd.ranges) == 6, httpd.ranges  # 2 ASR + 2 MT + 2 voices, each once
    assert len(calls) == 1 and (config.load().my_lang, config.load().their_lang) == ('de', 'ja')


def test_settings_window_changes_only_the_pair_without_a_restart(tmp_path, monkeypatch):
    from peresmeshnik.ui import tray as tray_mod
    t, calls = tray_with(monkeypatch, tmp_path, wire(config.Settings()))
    restarts, switched = [], []
    t.restart = lambda: restarts.append(1)
    t.switch_pair = lambda my, their, engines=None: switched.append((my, their))
    d = tray_mod.SettingsDialog(t)
    d.their_lang.setCurrentIndex(d.their_lang.findData('ja'))
    d.save()
    assert switched == [('ru', 'ja')] and restarts == [] and config.load().their_lang == 'en'  # saved once its models are here
    d = tray_mod.SettingsDialog(t)
    d.threads.setValue(3)
    d.save()
    assert restarts == [1] and config.load().local.threads == 3


class Direction:
    """A local direction: says one clause in its pair, lives until closed."""
    made: list = []

    def __init__(self, src, tgt):
        self.src, self.tgt, self.closed = src, tgt, threading.Event()
        self.tts = self.mt = None
        Direction.made.append(self)

    def warm_up(self):
        pass

    def feed(self, pcm):
        pass

    def events(self):
        yield Caption(f'{self.src} text', f'{self.tgt} text', final=True, clause=0)
        yield Speech(np.zeros(240, np.float32), 24000)
        self.closed.wait()

    def close(self):
        self.closed.set()


def test_controller_makes_its_sides_again_for_a_new_pair(tmp_path, monkeypatch):
    """Controller.set_languages while translating: both sides stop and start again in the new pair, as after an error;
    translation stays on; off, it waits for the next start."""
    from peresmeshnik import app, pipeline
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    stand_ins(monkeypatch)
    monkeypatch.setattr(pipeline, 'make_local', lambda src, tgt, **k: Direction(src, tgt))
    monkeypatch.setattr(app.Controller, '_output', lambda self, name, s: type('Out', (), {
        'play': lambda *a: None, 'flush': lambda *a: None, 'stop': lambda *a: None})())
    Direction.made = []
    sample = str(Path(__file__).parents[1] / 'peresmeshnik' / 'ui' / 'samples' / 'en.ogg')
    s = wire(config.Settings(them_to_me='local', me_to_them='local'))
    s.local.clone = 'piper'
    shown, states = [], []
    overlay = type('O', (), {'side': lambda self, name: type('S', (), {'show': lambda _, c: shown.append((name, c)),
                                                                        'hide': lambda _: None})()})()
    ctl = app.Controller(overlay, states.append, sample, sample, settings=s, test=True)
    ctl.set_translation(True).result(30)
    assert sorted((d.src, d.tgt) for d in Direction.made) == [('en', 'ru'), ('ru', 'en')] and ctl.on
    first = list(Direction.made)
    s.my_lang, s.their_lang = 'de', 'ja'
    ctl.set_languages().result(30)
    assert all(d.closed.is_set() for d in first) and ctl.on and len(ctl.sides) == 2
    assert sorted((d.src, d.tgt) for d in Direction.made[2:]) == [('de', 'ja'), ('ja', 'de')]
    assert states[-1].startswith('translating (') and states[-1].endswith('Deutsch (German) ⇄ 日本語 (Japanese))'), states
    ctl.set_translation(False).result(30)
    s.my_lang = 'fr'
    ctl.set_languages().result(30)
    assert len(Direction.made) == 4 and not ctl.on  # off: nothing made now
    assert states[-1] == 'idle, translation off (Français (French) ⇄ 日本語 (Japanese))'
    ctl.quit()


def test_while_a_side_is_made_again_the_call_and_i_hear_each_other(tmp_path, monkeypatch):
    """_drop / _make (a new pair, a restart after an error): me -> them — the microphone loop on before the old side
    stops, off once the new one is made, before it starts (the call hears me meanwhile, as in _on); them -> me — the
    route undone first, the pass-through (macOS < 14.4) on once the old input let go, off before the new one starts. A
    side that cannot be made or started leaves its stand-in on, the other side runs; none: translation off. Each side's
    rows take the new languages once it is made again. A clone that cannot speak the language: said at the start (a new
    pair: by the tray), once."""
    from peresmeshnik import app, pipeline
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    stand_ins(monkeypatch)
    log, fail = [], set()

    class D(Direction):
        def __init__(self, src, tgt):
            if (src, tgt) in fail:
                raise RuntimeError(f'no {src}-{tgt} today')
            super().__init__(src, tgt)
            log.append(('made', src))

        def close(self):
            log.append(('closed', self.src))
            super().close()
    monkeypatch.setattr(pipeline, 'make_local', lambda src, tgt, **k: D(src, tgt))
    monkeypatch.setattr(app.Controller, '_output', lambda self, name, s: type('Out', (), {
        'play': lambda *a: None, 'flush': lambda *a: None, 'stop': lambda *a: None})())
    for name in ('_loopback', '_through_start', '_through_stop', '_unroute'):
        monkeypatch.setattr(app.Controller, name, lambda self, *a, name=name: log.append((name, *a)))
    sample = str(Path(__file__).parents[1] / 'peresmeshnik' / 'ui' / 'samples' / 'en.ogg')
    s = wire(config.Settings(them_to_me='local', me_to_them='local'))
    s.local.clone = 'piper'
    states = []
    overlay = type('O', (), {'side': lambda self, name: type('S', (), {'show': lambda *a: None, 'hide': lambda *a: None})(),
                             'set_pair': lambda self, my, their, side=None: log.append(('pair', side, my, their))})()
    ctl = app.Controller(overlay, states.append, sample, sample, settings=s, test=True)
    ctl.set_translation(True).result(30)
    log.clear()
    s.my_lang, s.their_lang = 'de', 'ja'
    ctl.set_languages().result(30)
    assert log == [('_unroute',), ('closed', 'en'), ('_through_start',), ('made', 'ja'), ('_through_stop',),
                   ('pair', 'them', 'de', 'ja'), ('_loopback', True), ('closed', 'ru'), ('made', 'de'), ('_loopback', False),
                   ('pair', 'me', 'de', 'ja')], log

    real_start = app.Side.start
    monkeypatch.setattr(app.Side, 'start', lambda self: (_ for _ in ()).throw(RuntimeError('the microphone is gone'))
                        if self.name == 'me' else real_start(self))
    log.clear()
    ctl.set_languages().result(30)  # me -> them made, but its start fails: closed, the loop stays on
    assert [x.name for x in ctl.sides] == ['them'] and ctl.on
    assert log[-5:] == [('_loopback', True), ('closed', 'de'), ('made', 'de'), ('closed', 'de'), ('pair', 'me', 'de', 'ja')], log
    assert ('_loopback', False) not in log and any(
        'me → them cannot start for Deutsch (German) ⇄ 日本語 (Japanese): the microphone is gone' in x for x in states)
    monkeypatch.setattr(app.Side, 'start', real_start)

    log.clear()
    fail.add(('ja', 'de'))  # them -> me cannot be made in ja <-> de any more; me -> them can
    s.local.clone = 'clone-en'  # chosen, but it has no Japanese: the stock voice
    n = len(states)
    ctl.set_languages().result(30)
    assert [x.name for x in ctl.sides] == ['me'] and ctl.on
    assert ('_through_stop',) not in log and any(
        st.startswith('error: them → me cannot start for Deutsch (German) ⇄ 日本語 (Japanese): no ja-de') for st in states)
    note = 'note: me → them: your cloned voice cannot speak Japanese: a stock voice speaks instead'
    assert note not in states[n:]  # the tray said it with the pair
    ctl.set_translation(False).result(30)
    n = len(states)
    ctl.set_translation(True).result(30)
    assert states[n:].count(note) == 1, states[n:]
    fail.add(('de', 'ja'))  # and neither side now: translation off
    log.clear()
    ctl.set_languages().result(30)
    assert not ctl.on and not ctl.sides and [x for x in log if x[0] != 'pair'][:2] == [('_loopback', True), ('closed', 'de')], log
    ctl.quit()


def test_a_row_keeps_its_languages_fonts(tmp_path):
    from peresmeshnik.ui.overlay import SubtitleOverlay
    o = SubtitleOverlay(rows=2)
    o.set_pair('zh', 'ja')
    o.caption.emit('me', Caption('这是', 'これ', final=True, clause=0))
    o.set_pair('ko', 'ar')  # a new pair: its rows in its fonts, the old row in its own
    o.caption.emit('them', Caption('إنه', '마텔리', final=True, clause=0))
    settle()
    assert o._langs == (('zh', 'ja'), ('ar', 'ko'))
    o.clear.emit()
    settle()


def test_fallback_of_a_cloud_side_says_what_is_missing(tmp_path, monkeypatch):
    """A cloud side of a new pair: no local ASR + MT — its subtitles from the cloud, and going local says why (no
    KeyError); ASR + MT but no voice for the language — local subtitles, the fallback voice says why."""
    from peresmeshnik import pipeline
    from peresmeshnik.cloud import fallback, gemini
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    stand_ins(monkeypatch)
    monkeypatch.setattr(gemini, 'GeminiDirection', lambda src, tgt, *a, **k: Direction(src, tgt))
    monkeypatch.setattr(pipeline, 'make_local', lambda src, tgt, **k: Direction(src, tgt))
    s = config.Settings(my_lang='de', their_lang='ja', them_to_me='cloud', me_to_them='cloud')
    s.local.asr, s.local.mt, s.local.tts = {}, {}, {}
    why = fallback.voice_why('them', s)
    assert why.startswith('no local speech recognition for Japanese; no local translation from Japanese to German')
    fb = fallback.make('them', s, 'a-key')
    try:
        assert fb.cloud_subtitles and fb.local is None
        with pytest.raises(RuntimeError, match='no local speech recognition for Japanese'):
            fb.make_local(None)
    finally:
        fb.close()
    wire(s)
    s.local.tts = {'ja': 'tts-ja'}  # German has no stock voice: them -> me has subtitles but nothing to speak with
    assert fallback.voice_why('them', s) == 'no local voice for German'
    fb = fallback.make('them', s, 'a-key')
    try:
        assert not fb.cloud_subtitles and fb.local is not None  # local subtitles beside the cloud voice
        fb.muted.load()
        assert 'no local voice for German' in str(fb.muted.error)
    finally:
        fb.close()


def test_voice_page_needs_no_recording_where_the_clone_cannot_speak(tmp_path, monkeypatch):
    from peresmeshnik.ui import wizard
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    stand_ins(monkeypatch)
    s = wire(config.Settings(my_lang='de', their_lang='ja', me_to_them='local'))  # the clone (stand-in) has no Japanese
    page = wizard.VoicePage(type('W', (), {'s': s})())
    page.initializePage()
    assert page.isComplete() and page.record_button.isHidden() and page.check_button.isHidden()
    assert 'cannot speak Japanese' in page.subTitle() and 'No recording is needed' in page.subTitle()
    s.their_lang = 'en'
    page = wizard.VoicePage(type('W', (), {'s': s})())
    page.initializePage()
    assert not page.isComplete() and not page.record_button.isHidden()  # the clone speaks English: record


def test_pair_size_and_download_go_through_the_registry_functions(tmp_path, monkeypatch, server):
    """A registry row without files of its own (S1's `via`: a pivot through English): the wizard and the tray take its
    size from models.size and download it with models.fetch, which unfold it."""
    from peresmeshnik import models as models_mod
    from peresmeshnik.ui import wizard
    t, calls = tray_with(monkeypatch, tmp_path, wire(config.Settings(them_to_me='local', me_to_them='cloud')))  # ru / en
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='cloud'))
    reg = models_mod.registry()
    reg['mt-ja-de'] = models.Model('mt-ja-de', 'mt', ('ja-de',), 'MIT', 'u', 'u', {})  # no files: it is two others
    monkeypatch.setattr(models_mod, 'size', lambda m: 123_000_000 if m.id == 'mt-ja-de' else m.size)
    fetched = []
    monkeypatch.setattr(models_mod, 'fetch', lambda mid, *a, **k: fetched.append(mid))
    monkeypatch.setattr(models_mod, 'is_fetched', lambda m, base=None: m.id not in ('mt-ja-de',) and m.id in fetched + ['silero-vad'])
    page = wizard.LanguagesPage(type('W', (), {'s': s})())
    assert page.download.text().startswith('123 MB for this pair'), page.download.text()
    t.switch_pair('de', 'ja')
    assert wait_for(lambda: calls) and 'mt-ja-de' in fetched
    assert any('downloading 123 MB' in m for m in t.said), t.said


# FLEURS sentence 1515 (peresmeshnik/ui/samples): two pairs, each side's original and translation
S1515 = {'zh': '这是马特利四年来第五次入选海地临时选举委员会。', 'ja': 'マルテリーにとっては4年間で5度目のCEPです。',
         'ko': '마텔리가 4년 만에 다섯 번째 하는 CEP다.', 'ar': 'إنه المجلس الانتخابي المؤقت الخامس لمارتيلي خلال أربع سنوات.'}
DISTINCT = {'zh': ('你好吗', '我们他'), 'ja': ('あいう', 'かきく'), 'ko': ('가나다', '라마바'), 'ar': ('ابت', 'جحخ')}


def drawn(text: str, font):
    from PySide6.QtGui import QImage, QPainter
    img = QImage(240, 60, QImage.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    p.setFont(font)
    p.setPen(0xffffffff)
    p.drawText(10, 40, text)
    p.end()
    return img


def test_overlay_draws_chinese_japanese_korean_and_arabic(tmp_path):
    """Each script draws real glyphs where the font database has a font for it (two different words look different;
    empty boxes would look the same), and the app says what to install where it has none; macOS and Windows ship them.
    Chinese, Japanese and Korean rows are drawn in a font of their own language (Han forms differ). Arabic: the painter
    lays it out right to left by itself, and a long line keeps its end (the latest words). The overlay with zh ⇄ ja and
    ko ⇄ ar goes to $PERESMESHNIK_SHOTS (a CI artifact per OS)."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFontDatabase, QFontMetrics

    from peresmeshnik.ui.overlay import OWN_FONTS, SubtitleOverlay, font_note, missing_fonts
    o = SubtitleOverlay(rows=4)
    shots = Path(os.environ.get('PERESMESHNIK_SHOTS') or tmp_path)
    shots.mkdir(parents=True, exist_ok=True)
    for my, their in (('zh', 'ja'), ('ko', 'ar')):
        o.set_pair(my, their)
        for i in range(4):
            mine = i % 2 == 0
            o.caption.emit('me' if mine else 'them', Caption(S1515[my if mine else their], S1515[their if mine else my],
                                                            final=True, clause=i))
            settle(0.1)
        assert len(o._shown) == 4
        assert o.grab().save(str(shots / f'overlay-{sys.platform}-{my}-{their}.png'))
        o.clear.emit()
        settle()
    print('paint ms, zh ja then ko ar rows (the pair warmed by set_pair):', [round(x, 1) for x in o.paint_ms])
    own = {lang: o.own_fonts(lang)[1].family() for lang in ('zh', 'ja', 'ko')}
    print('own fonts:', own)
    if sys.platform in ('darwin', 'win32'):  # Han in its language's own forms, not a fallback of the system's choosing
        assert all(own[x] in OWN_FONTS[x] for x in own), own
    for lang in DISTINCT:
        fams = QFontDatabase.families(getattr(QFontDatabase.WritingSystem, languages.WRITING[lang]))
        print(lang, len(fams), 'fonts:', ', '.join(fams[:4]))
    miss = missing_fonts(list(DISTINCT))
    for lang, (a, b) in DISTINCT.items():
        assert (drawn(a, o.f_tr) != drawn(b, o.f_tr)) == (lang not in miss), f'{lang}: font database and glyphs disagree'
    if sys.platform in ('darwin', 'win32'):
        assert not miss, f'no font for {miss} on this OS'
    assert bool(font_note(list(DISTINCT))) == bool(miss) and (not miss or 'Install one' in font_note(miss))
    ar = S1515['ar'] * 3
    cut = QFontMetrics(o.f_tr).elidedText(ar, Qt.ElideLeft, 300)
    assert cut.startswith('…') and cut.endswith('أربع سنوات.')  # the logical end: the newest words stay


def test_arabic_from_the_cloud_is_said_to_be_egyptian(tmp_path, monkeypatch):
    from peresmeshnik.ui import wizard
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    monkeypatch.setattr(models, 'registry', lambda: before_stage6(REGISTRY))
    page = wizard.LanguagesPage(type('W', (), {'s': config.Settings()})())
    page.their_lang.setCurrentIndex(page.their_lang.findData('ar'))
    assert 'Egyptian Arabic' in page.why['them_to_me'].text() and 'Egyptian' not in page.why['me_to_them'].text()


def test_a_star_model_reaches_the_side(tmp_path, monkeypatch):
    """'*' in `[local]` asr and mt (one model for every language or pair): the app's local side, the cloud's local
    fallback and make_local's own defaults are built on it (languages.direction's ids), no KeyError."""
    from types import SimpleNamespace

    from peresmeshnik import app, pipeline
    from peresmeshnik.cloud import fallback, gemini
    from peresmeshnik.local import asr, mt, tts, vad
    reg = dict(models.registry())
    for mid, role, langs in (('asr-many', 'asr', ('de', 'ja')), ('mt-many', 'mt', ('de', 'ja')), ('tts-de', 'tts', ('de',)),
                             ('tts-ja', 'tts', ('ja',))):
        reg[mid] = models.Model(mid, role, langs, 'MIT', 'u', 'u', {})
    monkeypatch.setattr(models, 'registry', lambda: reg)
    made = []
    monkeypatch.setattr(asr, 'OnnxAsr', lambda mid, threads=None: made.append(mid))
    monkeypatch.setattr(mt, 'OpusMt', lambda src, tgt, mid, threads=None: made.append(mid))
    monkeypatch.setattr(tts, 'PiperTts', lambda mid, threads=None: made.append(mid))
    monkeypatch.setattr(vad, 'SileroVad', lambda: None)
    monkeypatch.setattr(pipeline, 'LocalDirection', lambda *a, **k: SimpleNamespace(
        tts=None, mt=None, warm_up=lambda: None, close=lambda: None))
    monkeypatch.setattr(gemini, 'GeminiDirection', lambda src, tgt, *a, **k: Direction(src, tgt))
    s = config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local')
    s.local.asr, s.local.mt, s.local.tts, s.local.clone = {'*': 'asr-many'}, {'*': 'mt-many'}, {'de': 'tts-de', 'ja': 'tts-ja'}, 'piper'
    pipeline.make_local('ja', 'de', settings=s)
    assert sorted(made) == ['asr-many', 'mt-many', 'tts-de']
    for name, src, tgt, voice in (('them', 'ja', 'de', 'tts-de'), ('me', 'de', 'ja', 'tts-ja')):
        made.clear()
        side = app.Side.__new__(app.Side)
        side.name, side.src, side.tgt = name, src, tgt
        side._local(s)
        assert sorted(made) == ['asr-many', 'mt-many', voice]
    s.them_to_me = 'cloud'
    made.clear()
    fb = fallback.make('them', s, 'a-key')  # local subtitles beside the cloud voice: the '*' models
    try:
        assert sorted(made) == ['asr-many', 'mt-many']
    finally:
        fb.close()


def test_tray_asks_before_a_side_goes_to_the_cloud_and_brings_it_back(tmp_path, monkeypatch):
    """H-02: a pair a local side cannot run makes the tray ask first (its sound goes to Google), declined nothing changes;
    accepted the side is remembered as moved by the app and comes back to local with a pair it can run. A side the user
    set to the cloud stays there; the user's own choice of an engine forgets the move."""
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local'))
    s.local.clone = 'piper'  # the stock voice
    t, calls = tray_with(monkeypatch, tmp_path, s)
    asked = []
    t.confirm = lambda text: asked.append(text) or False
    t.switch_pair('de', 'fr')  # no French models: both sides would go to the cloud
    assert len(asked) == 1 and 'Google' in asked[0] and 'the call' in asked[0] and 'your voice' in asked[0], asked
    got = config.load()
    assert (got.their_lang, got.them_to_me, got.me_to_them, got.forced_cloud) == ('ja', 'local', 'local', []) and not calls
    t.confirm = lambda text: asked.append(text) or True
    t.switch_pair('de', 'fr')
    assert wait_for(lambda: calls)
    got = config.load()
    assert (got.their_lang, got.them_to_me, got.me_to_them) == ('fr', 'cloud', 'cloud')
    assert sorted(got.forced_cloud) == ['me_to_them', 'them_to_me']
    t.switch_pair('de', 'ja')  # a pair both run locally: both come back, nothing asked
    assert wait_for(lambda: len(calls) == 2) and len(asked) == 2
    got = config.load()
    assert (got.their_lang, got.them_to_me, got.me_to_them, got.forced_cloud) == ('ja', 'local', 'local', [])
    assert any('back on this computer' in m for m in t.said), t.said

    got.me_to_them = 'cloud'  # the user's own cloud: not moved by the app, it stays
    config.save(got)
    t.sync()
    t.switch_pair('de', 'fr')
    assert wait_for(lambda: len(calls) == 3) and config.load().forced_cloud == ['them_to_me']
    t.switch_pair('de', 'ja')
    assert wait_for(lambda: len(calls) == 4)
    got = config.load()
    assert (got.them_to_me, got.me_to_them, got.forced_cloud) == ('local', 'cloud', [])

    t.switch_pair('de', 'fr')
    assert wait_for(lambda: len(calls) == 5) and config.load().forced_cloud == ['them_to_me']
    t.restart = lambda: None
    next(a for a in t.engines['them_to_me'].actions() if a.data()[1] == 'cloud').trigger()  # chosen by the user now
    settle(0.1)
    assert config.load().forced_cloud == []


def test_languages_page_brings_back_a_side_it_moved_to_the_cloud(tmp_path, monkeypatch):
    """H-02 in the wizard: a pair a side cannot run moves it to the cloud; the pair it can run brings it back to local, and
    the move is saved (`forced_cloud`) for the tray. A side the user set to the cloud by hand stays there."""
    from peresmeshnik.ui import wizard
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    monkeypatch.setenv('GEMINI_API_KEY', 'a-key')
    stand_ins(monkeypatch)
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local'))
    s.local.clone = 'piper'
    config.save(s)
    page = wizard.LanguagesPage(type('W', (), {'s': s, 'key': ''})())
    engines = lambda: tuple(page.engine[k].currentText() for k in languages.SIDES)  # noqa: E731
    page.their_lang.setCurrentIndex(page.their_lang.findData('fr'))  # nothing local for French
    assert engines() == ('cloud', 'cloud')
    assert page.validatePage() and sorted(page.w.s.forced_cloud) == ['me_to_them', 'them_to_me']
    page.their_lang.setCurrentIndex(page.their_lang.findData('ja'))
    assert engines() == ('local', 'local')
    assert page.validatePage() and page.w.s.forced_cloud == []
    page.engine['me_to_them'].setCurrentText('cloud')  # by hand
    page.their_lang.setCurrentIndex(page.their_lang.findData('fr'))
    page.their_lang.setCurrentIndex(page.their_lang.findData('ja'))
    assert engines() == ('local', 'cloud')


def test_local_from_the_tray_or_settings_downloads_its_models_first(tmp_path, monkeypatch, server):
    """M-07: "local" chosen in the tray menu (or the settings window) for a side whose models are not here: they are
    downloaded first, as for a new pair, then it is saved and the sides are made again — not saved at once into a start
    that fails. A pair the side cannot run locally: said why, nothing changes."""
    from peresmeshnik.ui import tray as tray_mod
    httpd, url = server
    blob = b'm' * 40_000
    (httpd.root / 'w.bin').write_bytes(blob)
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='cloud', me_to_them='cloud'))
    s.local.clone, s.cloud.subtitles = 'piper', True  # the cloud's subtitles: a cloud side needs nothing local
    t, calls = tray_with(monkeypatch, tmp_path, s)
    stand_ins(monkeypatch, url, files={'w.bin': {'sha256': hashlib.sha256(blob).hexdigest(), 'size': len(blob)}})
    local = {k: next(a for a in g.actions() if a.data()[1] == 'local') for k, g in t.engines.items()}
    local['them_to_me'].trigger()
    assert config.load().them_to_me == 'cloud' and any('downloading' in m for m in t.said), t.said  # not yet
    assert wait_for(lambda: calls, 30)
    got = config.load()
    assert (got.them_to_me, got.me_to_them) == ('local', 'cloud') and len(calls) == 1
    fetched = sorted(p.name for p in (tmp_path / 'models').iterdir() if (p / models.MARKER).exists())
    assert {'asr-ja', 'mt-ja-de', 'tts-de'} <= set(fetched) and 'asr-de' not in fetched  # them -> me's own, nothing else

    d = tray_mod.SettingsDialog(t)
    d.engine['me_to_them'].setCurrentText('local')
    d.save()
    assert wait_for(lambda: len(calls) == 2, 30) and config.load().me_to_them == 'local'
    assert 'asr-de' in [p.name for p in (tmp_path / 'models').iterdir() if (p / models.MARKER).exists()]

    got.my_lang, got.their_lang, got.them_to_me = 'de', 'fr', 'cloud'  # nothing local for French
    config.save(got)
    t.sync()
    local['them_to_me'].trigger()
    settle(0.1)
    assert config.load().them_to_me == 'cloud' and len(calls) == 2
    assert 'Them → me cannot run on this computer for German ⇄ French (no local speech recognition for French' in t.said[-1]
    assert local['them_to_me'].isChecked() is False  # the menu shows what runs


def test_voice_page_without_an_nvidia_gpu_asks_for_no_recording(tmp_path, monkeypatch):
    """M-23: no NVIDIA GPU (a Mac, most laptops): me -> them local speaks in the stock voice, says why, and wants no
    recording; nothing says the clone "runs on the CPU"."""
    from peresmeshnik.ui import wizard
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    s = config.Settings(me_to_them='local')
    page = wizard.VoicePage(type('W', (), {'s': s})())
    page.initializePage()
    assert page.isComplete() and page.record_button.isHidden() and page.own.isHidden()
    assert languages.clone_why(s.local.clone, 'en') in page.subTitle() and 'runs on the CPU' not in page.subTitle()


def test_languages_page_says_what_a_cloud_side_falls_back_on(tmp_path, monkeypatch):
    """H-09: a cloud side with nothing local to fall back on (subtitles from the cloud, or no local models) says it stops
    until the cloud is back; one that goes on locally (its stock voice is downloaded) says nothing more."""
    from peresmeshnik.ui import wizard
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    monkeypatch.setenv('GEMINI_API_KEY', 'a-key')
    stand_ins(monkeypatch)
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='cloud', me_to_them='cloud'))
    s.local.clone = 'piper'
    config.save(s)
    page = wizard.LanguagesPage(type('W', (), {'s': s})())
    assert 'If the cloud fails' not in page.why['them_to_me'].text()  # it goes on here: nothing to warn about
    stand_ins(monkeypatch, consent=('tts-ja',))  # me -> them falls back on a noncommercial voice: optional, said
    page.refresh()
    assert 'fallback voice is noncommercial' in page.why['me_to_them'].text()
    assert 'noncommercial' not in page.why['them_to_me'].text()
    page.subtitles.setChecked(True)
    assert 'If the cloud fails, it stops until it is back' in page.why['them_to_me'].text()


def test_settings_local_subtitles_download_their_models_first(tmp_path, monkeypatch, server):
    """FIXES-b 7: "Subtitles from the cloud" unticked in Settings… goes through switch_pair: the cloud sides' local ASR and
    MT are downloaded first, then it is saved (not saved at once into a fallback and local subtitles that cannot start)."""
    from peresmeshnik.ui import tray as tray_mod
    httpd, url = server
    blob = b'm' * 40_000
    (httpd.root / 'w.bin').write_bytes(blob)
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='cloud', me_to_them='cloud'))
    s.local.clone, s.cloud.subtitles = 'piper', True
    t, calls = tray_with(monkeypatch, tmp_path, s)
    stand_ins(monkeypatch, url, files={'w.bin': {'sha256': hashlib.sha256(blob).hexdigest(), 'size': len(blob)}})
    d = tray_mod.SettingsDialog(t)
    d.subtitles.setChecked(False)
    d.save()
    assert config.load().cloud.subtitles and any('downloading' in m and 'subtitles from the local' in m for m in t.said), t.said
    assert wait_for(lambda: calls, 30) and not config.load().cloud.subtitles
    assert {'asr-ja', 'asr-de'} <= {p.name for p in (tmp_path / 'models').iterdir() if (p / models.MARKER).exists()}


def test_settings_window_keeps_the_mark_of_a_side_the_app_moved(tmp_path, monkeypatch):
    """FIXES-b 1 (H-02 through Settings…): the settings window hands switch_pair only the engines the user changed. Its
    combo showing "cloud" for a side the app moved is not the user's choice: the pair it runs again brings it back; the
    other side's engine changed keeps the mark; a question answered No leaves the mark as it was."""
    from peresmeshnik.ui import tray as tray_mod
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local'))
    s.local.clone = 'piper'
    t, calls = tray_with(monkeypatch, tmp_path, s)
    t.switch_pair('de', 'fr')  # both sides moved (the question answered Yes)
    assert wait_for(lambda: calls) and sorted(config.load().forced_cloud) == ['me_to_them', 'them_to_me']
    d = tray_mod.SettingsDialog(t)
    d.their_lang.setCurrentIndex(d.their_lang.findData('ja'))
    d.save()
    assert wait_for(lambda: len(calls) == 2)
    got = config.load()
    assert (got.their_lang, got.them_to_me, got.me_to_them, got.forced_cloud) == ('ja', 'local', 'local', [])

    t.switch_pair('de', 'fr')
    assert wait_for(lambda: len(calls) == 3) and len(config.load().forced_cloud) == 2
    got = config.load()
    got.cloud.subtitles = True  # nothing local to download for a cloud side
    config.save(got)
    d = tray_mod.SettingsDialog(t)
    d.engine['me_to_them'].setCurrentText('local')  # not possible for de -> fr: said, nothing changes
    d.save()
    settle(0.1)
    assert sorted(config.load().forced_cloud) == ['me_to_them', 'them_to_me']

    asked = []
    t.confirm = lambda text: asked.append(text) or False
    got = wire(config.load())
    got.my_lang, got.their_lang, got.them_to_me, got.me_to_them, got.forced_cloud = 'de', 'ja', 'cloud', 'local', ['them_to_me']
    config.save(got)
    t.switch_pair('de', 'fr', {'them_to_me': 'cloud'})  # the user's "cloud" for the moved side, and a pair that asks: No
    assert asked and config.load().forced_cloud == ['them_to_me'] and config.load().their_lang == 'ja'


def test_an_optional_fallback_voice_does_not_hold_the_wizard_or_the_tray(tmp_path, monkeypatch, server):
    """FIXES-b 2: a cloud side's noncommercial fallback voice does not hold anything: the tray asks for it apart and a No
    changes the pair all the same (that side has no fallback); the models page completes without it and downloads it
    only with the box ticked."""
    from peresmeshnik.ui import wizard
    httpd, url = server
    blob = b'v' * 30_000
    (httpd.root / 'w.bin').write_bytes(blob)
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='cloud', me_to_them='cloud'))  # local subtitles
    s.local.clone = 'piper'
    t, calls = tray_with(monkeypatch, tmp_path, s)
    stand_ins(monkeypatch, url, files={'w.bin': {'sha256': hashlib.sha256(blob).hexdigest(), 'size': len(blob)}},
              consent=('tts-ja',))
    asked = []
    t.confirm = lambda text: asked.append(text) or False
    t.switch_pair('ja', 'de')  # them -> me de -> ja in the cloud: its fallback voice tts-ja (noncommercial) is asked apart
    assert wait_for(lambda: calls, 30) and config.load().my_lang == 'ja'  # No: the pair changed all the same
    assert len(asked) == 1 and 'tts-ja' in asked[0] and 'stops' in asked[0], asked
    assert not models.is_fetched(models.get('tts-ja')) and models.is_fetched(models.get('tts-de'))

    s = config.load()
    assert languages.optional_models(s) == ['tts-ja'] and 'tts-ja' not in languages.needed_models(s)
    page = wizard.ModelsPage(type('W', (), {'s': s})())
    page.initializePage()
    assert page.isComplete() and not page.consent.isHidden() and 'tts-ja' in page.consent.text()
    assert 'optional' in page.list.text() and 'stops until the cloud is back' in page.state.text()
    page.consent.setChecked(True)
    page.download()
    assert wait_for(lambda: not page.busy, 30) and models.is_fetched(models.get('tts-ja')), page.state.text()


def test_voice_page_offers_no_recording_where_the_clone_never_speaks(tmp_path, monkeypatch):
    """FIXES-b 5 (M-23): me -> them in the cloud on a computer where no clone speaks (a Mac, no suitable NVIDIA GPU): no
    "optional" recording, the reason instead; where the clone can speak it stays optional (the cloud's fallback)."""
    from peresmeshnik.ui import wizard
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    s = config.Settings(me_to_them='cloud')
    page = wizard.VoicePage(type('W', (), {'s': s})())
    page.initializePage()
    assert page.isComplete() and page.record_button.isHidden() and page.import_button.isHidden()
    assert languages.clone_why(s.local.clone, 'en') in page.subTitle() and 'Optional' not in page.subTitle()
    monkeypatch.setattr(languages, 'gpu', lambda engine='': True)  # nvidia-smi says nothing here: taken as able
    page.initializePage()  # the same page again (Back, then Next): the recording is offered anew
    assert not page.record_button.isHidden() and not page.prompt.isHidden() and page.subTitle().startswith('Optional')


def test_languages_page_says_the_clone_once_on_a_mac(tmp_path, monkeypatch):
    """FIXES-b 4 (L-30): no hard-coded "On a Mac …" line beside clone_why's own note."""
    from peresmeshnik.ui import wizard
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setenv('GEMINI_API_KEY', 'a-key')
    monkeypatch.setattr(wizard, 'MACOS', True)
    monkeypatch.setattr(languages.sys, 'platform', 'darwin')
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    s = config.Settings(me_to_them='local')
    config.save(s)
    page = wizard.LanguagesPage(type('W', (), {'s': s})())
    text = page.why['me_to_them'].text()
    monkeypatch.undo()
    assert text.count('keeps up') == 1 and 'On a Mac' not in text, text

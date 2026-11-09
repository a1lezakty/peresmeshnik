"""Languages in the shell: the wizard's languages page says what each side of a pair runs on before any download, the tray
switches the pair without restarting the app (its models fetched first, then the sides made again), the overlay draws
Chinese, Japanese, Korean and Arabic (right to left) — a snapshot per OS for CI. Pairs whose real models are not in the
registry yet run on stand-ins (`test_languages.stub`)."""
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
STAND_INS = {'asr': ('asr-de', 'asr-ja'), 'mt': ('mt-de-ja', 'mt-ja-de'), 'tts': ('tts-de', 'tts-ja')}


def settle(seconds=0.05):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def stand_ins(monkeypatch, url='https://example.invalid/', files=None, consent=()):
    """German and Japanese on stand-in models (served by `url`); `wire(s)` puts them into settings. Silero VAD counts as
    downloaded: a test never fetches a real model."""
    reg = dict(models.registry())
    for role, ids in STAND_INS.items():
        for mid in ids:
            reg[mid] = models.Model(mid, role, (mid.split('-', 1)[1],), 'CPML' if mid in consent else 'MIT',
                                    'https://example.invalid/lic', url, dict(files or {}), consent=mid in consent)
    monkeypatch.setattr(models, 'registry', lambda: reg)
    vad = models.root() / 'silero-vad'
    vad.mkdir(parents=True, exist_ok=True)
    (vad / models.MARKER).write_text(reg['silero-vad'].fingerprint + '\n')


def wire(s: config.Settings) -> config.Settings:
    s.local.asr.update(de='asr-de', ja='asr-ja')
    s.local.mt.update({'de-ja': 'mt-de-ja', 'ja-de': 'mt-ja-de'})
    s.local.tts.update(de='tts-de', ja='tts-ja')
    return s


def test_languages_page_says_before_the_download_what_runs_where(tmp_path, monkeypatch):
    from peresmeshnik.ui import wizard
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    monkeypatch.setenv('GEMINI_API_KEY', 'a-key')
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
    assert ('On a Mac your voice goes to the call through the cloud' in page.why['me_to_them'].text()) == (sys.platform == 'darwin')

    stand_ins(monkeypatch, files={'w.bin': {'sha256': '0' * 64, 'size': 50_000_000}}, consent=('tts-ja',))
    s = wire(config.Settings(my_lang='de', their_lang='ja', them_to_me='local', me_to_them='local'))
    config.save(s)  # saved settings: the page starts from their engines
    page = wizard.LanguagesPage(type('W', (), {'s': s})())
    for key in languages.SIDES:
        assert page.engine[key].model().item(local).isEnabled() and page.engine[key].currentText() == 'local'
    assert 'cannot speak Japanese: a stock voice speaks instead' in page.why['me_to_them'].text()  # XTTS has no ja here
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
    pick['my_lang']['de'].trigger()  # de <-> en: no German models -> both sides to the cloud, nothing to download
    settle(0.1)
    assert (config.load().my_lang, config.load().them_to_me, config.load().me_to_them) == ('de', 'cloud', 'cloud')
    assert len(calls) == 1 and pick['my_lang']['de'].isChecked()
    config.save(s)  # back to ru <-> en, both local
    t.sync()
    pick['my_lang']['de'].setChecked(True)
    t.switch_pair('de', 'ja')
    assert 'mt-de-ja (CPML' in asked[0] and config.load().my_lang == 'ru' and len(calls) == 1  # declined: nothing changes
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


class Direction:
    """A local direction: says one clause in its pair, lives until closed."""
    made: list = []

    def __init__(self, src, tgt):
        self.src, self.tgt, self.closed = src, tgt, threading.Event()
        self.tts = None
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
    assert states[-1].startswith('translating (') and states[-1].endswith('languages de ⇄ ja)'), states
    ctl.set_translation(False).result(30)
    s.my_lang = 'fr'
    ctl.set_languages().result(30)
    assert len(Direction.made) == 4 and not ctl.on  # off: nothing made now
    ctl.quit()


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

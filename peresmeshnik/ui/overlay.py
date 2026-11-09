"""Subtitle overlay: a frameless, always-on-top, click-through window at the bottom of the screen.

Each row is one clause: the original (grey) above its translation (white). Captions come from the
direction threads through a queued Qt signal; the window repaints only when the visible text
changes, and nothing in it runs on a timer, so in silence it costs nothing. With `show_latency`
(tray: "Show latency", `ui.show_latency`) the original's line ends with the clause's latencies
from the end of its speech (`engines.Latency`): to its subtitle, to its voice (a cloud phrase: to the
end of its voice).
"""
from __future__ import annotations

import logging
import sys
import time
from collections import OrderedDict, deque

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QWidget

from peresmeshnik.engines import Caption, Latency

log = logging.getLogger('peresmeshnik.ui')
GREY, WHITE, BACK = QColor(170, 170, 170), QColor(255, 255, 255), QColor(0, 0, 0, 170)
ME_PREFIX = '» '  # rows of what I said (me -> them)
FINAL_AFTER_END_S = 1.5  # a phrase's last final caption comes this long after its speech ends, at most
SHOWN = {'caption': 'subtitle', 'voice': 'voice', 'voice_end': 'voice ends'}  # Latency kinds shown, in this order
FONT_HINT = {'darwin': 'macOS has them: Font Book → File → Restore Standard Fonts',
             'win32': 'Settings → Time & language → Language & region → Add a language (its fonts come with it)'}.get(
    sys.platform, 'e.g. sudo apt install fonts-noto-cjk fonts-noto-core (Noto fonts of your distribution)')


# Han characters look different in Chinese, Japanese and Korean (one code point, three glyph forms): a row in one of them
# is drawn in the first installed font of its own language, not in whatever fallback the system picks for Han (on the
# Windows runner a Chinese row came out half in a Korean font). The fonts have Latin letters and digits too.
OWN_FONTS = {
    'zh': ('Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei',
           'Droid Sans Fallback', 'SimSun'),
    'ja': ('Yu Gothic UI', 'Meiryo', 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Noto Sans CJK JP', 'Source Han Sans JP',
           'IPAGothic', 'MS Gothic'),
    'ko': ('Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans CJK KR', 'Source Han Sans KR', 'NanumGothic', 'UnDotum', 'Gulim'),
}


WARM = {'uk': 'Привіт', 'zh': '你好，这是', 'ja': 'こんにちは、カタカナ漢字', 'ko': '안녕하세요', 'ar': 'مرحبا بكم',
        'de': 'Grüße', 'fr': 'Ça été', 'es': '¿Señor?', 'pt': 'São ação', 'pl': 'Cześć', 'tr': 'Ğüşı', 'it': 'Così'}


def missing_fonts(langs) -> list[str]:
    """Languages whose script no installed font has (Qt's font database): their subtitles would be empty boxes.
    Arabic needs nothing more: Qt lays out right-to-left text by itself (the painter's direction is automatic)."""
    from peresmeshnik import languages
    if QGuiApplication.instance() is None:  # no font database without the GUI (a headless run): nothing to say
        return []
    return [x for x in dict.fromkeys(langs)
            if not QFontDatabase.families(getattr(QFontDatabase.WritingSystem, languages.WRITING.get(x, 'Latin')))]


def font_note(langs) -> str:
    """'' or what to install, for the wizard, the tray and the start of translation."""
    from peresmeshnik import languages
    if not (miss := missing_fonts(langs)):
        return ''
    return (f'no font for {", ".join(languages.name(x) for x in miss)} on this computer: its subtitles would show empty '
            f'boxes. Install one: {FONT_HINT}')


class SubtitleOverlay(QWidget):
    caption = Signal(str, object)  # (side, Caption), emitted from any thread
    pair = Signal(str, str)  # (my, their): their fonts load on the GUI thread at translation on, not at the first row
    latency = Signal(str, object)  # (side, Latency), emitted from any thread
    clear = Signal()

    def __init__(self, rows: int = 2, font_px: int = 24):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)  # a Qt tool window on macOS hides while its app is not active
        self.setWindowTitle('Peresmeshnik subtitles')
        self.rows = rows
        self.f_orig, self.f_tr = QFont(), QFont()
        self.f_orig.setPixelSize(round(font_px * 0.75))
        self.f_tr.setPixelSize(font_px)
        self.f_tr.setBold(True)
        self.h_orig, self.h_tr = QFontMetrics(self.f_orig).height(), QFontMetrics(self.f_tr).height()
        warm = QImage(8, 8, QImage.Format_ARGB32_Premultiplied)  # load both fonts and the Cyrillic fallback now:
        p = QPainter(warm)  # the first caption's paint took 23-28 ms on the macOS / Windows runners without it
        for f in (self.f_orig, self.f_tr):
            p.setFont(f)
            p.drawText(0, 8, 'Hello, Привет 0…')
        p.end()
        self.pad = font_px // 2
        scr = QGuiApplication.primaryScreen().availableGeometry()
        w, h = int(scr.width() * 0.7), rows * (self.h_orig + self.h_tr) + 2 * self.pad
        self.setGeometry(scr.x() + (scr.width() - w) // 2, scr.y() + scr.height() - h - 60, w, h)
        self.show_latency = False
        self.langs = {}  # side -> (language of its originals, of its translations): set_pair; a row keeps its own
        self._fonts: dict[str, tuple[QFont, QFont]] = {}  # language -> (original, translation) fonts: own_fonts
        self._rows: OrderedDict[tuple, list] = OrderedDict()  # (side, n) -> [original, translation, {kind: s}]
        self._open: dict[str, int] = {}  # side -> number of its open row, for captions without a clause number
        self._next = 0
        self._final_at: dict = {}  # row -> when its final caption came (a cloud phrase's numbers find their row by it)
        self.clock = time.monotonic
        self._shown: tuple = ()
        self._langs: tuple = ()  # (language of the original, of the translation) of each row shown
        self.paints, self.updates = 0, 0  # repaints / changes of the visible text
        self.paint_ms: deque[float] = deque(maxlen=10_000)
        self.caption.connect(self._on_caption)
        self.pair.connect(self._warm)
        self.latency.connect(self._on_latency)
        self.clear.connect(self._on_clear)
        self.on_top = None
        if sys.platform == 'win32':  # above other always-on-top windows too, while shown
            from peresmeshnik.ui.windows import OnTop
            self.on_top = OnTop()

    def side(self, name: str) -> OverlaySide:
        return OverlaySide(self, name)

    def set_pair(self, my: str, their: str, side: str | None = None) -> None:
        """Any thread (the controller's, at translation on and a new pair): the languages of each side's new rows, for
        their fonts at the next paint; `side`: only its (made again for the pair, the other still in the old one)."""
        pair = {'them': (their, my), 'me': (my, their)}
        self.langs = pair if side is None else {**self.langs, side: pair[side]}
        self.pair.emit(my, their)

    def _warm(self, my: str, their: str) -> None:
        """Load the pair's fonts now: a Chinese or Japanese first row took 85-121 ms to paint cold on the macOS and
        Windows runners (BENCH stage 6), Latin and Cyrillic are warmed in __init__."""
        img = QImage(8, 8, QImage.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        for lang in (my, their):
            for f in self.own_fonts(lang):
                p.setFont(f)
                p.drawText(0, 8, WARM.get(lang, 'Hello'))
        p.end()

    def own_fonts(self, lang: str) -> tuple[QFont, QFont]:
        """GUI thread: the row fonts for a language — Chinese, Japanese, Korean in an installed font of their own."""
        if lang not in self._fonts:
            fonts = self.f_orig, self.f_tr
            if own := next((f for f in OWN_FONTS.get(lang, ()) if f in set(QFontDatabase.families())), None):
                fonts = tuple(QFont(f) for f in fonts)
                for f in fonts:
                    f.setFamily(own)
            self._fonts[lang] = fonts
        return self._fonts[lang]

    # GUI thread -------------------------------------------------------------------------------------
    def _key(self, side: str, c: Caption) -> tuple:
        if c.clause >= 0:
            return side, 'c', c.clause
        if c.translation and c.final:  # a cloud translation: give it to the oldest row still without one
            for k, (_, tr, *_) in self._rows.items():
                if k[0] == side and not tr:
                    return k
        if side not in self._open:
            self._open[side], self._next = self._next, self._next + 1
        k = side, 'o', self._open[side]
        if c.final:
            del self._open[side]
        return k

    def _on_caption(self, side: str, c: Caption) -> None:
        k = self._key(side, c)
        row = self._rows.setdefault(k, ['', '', {}, self.langs.get(side, ('', ''))])  # its languages: as it came
        row[0] = c.original or row[0]
        row[1] = c.translation or row[1]
        if c.final:
            self._final_at[k] = self.clock()
        while len(self._rows) > self.rows:
            self._final_at.pop(self._rows.popitem(last=False)[0], None)
        if self._refresh():
            log.info('overlay %s: %s | %s', side, row[0], row[1])

    def _on_latency(self, side: str, lat: Latency) -> None:
        if lat.what not in SHOWN:  # the pipeline's own links: the log has them
            return
        k = (side, 'c', lat.clause)
        if lat.clause < 0:  # a cloud phrase (its voice ended s after its speech): the newest row final by then
            end = self.clock() - lat.s + FINAL_AFTER_END_S
            k = next((k for k in reversed(self._rows) if k[0] == side and self._final_at.get(k, end + 1) <= end), None)
        if k in self._rows:
            self._rows[k][2][lat.what] = lat.s
            if self.show_latency:
                self._refresh()

    def set_show_latency(self, on: bool) -> None:  # the tray's switch: the rows shown change now, not at the next one
        if on != self.show_latency:
            self.show_latency = on
            if self._rows:
                self._refresh()

    def _lat(self, v: dict) -> str:
        if not (self.show_latency and v):
            return ''
        return '  ⏱ ' + ' · '.join(f'{SHOWN[w]} +{v[w]:.2f} s' for w in SHOWN if w in v)

    def _refresh(self) -> bool:
        """The rows as painted; False (and no repaint) when nothing visible changed."""
        shown = tuple((ME_PREFIX if k[0] == 'me' else '', v[0], v[1], self._lat(v[2])) for k, v in self._rows.items())
        if shown == self._shown:
            return False
        self._shown, self._langs = shown, tuple(v[3] for v in self._rows.values())
        self.updates += 1
        if not self.isVisible():
            self.show()
            if sys.platform == 'darwin':  # after show(): Qt sets the window's level and behaviour while showing it
                from peresmeshnik.ui import macos
                macos.overlay(int(self.winId()))
            if self.on_top:
                self.on_top.start(int(self.winId()))
        self.update()
        return True

    def _on_clear(self) -> None:
        self._rows.clear()
        self._open.clear()
        self._final_at.clear()
        self._shown = ()
        self.hide()
        if self.on_top:
            self.on_top.stop()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt name)
        t = time.perf_counter()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(BACK)
        p.drawRoundedRect(self.rect(), self.pad, self.pad)
        w, y = self.width() - 2 * self.pad, self.pad + (self.rows - len(self._shown)) * (self.h_orig + self.h_tr)
        for (src, tgt), (prefix, orig, tr, lat) in zip(self._langs, self._shown):  # newest at the bottom
            f_orig, f_tr = self.own_fonts(src)[0], self.own_fonts(tgt)[1]  # long lines keep their end (the latest words)
            for text, font, color, h in ((orig + lat, f_orig, GREY, self.h_orig), (tr, f_tr, WHITE, self.h_tr)):
                if text:
                    p.setFont(font)
                    p.setPen(color)
                    p.drawText(QRect(self.pad, y, w, h), Qt.AlignHCenter | Qt.AlignVCenter,
                               QFontMetrics(font).elidedText(prefix + text, Qt.ElideLeft, w))
                y += h
        p.end()
        self.paints += 1
        self.paint_ms.append((time.perf_counter() - t) * 1000)


    def stats(self) -> str:
        ms = sorted(self.paint_ms)
        if not ms:
            return f'{self.updates} text changes, no repaints'
        return (f'{self.updates} text changes, {self.paints} repaints, paint median {ms[len(ms) // 2]:.2f} ms, '
                f'p99 {ms[int(len(ms) * 0.99)]:.2f} ms, max {ms[-1]:.2f} ms')


class OverlaySide:
    """`Overlay` (engines.py) for one direction; safe to call from the direction's thread."""

    def __init__(self, overlay: SubtitleOverlay, name: str):
        self.overlay, self.name = overlay, name

    def show(self, caption: Caption) -> None:
        self.overlay.caption.emit(self.name, caption)

    def latency(self, lat: Latency) -> None:
        self.overlay.latency.emit(self.name, lat)

    def hide(self) -> None:
        self.overlay.clear.emit()

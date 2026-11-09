"""Tray icon and its menu, and the settings window over `config.py`.

Without a system tray (GNOME without the AppIndicator extension, a bare X server) the same menu
sits on a button in a small window. Nothing here repaints or polls: the menu changes only when
the state does.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QActionGroup, QDesktopServices, QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from peresmeshnik import config

log = logging.getLogger('peresmeshnik.ui')
ICONS = Path(__file__).resolve().parents[2] / 'assets' / 'icons'  # ponytail: repo path; packaging (stage 3) bundles them
LANGS = {'en': 'English', 'ru': 'Русский'}
MIC_LABEL = 'Microphone'  # me -> them and the loopback while translation is off
SIDES = {'them_to_me': 'Them → me', 'me_to_them': 'Me → them'}
MAC, WIN = sys.platform == 'darwin', sys.platform == 'win32'
BROWSER_NOTE = (f'A browser {"is taken" if MAC or WIN else "moves"} with all of its sound: YouTube in another tab is translated too. '
                'For calls a separate browser is better.')
CALL_NOTE = ('While translating, macOS mutes it on your speakers' if MAC else
             'While translating, its own sound is muted on your speakers' if WIN else 'Its sound moves to the translation while it is on')
CAPTURE = {'darwin': 'capture_macos', 'win32': 'capture_windows'}.get(sys.platform, 'capture_pipewire')


def tray_icon() -> QIcon:
    icon = QIcon()
    if WIN:
        from peresmeshnik.ui.windows import light_taskbar
        light = not light_taskbar()  # the white glyph on a dark taskbar
    else:
        light = sys.platform.startswith('linux')  # dark panels; macOS colours the template itself
    for px in (16, 22, 24, 32, 44, 48, 64):
        f = ICONS / f'tray{"-light" if light else ""}-{px}.png'
        if f.exists():
            icon.addFile(str(f))
    icon.setIsMask(sys.platform == 'darwin')  # the black glyph as a template image: macOS colours it for the menu bar
    return icon


def capture():
    """The call-sound module of this OS: `apps()` playing now as [(key, label)], `is_browser(key)`."""
    import importlib
    return importlib.import_module(f'peresmeshnik.audio.{CAPTURE}')


class Tray(QObject):
    call_apps = Signal(object)  # (apps playing, error): listed on a thread of its own

    def __init__(self, ctl):
        super().__init__()
        self.ctl = ctl
        self.toggle_live = lambda: None  # set by the app once the hotkey is known
        self.reload_hotkey = lambda combo: ''
        self.hotkey_why = ''  # why there is no global hotkey ('' when it works); the wizard shows it too
        self.windows: list[QWidget] = []
        m = self.menu = QMenu()
        self.status = m.addAction('idle, translation off')
        self.status.setEnabled(False)
        m.addSeparator()
        self.translation = m.addAction('Translation')
        self.translation.setCheckable(True)
        self.translation.toggled.connect(self.set_translation)
        self.live = m.addAction('Live voice')
        self.live.setCheckable(True)
        self.live.setEnabled(False)  # only while translating: off, the call hears the microphone as is
        self.live.triggered.connect(lambda: self.toggle_live())
        self.engines: dict[str, QActionGroup] = {}
        for key, title in SIDES.items():
            sub = m.addMenu(title)
            group = self.engines[key] = QActionGroup(sub)
            for engine in config.ENGINES:
                a = sub.addAction(engine)
                a.setCheckable(True)
                a.setData((key, engine))
                group.addAction(a)
            group.triggered.connect(self._engine_chosen)
        self._call_group = None
        self.call_menu = m.addMenu('Call app')
        self.call_menu.aboutToShow.connect(self._fill_call_menu)
        self.call_apps.connect(self._show_call_apps)
        m.addSeparator()
        m.addAction('Settings…', self.open_settings)
        m.addAction('Setup wizard…', self.open_wizard)
        self.show_latency = m.addAction('Show latency')  # per clause in the subtitles: end of speech -> subtitle, -> voice
        self.show_latency.setCheckable(True)
        self.show_latency.toggled.connect(self._latency_chosen)
        self.crash = m.addAction('Crash report…', self.open_report)
        self.crash.setVisible(bool(_reports()))
        m.addSeparator()
        m.addAction('Quit', self._quit)
        self.sync()
        self.icon = QSystemTrayIcon(tray_icon())
        if WIN:  # the taskbar went light or dark: the glyph that shows on it
            from peresmeshnik.ui import windows
            windows.host().theme.connect(lambda: self.icon.setIcon(tray_icon()))
        self.icon.setContextMenu(m)
        self.icon.setToolTip('Peresmeshnik')
        self.fallback = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.icon.show()
        else:  # no tray on this desktop: the menu on a button in a small window
            log.warning('no system tray: the menu is in the Peresmeshnik window')
            w = self.fallback = _Window()
            w.setWindowTitle('Peresmeshnik')
            w.setWindowIcon(tray_icon())
            lay = QVBoxLayout(w)
            self.fallback_status = QLabel(self.status.text())
            b = QPushButton('Peresmeshnik ▾')
            b.setMenu(m)
            lay.addWidget(self.fallback_status)
            lay.addWidget(b)
            w.show()

    def sync(self) -> None:
        """Menu checks from the settings and the controller (after a change elsewhere)."""
        s = config.load()
        self.show_latency.blockSignals(True)
        self.show_latency.setChecked(s.ui.show_latency)
        self.show_latency.blockSignals(False)
        if hasattr(overlay := getattr(self.ctl, 'overlay', None), 'set_show_latency'):
            overlay.set_show_latency(s.ui.show_latency)
        for key, group in self.engines.items():
            for a in group.actions():
                a.setChecked(a.data()[1] == getattr(s, key))
        self.live.setChecked(self.ctl.live)

    def hotkey_note(self, combo: str, why: str, announce: bool = False) -> None:
        """The hotkey in the menu item; why there is none in its tooltip, on the wizard's hotkey page and, with
        `announce`, once as a tray message (the SNI menus of GNOME / KDE show no tooltips)."""
        self.live.setText(f'Live voice ({QKeySequence(combo).toString(QKeySequence.NativeText)})' if combo else 'Live voice')
        self.live.setToolTip(why)
        self.hotkey_why = why
        if why and announce:
            self.message(why)

    def message(self, text: str) -> None:
        if self.icon.isVisible():
            self.icon.showMessage('Peresmeshnik', text, QSystemTrayIcon.Warning, 8000)
        elif self.fallback:
            self.fallback_status.setText(text)

    def set_state(self, state: str) -> None:
        self.status.setText(state[:120])
        self.icon.setToolTip(f'Peresmeshnik: {state}')
        if self.fallback:
            self.fallback_status.setText(state)
        on = state.startswith(('warming up', 'translating', 'note'))
        self.live.setEnabled(self.ctl.on or state.startswith(('translating', 'note')))
        self.live.setChecked(self.ctl.live)
        if state.startswith('note'):  # translation goes on, differently (a cloud side went local, or back): said once
            self.message(state.removeprefix('note: '))
        if not state.startswith('error'):
            self.translation.blockSignals(True)
            self.translation.setChecked(on)
            self.translation.blockSignals(False)
        else:
            self.translation.blockSignals(True)
            self.translation.setChecked(self.ctl.on)
            self.translation.blockSignals(False)
            self.message(state)

    def report_ready(self, path: str, quiet: bool = False, why: str = 'An error was recorded') -> None:
        """A crash report is there (crash.py): the menu item shows; one message per report, none when the app's own
        message about it says so already (`quiet`: a side restarted)."""
        self.crash.setVisible(True)
        if not quiet:
            self.message(f'{why}. A report without your data (nothing is sent): tray menu → Crash report…')

    def open_report(self) -> None:
        if reports := _reports():
            self._show(ReportDialog(reports[-1]))

    def _latency_chosen(self, on: bool) -> None:
        s = config.load()
        s.ui.show_latency = on
        config.save(s)
        self.sync()

    def set_translation(self, on: bool) -> None:
        self.translation.blockSignals(True)
        self.translation.setChecked(on)
        self.translation.blockSignals(False)
        self.ctl.set_translation(on)

    def _fill_call_menu(self) -> None:
        """The apps playing now, read when the menu opens (never polled) on a thread: pactl may take seconds."""
        self.call_menu.clear()
        self.call_menu.addAction('Looking for the apps playing now…').setEnabled(False)

        def work():
            try:
                self.call_apps.emit((capture().apps(), ''))
            except Exception as e:  # no sound server; macOS, Windows: no call sound capture in this version yet
                missing = isinstance(e, ModuleNotFoundError) and e.name == f'peresmeshnik.audio.{CAPTURE}' and CAPTURE != 'capture_pipewire'
                self.call_apps.emit(([], 'not in this version' if missing else str(e)))
        threading.Thread(target=work, name='call-apps', daemon=True).start()

    def _show_call_apps(self, result) -> None:
        """In the GUI thread: the menu of apps; the chosen one stays listed."""
        items, error = result
        m, cur = self.call_menu, config.load().ui.call_app
        m.clear()
        if self._call_group:
            self._call_group.deleteLater()
        if error:
            m.addAction(f'Cannot list the playing apps: {error}').setEnabled(False)
        if cur and cur not in dict(items):
            items.append((cur, f'{cur} (not playing now)'))
        if not items:
            m.addAction('Nothing plays now: start the call, then open this menu again').setEnabled(False)
        group = self._call_group = QActionGroup(m)
        for key, label in items:
            a = m.addAction(label)
            a.setCheckable(True)
            a.setChecked(key == cur)
            a.setData(key)
            group.addAction(a)
        group.triggered.connect(self._call_chosen)
        m.addSeparator()
        m.addAction(CALL_NOTE).setEnabled(False)
        m.addAction(BROWSER_NOTE.split(':')[0] + ' (other tabs too)').setEnabled(False)

    def _call_chosen(self, action) -> None:
        self.ctl.set_call_app(action.data())
        if getattr(capture(), 'is_browser', lambda key: False)(action.data()):
            self.message(BROWSER_NOTE)

    def _engine_chosen(self, action) -> None:
        key, engine = action.data()
        s = config.load()
        setattr(s, key, engine)
        config.save(s)
        self.restart()

    def restart(self) -> None:
        """New settings take effect: a running translation is turned off and on again."""
        if self.ctl.on:
            self.ctl.set_translation(False)
            self.ctl.set_translation(True)

    def _show(self, w: QWidget) -> None:
        # keep a reference while it is open or still working (a closed wizard may be downloading models)
        self.windows = [x for x in self.windows if x.isVisible() or getattr(x, 'busy', lambda: False)()] + [w]
        w.show()
        w.raise_()
        w.activateWindow()
        if sys.platform == 'darwin':  # a menu-bar app is not active by itself: its window would open behind the others
            from peresmeshnik.ui import macos
            macos.front(int(w.winId()))

    def open_settings(self) -> None:
        self._show(SettingsDialog(self))

    def open_wizard(self) -> None:
        from peresmeshnik.ui.wizard import SetupWizard
        self._show(SetupWizard(self))

    def _quit(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.quit()


def _reports() -> list[Path]:
    from peresmeshnik import crash
    return sorted(crash.reports_dir().glob('crash-*.txt')) if crash.reports_dir().is_dir() else []


class ReportDialog(QDialog):
    """The newest crash report as it is, to read before sending: copy it, or open a GitHub issue with it."""

    def __init__(self, path: Path):
        super().__init__()
        self.path, self.text = path, path.read_text('utf-8', 'replace')
        self.setWindowTitle('Peresmeshnik crash report')
        self.resize(760, 520)
        lay = QVBoxLayout(self)
        note = QLabel('The app sends nothing. This report has no subtitles, audio, device names or keys: read it, then '
                      'copy it or open a GitHub issue with it and send it yourself.')
        note.setWordWrap(True)
        view = QPlainTextEdit(self.text)
        view.setReadOnly(True)
        b = QDialogButtonBox(QDialogButtonBox.Close)
        for title, fn in (('Copy', self.copy), ('Open a GitHub issue', self.issue), ('Show the folder', self.folder)):
            b.addButton(title, QDialogButtonBox.ActionRole).clicked.connect(fn)
        b.rejected.connect(self.reject)
        for w in (note, view, b):
            lay.addWidget(w)

    def copy(self) -> None:
        QGuiApplication.clipboard().setText(self.text)

    def issue(self) -> None:
        from peresmeshnik import crash
        self.copy()  # the link may carry only its start
        QDesktopServices.openUrl(QUrl(crash.issue_url(self.text), QUrl.TolerantMode))

    def folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path.parent)))


class _Window(QWidget):
    """The menu's window when there is no tray: closing it quits (else the app would be gone from view for good)."""

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt name)
        from PySide6.QtWidgets import QApplication
        QApplication.quit()


def key_file(cfg: config.Cloud | None = None) -> Path:
    """Where the app reads the key from (cloud/__init__.py `api_key`): `cloud.key_file`, else the config folder."""
    return Path(cfg.key_file) if cfg and cfg.key_file else config.config_dir() / 'gemini-key'


def save_key(key: str, cfg: config.Cloud | None = None) -> None:
    """The Gemini key goes into its own file, readable only by the user (never into settings.toml)."""
    f = key_file(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(0o600)
    f.chmod(0o600)
    f.write_text(key.strip() + '\n')


def device_combo(kind: str, current: str) -> QComboBox:
    """PipeWire / PulseAudio sinks or sources; '' = the system default."""
    c = QComboBox()
    c.addItem('System default', '')
    try:
        from peresmeshnik.ui.selftest import devices
        for name, desc in devices(kind):
            if not name.startswith('peresmeshnik_'):  # our own devices are no choice here
                c.addItem(desc, name)
    except Exception as e:  # no sound server: only the default
        c.setToolTip(f'cannot list devices: {e}')
    if current and c.findData(current) < 0:
        c.addItem(current, current)
    c.setCurrentIndex(max(0, c.findData(current)))
    return c


class SettingsDialog(QDialog):
    def __init__(self, tray: Tray):
        super().__init__()
        self.tray, self.s = tray, config.load()
        s = self.s
        self.setWindowTitle('Peresmeshnik settings')
        f = QFormLayout(self)
        self.my_lang, self.their_lang = QComboBox(), QComboBox()
        for c, v in ((self.my_lang, s.my_lang), (self.their_lang, s.their_lang)):
            for code, name in LANGS.items():
                c.addItem(name, code)
            c.setCurrentIndex(c.findData(v))
        f.addRow('I speak', self.my_lang)
        f.addRow('They speak', self.their_lang)
        self.engine = {}
        for key, title in SIDES.items():
            c = self.engine[key] = QComboBox()
            c.addItems(config.ENGINES)
            c.setCurrentText(getattr(s, key))
            f.addRow(title, c)
        self.subtitles = QCheckBox('Subtitles from the cloud too (else from the local ASR: faster)')
        self.subtitles.setChecked(s.cloud.subtitles)
        f.addRow('Cloud', self.subtitles)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText('saved' if key_file(s.cloud).exists() else 'Gemini API key (aistudio.google.com)')
        f.addRow('Gemini key', self.key)
        self.threads = QSpinBox()
        self.threads.setRange(1, 8)
        self.threads.setValue(s.local.threads)
        f.addRow('Threads per engine', self.threads)
        self.voice = QComboBox()
        self.voice.addItem('none', '')
        from peresmeshnik import voice
        for p in sorted(voice.voices_dir().glob('*/voice.json')):
            if p.parent.suffix not in ('.new', '.old'):  # the wizard's draft of a new take, left by a crash
                self.voice.addItem(p.parent.name, p.parent.name)
        self.voice.setCurrentIndex(max(0, self.voice.findData(s.local.voice)))
        f.addRow('My voice (local clone)', self.voice)
        self.hotkey = QKeySequenceEdit(QKeySequence(s.ui.hotkey))
        f.addRow('Hotkey: translation ⇄ live', self.hotkey)
        if sys.platform.startswith('linux'):
            self.headphones = device_combo('sinks', s.ui.headphones)
            self.mic = device_combo('sources', s.ui.mic)
            f.addRow('Headphones', self.headphones)
            f.addRow(MIC_LABEL, self.mic)
        self.note = QLabel()
        self.note.setWordWrap(True)
        f.addRow(self.note)
        b = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        b.accepted.connect(self.save)
        b.rejected.connect(self.reject)
        f.addRow(b)

    def save(self) -> None:
        s = self.s
        s.my_lang, s.their_lang = self.my_lang.currentData(), self.their_lang.currentData()
        if s.my_lang == s.their_lang:
            self.note.setText('Choose two different languages.')
            return
        for key, c in self.engine.items():
            setattr(s, key, c.currentText())
        s.cloud.subtitles = self.subtitles.isChecked()
        s.local.threads = self.threads.value()
        s.local.voice = self.voice.currentData()
        combo = self.hotkey.keySequence().toString(QKeySequence.PortableText)
        if combo and combo != s.ui.hotkey:
            if why := self.tray.reload_hotkey(combo):
                self.note.setText(f'Hotkey not changed: {why}')
                return
            s.ui.hotkey = combo
        if hasattr(self, 'headphones'):
            s.ui.headphones, s.ui.mic = self.headphones.currentData(), self.mic.currentData()
        if self.key.text().strip():
            save_key(self.key.text(), s.cloud)
        config.save(s)
        self.tray.sync()
        self.tray.restart()
        self.accept()

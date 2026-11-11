"""Tray icon and its menu, and the settings window over `config.py`.

Without a system tray (GNOME without the AppIndicator extension, a bare X server) the same menu
sits on a button in a small window. Nothing here repaints or polls: the menu changes only when
the state does.
"""
from __future__ import annotations

import logging
import os
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
    QHBoxLayout,
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

from peresmeshnik import config, languages

log = logging.getLogger('peresmeshnik.ui')
ICONS = Path(__file__).resolve().parents[2] / 'assets' / 'icons'  # ponytail: repo path; packaging (stage 3) bundles them
LANGS = {c: languages.label(c) for c in languages.LANGS}
MIC_LABEL = 'Microphone'  # me -> them and the loopback while translation is off
SIDES = {'them_to_me': 'Them → me', 'me_to_them': 'Me → them'}
SENT = {'them_to_me': "the call's sound (the other person's voice)", 'me_to_them': 'your voice'}  # what a cloud side sends
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
    pair_ready = Signal(object, str)  # (the settings with the new pair, '' or the error): its models, fetched on a thread
    told = Signal(str)  # a tray message from that thread

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
        self.langs: dict[str, QActionGroup] = {}
        lang_menu = m.addMenu('Languages')
        for key, title in (('my_lang', 'I speak'), ('their_lang', 'They speak')):
            sub = lang_menu.addMenu(title)
            group = self.langs[key] = QActionGroup(sub)
            for code, label in LANGS.items():
                a = sub.addAction(label)
                a.setCheckable(True)
                a.setData((key, code))
                group.addAction(a)
            group.triggered.connect(self._lang_chosen)
        self.pair_ready.connect(self._pair_ready)
        self.told.connect(self._told)
        self._pair = None  # the pair being fetched now: a later choice wins
        self._call_group = None
        self.call_menu = m.addMenu('Call app')
        self.call_menu.aboutToShow.connect(self._fill_call_menu)
        self.call_apps.connect(self._show_call_apps)
        m.addSeparator()
        m.addAction('Settings…', self.open_settings)
        m.addAction('Server…', self.open_server)  # translation on another computer: pairing
        m.addAction('Setup wizard…', self.open_wizard)
        m.addAction('Voices…', self.open_voices)
        m.addAction('Components…', self.open_components)
        self.remember = m.addAction('Keep the voice of the person in this call…', self.remember_voice)
        self.their_voice = m.addAction('Their own voice (them → me)')  # [local] their_voice: in my headphones only (S2)
        self.their_voice.setCheckable(True)
        self.their_voice.toggled.connect(self._their_voice_chosen)
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
        self.their_voice.blockSignals(True)
        self.their_voice.setChecked(s.local.their_voice)
        self.their_voice.blockSignals(False)
        if hasattr(overlay := getattr(self.ctl, 'overlay', None), 'set_show_latency'):
            overlay.set_show_latency(s.ui.show_latency)
        for groups in (self.engines, self.langs):
            for key, group in groups.items():
                for a in group.actions():
                    a.setChecked(a.data()[1] == getattr(s, key))
        self.live.setChecked(self.ctl.live)
        self.remember.setVisible(bool(getattr(self.ctl, 'on', False)) and hasattr(self.ctl, 'them_voice')
                                 and self._keeps_their_voice(s))

    @staticmethod
    def _keeps_their_voice(s) -> bool:
        """Whether the them side's clone engine here (S2's voices/live.engine) keeps a voice to take to the next call:
        Pocket TTS (its state) yes; an engine of the NVIDIA voice pack (XTTS, Qwen3) not (COMMON 23:49), nor none."""
        from peresmeshnik.voices import live
        engine = live.engine(s)[0]
        return bool(engine) and not languages.needs_gpu(engine)

    def hotkey_note(self, combo: str, why: str, announce: bool = False) -> None:
        """The hotkey in the menu item; why there is none in its tooltip, on the wizard's hotkey page and, with
        `announce`, once as a tray message (the SNI menus of GNOME / KDE show no tooltips)."""
        self.live.setText(f'Live voice ({QKeySequence(combo).toString(QKeySequence.NativeText)})' if combo else 'Live voice')
        self.live.setToolTip(why)
        self.hotkey_why = why
        if why and announce:
            self.message(f'No global hotkey: {why}')

    def message(self, text: str) -> None:
        if self.icon.isVisible():
            self.icon.showMessage('Peresmeshnik', text, QSystemTrayIcon.Warning, 8000)
        elif self.fallback:
            self.fallback_status.setText(text)

    def set_state(self, state: str) -> None:
        if state.startswith('notice: '):  # about something else than translating (the call app's volume): said, no more
            self.message(state.removeprefix('notice: '))
            return
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

    def _their_voice_chosen(self, on: bool) -> None:
        """Them -> me in the voice of the person talking, or in the stock voice: from the next start of the translation
        (a running one restarts)."""
        from peresmeshnik import models
        from peresmeshnik.voices import live
        s = config.load()
        s.local.their_voice = on
        config.save(s)
        self.sync()
        self.restart()
        if on and (e := live.engine(s)[0]) and not models.is_fetched(models.get(e)):
            self.message(f'{SIDES["them_to_me"]}: their voice ({e}) is not downloaded: a stock voice speaks for them; '
                         'Components… in this menu downloads it')

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
        """A side's engine from the menu: through switch_pair, so "local" downloads its models first (M-07)."""
        key, engine = action.data()
        from peresmeshnik.remote import pairing
        if engine == 'remote' and pairing.load() is None:  # pair first: the window puts both directions there after it
            self.sync()
            self.open_server()
            return
        base, chosen = (self._pair[0], self._pair[3]) if self._pair else (config.load(), {})  # a change on its way: further
        self.switch_pair(base.my_lang, base.their_lang, {**chosen, key: engine})

    def _lang_chosen(self, action) -> None:
        key, code = action.data()
        base = self._pair[0] if self._pair else config.load()  # a pair still downloading is the one changed further
        pair = {'my_lang': base.my_lang, 'their_lang': base.their_lang}
        if code == pair['their_lang' if key == 'my_lang' else 'my_lang']:  # the other side's language: the pair turns round
            pair = {'my_lang': pair['their_lang'], 'their_lang': pair['my_lang']}
        pair[key] = code
        if (pair['my_lang'], pair['their_lang']) == (base.my_lang, base.their_lang):  # the language it has: nothing to do
            return
        self.switch_pair(pair['my_lang'], pair['their_lang'], self._pair[3] if self._pair else None)

    def confirm(self, text: str) -> bool:
        """Yes / No, in front: a tray app is not active by itself, its question would open behind the other windows."""
        return self.ask(text, choose=False) == 'yes'

    def ask(self, text: str, choose: bool = True) -> str:
        """'yes', 'no' or 'choose' (a "Choose…" button), in front as `confirm`."""
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(QMessageBox.Question, 'Peresmeshnik', text, QMessageBox.Yes | QMessageBox.No)
        pick = box.addButton('Choose…', QMessageBox.ActionRole) if choose else None
        box.setDefaultButton(QMessageBox.No)
        box.show()
        box.raise_()
        box.activateWindow()
        if sys.platform == 'darwin':
            from peresmeshnik.ui import macos
            macos.front(int(box.winId()))
        box.exec()
        return 'choose' if pick is not None and box.clickedButton() is pick else \
            'yes' if box.clickedButton() is box.button(QMessageBox.Yes) else 'no'

    def switch_pair(self, my: str, their: str, engines: dict[str, str] | None = None) -> None:
        """Another pair, or another engine for a side (`engines`, the user's choice; 'subtitles': 'cloud' or 'local' is
        Settings…' "Subtitles from the cloud", whose local ASR and MT may be missing), without restarting the app: a local
        side that cannot run for the pair goes to the cloud only when the user says yes (its sound goes to Google), and
        is remembered as moved (`forced_cloud`): a later pair it can run brings it back to local; a side the user sets to
        local where it cannot run is said why and nothing changes. What a local side does differently is said (a stock
        voice instead of the clone); the missing models are fetched first (size and licenses said, restricted ones asked)
        while translation goes on as it is; then it is saved and the sides are made again (Controller.set_languages)."""
        from peresmeshnik import models
        from peresmeshnik.ui.overlay import font_note
        from peresmeshnik.ui.wizard import has_key
        s = config.load()
        chosen = dict(engines or {})
        subs = chosen.pop('subtitles', None)
        subs = None if subs is None or (subs == 'cloud') == s.cloud.subtitles else subs  # only a change
        # "cloud" chosen for a side the app moved there: the user's own from now on (no move back). Saved with the rest,
        # after the questions: a No keeps the mark
        kept = [k for k, v in chosen.items() if v == getattr(s, k) and k in s.forced_cloud]
        engines = {k: v for k, v in chosen.items() if v != getattr(s, k)}
        if (my, their) == (s.my_lang, s.their_lang) and not engines and not subs:  # back to what runs: another's download is let go
            if kept:  # nothing else changes: nothing to ask
                s.forced_cloud = [k for k in s.forced_cloud if k not in kept]
                config.save(s)
            self._pair = None
            self.status.setText(self.icon.toolTip().removeprefix('Peresmeshnik: ')[:120])  # the last state, as before it
            self.sync()
            return
        pair = f'{languages.name(my)} ⇄ {languages.name(their)}'
        name = ', '.join(([pair] if (my, their) != (s.my_lang, s.their_lang) else []) + [f'{SIDES[k]}: {v}' for k, v in engines.items()]
                         + ([f'{subs} subtitles'] if subs else []))
        s.my_lang, s.their_lang = my, their
        if subs:
            s.cloud.subtitles = subs == 'cloud'
        for key, engine in engines.items():
            setattr(s, key, engine)
        s.forced_cloud = [k for k in s.forced_cloud if k not in [*engines, *kept]]  # the user's own: no move back
        notes, moved, back, why = [], [], [], {}
        for key in languages.SIDES:
            d = languages.direction(s, key)
            if getattr(s, key) == 'local' and not d.local and key in engines:
                self.message(f'{SIDES[key]} cannot run on this computer for {pair} ({"; ".join(d.why)}): not changed')
                self.sync()
                return
            if getattr(s, key) == 'local' and not d.local:
                setattr(s, key, 'cloud')
                moved.append(key)
                why[key] = '; '.join(d.why)
                notes.append(f'{SIDES[key]} goes to the cloud ({why[key]})')
            elif getattr(s, key) == 'cloud' and key in s.forced_cloud and d.local:  # the app moved it: this pair runs here
                setattr(s, key, 'local')
                back.append(key)
                notes.append(f'{SIDES[key]} is back on this computer')
            if getattr(s, key) == 'local':
                notes += [f'{SIDES[key]}: {n}' for n in d.notes]
        if 'cloud' in (s.them_to_me, s.me_to_them) and not has_key(s):
            self.message(f'{name}: not changed — it needs the cloud ({"; ".join(notes) or "chosen"}) and no Gemini key is '
                         'set (Settings…)')
            self.sync()
            return
        if moved and not self.confirm(
                f'{name}: ' + ' '.join(f'{SIDES[k]} cannot run on this computer ({why[k]}).' for k in moved)
                + f' In the cloud {" and ".join(SENT[k] for k in moved)} {"go" if len(moved) > 1 else "goes"} to Google '
                '(Gemini Live). Switch to the cloud for this pair? A pair that runs here brings it back.'):
            self.sync()
            return
        todo, more = clone_notes(s, [m for m in map(models.get, languages.needed_models(s)) if not models.is_fetched(m)])
        notes += more
        lic = ', '.join(dict.fromkeys(p.license for m in todo for p in models.parts(m)))  # a pair through English: both
        restricted = list(dict.fromkeys(f'{p.id} ({p.license}, {p.license_url})' for m in todo for p in models.parts(m)
                                        if p.consent))
        engines_now = {**engines, **{k: getattr(s, k) for k in kept}, **({'subtitles': subs} if subs else {})}
        answer = self.ask(f'{name}: download {sum(map(models.size, todo)) / 1e6:,.0f} MB ({lic})?'
                          + (f' Under restricted licenses (noncommercial use only, they bind you as the user): '
                             f'{", ".join(restricted)}; Yes accepts them.' if restricted else '')
                          + ' Choose… shows what is downloaded and lets you pick.') if todo else 'yes'
        if answer == 'no':
            self.sync()
            return
        if answer == 'choose':  # the pair changes once what it needs is downloaded there
            from peresmeshnik.ui.components import ComponentsDialog
            self._pair = s, moved, back, engines_now
            d = ComponentsDialog(s, translating=lambda: bool(getattr(self.ctl, 'on', False)))
            d.box.finished.connect(lambda err, s=s, box=d.box: box.missing() or self._pair_ready(s, ''))  # a refused clone
            # (403) is not missing: the pair changes, its stock voice speaks

            def let_go(*_, s=s, d=d):  # closed before what the pair needs is here: that pair is not on its way any more
                if self._pair and self._pair[0] is s and not d.box.busy:
                    self._pair = None
            d.finished.connect(let_go)  # closed with nothing downloading
            d.box.finished.connect(lambda err: d.isVisible() or let_go())  # its download ended after it was closed
            self._show(d)
            self.sync()
            return
        extra = [m for m in map(models.get, languages.optional_models(s)) if not models.is_fetched(m)]
        if extra and not self.confirm(  # a cloud side's fallback voice nobody chose: asked apart, a No changes nothing else
                f'{name}: when the cloud fails, a cloud direction goes on with the stock voice '
                f'{", ".join(f"{m.id} ({m.license}, {m.license_url})" for m in extra)}, under a restricted license '
                '(noncommercial use only, it binds you as the user). Download and accept it? No: that direction stops '
                'until the cloud is back.'):
            extra = []
        lic = ', '.join(dict.fromkeys(p.license for m in todo + extra for p in models.parts(m)))
        if note := font_note([my, their]):
            notes.append(note)
        self._pair = s, moved, back, engines_now
        if not todo and not extra:
            if notes:
                self.message(f'{name}: ' + '. '.join(notes))
            return self._pair_ready(s, '')
        self.message(f'{name}: downloading {sum(map(models.size, todo + extra)) / 1e6:,.0f} MB ({lic}); translation goes on as it is '
                     'meanwhile' + ''.join(f'. {n}' for n in notes))
        self.status.setText(f'downloading the models for {name}')  # the Translation check stays as it is

        def work():
            try:
                for m in todo:
                    try:
                        models.fetch(m.id)  # one download per model at a time (the wizard may fetch it too)
                    except models.Gated as e:  # its token refused: the others still download; the clone gives way
                        self.told.emit(f'{name}: {e}; a stock voice speaks meanwhile')
            except Exception as e:  # the network: the .part files stay, choosing it again resumes them
                return self.pair_ready.emit(s, str(e))
            for m in extra:  # optional: its failure does not hold the pair back
                try:
                    models.fetch(m.id)
                except Exception as e:
                    self.told.emit(f'{name}: the fallback voice {m.id} did not download ({e}): when the cloud fails, that '
                                   'direction stops until it is back; choosing the pair again resumes the download')
            self.pair_ready.emit(s, '')
        threading.Thread(target=work, name='pair-fetch', daemon=True).start()

    def _told(self, text: str) -> None:
        self.message(text)

    def _pair_ready(self, s: config.Settings, err: str) -> None:
        if not self._pair or s is not self._pair[0]:  # another pair was chosen since
            return
        (_, moved, back, engines), self._pair = self._pair, None
        name = f'{languages.name(s.my_lang)} ⇄ {languages.name(s.their_lang)}'
        if err:
            self.set_state(f'error: {name} not changed: the download failed ({err}); choose it again to resume')
            self.sync()
            return
        saved = config.load()  # the pair, and the sides it or the user moved; the rest as saved now (changed meanwhile)
        saved.my_lang, saved.their_lang = s.my_lang, s.their_lang
        for key in moved + back + [k for k in engines if k in languages.SIDES]:
            setattr(saved, key, getattr(s, key))
        if 'subtitles' in engines:
            saved.cloud.subtitles = s.cloud.subtitles
        saved.forced_cloud = [k for k in languages.SIDES if k in moved or (k in saved.forced_cloud and k not in back + list(engines))]
        config.save(saved)
        self.sync()
        self.ctl.set_languages()  # its control thread knows whether translation is on (a start may be under way)

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

    def open_server(self) -> None:
        from peresmeshnik.ui.server import ServerDialog
        self._show(ServerDialog(self))

    def open_wizard(self) -> None:
        from peresmeshnik.ui.wizard import SetupWizard
        self._show(SetupWizard(self))

    def open_voices(self) -> None:
        from peresmeshnik.ui.voices import VoicesDialog
        s = config.load()

        def changed(mine: bool) -> None:  # my voice deleted or confirmed: the me side takes it (or the stock voice)
            if mine:
                saved = config.load()
                saved.local.voice = s.local.voice
                config.save(saved)
                self.restart()
        self._show(VoicesDialog(s, changed, translating=lambda: bool(getattr(self.ctl, 'on', False))))

    def open_components(self) -> None:
        """What is downloaded and what can be: the wizard's Components page for the pair in the settings now."""
        from peresmeshnik.ui.components import ComponentsDialog
        self._show(ComponentsDialog(config.load(), translating=lambda: bool(getattr(self.ctl, 'on', False))))

    def remember_voice(self) -> None:
        """Keep the voice of the person in this call, only with their agreement (S2: Controller.them_voice() ->
        (({model id: (state, sha256 of the weights)}, language, seconds of their speech), '') or (None, why not)). The
        voice kept is the one of when the user said they agreed, not of when the dialog opened."""
        from peresmeshnik.ui.voices import RememberDialog
        _, why = self.ctl.them_voice()
        if why:
            self.message(f'Their voice cannot be kept now: {why}.')
            return
        d = RememberDialog()
        if d.exec():
            live, why = self.ctl.them_voice()
            if why:
                self.message(f'Their voice was not kept: {why}.')
                return
            d.keep(*live)
            self.message('Kept: their voice is in Voices…, heard only by you.')

    def _quit(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.quit()


def clone_notes(s, todo: list) -> tuple[list, list[str]]:
    """A pair's models to download (`todo`, registry rows) -> (those without the clone engines, the notes): nobody chose
    a clone here, Components… downloads it; said per side, mine or theirs (S2: the other person's voice). The speaker
    model waits with them: without a clone it is not needed."""
    from peresmeshnik import models
    from peresmeshnik.voice import SPEAKER
    from peresmeshnik.voices import live
    if not (clone := [m for m in todo if m.role == 'clone']):
        return todo, []
    me = languages.direction(s, 'me_to_them')
    mine = me.tts if s.me_to_them == 'local' and me.voice == 'clone' else None
    their = live.engine(s)[0] if s.them_to_me == 'local' else None
    notes = []
    for key, engine, what in (('me_to_them', mine, 'your cloned voice'), ('them_to_me', their, 'their voice')):
        if engine and (key == 'me_to_them' or engine != mine) and (  # one engine for both: said once, as mine
                ms := [m for m in clone if m.id in (engine, models.get(engine).fallback)]):
            notes.append(f'{SIDES[key]}: {what} ({ms[0].id}, {sum(map(models.size, ms)) / 1e9:.1f} GB) is not '
                         f'downloaded: a stock voice speaks{" for them" if key == "them_to_me" else ""}; Components… in '
                         'this menu downloads it')
    return [m for m in todo if m not in clone and m.id != SPEAKER], notes


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
        self.forget = QPushButton('Forget the key')  # an empty field keeps it: this removes the app's own file, on Save
        self.forget.clicked.connect(self.forget_key)
        self.forgotten = False
        self._key_state()
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.key)
        lay.addWidget(self.forget)
        f.addRow('Gemini key', row)
        self.threads = QSpinBox()
        self.threads.setRange(1, 8)
        self.threads.setValue(s.local.threads)
        f.addRow('Threads per engine', self.threads)
        self.voice = QComboBox()
        self.voice.addItem('none', '')
        from peresmeshnik import voice
        for v in voice.profiles():  # mine only: a contact's voice never speaks as me (the me side refuses it anyway)
            if v.meta['kind'] == 'me':
                self.voice.addItem(v.meta['name'], v.name)
        self.voice.setCurrentIndex(max(0, self.voice.findData(s.local.voice)))
        f.addRow('My voice (local clone)', self.voice)
        self.hotkey = QKeySequenceEdit(QKeySequence(s.ui.hotkey))
        f.addRow('Hotkey: translation ⇄ live', self.hotkey)
        self.headphones = device_combo('sinks', s.ui.headphones)  # every OS: the wizard's list (selftest.devices)
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

    def _key_state(self) -> None:
        env, mine = os.environ.get('GEMINI_API_KEY', '').strip(), not self.s.cloud.key_file  # cloud.api_key's order
        saved = key_file(self.s.cloud).exists()
        d = config.config_dir()
        old = d.with_name(d.name.lower()) / 'gemini-key'  # the lowercase folder api_key reads too (Linux)
        old = old if mine and not saved and old.is_file() else None
        self.key.setPlaceholderText('from $GEMINI_API_KEY' if env else f'from {old}' if old else 'forgotten on Save'
                                    if self.forgotten else 'saved' if saved else 'Gemini API key (aistudio.google.com)')
        self.forget.setEnabled(saved and mine and not env and not self.forgotten)
        self.forget.setToolTip('The key comes from the environment ($GEMINI_API_KEY): unset it there.' if env else
                               f'The key is read from the old folder: remove {old} there.' if old else '' if mine else
                               f'The key is read from your own file {self.s.cloud.key_file}: not removed here.')

    def forget_key(self) -> None:
        self.forgotten = True  # Cancel keeps it
        self._key_state()

    def save(self) -> None:
        s = self.s
        pair = self.my_lang.currentData(), self.their_lang.currentData()
        if pair[0] == pair[1]:
            self.note.setText('Choose two different languages.')
            return
        # only what the user changed (a side the app moved shows "cloud": not a choice), saved with the pair by switch_pair
        engines = {key: c.currentText() for key, c in self.engine.items() if c.currentText() != getattr(s, key)}
        if self.subtitles.isChecked() != s.cloud.subtitles:  # local subtitles need the local ASR and MT: fetched first
            engines['subtitles'] = 'cloud' if self.subtitles.isChecked() else 'local'
        s.local.threads = self.threads.value()
        s.local.voice = self.voice.currentData()
        combo = self.hotkey.keySequence().toString(QKeySequence.PortableText)
        if combo and combo != s.ui.hotkey:
            if why := self.tray.reload_hotkey(combo):
                self.note.setText(f'Hotkey not changed: {why}')
                return
            s.ui.hotkey = combo
        s.ui.headphones, s.ui.mic = self.headphones.currentData(), self.mic.currentData()
        if new_key := bool(self.key.text().strip()):
            save_key(self.key.text(), s.cloud)
        elif self.forgotten:
            key_file().unlink(missing_ok=True)  # the app's own file only (its button is off for one's own key_file, env)
        if new_key or self.forgotten or s != config.load():  # the pair and the engines as they were: switch_pair saves them, models first
            config.save(s)
            self.tray.sync()
            self.tray.restart()
        if pair != (s.my_lang, s.their_lang) or engines:
            self.tray.switch_pair(*pair, engines)
        self.accept()

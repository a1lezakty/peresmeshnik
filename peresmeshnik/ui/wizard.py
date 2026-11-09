"""Setup wizard (first run, and from the tray): languages and engines, devices, models with their
licenses, the loopback self-test, your voice (recorded or imported, checked, and the clone checked),
the hotkey and a test phrase in both directions.

Slow work (downloads, the self-test, the test phrase) runs on worker threads and reports through
Qt signals; nothing is polled. Settings are written when the wizard finishes.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

import psutil
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from peresmeshnik import config, languages, models
from peresmeshnik.languages import needed_models  # noqa: F401 (the models page asks it; tests patch it here)
from peresmeshnik.local import voicepack
from peresmeshnik.ui.overlay import font_note
from peresmeshnik.ui.tray import LANGS, MIC_LABEL, SIDES, device_combo, save_key

LINUX = sys.platform.startswith('linux')
log = logging.getLogger('peresmeshnik.ui')
MACOS = sys.platform == 'darwin'
WINDOWS = sys.platform == 'win32'
VMIC = 'BlackHole 2ch' if MACOS else 'CABLE Output' if WINDOWS else 'Peresmeshnik Mic'  # the call app's microphone
SAMPLES = Path(__file__).parent / 'samples'


def has_key(s: config.Settings) -> bool:
    from peresmeshnik.cloud import api_key
    try:
        return bool(api_key(s.cloud))
    except SystemExit:
        return False


MIN_VRAM_GB = 5.5  # the clone (XTTS-v2 ~2 GB) next to them -> me on CUDA (~0.6 GB): a 6 GB card (6144 MiB, some report less)


PACK_CONSENT = ('I accept the licenses of the voice pack: PyTorch (BSD-3-Clause), coqui-tts (MPL-2.0), and of the XTTS-v2 '
                'voice model it runs, the Coqui Public Model License (huggingface.co/coqui/XTTS-v2): noncommercial use '
                'only, it binds me as the user')


MIN_DRIVER = (570, 65)  # Windows: CUDA 12.8 GA's driver (the pack's torch 2.8.0+cu128; NVIDIA's CUDA 12.8.0 notes)
MIN_DRIVER_COMPAT = (528, 33)  # CUDA 12.x minor version compatibility on Windows: below 570.65 not checked, may work
MIN_CC = 6.1  # the pack's Windows torch is built for sm_61 .. sm_120 (its _cuda_getArchFlags, the voice-pack job)


class Gpu(NamedTuple):
    vram: float = 0.0  # GB
    driver: tuple[int, ...] = ()
    cc: float = 0.0  # compute capability; 0: not known (a driver too old to say)
    why: str = ''  # nvidia-smi is there but fails: the driver's own words

    @property
    def driver_text(self) -> str:
        return _version(self.driver)


def _smi(exe: str, fields: str) -> list[list[str]] | str:
    try:
        r = subprocess.run([exe, f'--query-gpu={fields}', '--format=csv,noheader,nounits'], capture_output=True, text=True,
                           timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError) as e:
        return str(e)
    rows = [[x.strip() for x in ln.split(',')] for ln in (r.stdout or '').splitlines() if ln.strip()]
    if rows and all(len(row) == fields.count(',') + 1 and row[0].replace('.', '', 1).isdigit() for row in rows):
        return rows
    return ' '.join((r.stdout or '').split())[:200] or f'nvidia-smi exited with {getattr(r, "returncode", "?")}'


def gpu() -> Gpu:
    """The biggest NVIDIA GPU by nvidia-smi (no torch, ~0.1 s): memory, driver, compute capability; Gpu() without one.
    A driver that does not answer is not "no GPU": `why` says what nvidia-smi said."""
    exe = shutil.which('nvidia-smi')
    if not exe:
        return Gpu()
    rows = _smi(exe, 'memory.total,driver_version,compute_cap')
    if isinstance(rows, str):  # a driver before compute_cap (R510) does not know the field
        rows = old if not isinstance(old := _smi(exe, 'memory.total,driver_version'), str) else rows
    if isinstance(rows, str):
        return Gpu(why=rows)
    best = max(rows, key=lambda row: float(row[0]))

    def num(x: str, kind=float):
        try:
            return kind(x)
        except ValueError:
            return kind()
    return Gpu(float(best[0]) / 1024, tuple(num(x, int) for x in best[1].split('.')), num(best[2]) if len(best) > 2 else 0.0)


def gpu_text(vram: float) -> str:
    return f'an NVIDIA GPU with {vram:.0f} GB' if vram else 'no NVIDIA GPU'


def _version(v: tuple[int, ...]) -> str:
    return '.'.join(f'{x:02d}' if i else str(x) for i, x in enumerate(v))


def gpu_note(g: Gpu) -> str:
    """A driver between CUDA 12.x's compatibility floor and CUDA 12.8's own: not refused, not checked either."""
    if not g.why and MIN_DRIVER_COMPAT <= g.driver < MIN_DRIVER:
        return (f'Your NVIDIA driver {g.driver_text} is older than {_version(MIN_DRIVER)} (CUDA 12.8\'s): not checked, it '
                'may work; if the clone does not start, update it from nvidia.com/drivers.')
    return ''


def gpu_refusal(g: Gpu) -> str:
    """Why the voice pack (Windows) would not run the clone on this GPU, said before its 3.7 GB download; '' if it would."""
    if g.why:
        return (f'The NVIDIA driver does not answer ({g.why}): reinstall or update it from nvidia.com/drivers, then open '
                'this page again.')
    if g.vram < MIN_VRAM_GB:
        return (f'The clone of your voice needs an NVIDIA GPU with at least 6 GB of memory; this computer has '
                f'{gpu_text(g.vram)}.')
    if g.driver < MIN_DRIVER_COMPAT:
        return (f'The clone of your voice needs NVIDIA driver {_version(MIN_DRIVER)} or newer; this computer has '
                f'{g.driver_text or "an older one"}: update it from nvidia.com/drivers (or the NVIDIA app), then open this '
                'page again.')
    if g.cc and g.cc < MIN_CC:
        return (f'This NVIDIA GPU is too old for the voice pack (compute capability {g.cc:g}; it needs {MIN_CC:g} or newer: '
                'GeForce GTX 10xx and later).')
    return ''


def tier() -> tuple[str, str]:
    """(what this machine is, the recommended me -> them engine)."""
    gb, vram = psutil.virtual_memory().total / 2 ** 30, gpu().vram
    rec = 'local' if vram >= MIN_VRAM_GB else 'cloud'  # the clone keeps up only on an NVIDIA GPU with room for it (BENCH)
    return f'{gb:.0f} GB RAM, {gpu_text(vram)}: recommended them → me local, me → them {rec}', rec


class LanguagesPage(QWizardPage):
    """The pair, and per side local or cloud: before anything is downloaded, what each side runs on, why local cannot be
    (languages.direction), what is downloaded for it, under which licenses, and whether the subtitles' fonts are here."""

    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        s = w.s
        self.setTitle('Languages and engines')
        what, self.rec = tier()
        self.setSubTitle(f'This computer: {what}.')
        f = QFormLayout(self)
        self.my_lang, self.their_lang = QComboBox(), QComboBox()
        for c, v in ((self.my_lang, s.my_lang), (self.their_lang, s.their_lang)):
            for code, name in LANGS.items():
                c.addItem(name, code)
            c.setCurrentIndex(max(0, c.findData(v)))
        f.addRow('I speak', self.my_lang)
        f.addRow('They speak', self.their_lang)
        self.engine, self.why = {}, {}
        for key, title in SIDES.items():
            c = self.engine[key] = QComboBox()
            c.addItems(config.ENGINES)
            c.setCurrentText(getattr(s, key) if config.settings_path().exists() else ('local' if key == 'them_to_me' else self.rec))
            self.why[key] = QLabel()
            self.why[key].setWordWrap(True)
            f.addRow(title, c)
            f.addRow('', self.why[key])
        self.subtitles = QCheckBox('Subtitles from the cloud (else from the local ASR: faster, needs 0.7–2.2 GB of models; '
                                   'this pair\'s below)')
        self.subtitles.setChecked(s.cloud.subtitles)
        f.addRow('Cloud', self.subtitles)
        self.download = QLabel()
        self.download.setWordWrap(True)
        self.download.setOpenExternalLinks(True)
        f.addRow('Download', self.download)
        self.fonts = QLabel()
        self.fonts.setWordWrap(True)
        f.addRow(self.fonts)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText('already set' if has_key(s) else 'Gemini API key: aistudio.google.com → Get API key')
        f.addRow('Gemini key', self.key)
        note = QLabel('The cloud is Google Gemini Live. On the free tier Google may use the audio to improve its models; '
                      'use a paid key for real calls. Local engines keep everything on this computer.')
        note.setWordWrap(True)
        f.addRow(note)
        self.err = QLabel()
        f.addRow(self.err)
        for c in (self.my_lang, self.their_lang, *self.engine.values()):
            c.currentIndexChanged.connect(self.refresh)
        self.subtitles.toggled.connect(self.refresh)
        self.refresh()

    def chosen(self) -> config.Settings:
        """The settings as this page has them now (the wizard's own are written in validatePage)."""
        s = copy.deepcopy(self.w.s)
        s.my_lang, s.their_lang = self.my_lang.currentData(), self.their_lang.currentData()
        for key, c in self.engine.items():
            setattr(s, key, c.currentText())
        s.cloud.subtitles = self.subtitles.isChecked()
        return s

    def refresh(self) -> None:
        """Per side: local is offered only where it can run (else it says why and the side goes to the cloud), with what
        runs differently; then the download for this pair with its licenses, and the fonts."""
        s = self.chosen()
        for key, c in self.engine.items():
            d = languages.direction(s, key)
            item = c.model().item(config.ENGINES.index('local'))
            item.setEnabled(d.local)
            if not d.local and c.currentText() == 'local':
                c.blockSignals(True)
                c.setCurrentText('cloud')
                c.blockSignals(False)
                setattr(s, key, 'cloud')
            if not d.local:
                text = f'✘ Not on this computer: {"; ".join(d.why)}. The cloud translates {languages.name(d.src)} → ' \
                       f'{languages.name(d.tgt)}.'
            else:
                text = '✔ Both on this computer and in the cloud.' + ''.join(f' ⚠ {n}.' for n in d.notes if c.currentText() == 'local')
            if c.currentText() == 'cloud' and languages.cloud_subtitles(s, key) and not s.cloud.subtitles:
                text += ' Its subtitles come from the cloud too.'
            if d.src == 'ar' and c.currentText() == 'cloud' and languages.cloud_subtitles(s, key):
                text += ' Arabic subtitles from the cloud are Egyptian Arabic (Gemini has no other): other dialects come out ' \
                        'worse.'
            if MACOS and key == 'me_to_them':  # BENCH stage 6: no clone keeps up on an M1's CPU (XTTS, LuxTTS measured)
                text += ' On a Mac your voice goes to the call through the cloud (or a stock voice): no local clone of it ' \
                        'keeps up here.'
            self.why[key].setText(text)
        ms, base = [models.get(i) for i in needed_models(s)], models.root()
        todo = [m for m in ms if not models.is_fetched(m, base)]
        lic = {}  # one link per license
        for m in [p for x in ms for p in models.parts(x)]:  # a pair through English: both of its models' licenses
            lic.setdefault(m.license, f'<a href="{m.license_url}">{m.license}</a>'
                           + (' (noncommercial: you accept it on the models page)' if m.consent else ''))
        self.download.setText((f'{sum(map(models.size, todo)) / 1e6:,.0f} MB for this pair ({len(todo)} of {len(ms)} models '
                               'not here yet)' if todo else 'Nothing: every model of this pair is here') +
                              '. Licenses: ' + ', '.join(lic.values()) + '.')
        self.fonts.setText(f'⚠ Subtitles: {note}' if (note := font_note([s.my_lang, s.their_lang])) else '')

    def validatePage(self) -> bool:  # noqa: N802 (Qt name)
        s = self.chosen()
        self.w.key = self.key.text().strip()
        if s.my_lang == s.their_lang:
            self.err.setText('Choose two different languages.')
            return False
        if 'cloud' in (s.them_to_me, s.me_to_them) and not (self.w.key or has_key(s)):
            self.err.setText('The cloud needs a Gemini API key.')
            return False
        if bad := [SIDES[k] for k in languages.SIDES if getattr(s, k) == 'local' and not languages.direction(s, k).local]:
            self.err.setText(f'{", ".join(bad)}: not on this computer for this pair; choose cloud.')
            return False
        self.w.s.my_lang, self.w.s.their_lang, self.w.s.cloud.subtitles = s.my_lang, s.their_lang, s.cloud.subtitles
        for k in languages.SIDES:
            setattr(self.w.s, k, getattr(s, k))
        if self.w.key:
            save_key(self.w.key, s.cloud)  # the test phrase on a later page already needs it
        return True


class DevicesPage(QWizardPage):
    mic_checked = Signal(bool)  # Windows: the microphone is denied (the worker's probe)

    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        self.setTitle('Sound devices')
        self.setSubTitle('Translated speech goes to your headphones; your translated voice goes to the virtual microphone '
                         f'"{VMIC}", which you choose in the call app.')
        self.f = QFormLayout(self)
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.mic_checked.connect(self._mic_checked)
        self._vmic_text, self._denied, self._probed = '', False, False

    def initializePage(self) -> None:  # noqa: N802
        while self.f.rowCount():
            self.f.removeRow(0)
        self.headphones = device_combo('sinks', self.w.s.ui.headphones)
        self.mic = device_combo('sources', self.w.s.ui.mic)
        self.headphones.currentIndexChanged.connect(self._echo)
        self.f.addRow('Headphones', self.headphones)
        self.f.addRow(MIC_LABEL, self.mic)
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.f.addRow(self.warn)
        self._echo()
        if MACOS or WINDOWS:
            self.vmic = QLabel()
            self.vmic.setWordWrap(True)
            self.vmic.setOpenExternalLinks(True)
            self.vmic.setTextInteractionFlags(Qt.TextBrowserInteraction)  # the brew command can be copied
            self.recheck = QPushButton('Check again')
            self.recheck.clicked.connect(lambda: self.check_vmic(probe=True))
            self.f.addRow('Virtual microphone', self.vmic)
            self.f.addRow('', self.recheck)
            self.check_vmic(probe=not self._probed)  # the microphone tried once, not on every visit to the page

    def check_vmic(self, probe: bool = False) -> bool:
        return self.check_cable(probe) if WINDOWS else self.check_blackhole()

    def check_cable(self, probe: bool = False) -> bool:
        """Windows: VB-Cable is the user's to install (the driver needs an administrator and a restart); missing -> how
        to, and "Check again". `probe`: also try the microphone (Windows' permission), in a worker: up to seconds."""
        from peresmeshnik.audio import windows
        try:
            devs = windows.devices()
            play, rec = windows.cable(devs)
            text, ok = ('✔ VB-Audio Virtual Cable is installed: in the call app choose the microphone '
                        f'<b>{windows.CABLE_OUT}</b>.'), True
            if any(d['default_output'] for d in devs if d['outputs'] and windows.is_ours(d)):
                text += (f'<br>⚠ <b>{play["name"]}</b> is Windows\' default output: what Windows plays, the translation of '
                         'the other side too, goes into the call instead of your headphones. Choose your headphones or '
                         'speakers in Settings → System → Sound → Output.')
            if rec['default_input']:
                text += (f'<br>⚠ <b>{rec["name"]}</b> is Windows\' default microphone: the call app may take it, but every '
                         'other app hears the translation instead of you. Choose your real microphone in Settings → System '
                         '→ Sound → Input, and CABLE Output in the call app.')
            if probe:
                self._probed = True

                def ask():
                    try:
                        self.mic_checked.emit(windows.permission() == 'denied')
                    except Exception as e:  # the helper is missing: said above already
                        log.warning('microphone permission: %s', e)
                threading.Thread(target=ask, name='mic-permission', daemon=True).start()
        except windows.CableMissing:
            text, ok = ('<b>✘ VB-Audio Virtual Cable is not installed.</b> It is the virtual microphone that carries your '
                        'translated voice to the call app (them → me works without it).<br>Install it: download the '
                        f'<b>VB-CABLE Driver Pack</b> from <a href="{windows.URL}">vb-audio.com/Cable</a>, unpack the zip, '
                        'right-click <b>VBCABLE_Setup_x64.exe</b> → <b>Run as administrator</b> → <b>Install Driver</b>, '
                        'restart Windows. Then press <b>Check again</b>.'), False
        except Exception as e:  # the helper is missing (a broken install)
            text, ok = f'✘ Cannot list the sound devices: {e}', False
        self._vmic_text = text if ok else ''
        self.vmic.setText(text + (f'<br>✘ {windows.NO_PERMISSION}.' if ok and self._denied else ''))
        self.recheck.setVisible(not ok or self._denied)
        return ok

    def _mic_checked(self, denied: bool) -> None:
        from peresmeshnik.audio import windows
        self._denied = denied
        if self._vmic_text:
            self.vmic.setText(self._vmic_text + (f'<br>✘ {windows.NO_PERMISSION}.' if denied else ''))
            self.recheck.setVisible(denied)

    def check_blackhole(self) -> bool:
        """macOS: BlackHole 2ch is the user's to install; missing -> how to, and "Check again". The microphone permission
        too: denied, macOS hands the app silence."""
        from peresmeshnik.audio import macos
        try:
            macos.blackhole()
            text, ok = '✔ BlackHole 2ch is installed: in the call app choose the microphone <b>BlackHole 2ch</b>.', True
        except macos.BlackHoleMissing:
            text, ok = ('<b>✘ BlackHole 2ch is not installed.</b> It is the virtual microphone that carries your translated '
                        'voice to the call app (them → me works without it).<br>Install it: in the Terminal run '
                        '<code>brew install blackhole-2ch</code>, or download the installer from '
                        '<a href="https://existential.audio/blackhole/">existential.audio/blackhole</a>. Then press '
                        '<b>Check again</b> (still not found: restart the Mac).'), False
        except Exception as e:  # the helper is missing (a broken install)
            text, ok = f'✘ Cannot list the sound devices: {e}', False
        if ok and macos.permission() in ('denied', 'restricted'):
            text += f'<br>✘ {macos.NO_PERMISSION}.'
        self.vmic.setText(text)
        self.recheck.setVisible(not ok)
        return ok

    def _echo(self) -> None:
        name = self.headphones.currentText().lower()
        speakers = any(k in name for k in ('speaker', 'built-in', 'встроен', 'динамик'))
        self.warn.setText('⚠ This looks like loudspeakers: the microphone would hear the translation and translate it back. '
                          'Use headphones.' if speakers else 'Use headphones, not loudspeakers (echo).')

    def validatePage(self) -> bool:  # noqa: N802
        self.w.s.ui.headphones, self.w.s.ui.mic = self.headphones.currentData(), self.mic.currentData()
        return True


class CallSoundPage(QWizardPage):
    """macOS: how the app hears the call (audio/capture_macos): a process tap of the call app (14.4+; macOS asks for
    "System Audio Recording" once, `request()` waits for that answer, so not on the GUI thread), or before 14.4 the call
    app's speaker set to BlackHole 16ch. Windows (audio/capture_windows, S1): a process loopback of the call app
    (Windows 10 2004+, no permission to ask), checked here. Either way the page can be left: them -> me says why when
    it cannot start."""
    answered = Signal(int, str)

    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        self.setTitle("The call's sound")
        lay = QVBoxLayout(self)
        self.text, self.state = QLabel(), QLabel()
        for x in (self.text, self.state):
            x.setWordWrap(True)
        self.state.setTextInteractionFlags(Qt.TextBrowserInteraction)  # the brew command can be copied
        self.state.setOpenExternalLinks(True)
        self.button = QPushButton()
        self.button.clicked.connect(self.act)
        for x in (self.text, self.button, self.state):
            lay.addWidget(x)
        self.answered.connect(self._answered)
        self.running, self.path = False, ''

    def initializePage(self) -> None:  # noqa: N802
        if WINDOWS:
            self.path = 'loopback'
            self.text.setText("While translating, Peresmeshnik hears your call app through Windows (only that app's sound) "
                              'and turns its volume down on your speakers; translation off gives the volume back.')
            self.button.setText('Check again')
            self.check_loopback()
            return
        from peresmeshnik.audio import capture_macos as cm
        self.path = cm.path()
        if self.path == 'tap':
            self.text.setText('Peresmeshnik hears your call app through macOS (System Audio Recording) and mutes it on your '
                              'speakers while translating.')
            self.button.setText('Allow')
            self._show(*self._safe(cm.check))
        else:
            self.text.setText(f'In your call app choose the speaker <b>{cm.CALL_DEVICE}</b>. Peresmeshnik plays the call to your '
                              'headphones itself; while translating you hear only the translation.')
            self.button.setText('Check again')
            self.check_blackhole()

    @staticmethod
    def _safe(fn) -> tuple[int, str]:
        try:
            return fn()
        except Exception as e:  # no tap helper (a broken install), the helper crashed
            return -1, str(e)

    def act(self) -> None:
        if WINDOWS:
            self.check_loopback()
            return
        if self.path != 'tap':
            self.check_blackhole()
            return
        from peresmeshnik.audio import capture_macos as cm
        self.running = True
        self.button.setEnabled(False)
        self.state.setText('Answer the macOS dialog: allow Peresmeshnik to record the call app\'s sound…')
        self.completeChanged.emit()
        threading.Thread(target=lambda: self.answered.emit(*self._safe(cm.request)), name='tap-permission', daemon=True).start()

    def _answered(self, code: int, why: str) -> None:
        self.running = False
        self.button.setEnabled(True)
        self._show(code, why)
        self.completeChanged.emit()

    def _show(self, code: int, why: str) -> None:
        self.state.setText('✔ Allowed: while translating, the call app is heard through macOS.' if code == 0 else f'✘ {why}')
        self.button.setVisible(code != 0)

    def check_loopback(self) -> bool:
        """Windows: the loopback helper and Windows itself can take the call app's sound; which app plays now, if any."""
        from peresmeshnik.audio import capture_windows as cw
        code, why = self._safe(cw.check)
        if code:
            self.state.setText(f'✘ {why}')
        else:
            try:
                app = cw.guess_call_app()
            except Exception:  # the session list failed: the check passed, the app is found at the call
                app = ''
            self.state.setText('✔ Windows gives Peresmeshnik the call app\'s sound. ' + (
                f'Playing now: <b>{app}</b>.' if app else 'Telegram, Zoom, Discord and the like are found by themselves '
                'once the call plays; else choose the app in the tray menu <b>Call app</b>.'))
        self.button.setVisible(code != 0)
        return code == 0

    def check_blackhole(self) -> bool:
        from peresmeshnik.audio import capture_macos as cm
        from peresmeshnik.audio import macos
        try:
            ok = cm.CALL_DEVICE in {d['name'] for d in macos.devices()}
        except Exception as e:
            self.state.setText(f'✘ Cannot list the sound devices: {e}')
            return False
        self.state.setText(f'✔ {cm.CALL_DEVICE} is installed.' if ok else
                           f'<b>✘ {cm.CALL_DEVICE} is not installed.</b> In the Terminal run <code>brew install --cask '
                           'blackhole-16ch</code>, or download the installer from <a href="https://existential.audio/blackhole/">'
                           'existential.audio/blackhole</a>; then press <b>Check again</b> (still not found: restart the Mac).')
        self.button.setVisible(not ok)
        return ok

    def isComplete(self) -> bool:  # noqa: N802
        return not self.running


class ModelsPage(QWizardPage):
    progress = Signal(int, int, str)  # done bytes, total bytes, current model
    finished = Signal(str)  # '' or the error

    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        self.setTitle('Models')
        self.setSubTitle('Downloaded once from pinned addresses and checked by SHA-256; then everything local works offline.')
        lay = QVBoxLayout(self)
        self.list = QLabel()
        self.list.setWordWrap(True)
        self.list.setOpenExternalLinks(True)
        self.consent = QCheckBox()
        self.consent.toggled.connect(self.completeChanged)
        self.button = QPushButton('Download')
        self.button.clicked.connect(self.download)
        self.bar = QProgressBar()
        self.state = QLabel()
        self.state.setWordWrap(True)
        for x in (self.list, self.consent, self.button, self.bar, self.state):
            lay.addWidget(x)
        self.progress.connect(self._progress)
        self.finished.connect(self._finished)
        self.busy = False

    def initializePage(self) -> None:  # noqa: N802
        self.ms = [models.get(i) for i in needed_models(self.w.s)]
        base = models.root()
        lic = {m.id: ' + '.join(f'<a href="{p.license_url}">{p.license}</a>' for p in models.parts(m)) for m in self.ms}
        rows = [f'<tr><td>{"✔" if models.is_fetched(m, base) else "·"}</td><td>{m.id}</td>'
                f'<td align=right>{models.size(m) / 1e6:.0f} MB</td><td>{lic[m.id]}</td></tr>' for m in self.ms]
        self.list.setText(f'<table cellspacing=6>{"".join(rows)}</table><p>Folder: {base}</p>')
        restricted = [m for m in self.ms if m.consent and not models.is_fetched(m, base)]
        self.consent.setVisible(bool(restricted))
        self.consent.setText('I accept the restricted licenses (noncommercial use only, they bind me as the user): '
                             + ', '.join(f'{m.id} ({m.license})' for m in restricted))
        self._finished('')

    def missing(self) -> list:
        base = models.root()
        return [m for m in self.ms if not models.is_fetched(m, base)]

    def isComplete(self) -> bool:  # noqa: N802
        return not self.busy and not self.missing()

    def download(self) -> None:
        todo = self.missing()
        if any(m.consent for m in todo) and not self.consent.isChecked():
            self.state.setText('Accept the restricted licenses first, or choose cloud for that direction.')
            return
        self.busy = True
        self.button.setEnabled(False)
        self.completeChanged.emit()
        total = sum(map(models.size, todo))

        def run():
            done = [0]
            try:
                for m in todo:
                    def tick(n, m=m):
                        done[0] += n
                        self.progress.emit(done[0], total, m.id)
                    models.fetch(m.id, progress=tick)
                self.finished.emit('')
            except Exception as e:
                self.finished.emit(str(e))
        threading.Thread(target=run, name='models-fetch', daemon=True).start()

    def _progress(self, done: int, total: int, mid: str) -> None:
        self.bar.setMaximum(max(total // 1024, 1))
        self.bar.setValue(done // 1024)
        self.state.setText(f'{mid}: {done / 1e6:.0f} of {total / 1e6:.0f} MB')

    def _finished(self, err: str) -> None:
        self.busy = False
        todo = self.missing()
        self.button.setEnabled(bool(todo))
        self.bar.setVisible(bool(todo) or bool(err))
        self.state.setText(f'Download failed: {err}. Press Download again to resume.' if err else
                           f'{len(todo)} to download, {sum(map(models.size, todo)) / 1e6:.0f} MB.' if todo else 'All models are here.')
        self.completeChanged.emit()


class SelfTestPage(QWizardPage):
    result = Signal(bool, str)
    done = Signal()

    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        self.setTitle('Self-test')
        self.setSubTitle('A short chirp goes through each of the app\'s sound paths and is recorded back: no call needed. '
                         'Translation is turned off for the test. Say something during the microphone check.')
        lay = QVBoxLayout(self)
        self.button = QPushButton('Run the self-test')
        self.button.clicked.connect(self.run)
        self.out = QLabel()
        self.out.setWordWrap(True)
        lay.addWidget(self.button)
        lay.addWidget(self.out)
        self.result.connect(lambda ok, text: self.out.setText(self.out.text() + ('✔ ' if ok else '✘ ') + text + '<br>'))
        self.done.connect(self._done)
        self.ran = self.running = False
        self.ok = False

    def isComplete(self) -> bool:  # noqa: N802
        return self.ran

    def run(self) -> None:
        self.button.setEnabled(False)
        self.out.setText('')
        ctl, s = self.w.tray.ctl, self.w.s

        def work():
            self.ok = False
            try:
                from peresmeshnik.ui import selftest
                ctl.set_translation(False).result()  # the call sink must carry only the chirp
                vmic = ctl.submit(ctl.virtual_mic).result()  # the app's own: a second one would replace it
                if MACOS or WINDOWS:  # the test records the virtual mic like a call app: the loop would join the chirp
                    ctl.submit(ctl._loopback, False).result()
                try:
                    rs = selftest.run_all(s.ui.headphones or None, s.ui.mic or None, vmic,
                                          report=lambda r: self.result.emit(r.ok, f'{r.name}: {r.detail}'))
                finally:
                    if (MACOS or WINDOWS) and not ctl.on:
                        ctl.submit(ctl._loopback, True).result()
                self.ok = all(r.ok for r in rs)
            except Exception as e:  # no pactl, no sound server: say so instead of waiting forever
                self.result.emit(False, f'self-test: {e}')
            finally:
                self.done.emit()
        self.running = True
        threading.Thread(target=work, name='selftest', daemon=True).start()

    def _done(self) -> None:
        self.ran, self.running = True, False
        self.button.setEnabled(True)
        self.out.setText(self.out.text() + ('<b>All paths work.</b>' if self.ok else '<b>Some paths failed</b>: check the devices, '
                                            'run again, or continue and fix it later.'))
        self.completeChanged.emit()


class VoicePage(QWizardPage):
    """Voice enrollment (ARCHITECTURE §7, step 5): record or import 30-60 s of the user's own voice, the quality report
    with its verdict; with the local clone also the clone check (voice.check_clone). Optional for a cloud me -> them.
    A new take is kept as the profile "<name>.new" (the test phrase already speaks with it) and replaces "<name>" only
    when the wizard finishes; cancelled, it is dropped."""
    progress = Signal(float)
    result = Signal(object)  # (Report, Voice | None), or an error text
    checked = Signal(str)
    pack_progress = Signal(object)  # voicepack.install's (stage, done, total[, text]): bytes overflow a Qt int
    pack_done = Signal(str)  # '' or why it stopped
    gpu_found = Signal(object)  # Gpu: nvidia-smi runs in a worker (up to its 10 s), not in the GUI thread
    seconds = 45.0
    name = 'me'

    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        self.setTitle('Your voice')
        lay = QVBoxLayout(self)
        self.prompt = QLabel()
        self.prompt.setWordWrap(True)
        row = QHBoxLayout()
        self.record_button = QPushButton(f'Record {self.seconds:.0f} s')
        self.record_button.clicked.connect(self.record)
        self.import_button = QPushButton('Import a recording…')
        self.import_button.clicked.connect(lambda: self.import_file())
        self.stock = QCheckBox('Speak with a stock voice (Piper), not mine: it keeps up on this computer')
        self.stock.toggled.connect(self._stock)
        self.check_button = QPushButton('Check the clone')
        self.check_button.clicked.connect(self.check_clone)
        for b in (self.record_button, self.import_button, self.check_button):
            row.addWidget(b)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.out, self.clone_out, self.pack_out = QLabel(), QLabel(), QLabel()
        for x in (self.out, self.clone_out, self.pack_out):
            x.setWordWrap(True)
        self.pack_button, self.pack_cancel = QPushButton('Install the voice pack'), QPushButton('Cancel')
        self.pack_button.clicked.connect(self.install_pack)
        self.pack_consent = QCheckBox(PACK_CONSENT)  # before anything is downloaded
        self.pack_consent.toggled.connect(self._buttons)
        self.pack_cancel.clicked.connect(lambda: self._cancel_pack.set())
        self.pack_bar = QProgressBar()
        for x in (self.pack_out, self.pack_consent, self.pack_button, self.pack_cancel, self.pack_bar):
            x.setVisible(False)  # _show_pack: only on Windows with the clone
        prow = QHBoxLayout()
        prow.addWidget(self.pack_button)
        prow.addWidget(self.pack_cancel)
        prow.addStretch()
        lay.addWidget(self.stock)
        lay.addWidget(self.pack_out)
        lay.addWidget(self.pack_consent)
        lay.addLayout(prow)
        lay.addWidget(self.pack_bar)
        lay.addWidget(self.prompt)
        lay.addLayout(row)
        for x in (self.bar, self.out, self.clone_out):
            lay.addWidget(x)
        self.progress.connect(lambda t: self.bar.setValue(int(t)))
        self.result.connect(self._result)
        self.checked.connect(self._checked)
        self.pack_progress.connect(self._pack_progress)
        self.pack_done.connect(self._pack_done)
        self.voice = None
        self.running = False
        self.packing, self.pack_note, self.gpu = False, '', None  # None: not known yet (the worker asks)
        self.gpu_found.connect(self._gpu_found)
        self._cancel_pack = threading.Event()
        self.draft = f'{self.name}.new'

    def local(self) -> bool:
        return self.w.s.me_to_them == 'local'

    def clone_off(self) -> str:
        """Why me -> them local speaks in a stock voice though the clone is chosen ('' = the clone speaks): it cannot
        speak their language, or needs an NVIDIA GPU here (languages.direction)."""
        from peresmeshnik.voice import STOCK
        s = self.w.s
        d = languages.direction(s, 'me_to_them')
        return '; '.join(d.notes) if self.local() and s.local.clone != STOCK and d.voice == 'stock' else ''

    def initializePage(self) -> None:  # noqa: N802
        from peresmeshnik import voice
        s = self.w.s
        self.setSubTitle('"Me → them" on this computer speaks the translation in your voice. Read the text below aloud for '
                         f'about {self.seconds:.0f} s, at your normal pace, in a quiet room — or import 30-60 s of your own '
                         'voice. The recording stays on this computer.' if self.local() else
                         'Optional: "me → them" runs in the cloud, in Gemini\'s voice. Your recording is only needed for the '
                         'local clone.')
        if self.local() and MACOS:  # no Apple GPU for torch in CI: the clone ran on the CPU there (BENCH)
            self.setSubTitle(self.subTitle() + ' On this Mac the clone of your voice runs on the CPU and falls behind the '
                             'conversation: choose cloud for "me → them", or a stock voice.')
        self.prompt.setText(voice.PROMPTS.get(s.my_lang, 'Read any text in your language aloud, at your normal pace.'))
        self.stock.setVisible(self.local())
        self.stock.setChecked(s.local.clone == voice.STOCK)
        self.check_button.setVisible(self.local() and not self.stock.isChecked())
        if not WINDOWS:
            self.gpu = Gpu()
        elif self.gpu is None and not getattr(self, '_asking', False):  # once: the GPU does not change meanwhile
            self._asking = True
            threading.Thread(target=lambda: self.gpu_found.emit(gpu()), name='gpu', daemon=True).start()
        if why := self.clone_off():  # nothing to record for it: said, and the page can be left
            self.setSubTitle(f'"Me → them" speaks in a stock voice for this pair: {why}. No recording is needed.')
            for x in (self.stock, self.prompt, self.record_button, self.import_button, self.check_button):
                x.setVisible(False)
        if self.voice is None and s.local.voice and (voice.voices_dir() / s.local.voice / 'voice.json').exists():
            self.voice = voice.load(s.local.voice)
            q = self.voice.meta.get('quality', {})
            self.out.setText(f'Saved voice "{self.voice.name}": {q.get("seconds", 0):.0f} s, '
                             f'recorded {self.voice.meta.get("created", "")}.')
        self._show_pack()
        self._buttons()

    def isComplete(self) -> bool:  # noqa: N802
        return not self.running and not self.packing and (not self.local() or self.stock.isChecked() or self.voice is not None
                                                          or bool(self.clone_off()))

    def pack(self) -> str:
        """The clone's voice pack here: '' where it is not this page's business (not Windows, cloud, the stock voice, a
        runner of one's own in $PERESMESHNIK_VOICE_CMD, a build without the pack's files), 'installing', 'gpu' (gpu_refusal:
        too little memory, an old driver or GPU, a driver that does not answer), else voicepack.state()."""
        if not (WINDOWS and self.local() and not self.stock.isChecked()) or self.clone_off() or os.environ.get('PERESMESHNIK_VOICE_CMD') \
                or not voicepack.lock_file().is_file():
            return ''
        if self.packing:
            return 'installing'
        if self.gpu is None:
            return 'checking'
        return 'gpu' if gpu_refusal(self.gpu) else voicepack.state()

    def _gpu_found(self, g) -> None:
        self.gpu = g
        self._show_pack()
        self._buttons()

    def _show_pack(self) -> None:
        st = self.pack()
        for x in (self.pack_out, self.pack_consent, self.pack_button, self.pack_cancel, self.pack_bar):
            x.setVisible(False)
        if not st:
            return
        self.pack_out.setVisible(True)
        self.pack_cancel.setVisible(st == 'installing')
        self.pack_bar.setVisible(st == 'installing')
        if st == 'installing':
            return
        if st == 'checking':
            self.pack_out.setText('Checking the NVIDIA GPU…')
            return
        if st == 'gpu':
            text = gpu_refusal(self.gpu) + ' Choose a stock voice above, or cloud for "me → them" on the first page.'
        elif st == 'ready':
            text = 'The voice pack is installed.'
        else:
            spec = json.loads(voicepack.lock_file().read_text())
            dl = sum(f['size'] for f in [spec['python'], spec['pip'], *spec['wheels']])
            text = {'none': 'The clone runs in the voice pack: its own Python with torch for NVIDIA GPUs.',
                    'partial': 'The voice pack install did not finish: it goes on from where it stopped.',
                    'outdated': 'The voice pack was installed for another version of Peresmeshnik: install it again.'}[st]
            text += (f' {dl / 1e9:.1f} GB to download, {spec["installed_bytes"] / 1e9:.0f} GB on disk '
                     f'({voicepack.need_bytes() / 1e9:.0f} GB free space needed while installing).')
            self.pack_button.setText({'partial': 'Resume the install'}.get(st, 'Install the voice pack'))
            self.pack_button.setVisible(True)
            self.pack_consent.setVisible(True)
        if st != 'gpu' and (note := gpu_note(self.gpu)):
            text = f'{note} {text}'
        self.pack_out.setText(f'{self.pack_note} {text}'.strip())

    def install_pack(self) -> None:
        if not self.pack_consent.isChecked():  # the button is off until then
            return
        self.packing, self.pack_note = True, ''
        self._cancel_pack.clear()
        self.pack_bar.setRange(0, 0)  # busy until bytes come
        self.pack_out.setText('Starting the voice pack install…')
        self._show_pack()
        self._buttons()

        def work():
            try:
                voicepack.install(lambda *a: self.pack_progress.emit(a), self._cancel_pack)
                self.pack_done.emit('')
            except voicepack.Cancelled:
                self.pack_done.emit('Cancelled: what is downloaded stays for the next try.')
            except Exception as e:  # no space, the network, pip
                self.pack_done.emit(f'error: {e}.')
        threading.Thread(target=work, name='voice-pack', daemon=True).start()

    def _pack_progress(self, a: tuple) -> None:
        stage, done, total, text = (*a, '')[:4]
        if stage == 'downloading' and total:
            self.pack_bar.setRange(0, 1000)
            self.pack_bar.setValue(int(1000 * done / total))
            self.pack_out.setText(f'Downloading the voice pack: {done / 1e9:.2f} of {total / 1e9:.2f} GB ({text})')
        else:
            self.pack_bar.setRange(0, 0)
            self.pack_out.setText({'cleaning': 'Removing the old voice pack…', 'unpacking': 'Unpacking its Python…',
                                   'installing': f'Installing, a few minutes: {text}'}.get(stage, stage))

    def _pack_done(self, note: str) -> None:
        self.packing, self.pack_note = False, note
        self._show_pack()
        self._buttons()

    def _stock(self, on: bool) -> None:
        """The stock voice instead of the clone; its model (~60 MB) is fetched here when the models page took the clone."""
        from peresmeshnik import voice
        s = self.w.s
        s.local.clone = voice.STOCK if on else config.Local().clone
        self.check_button.setVisible(self.local() and not on)
        self._show_pack()
        vid = languages.direction(s, 'me_to_them').tts  # the stock voice now ('' = none for the language)
        if on and vid and not models.is_fetched(models.get(vid), models.root()):
            self.out.setText(f'Downloading the stock voice {vid}…')
            self._run(lambda: (models.fetch(vid), self.result.emit(f'The stock voice {vid} is here.')))
        self.completeChanged.emit()

    def translating(self) -> bool:
        """Translation on: a local me -> them already holds the clone; the check would load a second one (~3 GB)."""
        return bool(getattr(getattr(getattr(self.w, 'tray', None), 'ctl', None), 'on', False))

    def _buttons(self) -> None:
        for b in (self.record_button, self.import_button):
            b.setEnabled(not self.running)
        why = ('Turn translation off first: the check loads a second clone (~3 GB)' if self.translating() else
               self.no_check())
        self.check_button.setEnabled(not self.running and self.voice is not None and not why and self.pack() in ('', 'ready'))
        self.pack_button.setEnabled(not self.running and self.pack_consent.isChecked())
        self.stock.setEnabled(not self.packing)
        self.check_button.setToolTip(why)
        self.completeChanged.emit()

    def _run(self, work) -> None:
        self.running = True
        self._buttons()

        def go():
            try:
                work()
            except Exception as e:  # no microphone, an unreadable file, no voice pack: say so
                self.result.emit(f'error: {e}')
        threading.Thread(target=go, name='voice', daemon=True).start()

    def record(self) -> None:
        from peresmeshnik import voice
        from peresmeshnik.audio.mic import record
        s, sr = self.w.s, 24_000  # above the 16 kHz pipeline rate: the clone conditions at 22.05 kHz
        self.out.setText('Recording… read the text aloud.')
        self.clone_out.setText('')
        self.bar.setMaximum(int(self.seconds))
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self._run(lambda: self.result.emit(voice.enroll(record(self.seconds, s.ui.mic or None, sr, self.progress.emit), sr,
                                                        self.draft, s.my_lang, voice.PROMPTS.get(s.my_lang))))

    def import_file(self, path: str | None = None) -> None:
        from peresmeshnik import voice
        path = path or QFileDialog.getOpenFileName(self, 'Your voice: a recording of 30-60 s', '',
                                                   'Audio (*.wav *.flac *.ogg *.mp3)')[0]
        if path:
            self.clone_out.setText('')
            self._run(lambda: self.result.emit(voice.enroll(*voice.read_audio(Path(path)), self.draft, self.w.s.my_lang)))

    def _result(self, res) -> None:
        from peresmeshnik import voice
        self.running = False
        self.bar.setVisible(False)
        if isinstance(res, str):
            self.out.setText(res)
        else:
            r, v = res
            saved = ' Kept as your new voice: it is saved when you finish the wizard.' if v else ' Not saved.'
            self.out.setText(f'<b>{voice.verdict(r)}</b>{saved}<br>{voice.summary(r)}')
            if v:
                self.voice, self.w.s.local.voice = v, v.name
        self._buttons()

    def no_check(self) -> str:
        """Why the clone cannot be checked here ('' = it can): the check hears it with an ASR model of their language
        (offline: also one that does not keep up live)."""
        s = self.w.s
        if languages.pick(s.local.asr, s.their_lang):
            return ''
        return (f'No speech recognition model for {languages.name(s.their_lang)} to check the clone with: listen to it in '
                'a test call')

    def check_clone(self) -> None:
        """The clone says the test phrases in their language; the app's ASR must understand them (voice.check_clone)."""
        from peresmeshnik import models, voice
        s, v = self.w.s, self.voice
        if self.translating():
            self.clone_out.setText('Turn translation off first: the check loads a second clone (~3 GB).')
            return
        if why := self.no_check():
            self.clone_out.setText(why + '.')
            return
        self.clone_out.setText('Checking the clone: it says two test phrases… (first time: loading the voice pack)')

        def work():
            from peresmeshnik.local.asr import OnnxAsr
            from peresmeshnik.local.clone import make_clone
            tts = None
            try:
                tts = make_clone(models.path(s.local.clone), threads=s.local.threads)
                # their language's ASR, also where it does not keep up live (Whisper turbo without CUDA): two phrases
                asr = OnnxAsr(languages.pick(s.local.asr, s.their_lang), s.local.threads, live=False)
                self.checked.emit(voice.clone_verdict(voice.check_clone(tts, v, s.their_lang, asr)))
            except Exception as e:  # no voice pack or GPU, a model not downloaded yet, out of memory
                self.checked.emit(f'error: {e}')
            finally:
                getattr(tts, 'close', lambda: None)()  # the voice pack's process and its VRAM
        self.running = True
        self._buttons()
        threading.Thread(target=work, name='clone-check', daemon=True).start()

    def _checked(self, text: str) -> None:
        self.running = False
        self.clone_out.setText(text)
        self._buttons()

    def commit(self) -> None:
        """The wizard finished: a new take becomes the profile."""
        from peresmeshnik import voice
        if self.voice is not None and self.voice.name == self.draft:
            self.voice = voice.promote(self.draft, self.name)
            self.w.s.local.voice = self.name

    def rollback(self) -> None:
        """The wizard was cancelled: the saved profile stays as it was; a voice pack install stops (it resumes later)."""
        from peresmeshnik import voice
        self._cancel_pack.set()
        voice.discard(self.draft)


class HotkeyPage(QWizardPage):
    state = Signal(str)

    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        self.setTitle('Hotkey and a test phrase')
        self.setSubTitle('The hotkey switches between translation and your live voice during a call.')
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.edit = QKeySequenceEdit(QKeySequence(w.s.ui.hotkey))
        f.addRow('Hotkey', self.edit)
        lay.addLayout(f)
        row = QHBoxLayout()
        self.tests = {}
        for side, title in (('them', 'Test: them → me'), ('me', 'Test: me → them')):
            b = self.tests[side] = QPushButton(title)
            b.clicked.connect(lambda _=False, side=side: self.test(side))
            row.addWidget(b)
        lay.addLayout(row)
        self.out = QLabel('Each test plays a recorded phrase through that direction: you hear the translation in your '
                          'headphones and see the subtitles at the bottom of the screen.')
        self.out.setWordWrap(True)
        lay.addWidget(self.out)
        self.err = QLabel()
        self.err.setWordWrap(True)
        lay.addWidget(self.err)
        self.state.connect(self._state)
        self.running, self.error = False, ''

    def initializePage(self) -> None:  # noqa: N802
        if why := self.w.tray.hotkey_why:  # e.g. taken by another app, no portal on this desktop
            self.err.setText(f'No global hotkey now: {why}')

    def test(self, side: str) -> None:
        from peresmeshnik.app import Controller
        s = self.w.s
        lang = s.their_lang if side == 'them' else s.my_lang
        sample = str(SAMPLES / f'{lang}.ogg')
        for b in self.tests.values():
            b.setEnabled(False)
        self.out.setText('Loading the engines…')
        self.running, self.error = True, ''
        ctl = Controller(self.w.tray.ctl.overlay, self.state.emit, sample if side == 'them' else None,
                         sample if side == 'me' else None, settings=s, test=True)

        def work():
            try:
                ctl.set_translation(True).result()
                for side_ in ctl.sides:
                    side_.inp.done.wait(60)
                cloud = any(x.engine == 'cloud' for x in ctl.sides)
                time.sleep(6.0 if cloud else 3.0)  # the cloud voice ends ~3-4 s after the speech (BENCH)
                ctl.quit()
            finally:
                self.state.emit('test done')
        threading.Thread(target=work, name='test-phrase', daemon=True).start()

    def _state(self, st: str) -> None:
        if st.startswith('error'):
            self.error = st  # kept: a wrong key flashes for a second otherwise and the test looks fine
        if st == 'test done':
            self.running = False
            for b in self.tests.values():
                b.setEnabled(True)
            st = f'test done with an {self.error}' if self.error else st
        self.out.setText(st)

    def validatePage(self) -> bool:  # noqa: N802
        combo = self.edit.keySequence().toString(QKeySequence.PortableText)
        if combo and combo != self.w.s.ui.hotkey:
            if why := self.w.tray.reload_hotkey(combo):
                self.err.setText(f'Hotkey not changed: {why}')
                return False
            self.w.s.ui.hotkey = combo
        return True


class DonePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle('Ready')
        lab = QLabel({'darwin': DONE_MACOS, 'win32': DONE_WINDOWS}.get(sys.platform, DONE_LINUX))
        lab.setWordWrap(True)
        QVBoxLayout(self).addWidget(lab)


DONE_MACOS = ('In your call app (Meet, Zoom, Telegram) choose the microphone <b>BlackHole 2ch</b>. While translation '
              'is off it carries your own voice, so the call always hears you.<br><br>'
              "Their voice: while translation is on, the app records the call app's sound and mutes its original for you "
              '(you hear the translation instead); macOS asks once to allow "System Audio Recording". Telegram, Zoom, '
              'Discord and the like are found by themselves; else choose the app in the menu <b>Call app</b> while the '
              'call plays. A browser (Meet) is taken with <b>all</b> of its sound: YouTube in another tab is translated '
              'too, so for calls a separate browser is better.<br><br>'
              'Turn translation on from the Peresmeshnik icon in the menu bar. The hotkey switches to your live voice '
              'and back.<br><br>Settings and this wizard are in the same menu.')
DONE_WINDOWS = ('In your call app (Meet, Zoom, Telegram) choose the microphone <b>CABLE Output</b> (VB-Audio Virtual Cable). '
                'While translation is off it carries your own voice, so the call always hears you.<br><br>'
                "Their voice: while translation is on, the app records the call app's sound and mutes its original on your "
                'speakers (you hear the translation instead), and gives it back when translation is off. Telegram, Zoom, '
                'Discord and the like are found by themselves; else choose the app in the tray menu <b>Call app</b> while '
                'the call plays. A browser (Meet) is taken with <b>all</b> of its sound: YouTube in another tab is '
                'translated too, so for calls a separate browser is better.<br><br>'
                'Turn translation on from the Peresmeshnik icon in the notification area (at first Windows may keep it '
                'behind the ^ arrow). The hotkey switches to your live voice and back.<br><br>'
                'Settings and this wizard are in the same menu.')
DONE_LINUX = ('In your call app (Meet, Zoom, Telegram) choose the microphone <b>Peresmeshnik Mic</b>. While '
              'translation is off it carries your own voice, so the call always hears you.<br><br>'
              'Their voice: while translation is on, the app moves the call app\'s sound to <b>Peresmeshnik Call</b> '
              '(you hear the translation instead) and gives it back when translation is off. Telegram, Zoom, Discord '
              'and the like are found by themselves; else choose the app in the tray menu <b>Call app</b> while the '
              'call plays. A browser (Meet) moves with <b>all</b> of its sound: YouTube in another tab is translated '
              'too, so for calls a separate browser is better.<br><br>'
              'Turn translation on from the tray icon. The hotkey switches to your live voice and back.<br><br>'
              'Settings and this wizard are in the tray menu.')


class SetupWizard(QWizard):
    def __init__(self, tray):
        super().__init__()
        self.tray, self.s, self.key = tray, config.load(), ''
        self.setWindowTitle('Peresmeshnik setup')
        routed = LINUX or MACOS or WINDOWS
        self.pages = [LanguagesPage(self)] + ([DevicesPage(self)] if routed else []) + ([CallSoundPage(self)] if MACOS or WINDOWS else []) \
            + [ModelsPage(self)] \
            + ([SelfTestPage(self)] if routed else []) + [VoicePage(self), HotkeyPage(self), DonePage()]
        self.voice_page = next(p for p in self.pages if isinstance(p, VoicePage))
        for p in self.pages:
            self.addPage(p)
        self.accepted.connect(self.finish)
        self.rejected.connect(self.cancel)
        self.finished.connect(lambda _: tray.ctl.on or tray.ctl.overlay.clear.emit())  # the test phrase's subtitles
        self.currentIdChanged.connect(lambda _: QTimer.singleShot(0, self.fit))

    def showEvent(self, event) -> None:  # noqa: N802 (Qt name)
        super().showEvent(event)
        QTimer.singleShot(0, self.fit)

    def fit(self) -> None:
        """Taller when the current page's wrapped text needs it: QWizard sizes pages as if nothing wrapped (on Windows
        the last lines of a page were cut off)."""
        p = self.currentPage()
        if p and p.layout() and (need := p.layout().totalHeightForWidth(p.width())) > p.height():
            room = self.screen().availableGeometry().height() - (self.frameGeometry().height() - self.height())
            self.resize(self.width(), min(room, self.height() + need - p.height()))  # the buttons stay on screen

    def busy(self) -> bool:
        """Still downloading, testing or playing a phrase: the tray keeps this wizard alive after it is closed."""
        return any(getattr(p, 'busy', False) is True or getattr(p, 'running', False) for p in self.pages)

    def cancel(self) -> None:
        if (saved := config.load().ui.hotkey) != self.s.ui.hotkey:  # the hotkey page already bound the new one
            self.tray.reload_hotkey(saved)
        self.voice_page.rollback()

    def finish(self) -> None:
        self.voice_page.commit()  # before saving: the settings name the profile, not the draft
        config.save(self.s)
        self.tray.sync()
        self.tray.restart()

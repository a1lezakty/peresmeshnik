"""Setup wizard (first run, and from the tray): languages and engines, devices, models with their
licenses, the loopback self-test, the hotkey and a test phrase in both directions.

Slow work (downloads, the self-test, the test phrase) runs on worker threads and reports through
Qt signals; nothing is polled. Settings are written when the wizard finishes.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

import psutil
from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from peresmeshnik import config, models
from peresmeshnik.ui.tray import LANGS, MIC_LABEL, SIDES, device_combo, save_key

LINUX = sys.platform.startswith('linux')
SAMPLES = Path(__file__).parent / 'samples'


def has_key(s: config.Settings) -> bool:
    from peresmeshnik.cloud import api_key
    try:
        return bool(api_key(s.cloud))
    except SystemExit:
        return False


def needed_models(s: config.Settings) -> list[str]:
    """Registry ids the chosen setup runs on (ARCHITECTURE §6). Silero VAD is always there: it gates the cloud too."""
    ids = ['silero-vad']
    for side, (src, tgt) in (('them_to_me', (s.their_lang, s.my_lang)), ('me_to_them', (s.my_lang, s.their_lang))):
        local = getattr(s, side) == 'local'
        if local or not s.cloud.subtitles:  # local voice, or local subtitles next to the cloud voice
            ids += [s.local.asr[src], s.local.mt[f'{src}-{tgt}']]
        if local:
            ids.append(s.local.tts[tgt] if side == 'them_to_me' else s.local.clone)
    return list(dict.fromkeys(ids))


def tier() -> tuple[str, str]:
    """(what this machine is, the recommended me -> them engine)."""
    gb = psutil.virtual_memory().total / 2 ** 30
    gpu = bool(shutil.which('nvidia-smi'))
    rec = 'local' if gpu else 'cloud'  # no local clone is live without an NVIDIA GPU (BENCH)
    return f'{gb:.0f} GB RAM, {"an NVIDIA GPU" if gpu else "no NVIDIA GPU"}: recommended them → me local, me → them {rec}', rec


class LanguagesPage(QWizardPage):
    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        s = w.s
        self.setTitle('Languages and engines')
        what, rec = tier()
        self.setSubTitle(f'This computer: {what}.')
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
            c.setCurrentText(getattr(s, key) if config.settings_path().exists() else ('local' if key == 'them_to_me' else rec))
            f.addRow(title, c)
        self.subtitles = QCheckBox('Subtitles from the cloud (else from the local ASR: faster, needs ~1 GB of models)')
        self.subtitles.setChecked(s.cloud.subtitles)
        f.addRow('Cloud', self.subtitles)
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

    def validatePage(self) -> bool:  # noqa: N802 (Qt name)
        s = self.w.s
        s.my_lang, s.their_lang = self.my_lang.currentData(), self.their_lang.currentData()
        for key, c in self.engine.items():
            setattr(s, key, c.currentText())
        s.cloud.subtitles = self.subtitles.isChecked()
        self.w.key = self.key.text().strip()
        if s.my_lang == s.their_lang:
            self.err.setText('Choose two different languages.')
            return False
        if 'cloud' in (s.them_to_me, s.me_to_them) and not (self.w.key or has_key(s)):
            self.err.setText('The cloud needs a Gemini API key.')
            return False
        if self.w.key:
            save_key(self.w.key, s.cloud)  # the test phrase on a later page already needs it
        return True


class DevicesPage(QWizardPage):
    def __init__(self, w: SetupWizard):
        super().__init__()
        self.w = w
        self.setTitle('Sound devices')
        self.setSubTitle('Translated speech goes to your headphones; your translated voice goes to the virtual microphone '
                         '"Peresmeshnik Mic", which you choose in the call app.')
        self.f = QFormLayout(self)
        self.warn = QLabel()
        self.warn.setWordWrap(True)

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

    def _echo(self) -> None:
        name = self.headphones.currentText().lower()
        speakers = any(k in name for k in ('speaker', 'built-in', 'встроен', 'динамик'))
        self.warn.setText('⚠ This looks like loudspeakers: the microphone would hear the translation and translate it back. '
                          'Use headphones.' if speakers else 'Use headphones, not loudspeakers (echo).')

    def validatePage(self) -> bool:  # noqa: N802
        self.w.s.ui.headphones, self.w.s.ui.mic = self.headphones.currentData(), self.mic.currentData()
        return True


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
        rows = [f'<tr><td>{"✔" if models.is_fetched(m, base) else "·"}</td><td>{m.id}</td><td align=right>{m.size / 1e6:.0f} MB</td>'
                f'<td><a href="{m.license_url}">{m.license}</a></td></tr>' for m in self.ms]
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
        total = sum(m.size for m in todo)

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
                           f'{len(todo)} to download, {sum(m.size for m in todo) / 1e6:.0f} MB.' if todo else 'All models are here.')
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
        self.ran = False
        self.ok = False

    def isComplete(self) -> bool:  # noqa: N802
        return self.ran

    def run(self) -> None:
        self.button.setEnabled(False)
        self.out.setText('')
        ctl, s = self.w.tray.ctl, self.w.s

        def work():
            from peresmeshnik.ui import selftest
            ctl.set_translation(False).result()  # our capture sink must not be in use
            vmic = ctl.submit(ctl.virtual_mic).result()  # the app's own: a second one would replace it
            rs = selftest.run_all(s.ui.headphones or None, s.ui.mic or None, vmic,
                                  report=lambda r: self.result.emit(r.ok, f'{r.name}: {r.detail}'))
            self.ok = all(r.ok for r in rs)
            self.done.emit()
        threading.Thread(target=work, name='selftest', daemon=True).start()

    def _done(self) -> None:
        self.ran = True
        self.button.setEnabled(True)
        self.out.setText(self.out.text() + ('<b>All paths work.</b>' if self.ok else '<b>Some paths failed</b>: check the devices, '
                                            'run again, or continue and fix it later.'))
        self.completeChanged.emit()


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
        lay.addWidget(self.err)
        self.state.connect(self._state)

    def test(self, side: str) -> None:
        from peresmeshnik.app import Controller
        s = self.w.s
        lang = s.their_lang if side == 'them' else s.my_lang
        sample = str(SAMPLES / f'{lang}.ogg')
        for b in self.tests.values():
            b.setEnabled(False)
        self.out.setText('Loading the engines…')
        ctl = Controller(self.w.tray.ctl.overlay, self.state.emit, sample if side == 'them' else None,
                         sample if side == 'me' else None, settings=s, test=True)

        def work():
            ctl.set_translation(True).result()
            for side_ in ctl.sides:
                side_.inp.done.wait(60)
            cloud = any(x.engine == 'cloud' for x in ctl.sides)
            time.sleep(6.0 if cloud else 3.0)  # the cloud voice ends ~3-4 s after the speech (BENCH)
            ctl.quit()
            self.state.emit('test done')
        threading.Thread(target=work, name='test-phrase', daemon=True).start()

    def _state(self, st: str) -> None:
        self.out.setText(st)
        if st == 'test done' or st.startswith('error'):
            for b in self.tests.values():
                b.setEnabled(True)

    def validatePage(self) -> bool:  # noqa: N802
        combo = self.edit.keySequence().toString(QKeySequence.PortableText)
        if combo and combo != self.w.s.ui.hotkey:
            if why := self.w.tray.reload_hotkey(combo):
                self.err.setText(f'Cannot use {combo}: {why}')
                return False
            self.w.s.ui.hotkey = combo
        return True


class DonePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle('Ready')
        lab = QLabel('In your call app (Meet, Zoom, Telegram) choose the microphone <b>Peresmeshnik Mic</b>.<br><br>'
                     'Turn translation on from the tray icon. The hotkey switches to your live voice and back.<br><br>'
                     'Settings and this wizard are in the tray menu.')
        lab.setWordWrap(True)
        QVBoxLayout(self).addWidget(lab)


class SetupWizard(QWizard):
    def __init__(self, tray):
        super().__init__()
        self.tray, self.s, self.key = tray, config.load(), ''
        self.setWindowTitle('Peresmeshnik setup')
        self.pages = [LanguagesPage(self)] + ([DevicesPage(self)] if LINUX else []) + [ModelsPage(self)] \
            + ([SelfTestPage(self)] if LINUX else []) + [HotkeyPage(self), DonePage()]
        for p in self.pages:
            self.addPage(p)
        self.accepted.connect(self.finish)
        self.rejected.connect(self.cancel)
        self.finished.connect(lambda _: tray.ctl.on or tray.ctl.overlay.clear.emit())  # the test phrase's subtitles

    def cancel(self) -> None:
        if (saved := config.load().ui.hotkey) != self.s.ui.hotkey:  # the hotkey page already bound the new one
            self.tray.reload_hotkey(saved)

    def finish(self) -> None:
        config.save(self.s)
        self.tray.sync()
        self.tray.restart()

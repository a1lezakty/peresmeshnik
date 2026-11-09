"""Setup wizard (first run, and from the tray): languages and engines, devices, models with their
licenses, the loopback self-test, your voice (recorded or imported, checked, and the clone checked),
the hotkey and a test phrase in both directions.

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

from peresmeshnik import config, models
from peresmeshnik.ui.tray import LANGS, MIC_LABEL, SIDES, device_combo, save_key

LINUX = sys.platform.startswith('linux')
MACOS = sys.platform == 'darwin'
VMIC = 'BlackHole 2ch' if MACOS else 'Peresmeshnik Mic'  # what the call app takes as its microphone
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
            stock = side == 'them_to_me' or s.local.clone == 'piper'  # voice.STOCK: me -> them in a stock voice
            ids.append(s.local.tts[tgt] if stock else s.local.clone)
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
                         f'"{VMIC}", which you choose in the call app.')
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
        if MACOS:
            self.blackhole = QLabel()
            self.blackhole.setWordWrap(True)
            self.blackhole.setOpenExternalLinks(True)
            self.blackhole.setTextInteractionFlags(Qt.TextBrowserInteraction)  # the brew command can be copied
            self.recheck = QPushButton('Check again')
            self.recheck.clicked.connect(self.check_blackhole)
            self.f.addRow('Virtual microphone', self.blackhole)
            self.f.addRow('', self.recheck)
            self.check_blackhole()

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
        self.blackhole.setText(text)
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
    app's speaker set to BlackHole 16ch. Either way the page can be left: them -> me says why when it cannot start."""
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
                if MACOS:  # the test records BlackHole 2ch like a call app: the loop would open the microphone into it
                    ctl.submit(ctl._loopback, False).result()
                try:
                    rs = selftest.run_all(s.ui.headphones or None, s.ui.mic or None, vmic,
                                          report=lambda r: self.result.emit(r.ok, f'{r.name}: {r.detail}'))
                finally:
                    if MACOS and not ctl.on:
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
        self.out, self.clone_out = QLabel(), QLabel()
        for x in (self.out, self.clone_out):
            x.setWordWrap(True)
        lay.addWidget(self.stock)
        lay.addWidget(self.prompt)
        lay.addLayout(row)
        for x in (self.bar, self.out, self.clone_out):
            lay.addWidget(x)
        self.progress.connect(lambda t: self.bar.setValue(int(t)))
        self.result.connect(self._result)
        self.checked.connect(self._checked)
        self.voice = None
        self.running = False
        self.draft = f'{self.name}.new'

    def local(self) -> bool:
        return self.w.s.me_to_them == 'local'

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
        if self.voice is None and s.local.voice and (voice.voices_dir() / s.local.voice / 'voice.json').exists():
            self.voice = voice.load(s.local.voice)
            q = self.voice.meta.get('quality', {})
            self.out.setText(f'Saved voice "{self.voice.name}": {q.get("seconds", 0):.0f} s, '
                             f'recorded {self.voice.meta.get("created", "")}.')
        self._buttons()

    def isComplete(self) -> bool:  # noqa: N802
        return not self.running and (not self.local() or self.stock.isChecked() or self.voice is not None)

    def _stock(self, on: bool) -> None:
        """The stock voice instead of the clone; its model (~60 MB) is fetched here when the models page took the clone."""
        from peresmeshnik import voice
        s = self.w.s
        s.local.clone = voice.STOCK if on else config.Local().clone
        self.check_button.setVisible(self.local() and not on)
        vid = s.local.tts[s.their_lang]
        if on and not models.is_fetched(models.get(vid), models.root()):
            self.out.setText(f'Downloading the stock voice {vid}…')
            self._run(lambda: (models.fetch(vid), self.result.emit(f'The stock voice {vid} is here.')))
        self.completeChanged.emit()

    def translating(self) -> bool:
        """Translation on: a local me -> them already holds the clone; the check would load a second one (~3 GB)."""
        return bool(getattr(getattr(getattr(self.w, 'tray', None), 'ctl', None), 'on', False))

    def _buttons(self) -> None:
        for b in (self.record_button, self.import_button):
            b.setEnabled(not self.running)
        self.check_button.setEnabled(not self.running and self.voice is not None and not self.translating())
        self.check_button.setToolTip('Turn translation off first: the check loads a second clone (~3 GB)'
                                     if self.translating() else '')
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

    def check_clone(self) -> None:
        """The clone says the test phrases in their language; the app's ASR must understand them (voice.check_clone)."""
        from peresmeshnik import models, voice
        s, v = self.w.s, self.voice
        if self.translating():
            self.clone_out.setText('Turn translation off first: the check loads a second clone (~3 GB).')
            return
        self.clone_out.setText('Checking the clone: it says two test phrases… (first time: loading the voice pack)')

        def work():
            try:
                from peresmeshnik.local.clone import XttsClone
            except ImportError as e:
                self.checked.emit(f'The clone check needs the voice pack (torch, coqui-tts): {e}')
                return
            from peresmeshnik.local.asr import OnnxAsr
            try:
                tts = XttsClone(models.path(s.local.clone), threads=s.local.threads)
                asr = OnnxAsr(s.local.asr[s.their_lang], s.local.threads)
                self.checked.emit(voice.clone_verdict(voice.check_clone(tts, v, s.their_lang, asr)))
            except Exception as e:  # a model not downloaded yet, out of memory
                self.checked.emit(f'error: {e}')
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
        """The wizard was cancelled: the saved profile stays as it was."""
        from peresmeshnik import voice
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
        routed = LINUX or MACOS
        self.pages = [LanguagesPage(self)] + ([DevicesPage(self)] if routed else []) + ([CallSoundPage(self)] if MACOS else []) \
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

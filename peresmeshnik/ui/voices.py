"""The voices (tray "Voices…"): my own voice and the voices of people I talk to that I chose to keep (voice.json v2),
delete them. My voice speaks into the call only once confirmed: `ConfirmBox` (here and on the wizard's voice page) shows
two random phrases, records them at the microphone and checks them (voice.confirm_live: the app's ASR heard them, the
speaker vector is the recording's). `RememberDialog`: keep the voice of the person in this call (S2's live profile), only
with their agreement. Recording, downloads and checks take seconds: worker threads, answers come back as signals."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from peresmeshnik import config, voice

READ_S = 8.0  # the live reading: two short phrases at a normal pace, 3-4 s each
THEIRS = ('This voice is heard only by you, in your headphones; it never speaks into the call. It is kept as the clone '
          "engine's state, not as a recording of their speech, and stays on this computer.")


def described(v) -> str:
    """One line of the list: whose, the language, how long, and whether it may speak into the call."""
    m, c = v.meta, v.meta.get('consent') or {}
    when = (c.get('at') or m.get('created', ''))[:10]
    if m['kind'] == 'them':
        return f'{m["name"]}: someone you talk to, {m["lang"]}, {m["seconds"]:.0f} s of their speech, kept {when}'
    how = {'mic-live': f'confirmed at the microphone {when}',
           'import+mic-match': f'from a file, confirmed at the microphone {when}'}.get(
        c.get('how'), 'not confirmed yet: it does not speak into the call until you read two phrases (Confirm)')
    return f'Your voice: {m["lang"]}, {m["seconds"]:.0f} s, {how}'


def check_models(s: config.Settings, lang: str, say=lambda text: None):
    """The live check's ASR (the profile's language) and speaker vector, downloaded first if missing."""
    from peresmeshnik import languages, models
    from peresmeshnik.local.asr import OnnxAsr
    asr = languages.pick(s.local.asr, lang)
    for i in (asr, voice.SPEAKER):
        if not models.is_fetched(models.get(i)):
            say(f'Downloading what the check needs ({models.size(models.get(i)) / 1e6:.0f} MB, once)…')
            models.fetch(i)
    say('Checking…')
    return OnnxAsr(asr, s.local.threads, live=False), voice.Speaker(models.path(voice.SPEAKER))


class _Sig(QObject):
    done = Signal(object)  # (consent | None, text)
    say = Signal(str)
    progress = Signal(float)


class ConfirmBox(QWidget):
    """Read two phrases at the microphone: my profile `name` is confirmed (its voice.json gets the consent, `confirmed`
    is emitted) or not, and why. `how`: 'mic-live' (recorded just now) or 'import+mic-match' (a file)."""
    confirmed = Signal(dict)
    busy = Signal(bool)  # reading or checking: True, then False

    def __init__(self, s: config.Settings, name: str, how: str, root=None, parent=None):
        super().__init__(parent)
        self.s, self.name, self.how, self.root, self.running = s, name, how, root, False
        self.lang = voice.load(name, root).meta['lang']
        self.phrases = QLabel()
        self.phrases.setWordWrap(True)
        self.button = QPushButton('Read aloud')
        self.bar = QProgressBar()
        self.bar.setMaximum(int(READ_S))
        self.bar.setVisible(False)
        self.out = QLabel('')
        self.out.setWordWrap(True)
        lay = QVBoxLayout(self)
        intro = QLabel('To confirm this is your voice, press "Read aloud" and read these two phrases at your normal pace '
                       f'({READ_S:.0f} s). A recording made before cannot say them: only your live voice is accepted.')
        intro.setWordWrap(True)
        for w in (intro, self.phrases, self.button, self.bar, self.out):
            lay.addWidget(w)
        self._sig = _Sig()
        self._sig.done.connect(self._finished)
        self._sig.say.connect(self.out.setText)
        self._sig.progress.connect(lambda x: self.bar.setValue(int(x)))
        self.button.clicked.connect(self.start)
        self._new_phrases()

    def _new_phrases(self) -> None:
        self.said = voice.live_phrases(self.lang)
        self.phrases.setText('<br>'.join(f'<b>{p}</b>' for p in self.said))

    def start(self) -> None:
        self.running = True
        self.busy.emit(True)
        self.button.setEnabled(False)
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.out.setText('Recording… read the phrases aloud.')
        threading.Thread(target=self._work, name='confirm-voice', daemon=True).start()

    def _work(self) -> None:
        try:
            from peresmeshnik.audio.mic import record
            sr = 24_000
            pcm = record(READ_S, self.s.ui.mic or None, sr, self._sig.progress.emit)
            asr, spk = check_models(self.s, self.lang, self._sig.say.emit)
            consent, why = voice.confirm_live(voice.load(self.name, self.root), pcm, sr, self.said, asr, spk, self.how)
            if consent:
                voice.set_consent(self.name, consent, self.root)
            self._sig.done.emit((consent, why))
        except Exception as e:  # no microphone, no network for the download
            self._sig.done.emit((None, f'The check did not run: {e}'))

    def _finished(self, r) -> None:
        consent, why = r
        self.running = False
        self.busy.emit(False)
        self.bar.setVisible(False)
        self.button.setEnabled(True)
        if consent:
            self.out.setText(f'✔ Confirmed: your voice (similarity {consent["similarity"]:.2f}).')
            self.button.setEnabled(False)
            self.confirmed.emit(consent)
        else:
            self.out.setText(f'⚠ {why} Try again with the new phrases.')
            self._new_phrases()


class VoicesDialog(QDialog):
    """The tray's "Voices…": the list, Delete, Delete all voices of people you talk to, Confirm my voice.
    `on_change()`: after a delete or a confirmation (the tray restarts translation if my voice changed)."""

    def __init__(self, s: config.Settings, on_change=lambda: None, root=None):
        super().__init__()
        self.s, self.on_change, self.root = s, on_change, root
        self.setWindowTitle('Peresmeshnik: voices')
        self.list = QListWidget()
        self.delete_button = QPushButton('Delete')
        self.delete_theirs = QPushButton('Delete all voices of people you talk to')
        self.confirm_button = QPushButton('Confirm it is my voice…')
        self.confirm: ConfirmBox | None = None
        note = QLabel('Your voice speaks for you in the call once confirmed. The voices of people you talk to are heard '
                      'only by you, in your headphones; a call forgets them unless you keep one. All stay on this '
                      'computer.')
        note.setWordWrap(True)
        row = QHBoxLayout()
        for b in (self.confirm_button, self.delete_button, self.delete_theirs):
            row.addWidget(b)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        self.lay = QVBoxLayout(self)
        for w in (note, self.list):
            self.lay.addWidget(w)
        self.lay.addLayout(row)
        self.lay.addWidget(close)
        self.list.currentRowChanged.connect(self._buttons)
        self.delete_button.clicked.connect(self.delete)
        self.delete_theirs.clicked.connect(self.delete_all_theirs)
        self.confirm_button.clicked.connect(self.start_confirm)
        self.refresh()

    def refresh(self) -> None:
        self.voices = voice.profiles(self.root)
        self.list.clear()
        for v in self.voices:
            self.list.addItem(QListWidgetItem(described(v)))
        if not self.voices:
            self.list.addItem('No voices yet: record yours in the setup wizard ("Your voice").')
        self._buttons()

    def selected(self):
        i = self.list.currentRow()
        return self.voices[i] if 0 <= i < len(self.voices) else None

    def _buttons(self, *_) -> None:
        v = self.selected()
        self.delete_button.setEnabled(v is not None)
        self.delete_theirs.setEnabled(any(x.meta['kind'] == 'them' for x in self.voices))
        self.confirm_button.setEnabled(v is not None and v.meta['kind'] == 'me' and bool(voice.why_not_mine(v.meta)))

    def _ask(self, text: str) -> bool:
        return QMessageBox.question(self, 'Peresmeshnik', text) == QMessageBox.StandardButton.Yes

    def delete(self) -> None:
        v = self.selected()
        if v is None or not self._ask(f'Delete "{v.meta["name"]}"? ' + (
                'Your cloned voice will need a new recording (setup wizard).' if v.meta['kind'] == 'me' else '')):
            return
        voice.discard(v.name, self.root)
        if v.meta['kind'] == 'me' and self.s.local.voice == v.name:
            self.s.local.voice = ''  # the me side says "record your voice" instead of failing on a missing folder
        self.refresh()
        self.on_change()

    def delete_all_theirs(self) -> None:
        theirs = [v for v in self.voices if v.meta['kind'] == 'them']
        if theirs and self._ask(f'Delete the {len(theirs)} kept voices of people you talk to?'):
            for v in theirs:
                voice.discard(v.name, self.root)
            self.refresh()
            self.on_change()

    def start_confirm(self) -> None:
        v = self.selected()
        if v is None or self.confirm is not None:
            return
        self.confirm = ConfirmBox(self.s, v.name, voice.how_to_confirm(v.meta), self.root, self)
        self.confirm.confirmed.connect(lambda c: (self.refresh(), self.on_change()))
        self.lay.insertWidget(self.lay.count() - 1, self.confirm)



class RememberDialog(QDialog):
    """Keep the voice of the person in this call (S2's live profile), under a name, only once the user says the person
    agreed. The tray: `if d.exec(): d.keep(states, lang, seconds)`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Peresmeshnik: keep this voice')
        self.name = QLineEdit()
        self.name.setPlaceholderText('Whose voice is it (shown only to you)')
        self.agreed = QCheckBox('They agreed that I keep their voice for our next calls')
        text = QLabel(THEIRS)
        text.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save.setEnabled(False)
        self.name.textChanged.connect(self._ok)
        self.agreed.toggled.connect(self._ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        for w in (self.name, text, self.agreed, buttons):
            lay.addWidget(w)

    def _ok(self, *_) -> None:
        self.save.setEnabled(bool(self.name.text().strip()) and self.agreed.isChecked())

    def keep(self, states: dict, lang: str, seconds: float, root=None):
        """{model id: (state, sha256 of the weights)} -> the saved profile (voice.save_them), only as agreed."""
        assert self.agreed.isChecked() and self.name.text().strip()
        return voice.save_them(self.name.text().strip(), lang, seconds, states, root)

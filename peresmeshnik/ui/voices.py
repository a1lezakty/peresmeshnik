"""The voices (tray "Voices…"): my own voice and the voices of people I talk to that I chose to keep (voice.json v2),
delete them. My voice speaks into the call only once confirmed: `ConfirmBox` (here and on the wizard's voice page) shows
two random phrases, records them at the microphone and checks them (voice.confirm_live: the app's ASR heard them, the
speaker vector is each window of the recording's). `RememberDialog`: keep the voice of the person in this call (S2's live profile), only
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
    if why := voice.why_not_mine(v):
        return f'Your voice: {m["lang"]}, {m["seconds"]:.0f} s, {why}'
    how = {'mic-live': 'recorded here', 'import+mic-match': 'from a file'}.get(c.get('how'), '')
    return f'Your voice: {m["lang"]}, {m["seconds"]:.0f} s, {how}, confirmed at the microphone {when}'


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
    is emitted) or not, and why. Not while translating (`translating()`: the microphone and a second ASR), not during a
    pause after failed tries (voice.wait_s). `retire()`: its result, if one still comes, is ignored."""
    confirmed = Signal(dict)
    busy = Signal(bool)  # reading or checking: True, then False

    def __init__(self, s: config.Settings, name: str, root=None, parent=None, translating=lambda: False):
        super().__init__(parent)
        self.s, self.name, self.root, self.translating = s, name, root, translating
        self.running = self.dead = False
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
        if self.lang not in voice.LIVE_PHRASES:  # a profile in a language the app has no phrases for (CLI --lang)
            self.said = []
            self.phrases.setText(f'No phrases to read in "{self.lang}": this voice cannot be confirmed. Record it again '
                                 'in the setup wizard, in your language.')
            self.button.setEnabled(False)
            return
        self.said = voice.live_phrases(self.lang)
        self.phrases.setText('<br>'.join(f'<b>{p}</b>' for p in self.said))

    def start(self) -> None:
        if not self.said or self.running:
            return
        if self.translating():
            self.out.setText('Turn translation off first: the check records from the microphone.')
            return
        if w := voice.wait_s(self.root):
            self.out.setText(f'Too many tries: wait {max(w / 60, 1):.0f} min before the next one.')
            return
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
            consent, why = voice.confirm_live(voice.load(self.name, self.root), pcm, sr, self.said, asr, spk)
            if consent:  # only onto the recording it checked (voice.set_consent: ValueError if another one is there)
                voice.set_consent(self.name, consent, self.root)
            self._sig.done.emit((consent, why))
        except Exception as e:  # no microphone, no network for the download, a new take meanwhile
            self._sig.done.emit((None, f'The check did not run: {e}.'))

    def retire(self) -> None:
        self.dead = True
        self.setParent(None)
        self.hide()

    def _finished(self, r) -> None:
        if self.dead:  # a new take replaced it: its result is not about the voice shown
            return
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
    """The tray's "Voices…": the list (damaged folders too), Delete, Delete all voices of people you talk to, Confirm
    my voice (a box per profile). `on_change(mine)`: after a delete or a confirmation, `mine` when it was my voice (the
    tray saves the settings' voice and restarts translation)."""

    def __init__(self, s: config.Settings, on_change=lambda mine: None, root=None, translating=lambda: False):
        super().__init__()
        self.s, self.on_change, self.root, self.translating = s, on_change, root, translating
        self.setWindowTitle('Peresmeshnik: voices')
        self.list = QListWidget()
        self.delete_button = QPushButton('Delete')
        self.delete_theirs = QPushButton('Delete all voices of people you talk to')
        self.confirm_button = QPushButton('Confirm it is my voice…')
        self.boxes: dict[str, ConfirmBox] = {}
        note = QLabel('Your voice speaks for you in the call once confirmed. The voices of people you talk to are heard '
                      'only by you, in your headphones; a call forgets them unless you keep one. All stay on this '
                      'computer.')
        note.setWordWrap(True)
        row = QHBoxLayout()
        for b in (self.confirm_button, self.delete_button, self.delete_theirs):
            row.addWidget(b)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        from peresmeshnik.ui.components import HfTokenBox
        self.hf = HfTokenBox()  # the clone of my language waits for the user's Hugging Face token (Pocket TTS)
        self.hf.saved.connect(self._token_saved)
        self.token_note = QLabel()
        self.token_note.setWordWrap(True)
        self.lay = QVBoxLayout(self)
        for w in (note, self.list, self.hf, self.token_note):
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
        self.broken = voice.damaged(self.root)
        self.list.clear()
        for v in self.voices:
            self.list.addItem(QListWidgetItem(described(v)))
        for name in self.broken:
            self.list.addItem(QListWidgetItem(f'{name}: damaged (no readable voice.json) — delete'))
        if not self.voices and not self.broken:
            self.list.addItem('No voices yet: record yours in the setup wizard ("Your voice").')
        from peresmeshnik import components
        self.hf.show_for(components.token_page(self.s.their_lang))
        self._buttons()

    def _token_saved(self) -> None:
        """Another token: the clone engine is chosen again the next time translation starts (its weights: Components…);
        translation going on now is not restarted for it."""
        from peresmeshnik import languages, models
        engine = languages.clone_engine(self.s.their_lang)[0]
        self.token_note.setText(f'Saved. Your cloned voice ({engine}) is downloaded in Components… (tray menu).'
                                if engine and not models.is_fetched(models.get(engine)) else
                                'Saved.' if models.hf_token() else 'The token is forgotten.')
        self.refresh()

    def selected(self):
        """The voice of the selected row, or the damaged folder's name (str), or None."""
        i = self.list.currentRow()
        rows = [*self.voices, *self.broken]
        return rows[i] if 0 <= i < len(rows) else None

    def _buttons(self, *_) -> None:
        v = self.selected()
        name = v if isinstance(v, str) else getattr(v, 'name', None)
        checking = name in self.boxes and self.boxes[name].running
        for n, b in self.boxes.items():  # the box of the selected profile only
            b.setVisible(n == name)
        self.delete_button.setEnabled(v is not None and not checking)
        self.delete_theirs.setEnabled(any(x.meta['kind'] == 'them' for x in self.voices))
        self.confirm_button.setEnabled(not isinstance(v, (str, type(None))) and v.meta['kind'] == 'me'
                                       and bool(voice.why_not_mine(v)) and name not in self.boxes)

    def _ask(self, text: str) -> bool:
        return QMessageBox.question(self, 'Peresmeshnik', text) == QMessageBox.StandardButton.Yes

    def _forget_box(self, name: str) -> None:
        if b := self.boxes.pop(name, None):
            b.retire()

    def delete(self) -> None:
        v = self.selected()
        if isinstance(v, str):  # a damaged folder
            if self._ask(f'Delete the damaged folder "{v}"?'):
                voice.discard(v, self.root)
                self.refresh()
            return
        if v is None or not self._ask(f'Delete "{v.meta["name"]}"? ' + (
                'Your cloned voice will need a new recording (setup wizard).' if v.meta['kind'] == 'me' else '')):
            return
        mine = v.meta['kind'] == 'me'
        for n in (v.name, f'{v.name}.new', f'{v.name}.old') if mine else (v.name,):  # a take kept aside goes too
            voice.discard(n, self.root)
        self._forget_box(v.name)
        if mine and self.s.local.voice == v.name:
            self.s.local.voice = ''  # the me side says "record your voice" instead of failing on a missing folder
        self.refresh()
        self.on_change(mine)

    def delete_all_theirs(self) -> None:
        theirs = [v for v in self.voices if v.meta['kind'] == 'them']
        if theirs and self._ask(f'Delete the {len(theirs)} kept voices of people you talk to?'):
            for v in theirs:
                voice.discard(v.name, self.root)
            self.refresh()
            self.on_change(False)

    def start_confirm(self) -> None:
        v = self.selected()
        if isinstance(v, (str, type(None))) or v.name in self.boxes:
            return
        box = self.boxes[v.name] = ConfirmBox(self.s, v.name, self.root, self, self.translating)
        box.confirmed.connect(lambda c, n=v.name: (self._forget_box(n), self.refresh(), self.on_change(True)))
        box.busy.connect(self._buttons)
        self.lay.insertWidget(self.lay.count() - 1, box)
        self._buttons()


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

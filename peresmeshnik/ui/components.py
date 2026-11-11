"""The Components box: what the app downloads after it is installed (peresmeshnik/components.py), ticked by default where
it suits this computer and the chosen pair; nothing downloads before Download, anything installed can be removed. The
setup wizard's Components page and the tray's Components… window show this one box."""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from peresmeshnik import components, models
from peresmeshnik.local import voicepack

PACKS = (components.VOICE_PACK, components.ADDON)
PACK_CONSENT = ('I accept the licenses of the voice pack: PyTorch (BSD-3-Clause) with the NVIDIA CUDA and cuDNN libraries '
                'inside it (NVIDIA\'s license terms: docs.nvidia.com/cuda/eula, most of the download), coqui-tts '
                '(MPL-2.0), and of the XTTS-v2 voice model it runs, the Coqui Public Model License '
                '(huggingface.co/coqui/XTTS-v2): noncommercial use only, it binds me as the user')

HF_TEXT = ('Your cloned voice without an NVIDIA GPU speaks through Kyutai\'s Pocket TTS. Its weights are behind a '
           'button on Hugging Face: open the page, sign in, accept the conditions, then make a token of type Read '
           '(Settings → Access Tokens) and paste it here. It is kept in the settings folder, readable only by you, and '
           'sent only to huggingface.co.')


def mb(n: int) -> str:
    return f'{n / 1e9:.1f} GB' if n >= 1e9 else f'{n / 1e6:.0f} MB'


class HfTokenBox(QWidget):
    """For a gated model (`token = "hf"`): its page (accept the conditions there) and the user's token; saved empty, the
    token is forgotten (its file goes)."""
    saved = Signal()  # the token changed: the clone engine and the list are chosen again

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = QLabel(HF_TEXT)
        self.text.setWordWrap(True)
        self.open = QPushButton('Open the page')
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.save = QPushButton('Save the token')
        self.save.clicked.connect(self._save)
        self.token.returnPressed.connect(self._save)
        self.open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.page)))
        row = QHBoxLayout()
        for w in (self.open, self.token, self.save):
            row.addWidget(w)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.text)
        lay.addLayout(row)
        self.page = ''

    def show_for(self, page: str) -> None:
        """Shown while a gated model waits for the token or was refused: `page` is its model card ('' hides the box)."""
        self.setVisible(bool(page))
        if page:
            self.page = page
            self.token.setPlaceholderText('a token is saved (save it empty to forget it)' if models.hf_token() else 'hf_…')

    def _save(self) -> None:
        models.save_hf_token(self.token.text().strip())  # '' forgets it
        self.token.clear()
        self.token.setPlaceholderText('a token is saved (save it empty to forget it)' if models.hf_token() else 'hf_…')
        self.saved.emit()


class ComponentsBox(QWidget):
    progress = Signal(float, float, str)  # done bytes, total bytes (0: busy, no count), text: past 2 GB a Qt int overflows
    finished = Signal(str)  # '' or what went wrong
    changed = Signal()  # ticked, downloaded or removed: the wizard page's Next

    def __init__(self, s, translating=lambda: False, parent=None):
        super().__init__(parent)
        self.s, self.translating, self.busy, self.rows, self.ticked = s, translating, False, [], {}
        self.refused = {}  # id -> its model card: Hugging Face said no (models.Gated); Next does not wait for it
        self.cancel = threading.Event()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Component', 'Size', 'License', 'What for', ''])
        self.tree.setRootIsDecorated(False)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemChanged.connect(self._ticked)
        self.without = QLabel()  # unticked: what the pair does instead
        self.without.setWordWrap(True)
        self.consent = QCheckBox()
        self.consent.toggled.connect(lambda: self._update(''))
        self.pack_consent = QCheckBox(PACK_CONSENT)  # the voice pack (and its addon): its own licenses, said as they are
        self.pack_consent.toggled.connect(lambda: self._update(''))
        self.hf = HfTokenBox()
        self.hf.setVisible(False)
        self.hf.saved.connect(self._token_saved)
        self.button = QPushButton('Download')
        self.button.clicked.connect(self.download)
        self.stop = QPushButton('Cancel')
        self.stop.clicked.connect(self.cancel.set)
        self.bar = QProgressBar()
        self.state = QLabel()
        self.state.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.button)
        row.addWidget(self.stop)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        for x in (self.tree, self.without, self.consent, self.pack_consent, self.hf):
            lay.addWidget(x)
        lay.addLayout(row)
        lay.addWidget(self.bar)
        lay.addWidget(self.state)
        self.progress.connect(self._progress)
        self.finished.connect(self._done)

    def refresh(self) -> None:
        """The list anew (after a pair change, a download, a removal); a tick the user changed stays."""
        self.rows = components.listing(self.s)
        self.tree.blockSignals(True)
        self.tree.clear()
        for c in self.rows:
            it = QTreeWidgetItem([c.title, f'{mb(c.on_disk)} on disk' if c.installed else mb(c.size), c.license,
                                  c.unavailable or c.why])
            it.setData(0, Qt.ItemDataRole.UserRole, c.id)
            for col in range(4):
                it.setToolTip(col, '\n'.join(x for x in (c.unavailable or c.why, c.instead and f'Without it: {c.instead}.',
                                                         c.note, c.license_url) if x))
            if c.installed:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            elif c.unavailable:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            else:
                it.setCheckState(0, Qt.CheckState.Checked if self.wanted(c) else Qt.CheckState.Unchecked)
                if c.required:  # ticked for good: the hint says how not to need it
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            self.tree.addTopLevelItem(it)
            if c.on_disk:  # installed, or a download of another pair that stopped half way
                rm = QPushButton('Remove')
                rm.clicked.connect(lambda _=False, c=c: self.remove(c))
                self.tree.setItemWidget(it, 4, rm)
        self.tree.blockSignals(False)
        self._update('')

    def wanted(self, c) -> bool:
        """Ticked: what the user ticked or unticked, else the default (another pair's default changes with it)."""
        return c.required or self.ticked.get(c.id, c.recommended)

    def _ticked(self, it: QTreeWidgetItem) -> None:
        self.ticked[it.data(0, Qt.ItemDataRole.UserRole)] = it.checkState(0) == Qt.CheckState.Checked
        self._update('')

    def todo(self) -> list:
        """Ticked and not here, the pair's own first: an optional one failing does not hold them."""
        out = [c for c in self.rows if not c.installed and not c.unavailable and self.wanted(c)]
        return sorted(out, key=lambda c: not c.recommended)

    def missing(self) -> list:
        """What the pair runs on, ticked and not here yet: the wizard's Next waits for it; not for a clone engine Hugging
        Face refused (its conditions not accepted): a stock voice speaks, the reason is said."""
        return [c for c in self.todo() if c.recommended and c.id not in self.refused]

    def _token_saved(self) -> None:
        """Another token, or none: the engine is chosen again (languages.clone_engine) and the list with it."""
        self.refused.clear()
        if not self.busy:
            self.refresh()

    def download(self) -> None:
        if self.busy:  # one download at a time: a second install into the same voice pack breaks it
            return
        todo = self.todo()
        pack = [c for c in todo if c.id in PACKS]
        if any(c.consent and c.id not in PACKS for c in todo) and not self.consent.isChecked() or \
                pack and not self.pack_consent.isChecked():
            self.state.setText('Accept the licenses first, or untick what they are for.')
            return
        self.busy = True
        self.cancel.clear()
        self._buttons()
        self.changed.emit()
        total = sum(c.size for c in todo)

        def run():
            done, errors = [0], []
            for c in todo:
                if self.cancel.is_set():
                    errors.append('cancelled: what is downloaded stays for the next try')
                    break
                start = done[0]

                def tick(stage, got, size, text='', c=c, start=start):
                    if stage == 'downloading':
                        done[0] = start + got
                        self.progress.emit(done[0], total, f'{c.title}: {text}' if text != c.id else c.title)
                    else:
                        self.progress.emit(0, 0, f'{c.title}: {stage} {text}'.strip())
                try:
                    c.fetch(tick, self.cancel)
                    done[0] = start + c.size
                except Exception as e:  # the network, the disk, a token refused (models.Gated): the others still go
                    errors.append(f'{c.title}: {e}')
                    if isinstance(e, getattr(models, 'Gated', ())):
                        self.refused[c.id] = e.page
                    if isinstance(e, (components.Cancelled, voicepack.Cancelled)) or \
                            c.recommended and not isinstance(e, getattr(models, 'Gated', ())):
                        break  # cancelled, or the pair's own: stop here, Download resumes it
            self.finished.emit('; '.join(errors))
        threading.Thread(target=run, name='components-fetch', daemon=True).start()

    def remove(self, c) -> None:
        if self.busy or self.translating():
            self.state.setText('Turn translation off first: the app may be using it.' if self.translating() else
                               'Wait for the download to finish.')
            return
        if not self.confirm(f'Remove {c.title} ({mb(c.on_disk)})?'
                            + (f' Without it {c.instead}; choosing it again downloads it again.' if c.instead else '')):
            return
        try:
            c.remove()
            err = ''
        except Exception as e:  # the voice pack's files still held (Windows): said, as a failed download is
            err = f'Could not remove {c.title}: {e}'
        self.refresh()
        self.changed.emit()
        if err:
            self.state.setText(err)

    def confirm(self, text: str) -> bool:
        return QMessageBox.question(self, 'Remove', text) == QMessageBox.StandardButton.Yes

    def _progress(self, done: float, total: float, text: str) -> None:
        if total:
            self.bar.setRange(0, max(int(total) // 1024, 1))
            self.bar.setValue(int(done) // 1024)
            self.state.setText(f'{text}: {done / 1e6:.0f} of {total / 1e6:.0f} MB')
        else:
            self.bar.setRange(0, 0)
            self.state.setText(text)

    def _buttons(self) -> None:
        todo = self.todo()
        self.tree.setEnabled(not self.busy)  # no ticks, no Remove while it downloads
        self.button.setEnabled(not self.busy and bool(todo))
        self.stop.setVisible(self.busy)
        self.bar.setVisible(self.busy or bool(todo))
        for i in range(self.tree.topLevelItemCount()):
            if w := self.tree.itemWidget(self.tree.topLevelItem(i), 4):
                w.setEnabled(not self.busy)

    def _done(self, err: str) -> None:
        """The download thread ended: only here it is not busy any more."""
        self.busy = False
        self.refresh()  # what is here now
        self._update(err)

    def _update(self, err: str) -> None:
        todo = self.todo()
        need = [c for c in todo if c.consent and c.id not in PACKS]
        self.consent.setVisible(bool(need))
        self.consent.setText('I accept the restricted licenses (noncommercial use only, they bind me as the user): '
                             + ', '.join(f'{c.title} ({c.license})' for c in need))
        self.pack_consent.setVisible(any(c.id in PACKS for c in todo))
        self.hf.show_for(next(iter(self.refused.values()), '') or components.token_page(self.s.their_lang)
                         or next((c.license_url for c in todo if c.token and not models.hf_token()), ''))
        off = [c for c in self.rows if (c.id in self.refused or not (c.installed or c.unavailable or self.wanted(c)))
               and c.instead]  # unticked, or refused by Hugging Face
        self.without.setText('<br>'.join(f'Without {c.title}: {c.instead}.' for c in off))
        self.without.setVisible(bool(off))
        self.state.setText(f'Download failed: {err}. Press Download again to resume.' if err else
                           self.state.text() if self.busy else
                           f'{len(todo)} to download, {mb(sum(c.size for c in todo))}.' if todo else
                           f'Everything ticked is here. Folder: {models.root()}')
        self._buttons()
        self.changed.emit()


class ComponentsDialog(QDialog):
    """The tray's Components…: the same box, for the pair in the settings now."""

    def __init__(self, s, translating=lambda: False, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Peresmeshnik: components')
        self.resize(900, 520)
        self.box = ComponentsBox(s, translating, self)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Only the app was installed. Tick what to download: nothing downloads before Download, and '
                             'anything installed can be removed.'))
        lay.addWidget(self.box)
        self.box.refresh()

    def busy(self) -> bool:  # closed while it downloads: the tray keeps it alive (Tray._show)
        return self.box.busy

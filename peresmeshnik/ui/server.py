"""The "Server…" window (tray menu; the wizard shows the same box): pair this computer with `peresmeshnik serve` on a
stronger one - paste the line the server shows (https://<address>:47000/#<code>), and the translation runs there; forget
a pairing. Pairing waits for the network (seconds): it runs on a worker thread and its answer comes back as a signal."""
from __future__ import annotations

import socket
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from peresmeshnik import config, languages
from peresmeshnik.remote import pairing

HOW = ('On the stronger computer run <code>peresmeshnik serve</code>. It shows a line like '
       '<code>https://192.168.1.20:47000/#K7QM-2XPA-4DFE</code> (the address and a one-time code, 10 minutes): paste it '
       'here. Both computers must be on one network (home LAN or Tailscale). With "Me → them" on the server your voice '
       'profile goes to it for each session and is deleted there when the session ends.')


def described(p: dict | None) -> str:
    if p is None:
        return 'Not paired with a server.'
    return f'Paired with the server at {p["address"]} (certificate {p["fp"][:16]}…) as "{p.get("name", "")}".'


class _Done(QObject):
    done = Signal(object)  # the pairing (dict) or the error's text (str)


class PairBox(QWidget):
    """The address line, this computer's name, Pair and Forget; `paired(dict)` after a pairing."""
    paired = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = QLabel(described(pairing.load()))
        self.state.setWordWrap(True)
        how = QLabel(HOW)
        how.setWordWrap(True)
        self.url = QLineEdit()
        self.url.setPlaceholderText('https://192.168.1.20:47000/#K7QM-2XPA-4DFE')
        self.name = QLineEdit(socket.gethostname()[:64])
        self.name.setToolTip('This computer, as the server lists it (peresmeshnik serve --list, --revoke)')
        self.pair_button = QPushButton('Pair')
        self.forget_button = QPushButton('Forget this server')
        self.result = QLabel('')
        self.result.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(QLabel('This computer:'))
        row.addWidget(self.name)
        row.addWidget(self.pair_button)
        row.addWidget(self.forget_button)
        lay = QVBoxLayout(self)
        for w in (self.state, how, self.url):
            lay.addWidget(w)
        lay.addLayout(row)
        lay.addWidget(self.result)
        self._done = _Done()
        self._done.done.connect(self._finished)
        self.pair_button.clicked.connect(self.pair)
        self.forget_button.clicked.connect(self.forget)
        self.url.returnPressed.connect(self.pair)
        self.forget_button.setEnabled(pairing.load() is not None)

    def pair(self) -> None:
        url, name = self.url.text().strip(), self.name.text().strip() or 'laptop'
        try:
            pairing.parse(url)
        except ValueError as e:
            self.result.setText(f'⚠ {e}')
            return
        self.pair_button.setEnabled(False)
        self.result.setText('Pairing…')

        def work():
            try:
                self._done.done.emit(pairing.pair(url, name))
            except Exception as e:  # ProtocolError: the server's reason, a wrong proof, no connection
                self._done.done.emit(str(e))
        threading.Thread(target=work, name='pairing', daemon=True).start()

    def _finished(self, r) -> None:
        self.pair_button.setEnabled(True)
        if isinstance(r, str):
            self.result.setText(f'⚠ {r}')
            return
        self.url.clear()  # the code is used up
        self.state.setText(described(r))
        self.forget_button.setEnabled(True)
        self.result.setText('✔ Paired.')
        self.paired.emit(r)

    def forget(self) -> None:
        pairing.forget()
        self.state.setText(described(None))
        self.forget_button.setEnabled(False)
        self.result.setText('Forgotten here. On the server: peresmeshnik serve --revoke "<this computer>".')


class ServerDialog(QDialog):
    """The tray's "Server…": the pairing, and after it both directions on the server (when "use it" is checked)."""

    def __init__(self, tray):
        super().__init__()
        self.tray = tray
        self.setWindowTitle('Peresmeshnik: translation on another computer')
        self.box = PairBox(self)
        self.use = QCheckBox('Translate both directions on this server once paired')
        self.use.setChecked(True)
        self.now = QLabel(self._sides())
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        lay = QVBoxLayout(self)
        for w in (self.box, self.use, self.now, close):
            lay.addWidget(w)
        self.box.paired.connect(self._paired)

    def _sides(self, s: config.Settings | None = None) -> str:
        s = s or config.load()
        on = [k.replace('_to_', ' → ') for k in languages.SIDES if getattr(s, k) == 'remote']
        return f'On the server now: {", ".join(on)}.' if on else 'No direction runs on the server now (tray menu: remote).'

    def _paired(self, _p: dict) -> None:
        if not self.use.isChecked():
            return
        s = config.load()
        s.them_to_me = s.me_to_them = 'remote'
        config.save(s)
        self.now.setText(self._sides(s))
        self.tray.sync()
        self.tray.restart()

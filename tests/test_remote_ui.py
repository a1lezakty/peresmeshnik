"""The "Server…" window (peresmeshnik/ui/server.py): a bad line is said at once, a pairing runs off the UI thread and
its answer lands in the window, both directions go to the server after it, a refusal is shown, Forget forgets."""
import os
import sys
import time

import pytest

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from peresmeshnik import config  # noqa: E402
from peresmeshnik.remote import pairing  # noqa: E402
from peresmeshnik.remote.protocol import ProtocolError  # noqa: E402

app = QApplication.instance() or QApplication([])
URL = 'https://192.168.1.20:47000/#K7QM-2XPA-4DFE'


def until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, 'timed out'
        QCoreApplication.processEvents()
        time.sleep(0.01)


class FakeTray:
    synced = restarted = 0

    def sync(self):
        self.synced += 1

    def restart(self):
        self.restarted += 1


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setenv('PERESMESHNIK_CONFIG', str(tmp_path / 'settings.toml'))
    store = {}
    monkeypatch.setattr(pairing, 'load', lambda path=None: store.get('p'))
    monkeypatch.setattr(pairing, 'forget', lambda path=None: store.pop('p', None))
    return store


def test_pair_then_both_directions_on_the_server(saved, monkeypatch):
    from peresmeshnik.ui.server import ServerDialog
    calls = []

    def pair(url, name, path=None):
        calls.append((url, name))
        time.sleep(0.2)  # the network: the window stays responsive
        saved['p'] = {'address': '192.168.1.20:47000', 'fp': 'ab' * 32, 'token': 't', 'name': name}
        return saved['p']
    monkeypatch.setattr(pairing, 'pair', pair)
    tray = FakeTray()
    d = ServerDialog(tray)
    box = d.box
    assert box.state.text() == 'Not paired with a server.' and not box.forget_button.isEnabled()
    box.url.setText('192.168.1.20')  # no code
    box.pair()
    assert 'no pairing code' in box.result.text() and calls == []
    box.url.setText(URL)
    box.name.setText('laptop')
    box.pair()
    assert box.result.text() == 'Pairing…' and not box.pair_button.isEnabled()  # returned before the network answered
    until(lambda: box.result.text() == '✔ Paired.')
    assert calls == [(URL, 'laptop')] and '192.168.1.20:47000' in box.state.text() and box.url.text() == ''
    s = config.load()
    assert (s.them_to_me, s.me_to_them) == ('remote', 'remote') and tray.restarted == 1
    assert 'them → me, me → them' in d.now.text()
    box.forget()
    assert saved == {} and box.state.text() == 'Not paired with a server.'


def test_a_refusal_is_shown_and_nothing_changes(saved, monkeypatch):
    from peresmeshnik.ui.server import ServerDialog

    def pair(url, name, path=None):
        raise ProtocolError('code', 'the code was used or has expired')
    monkeypatch.setattr(pairing, 'pair', pair)
    d = ServerDialog(FakeTray())
    d.box.url.setText(URL)
    d.box.pair()
    until(lambda: d.box.result.text().startswith('⚠'))
    assert 'expired' in d.box.result.text() and d.box.pair_button.isEnabled()
    assert config.load().them_to_me == 'local'


def test_wizard_pairs_a_side_on_the_server(saved, tmp_path, monkeypatch):
    from peresmeshnik import languages
    from peresmeshnik.ui import wizard
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))
    s = config.Settings(them_to_me='remote', me_to_them='remote')
    config.save(s)
    page = wizard.LanguagesPage(type('W', (), {'s': s, 'key': ''})())
    assert not page.server.isHidden()
    for key in languages.SIDES:
        assert 'On your own server: pair it below' in page.why[key].text()
    assert 'you hear the call as it is' in page.why['them_to_me'].text()
    assert 'the call hears your own voice' in page.why['me_to_them'].text()
    assert not page.validatePage() and 'Pair with your server first' in page.err.text()
    saved['p'] = {'address': '192.168.1.20:47000', 'fp': 'ab' * 32, 'token': 't', 'name': 'laptop'}
    page.server.paired.emit(saved['p'])
    assert 'On your own server at 192.168.1.20:47000' in page.why['me_to_them'].text()
    assert page.validatePage() and (page.w.s.them_to_me, page.w.s.me_to_them) == ('remote', 'remote')
    assert languages.needed_models(page.chosen()) == ['silero-vad']  # nothing of the pair is downloaded here
    page.engine['them_to_me'].setCurrentText('local')
    page.engine['me_to_them'].setCurrentText('cloud')
    assert page.server.isHidden()


def test_tray_remote_without_a_pairing_opens_the_server_window(saved):
    from PySide6.QtGui import QAction

    from peresmeshnik.ui.tray import Tray
    calls = []
    fake = type('T', (), {'sync': lambda self: calls.append('sync'), 'open_server': lambda self: calls.append('server'),
                          'switch_pair': lambda self, *a: calls.append('switch'), '_pair': None})()
    a = QAction('remote')
    a.setData(('me_to_them', 'remote'))
    Tray._engine_chosen(fake, a)
    assert calls == ['sync', 'server']
    saved['p'] = {'address': 'x:1', 'fp': 'f', 'token': 't'}
    Tray._engine_chosen(fake, a)  # paired: as any engine
    assert calls[-1] == 'switch'

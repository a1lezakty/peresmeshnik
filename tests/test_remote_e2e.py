"""The app's whole path to its own server (package 8.1): this computer paired as the tray's Server… window does, both
sides (app.Side -> cloud/fallback -> remote/client) against the real `peresmeshnik serve` (remote/server.py) with stub
engines (tests/remote_stubs.py, no models), and what the user meets when the server goes: it stops and comes back on the
same port, the device is unpaired during a call, the server's side dies and loads again for longer than a reconnect waits.
The server's own end (pairing, hello, the voice, warm, revoke) is in test_remote_server.py; the client against a fake
server (faults the real one cannot make) in test_remote_client.py and test_remote_app.py."""
import time
from pathlib import Path

import numpy as np
import pytest
from remote_stubs import profile, start
from test_remote_client import until
from test_ui import FakeIn, FakeOut

from peresmeshnik import app, config, voice
from peresmeshnik.cloud import fallback
from peresmeshnik.engines import Caption, Speech
from peresmeshnik.remote import client, pairing, server

SIDES = ('them', 'me')
FRAME = np.zeros(512, np.float32)


class Net:
    """The server (stub engines, its folder: certificate, paired.json) and this computer paired with it."""

    def __init__(self, folder: Path, paired: Path):
        self.folder = folder
        self.srv = start(folder, warm_s=0.5)
        pairing.pair(server.url('127.0.0.1', self.srv.ports[0], self.srv.pairing.new_code()), 'laptop', paired)

    def stop(self):
        self.srv.stop()
        self.srv.thread.join(10)

    def again(self):
        """`peresmeshnik serve` started again: the same folder and port."""
        self.srv = start(self.folder, self.srv.ports[0], warm_s=0.5)


@pytest.fixture
def net(tmp_path, made, monkeypatch):
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path / 'models'))  # no local models here: the live voice when down
    monkeypatch.setattr(server, 'REVOKE_S', 0.1)
    monkeypatch.setattr(pairing, 'remote_path', lambda: tmp_path / 'remote')
    monkeypatch.setattr(voice, 'voices_dir', lambda: tmp_path / 'voices')
    n = Net(tmp_path / 'server', tmp_path / 'remote')
    yield n
    n.stop()


class Call:
    """Both sides on the server, as the app runs them: my confirmed voice goes up with "me -> them"."""

    def __init__(self, tmp_path, sides=SIDES):
        s = config.Settings()
        s.them_to_me = s.me_to_them = 'remote'
        s.remote.retry_s = 0.3
        (tmp_path / 'voices').mkdir(exist_ok=True)
        profile(tmp_path / 'voices' / 'mine', 1000)
        s.local.voice = 'mine'
        self.states, self.side, self.out, self.inp, self.shown = [], {}, {}, {}, {}
        for name in sides:
            self.inp[name], self.out[name], self.shown[name] = FakeIn(), FakeOut(), []
            overlay = type('O', (), {'show': lambda _, c, shown=self.shown[name]: shown.append(c)})()
            self.side[name] = app.Side(name, s, self.inp[name], self.out[name], overlay, on_error=self.states.append)
            self.side[name].start()
        for side in self.side.values():
            assert self.fb(side).cloud.ready.wait(10), self.states
        assert not any(side.auto for side in self.side.values())

    @staticmethod
    def fb(side):
        d = side.directions[0][0]
        return getattr(d, 'inner', d)  # me -> them: in its LiveSwitch

    def down(self):
        until(lambda: all(side.auto for side in self.side.values()), 10, lambda: self.states)

    def back(self):
        """The frames keep coming (the call goes on): each side finds the server again by itself."""
        t = time.monotonic()
        while any(side.auto for side in self.side.values()):
            for inp in self.inp.values():
                inp.cb(FRAME)
            time.sleep(0.03)
            assert time.monotonic() - t < 10, self.states
        assert all(self.fb(side).mode == 'cloud' for side in self.side.values())
        assert sum('back on the server' in x for x in self.states) == len(self.side), self.states

    def heard(self, made):
        """The call reaches the server's "them -> me", and its subtitle and voice come back."""
        them = [f for f in made if f.side == 'them'][-1]
        n = len(them.audio)
        self.inp['them'].cb(np.full(512, 0.1, np.float32))
        until(lambda: len(them.audio) > n)
        them.say(Caption('hello', 'привет', True, 0), Speech(np.full(4800, 0.3, np.float32), 24_000))
        until(lambda: Caption('hello', 'привет', True, 0) in self.shown['them']
              and sum(n for n, sr in self.out['them'].played if sr == 24_000) >= 4800)

    def stop(self):
        for side in self.side.values():
            side.stop()


def test_a_call_through_the_server(net, made, tmp_path):
    c = Call(tmp_path)
    try:
        them, me = ([f for f in made if f.side == name][0] for name in SIDES)
        assert (them.src, them.tgt, me.src, me.tgt) == ('en', 'ru', 'ru', 'en')
        assert me.voice_dir and (Path(me.voice_dir) / 'state.npy').stat().st_size == 1000  # my confirmed voice speaks
        assert them.voice_dir == ''
        c.heard(made)
        c.inp['me'].cb(np.full(512, 0.2, np.float32))  # my microphone, up to "me -> them"
        until(lambda: any(float(x[0]) == pytest.approx(0.2, abs=1e-3) for x in me.audio))
        assert not [x for x in c.states if x.startswith('error')], c.states
    finally:
        c.stop()


def test_the_server_stops_and_comes_back(net, made, tmp_path):
    c = Call(tmp_path)
    try:
        fp = net.srv.fp
        net.stop()
        c.down()  # the call as it is in my headphones, my own voice to the call
        assert sum('until the server answers again' in x for x in c.states if x.startswith('error')) == 2, c.states
        net.again()
        assert net.srv.fp == fp  # its certificate kept: the pairing still holds
        c.back()
        c.heard(made)
    finally:
        c.stop()


def test_unpaired_during_a_call(net, made, tmp_path):
    c = Call(tmp_path)
    try:
        assert pairing.Pairing(net.folder).revoke('laptop')  # `serve --revoke laptop`, another process
        c.down()
        n = len(made)
        for _ in range(20):  # 0.6 s of the call: tries at 0.3 s doubling, were it not for good
            for inp in c.inp.values():
                inp.cb(FRAME)
            time.sleep(0.03)
        assert len(made) == n and all(c.fb(side).probe is None for side in c.side.values())
        said = [x for x in c.states if x.startswith('error') and 'pair it again (tray menu, Server…)' in x
                and fallback.NO_MORE in x]
        assert len(said) == 2, c.states
    finally:
        c.stop()


def test_the_servers_side_dies_and_loads_longer_than_a_reconnect_waits(net, made, tmp_path, monkeypatch):
    """RS M5: after 'internal' the one new connection waits RECONNECT_READY_S for `ready`; the side's engines load longer
    (cold models): the call as it is meanwhile, then back on the server by the usual tries (READY_S each)."""
    c = Call(tmp_path, ('them',))
    try:
        monkeypatch.setattr(client, 'RECONNECT_READY_S', 0.5)
        stub = app.Side._local
        monkeypatch.setattr(app.Side, '_local', lambda side, s: time.sleep(1.5) or stub(side, s))
        made[0].q.put(None)  # its engines' events end: the server says 'internal', the side goes
        c.down()
        assert [x for x in c.states if x.startswith('error') and 'was not ready in' in x], c.states
        c.back()
        c.heard(made)
    finally:
        c.stop()

"""`peresmeshnik serve` (peresmeshnik/remote/server.py) on 127.0.0.1 with stub engines (no models) behind the app's own
Side, against the app's real client (remote/client.RemoteDirection) and raw WebSocket clients: pairing, then a side end
to end (audio up; subtitles, latency and voice down in pieces; barge-in), hello / version / auth / busy, the voice (zip,
sha256, its 0700 folder and its removal), the warm session, engines that die, logs without text, the command, headless."""
import asyncio
import json
import logging
import os
import queue
import ssl
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from websockets.sync.client import connect

from peresmeshnik import __version__, app, config, voice
from peresmeshnik.engines import Caption, Latency, Speech
from peresmeshnik.remote import client, pairing, protocol, server
from peresmeshnik.remote.client import RemoteDirection, ServerError

FRAME = 512


class Fake:
    """A side's local direction without models: records what it hears, says what the test tells it."""

    def __init__(self, side, s):
        self.side, self.src, self.tgt = side.name, side.src, side.tgt
        self.voice_dir, self.clone = s.local.voice, s.local.clone
        self.audio, self.interrupts, self.closed = [], 0, False
        self.q = queue.Queue()

    def feed(self, pcm):
        self.audio.append(pcm)

    def events(self):
        while (e := self.q.get()) is not None:
            yield e

    def interrupt(self):
        self.interrupts += 1

    def close(self):
        self.closed = True
        self.q.put(None)

    def say(self, *events):
        for e in events:
            self.q.put(e)


@pytest.fixture
def made(monkeypatch):
    """Every Fake the server's sides made, in order (app.Side._local, so ServerSide's own code runs as in the app)."""
    out = []

    def local(side, s):
        out.append(Fake(side, s))
        return out[-1]
    monkeypatch.setattr(app.Side, '_local', local)
    return out


def start(folder, **kw):
    srv = server.Server(folder, config.Settings(), '127.0.0.1', 0, **kw)
    srv.thread = threading.Thread(target=asyncio.run, args=(srv.serve(),), daemon=True)
    srv.thread.start()
    assert srv.started.wait(10)
    srv.address = f'127.0.0.1:{srv.ports[0]}'
    return srv


@pytest.fixture
def srv(tmp_path, made):
    s = start(tmp_path / 'server', warm_s=0.5)
    yield s
    s.stop()
    s.thread.join(10)


def until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, 'timed out'
        time.sleep(0.01)


class Events:
    """The client direction's events on a thread, as the app's side consumes them."""

    def __init__(self, d):
        self.q, self.error = queue.Queue(), None
        threading.Thread(target=self._run, args=(d,), daemon=True).start()

    def _run(self, d):
        try:
            for e in d.events():
                self.q.put(e)
        except Exception as e:
            self.error = e
        self.q.put(None)

    def get(self, timeout=5.0):
        return self.q.get(timeout=timeout)


def raw(srv, side='them'):
    """A bare client for what the app's never sends."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    return connect(f'wss://{srv.address}{protocol.side_path(side)}', ssl=ctx, max_size=protocol.CHUNK, open_timeout=5)


def answer(ws) -> dict:
    return json.loads(ws.recv(timeout=10))


def hello(token, side='them', src='en', tgt='ru', **kw):
    return json.dumps({**json.loads(protocol.hello(token, side, src, tgt)), **kw})


def frame(v):
    return np.full(FRAME, v, np.float32)


def profile(folder, size=int(2.5 * protocol.CHUNK)):
    folder.mkdir()
    (folder / 'voice.json').write_text('{"kind": "me", "name": "me"}')
    (folder / 'state.bin').write_bytes(os.urandom(size))
    return protocol.pack_voice(folder)


def test_pair_then_a_side_end_to_end(srv, made, tmp_path):
    """The main path: pairing with the line `serve` shows, then them -> me through the app's client and Side."""
    code = srv.pairing.new_code()
    paired = pairing.pair(server.url('127.0.0.1', srv.ports[0], code), 'laptop', tmp_path / 'remote')
    assert paired['fp'] == srv.fp and [d['name'] for d in srv.pairing.devices()] == ['laptop']
    d = RemoteDirection(paired['address'], 'them', 'en', 'ru', paired['token'], paired['fp'])
    ev = Events(d)
    assert d.ready.wait(10), ev.error
    assert (d.server['warm'], d.server['device'], d.server['voice']) == (False, 'cpu', 'stock')
    [f] = made
    assert (f.side, f.src, f.tgt) == ('them', 'en', 'ru')
    side = srv.slots['them'].engines
    assert isinstance(side, server.ServerSide) and side.echo is None

    for i in range(10):  # up: int16 frames, as they come
        d.feed(frame(i / 100))
    until(lambda: len(f.audio) == 10)
    assert [round(float(x[0]), 3) for x in f.audio] == [i / 100 for i in range(10)]

    pcm = np.full(int(24_000 * 0.25), 0.3, np.float32)  # down: what the side's engines say
    f.say(Caption('hel'), Caption('hello', 'привет', True, 0), Latency(0, 'voice', 0.3), Speech(pcm, 24_000))
    assert ev.get() == Caption('hel')
    assert ev.get() == Caption('hello', 'привет', True, 0)
    lat = ev.get()
    assert lat.what == 'voice' and 0.3 <= lat.s < 0.35
    hold = 24_000 * client.JITTER_MS // 1000
    assert [len(ev.get().pcm) for _ in range(3)] == [hold + 2400, 2400, 1200]  # SPEECH_MS pieces

    d.interrupt()  # barge-in: the server's pipeline is told, and it answers
    until(lambda: f.interrupts == 1)
    until(lambda: not d._hush)  # 'interrupted' came: the client plays again
    f.say(Speech(np.full(2400, 0.5, np.float32), 24_000))
    assert np.allclose(ev.get().pcm[-2400:], 0.5, atol=1e-4)

    states = []  # the server side's own errors and notes: a tray line on the client
    d.on_state = states.append
    side.on_error('error: them → me: something')
    until(lambda: states == ['error: server: them → me: something'])
    d.close()
    until(lambda: srv.slots['them'].conn is None)
    until(lambda: f.closed, 3)  # warm_s 0.5: then the engines go
    assert not srv.slots


def test_hello_version_auth_lang(srv, made):
    token = srv.pairing.add('laptop')
    for msg, code in [('{"type": "pair"}', 'hello'), (hello(token, version=0), 'version'),
                      (hello('cd' * 32, version=0), 'version'),  # an old client hears 'version', not 'auth'
                      (hello('cd' * 32), 'auth'), (hello(token, side='me'), 'hello'),
                      (hello(token, src='en', tgt='xx'), 'lang')]:
        with raw(srv) as ws:
            ws.send(msg)
            a = answer(ws)
            assert (a['type'], a['code']) == ('error', code), (msg, a)
    with raw(srv) as ws:  # a pair this server cannot run: no stock voice for Chinese
        ws.send(hello(token, src='en', tgt='zh'))
        assert answer(ws)['code'] == 'lang'
    d = RemoteDirection(srv.address, 'them', 'en', 'ru', 'cd' * 32, srv.fp)
    ev = Events(d)
    assert ev.get() is None and isinstance(ev.error, ServerError) and ev.error.permanent and ev.error.code == 'auth'
    assert not made


def test_one_computer_at_a_time(srv, made):
    a, b = srv.pairing.add('laptop'), srv.pairing.add('other')
    them = RemoteDirection(srv.address, 'them', 'en', 'ru', a, srv.fp)
    me = RemoteDirection(srv.address, 'me', 'ru', 'en', a, srv.fp)  # both sides of one computer: fine
    assert them.ready.wait(10) and me.ready.wait(10)
    with raw(srv, 'me') as ws:
        ws.send(hello(b, 'me', 'ru', 'en'))
        a_ = answer(ws)
        assert (a_['code'], a_['code'] in protocol.PERMANENT) == ('busy', False)
    them.close()
    with raw(srv) as ws:  # one side still on
        ws.send(hello(b))
        assert answer(ws)['code'] == 'busy'
    me.close()
    until(lambda: all(s.conn is None for s in srv.slots.values()))
    with raw(srv) as ws:  # all off: the other computer takes the server, the first one's warm engines go
        ws.send(hello(b))
        assert answer(ws)['type'] == 'ready'
        until(lambda: made[0].closed and made[1].closed)
        assert srv.slots['them'].device == 'other' and 'me' not in srv.slots


def test_the_voice_its_folder_and_its_side(srv, made, tmp_path):
    token = srv.pairing.add('laptop')
    mine, theirs = profile(tmp_path / 'mine'), profile(tmp_path / 'theirs', 1000)
    me = RemoteDirection(srv.address, 'me', 'ru', 'en', token, srv.fp, voice=mine)
    them = RemoteDirection(srv.address, 'them', 'en', 'ru', token, srv.fp, voice=theirs)
    assert me.ready.wait(10) and them.ready.wait(10)
    f = {x.side: x for x in made}
    folder = srv.slots['me'].folder
    assert f['me'].voice_dir == str(folder) and folder.parent == srv.voices  # me -> them: only the voice of its own hello
    assert (folder / 'state.bin').read_bytes() == (tmp_path / 'mine' / 'state.bin').read_bytes()
    assert voice.load(f['me'].voice_dir).meta['kind'] == 'me'  # the app's loader takes the folder as it is
    assert f['them'].voice_dir == '' and srv.slots['them'].folder != folder  # the contact's voice stays on 'them'
    assert isinstance(srv.slots['me'].engines.directions[0][0], server._Unswitched)  # no LiveSwitch here: the client's
    if os.name == 'posix':
        assert folder.stat().st_mode & 0o777 == 0o700 and srv.voices.stat().st_mode & 0o777 == 0o700
    me.close()
    them.close()
    until(lambda: not srv.slots and not any(srv.voices.iterdir()), 5)  # warm_s 0.5, then the folders go

    me = RemoteDirection(srv.address, 'me', 'ru', 'en', token, srv.fp)  # no voice: the server's stock one
    assert me.ready.wait(10) and me.server['voice'] == 'stock'
    assert (made[-1].voice_dir, made[-1].clone) == ('', 'piper')
    me.close()


def test_a_damaged_voice_and_leftovers(tmp_path, made):
    stale = tmp_path / 'server' / 'voices' / 'me-crashed'
    stale.mkdir(parents=True)
    (stale / 'voice.json').write_text('{}')
    srv = start(tmp_path / 'server')
    try:
        assert not stale.exists()  # a crashed server's voice goes at the start
        token = srv.pairing.add('laptop')
        data = profile(tmp_path / 'v', 1000)
        for sent, ref in [(data[:-1], protocol.voice_ref(data)), (b'x' * len(data), protocol.voice_ref(data)),
                          (data, {'sha256': 'ab' * 32, 'bytes': len(data)})]:
            with raw(srv, 'me') as ws:
                ws.send(hello(token, 'me', 'ru', 'en', voice=ref))
                assert answer(ws)['type'] == 'send_voice'
                ws.send(sent)
                if len(sent) < ref['bytes']:
                    ws.send(b'\1')  # its last byte is 0
                a = answer(ws)
                assert (a['type'], a['code']) == ('error', 'voice'), a
        not_zip = b'not a zip'
        with raw(srv, 'me') as ws:
            ws.send(hello(token, 'me', 'ru', 'en', voice=protocol.voice_ref(not_zip)))
            answer(ws)
            ws.send(not_zip)
            assert answer(ws)['code'] == 'voice'
        until(lambda: not any(srv.voices.iterdir()))
        assert not made and not srv.slots
    finally:
        srv.stop()
        srv.thread.join(10)


def test_warm_session(tmp_path, made):
    srv = start(tmp_path / 'server', warm_s=30)
    try:
        token = srv.pairing.add('laptop')
        v = profile(tmp_path / 'v', 1000)
        d = RemoteDirection(srv.address, 'me', 'ru', 'en', token, srv.fp, voice=v)
        assert d.ready.wait(10) and d.server['warm'] is False
        d.close()
        until(lambda: srv.slots['me'].conn is None)
        f = made[0]
        until(lambda: len(f.audio) == 1 and f.interrupts == 1)  # a pause closes the cut clause, its voice is dropped
        assert not f.audio[0].any()
        d = RemoteDirection(srv.address, 'me', 'ru', 'en', token, srv.fp, voice=v)  # the same: no upload, no loading
        assert d.ready.wait(10) and d.server['warm'] is True and len(made) == 1
        d.feed(frame(0.25))
        until(lambda: len(f.audio) == 2)
        f.say(Caption('a', 'б', True, 0))
        ev = Events(d)
        assert ev.get() == Caption('a', 'б', True, 0)
        d2 = RemoteDirection(srv.address, 'me', 'ru', 'en', token, srv.fp, voice=v)  # a new connection of the same side
        assert d2.ready.wait(10) and len(made) == 1  # takes it over; the old one is closed
        d2.close()
        d.close()
        until(lambda: srv.slots['me'].conn is None)
        d = RemoteDirection(srv.address, 'me', 'ru', 'de', token, srv.fp, voice=v)  # other languages: new engines
        assert d.ready.wait(10) and d.server['warm'] is False and len(made) == 2 and made[1].tgt == 'de'
        until(lambda: f.closed)
        d.close()
    finally:
        srv.stop()
        srv.thread.join(10)
    assert made[1].closed and not any(srv.voices.iterdir())  # the server stops: the warm ones go too


def test_engines_that_die(srv, made):
    token = srv.pairing.add('laptop')
    with raw(srv) as ws:
        ws.send(hello(token))
        assert answer(ws)['type'] == 'ready'
        made[0].q.put(None)  # its events ended by themselves: Side.on_failure
        a = answer(ws)
        assert (a['type'], a['code']) == ('error', 'internal') and 'ended' in a['text']
    until(lambda: not srv.slots)
    with raw(srv) as ws:  # not kept warm: new ones
        ws.send(hello(token))
        assert answer(ws) == {'type': 'ready', 'warm': False, 'device': 'cpu', 'voice': 'stock', 'app': __version__}
    assert len(made) == 2


@pytest.mark.parametrize('log_text', [False, True])
def test_logs_keep_numbers_only(tmp_path, made, caplog, log_text):
    srv = start(tmp_path / 'server', log_text=log_text)
    try:
        caplog.set_level(logging.INFO, logger='peresmeshnik.latency')
        logging.getLogger('peresmeshnik.latency').info(json.dumps(
            {'clause': 0, 'text': 'my secret words', 'translation': 'мои тайные слова', 'tts_ms': 120}, ensure_ascii=False))
        assert 'tts_ms' in caplog.text
        assert ('secret' in caplog.text and 'тайные' in caplog.text) is log_text
        assert ('secret' in caplog.text or 'тайные' in caplog.text) is log_text
    finally:
        srv.stop()
        srv.thread.join(10)
    assert not logging.getLogger('peresmeshnik.latency').filters


def test_the_command(tmp_path, monkeypatch, capsys):
    from peresmeshnik.__main__ import main
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(config, 'load', lambda: config.Settings())
    pairing.Pairing(tmp_path / 'server').add('laptop')
    assert main(['serve', '--list']) == 0 and capsys.readouterr().out.startswith('laptop\tpaired ')
    assert main(['serve', '--revoke', 'laptop']) == 0 and main(['serve', '--revoke', 'laptop']) == 1
    assert main(['serve', '--host', '8.8.8.8']) == 2 and 'not a private address' in capsys.readouterr().err
    assert server.url('fd00::5', 47000, 'K7QM') == 'https://[fd00::5]:47000/#K7QM'


def test_headless():
    """`serve` loads neither Qt nor PortAudio (a server without a display or a sound card)."""
    code = ('import sys, peresmeshnik.remote.server, peresmeshnik.pipeline, peresmeshnik.audio.mic; '
            'print(sorted(m for m in ("PySide6", "sounddevice") if m in sys.modules))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == '[]'

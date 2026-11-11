"""`peresmeshnik serve` (peresmeshnik/remote/server.py) on 127.0.0.1 with stub engines (no models) behind the app's own
Side, against the app's real client (remote/client.RemoteDirection) and raw WebSocket clients: pairing, then a side end
to end (audio up; subtitles, latency and voice down in pieces; barge-in), hello / version / auth / busy, the voice (zip,
sha256, its 0700 folder and its removal), the warm session, engines that die, logs without text, the command, headless."""
import asyncio
import contextlib
import json
import logging
import os
import queue
import re
import socket
import ssl
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from remote_stubs import CONFIRMED, Fake, profile, start
from websockets.sync.client import connect

from peresmeshnik import __version__, app, config, voice
from peresmeshnik.engines import Caption, Latency, Speech
from peresmeshnik.remote import client, pairing, protocol, server
from peresmeshnik.remote.client import RemoteDirection, ServerError

FRAME = 512


@pytest.fixture(autouse=True)
def clone_here(monkeypatch):
    """The server's GPU can speak my cloned voice (CI has none): it asks for the voice. A test says otherwise itself."""
    from peresmeshnik import languages
    monkeypatch.setattr(languages, 'clone_why', lambda *a, **k: '')


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


def raw(srv, side='them', path='', addr='', **kw):
    """A bare client for what the app's never sends."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    # ponytail: TLS 1.2 here: this sync client over TLS 1.3 got no handshake answer in ~1 of 5 runs of this file (never
    # alone, never the app's asyncio client in hundreds of connections); look into it if a real client ever hangs so
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    return connect(f'wss://{addr or srv.address}{path or protocol.side_path(side)}', ssl=ctx, max_size=protocol.CHUNK,
                   open_timeout=5, **kw)


def answer(ws) -> dict:
    return json.loads(ws.recv(timeout=10))


def hello(token, side='them', src='en', tgt='ru', **kw):
    return json.dumps({**json.loads(protocol.hello(token, side, src, tgt)), **kw})


def frame(v):
    return np.full(FRAME, v, np.float32)


def test_pair_then_a_side_end_to_end(srv, made, tmp_path):
    """The main path: pairing with the line `serve` shows, then them -> me through the app's client and Side."""
    code = srv.pairing.new_code()
    paired = pairing.pair(server.url('127.0.0.1', srv.ports[0], code), 'laptop', tmp_path / 'remote')
    assert paired['fp'] == srv.fp == srv.pairing.fp != '' and [d['name'] for d in srv.pairing.devices()] == ['laptop']
    d = RemoteDirection(paired['address'], 'them', 'en', 'ru', paired['token'], paired['fp'])
    ev = Events(d)
    assert d.ready.wait(10), ev.error
    assert (d.server['warm'], d.server['device'], d.server['voice']) == (False, 'cpu', 'stock')
    [f] = made
    assert (f.side, f.src, f.tgt) == ('them', 'en', 'ru')
    side = srv.slots['them'].engines
    assert isinstance(side, server.ServerSide) and side.echo is None
    sock = srv.slots['them'].conn.ws.transport.get_extra_info('socket')  # small messages not held back (BENCH remote-0)
    assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)

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
    assert (folder / 'reference.wav').read_bytes() == (tmp_path / 'mine' / 'reference.wav').read_bytes()
    assert sorted(x.name for x in folder.iterdir()) == ['reference.wav', 'voice.json']  # no cache from the laptop
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
                          (data, {'sha256': 'ab' * 32, 'bytes': len(data)}),
                          (data + b'\0', protocol.voice_ref(data))]:  # never more than hello said
            with raw(srv, 'me') as ws:
                ws.send(hello(token, 'me', 'ru', 'en', voice=ref))
                assert answer(ws)['type'] == 'send_voice'
                ws.send(sent)
                if len(sent) < ref['bytes']:
                    ws.send(b'\1')  # its last byte is 0
                a = answer(ws)
                assert (a['type'], a['code']) == ('error', 'voice'), a
                assert ('more' in a['text']) is (len(sent) > ref['bytes'])  # refused before the extra is kept
        bad_ones = [b'not a zip']  # a zip whose voice.json does not read, or is not an object: refused once (not 'internal',
        for i, bad in enumerate(['{not json', '[]', '"x"', '{"version": "2"}',  # retried forever)
                                 '{"version": 2, "kind": "me", "consent": "x"}', '[' * 100_000]):
            (tmp_path / f'bad{i}').mkdir()
            (tmp_path / f'bad{i}' / 'voice.json').write_text(bad)
            bad_ones.append(protocol.pack_voice(tmp_path / f'bad{i}'))
        for sent in bad_ones:
            with raw(srv, 'me') as ws:
                ws.send(hello(token, 'me', 'ru', 'en', voice=protocol.voice_ref(sent)))
                answer(ws)
                ws.send(sent)
                assert answer(ws)['code'] == 'voice'
        until(lambda: not any(srv.voices.iterdir()))
        assert not made and not srv.slots
    finally:
        srv.stop()
        srv.thread.join(10)


def test_a_new_hello_while_the_voice_unpacks(srv, made, tmp_path, monkeypatch):
    """The side's new connection comes while the old one's voice is unpacked: the old session goes, its folder too."""
    gate, inside, done, unpack = threading.Event(), threading.Event(), threading.Event(), server.Server._unpack

    def slow(data, folder, side):
        inside.set()
        gate.wait(5)
        unpack(data, folder, side)  # into the folder its _drop removed: made again
        done.set()
    monkeypatch.setattr(server.Server, '_unpack', staticmethod(slow))
    token = srv.pairing.add('laptop')
    data = profile(tmp_path / 'v', 1000)
    with raw(srv, 'me') as old, raw(srv, 'me') as new:
        old.send(hello(token, 'me', 'ru', 'en', voice=protocol.voice_ref(data)))
        assert answer(old)['type'] == 'send_voice'
        old.send(data)
        assert inside.wait(5)
        new.send(hello(token, 'me', 'ru', 'en'))
        assert answer(new)['type'] == 'ready'
        gate.set()
        assert done.wait(5)
        until(lambda: not any(srv.voices.iterdir()))
        assert srv.slots['me'].folder is None and len(made) == 1


def test_the_clone_falls_back_at_the_start(srv, monkeypatch):
    """The clone cannot speak as the side starts (no voice pack, out of VRAM, too slow): a stock voice speaks, `ready`
    says so and the side's note comes after it; with no stock voice either (zh, ar): 'lang', not tried again."""
    made, nostock = [], []
    note = 'note: me → them: a stock voice speaks instead of your cloned voice (no voice pack)'

    def fallback(side, s, why):
        if nostock:
            raise RuntimeError(f'your cloned voice cannot speak ({why}) and there is no stock voice')
        side.on_error(note)
        made.append(Fake(side, s))
        return made[-1]
    monkeypatch.setattr(app.Side, '_local', lambda side, s: side._fallback(s, 'no voice pack'))
    monkeypatch.setattr(app.Side, '_fallback', fallback)
    token = srv.pairing.add('laptop')
    with raw(srv, 'me') as ws:
        ws.send(hello(token, 'me', 'ru', 'en'))
        a = answer(ws)
        assert (a['type'], a['voice']) == ('ready', 'stock') and srv.slots['me'].engines.stock
        notes = srv.slots['me'].notes
        assert notes[-1] == note and [answer(ws)['text'] for _ in notes] == list(notes)
    until(lambda: not srv.slots)
    nostock.append(1)
    with raw(srv, 'me') as ws:
        ws.send(hello(token, 'me', 'ru', 'de'))
        a = answer(ws)
        assert (a['type'], a['code']) == ('error', 'lang') and 'no stock voice' in a['text']
    assert len(made) == 1


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


def test_a_warm_side_forgets_their_voices_when_the_client_leaves(tmp_path, made):
    """FIXES-s2 r2 2: the next call on warm engines may be someone else: the people of the last one, their speech and
    voices go when its client leaves (the engine stays loaded); the next hello starts with nobody."""
    from test_live import AmpSpeaker, FakeClone, FakeStock, speech

    from peresmeshnik.voices import live
    srv = start(tmp_path / 'server', warm_s=30)
    try:
        token = srv.pairing.add('laptop')
        d = RemoteDirection(srv.address, 'them', 'en', 'ru', token, srv.fp)
        assert d.ready.wait(10)
        lv = live.LiveVoice(FakeClone, AmpSpeaker, 'pocket-tts-ru', 'en')
        made[0].tts = live.TheirVoice(FakeStock(), lv)
        for i in range(3):
            lv.offer(speech(1.2, 0.3, i), 'a', 0, 1.2)
        until(lambda: lv.speaking is not None and lv.speaking.voice is not None)
        d.close()
        until(lambda: srv.slots['them'].conn is None)
        until(lambda: lv.people == [] and lv.speaking is None)
        assert lv.engine is not None  # warm: loaded still
        d = RemoteDirection(srv.address, 'them', 'en', 'ru', token, srv.fp)
        assert d.ready.wait(10) and d.server['warm'] is True and len(made) == 1 and lv.people == []
        d.close()
        lv.close()
    finally:
        srv.stop()
        srv.thread.join(10)


@pytest.mark.parametrize('noticed', [True, False])
def test_a_drop_reconnects_to_the_warm_side(srv, made, noticed, monkeypatch):
    """The client's one new connection after a drop gets the same engines; the clause the drop cut is closed by a pause
    (its voice not said) and comes again from the client. `noticed`: the server saw the old connection go (a reset);
    else (Wi-Fi gone) the new one takes over before it does."""
    monkeypatch.setattr(protocol, 'PING_S', 0.1)  # the client's
    monkeypatch.setattr(protocol, 'PONG_S', 0.3)
    token = srv.pairing.add('laptop')
    d = RemoteDirection(srv.address, 'them', 'en', 'ru', token, srv.fp)
    ev = Events(d)
    assert d.ready.wait(10)
    f = made[0]
    for _ in range(5):
        d.feed(frame(0.1))
        time.sleep(0.02)
    until(lambda: len(f.audio) == 5)
    f.say(Caption('on', clause=0))  # the clause the drop cuts, a partial of it heard
    assert ev.get() == Caption('on', clause=0)
    old = srv.slots['them'].conn
    if noticed:
        srv._loop.call_soon_threadsafe(old.ws.transport.abort)
    else:
        srv._loop.call_soon_threadsafe(old.ws.transport.pause_reading)  # no pongs: the client gives it up, not the server
    until(lambda: srv.slots['them'].conn not in (None, old) and len(f.audio) >= 11, 8)
    assert not f.audio[5].any() and f.interrupts == 1  # the pause, the cut clause's voice dropped
    assert [round(float(x[0]), 3) for x in f.audio[6:11]] == [0.1] * 5  # the cut clause again, whole
    assert len(made) == 1
    f.say(Caption('on', 'о', True, 0), Latency(0, 'caption', 0.1), Caption('one', 'один', True, 1))  # the cut clause's
    assert ev.get() == Caption('one', 'один', True, 1) and ev.error is None  # final goes nowhere: it comes again whole
    d.close()


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


def test_a_second_server_on_the_port(srv, monkeypatch, capsys):
    """`serve --code` while a server runs here: refused with why, no pairing line (that code would never work), and
    the running server's voices stay."""
    from peresmeshnik.__main__ import main
    monkeypatch.setattr(config, 'data_dir', lambda: srv.folder.parent)
    monkeypatch.setattr(config, 'load', lambda: config.Settings())
    live = srv.voices / 'me-live'
    live.mkdir()
    assert main(['serve', '--host', '127.0.0.1', '--port', str(srv.ports[0]), '--code']) == 1
    out = capsys.readouterr()
    assert 'already runs here' in out.err and 'Pair a computer' not in out.out and live.exists()


def test_fetch_a_pair_then_serve(tmp_path, monkeypatch):
    """`serve --fetch en-ru`: the pair's models both ways round, through `models fetch` (its license screen), then the
    server starts; a failed download does not start it."""
    from peresmeshnik import languages, models
    from peresmeshnik.__main__ import main
    asked, started = [], []
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(config, 'load', lambda: config.Settings())
    monkeypatch.setattr(models, 'cli', lambda argv: asked.append(argv) or rc)
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)

    async def serve(self):
        started.append(self)
    monkeypatch.setattr(server.Server, 'serve', serve)
    rc = 0
    assert main(['serve', '--host', '127.0.0.1', '--fetch', 'en-ru', '--accept-licenses']) == 0 and len(started) == 1
    s = config.Settings()
    want = set()
    for mine, theirs in (('en', 'ru'), ('ru', 'en')):
        s.my_lang, s.their_lang, s.them_to_me, s.me_to_them = mine, theirs, 'local', 'local'
        want |= set(languages.needed_models(s))
    [argv] = asked
    assert argv[0] == 'fetch' and argv[-1] == '--accept-licenses' and set(argv[1:-1]) == want and len(argv) == len(want) + 2
    assert {'gigaam-v3-e2e-ctc', 'parakeet-tdt-0.6b-v3', 'piper-ru_RU-dmitri-medium', 'piper-en_US-john-medium'} <= want
    rc = 1
    assert main(['serve', '--host', '127.0.0.1', '--fetch', 'de-fr', '--fetch', 'en-ru']) == 1 and len(started) == 1
    assert '--accept-licenses' not in asked[-1]  # asked on the terminal, as `models fetch` does
    assert main(['serve', '--host', '127.0.0.1', '--fetch', 'en-xx']) == 2 and len(asked) == 2


def test_headless():
    """`serve` loads neither Qt nor PortAudio (a server without a display or a sound card)."""
    code = ('import sys, peresmeshnik.remote.server, peresmeshnik.pipeline, peresmeshnik.audio.mic; '
            'print(sorted(m for m in ("PySide6", "sounddevice") if m in sys.modules))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == '[]'


# --- RS-MUST: what the server must do (the security review of the client, PR #24) ---------------------------------

def test_must_auth_first_and_limits(srv, made, monkeypatch):
    """1, 2: the token right after hello (before the voice, before busy); the first message within HELLO_S; a message
    at most CHUNK; at most MAX_UNAUTH connections without a checked token."""
    token = srv.pairing.add('laptop')
    with raw(srv, 'me') as ws:  # an unpaired client with a voice: 'auth', never 'send_voice'
        ws.send(hello('cd' * 32, 'me', 'ru', 'en', voice={'sha256': 'ab' * 32, 'bytes': protocol.VOICE_MAX}))
        assert answer(ws)['code'] == 'auth'
    with raw(srv) as ws:
        ws.send(hello(token))
        assert answer(ws)['type'] == 'ready'
        with raw(srv, 'me') as other:  # the server is busy, but an unpaired one hears 'auth', not 'busy'
            other.send(hello('cd' * 32, 'me', 'ru', 'en'))
            assert answer(other)['code'] == 'auth'
    from websockets.exceptions import ConnectionClosed
    monkeypatch.setattr(server, 'HELLO_S', 0.3)
    with raw(srv) as ws:  # silent: closed
        t = time.monotonic()
        with pytest.raises(ConnectionClosed):
            ws.recv(timeout=5)
        assert time.monotonic() - t < 3
    with raw(srv) as ws, pytest.raises(ConnectionClosed) as e:  # a message over CHUNK: closed by the WebSocket (1009),
        ws.send(b'x' * (protocol.CHUNK + 1))  # sometimes while it is still being sent
        ws.recv(timeout=5)
    assert getattr(e.value.rcvd, 'code', 1009) == 1009  # Windows: the rest of the send is reset before the close is read
    monkeypatch.setattr(server, 'HELLO_S', 10.0)
    monkeypatch.setattr(server, 'MAX_UNAUTH', 2)
    monkeypatch.setattr(server, 'PER_IP', 10)  # here: silent ones from many addresses
    until(lambda: not srv.unauth)
    with raw(srv) as a, raw(srv, 'me') as b:
        until(lambda: sum(srv.unauth.values()) == 2)
        with raw(srv) as c:  # all places taken: the oldest silent one goes, the newcomer stays
            with pytest.raises(ConnectionClosed):
                a.recv(timeout=5)
            c.send(hello(token))
            assert answer(c)['type'] == 'ready'
            assert sum(srv.unauth.values()) == 1  # b only: a token shown is no longer counted
        b.send(hello(token, 'me', 'ru', 'en'))
        assert answer(b)['type'] == 'ready' and not srv.unauth


def test_must_a_voice_is_per_device(tmp_path, made):
    """3: the warm session and the voice it keeps are the device's: another paired computer that knows the sha256 sends
    the voice itself and gets new engines, never the first one's."""
    srv = start(tmp_path / 'server', warm_s=30)
    try:
        a, b = srv.pairing.add('laptop'), srv.pairing.add('other')
        v = profile(tmp_path / 'v', 1000)
        d = RemoteDirection(srv.address, 'me', 'ru', 'en', a, srv.fp, voice=v)
        assert d.ready.wait(10)
        d.close()
        until(lambda: srv.slots['me'].conn is None)
        with raw(srv, 'me') as ws:
            ws.send(hello(b, 'me', 'ru', 'en', voice=protocol.voice_ref(v)))
            assert answer(ws)['type'] == 'send_voice'
            ws.send(v)
            assert answer(ws)['warm'] is False and len(made) == 2 and made[0].closed
    finally:
        srv.stop()
        srv.thread.join(10)


def test_must_bind_private_only(tmp_path):
    """6: never every address or a public one unless --public (the command checks each of `hosts('')` too)."""
    for host in ('', '0.0.0.0', '::', '8.8.8.8', ['127.0.0.1', '0.0.0.0']):
        with pytest.raises(ValueError):
            server.Server(tmp_path / 'server', config.Settings(), host)
    assert server.Server(tmp_path / 'server', config.Settings(), '0.0.0.0', public=True).host == '0.0.0.0'
    assert all(pairing.private(h) for h in server.hosts(''))


def test_must_pairing(srv, made, tmp_path, capsys, monkeypatch):
    """7: /v1/pair only while a code is open; one use; at most 10 minutes; the device paired said on the terminal."""
    assert pairing.CODE_S <= 600
    with raw(srv, path=protocol.PAIR_PATH) as ws:  # no code open: refused before anything is read
        assert answer(ws)['code'] == 'code'
    code = srv.pairing.new_code()
    check = pairing.Pairing.check  # the line is said after the answer: by then the client has closed (its address
    monkeypatch.setattr(pairing.Pairing, 'check', lambda self, t: time.sleep(0.3) or check(self, t))  # gone with it)
    out = []  # read until it is there

    def said():
        out.append(capsys.readouterr().out)
        return ''.join(out)
    pairing.pair(server.url('127.0.0.1', srv.ports[0], code), 'laptop', tmp_path / 'remote')
    until(lambda: said() == 'paired: laptop from 127.0.0.1\n')
    with raw(srv, path=protocol.PAIR_PATH) as ws:  # used: closed again
        assert answer(ws)['code'] == 'code'
    pairing.pair(server.url('127.0.0.1', srv.ports[0], srv.pairing.new_code()), 'laptop', tmp_path / 'remote2')
    second = srv.pairing.devices()[-1]['name']  # another device of the same name: the name it is kept (and revoked) by
    out.clear()
    until(lambda: said() == f'paired: {second} from 127.0.0.1\n')
    assert second != 'laptop'


def test_must_a_revoked_device_is_cut_off(tmp_path, made, monkeypatch):
    """8: `serve --revoke` from another process: the device's live session gets 'auth' (it stops trying) and its warm
    side goes too, within REVOKE_S."""
    monkeypatch.setattr(server, 'REVOKE_S', 0.1)
    srv = start(tmp_path / 'server', warm_s=30)
    try:
        token = srv.pairing.add('laptop')
        with raw(srv, 'me') as ws:
            ws.send(hello(token, 'me', 'ru', 'en'))
            assert answer(ws)['type'] == 'ready'
        until(lambda: srv.slots['me'].conn is None)  # warm
        with raw(srv) as ws:
            ws.send(hello(token))
            assert answer(ws)['type'] == 'ready'
            assert pairing.Pairing(tmp_path / 'server').revoke('laptop')  # another process: its own Pairing
            a = answer(ws)
            assert (a['type'], a['code']) == ('error', 'auth')
        until(lambda: not srv.slots and all(f.closed for f in made))
    finally:
        srv.stop()
        srv.thread.join(10)


def test_must_no_token_in_the_log(tmp_path, made, caplog, capsys):
    """9: neither hello nor pair is logged (token, nonce, mac), whatever the log level; websockets never below INFO."""
    caplog.set_level(logging.DEBUG)
    srv = start(tmp_path / 'server')
    try:
        assert logging.getLogger('websockets').getEffectiveLevel() >= logging.INFO
        code, nonce = srv.pairing.new_code(), 'ab' * 32
        with raw(srv, path=protocol.PAIR_PATH) as ws:
            ws.send(protocol.pair(code, srv.fp, nonce, 'laptop'))
            token = answer(ws)['token']
        with raw(srv) as ws:
            ws.send(hello(token))
            assert answer(ws)['type'] == 'ready'
    finally:
        srv.stop()
        srv.thread.join(10)
    for secret in (token, nonce, protocol.pair_mac(code, srv.fp, nonce), protocol.norm_code(code)):
        assert secret not in caplog.text
    assert 'ready' in caplog.text or 'loading' in caplog.text  # it did log the session


def test_must_not_a_browser(srv, made):
    """10: a request with an Origin header (a web page) is refused before any message, on every path."""
    from websockets.exceptions import InvalidStatus
    srv.pairing.new_code()
    for path in (protocol.PAIR_PATH, protocol.side_path('them')):
        with pytest.raises(InvalidStatus) as e:
            raw(srv, path=path, origin='https://example.org')
        assert e.value.response.status_code == 403


@pytest.mark.parametrize('meta', [{'kind': 'me', 'name': 'me'},  # v1: from an older app
                                  {**CONFIRMED, 'kind': 'them', 'consent': {'how': 'call-saved'}},  # a contact's
                                  {**CONFIRMED, 'consent': None},  # mine, not confirmed
                                  {**CONFIRMED, 'consent': {'how': 'mic-live', 'ref_sha256': 'ab' * 32}}])  # another take
def test_must_only_my_confirmed_voice_speaks_as_me(srv, made, tmp_path, meta):
    """12, FIXES 3: me -> them speaks only with my own voice confirmed live, this very recording (the app's one rule,
    voice.why_not_mine); anything else sent on 'me': a stock voice and why, and that voice is gone at once."""
    token = srv.pairing.add('laptop')
    v = profile(tmp_path / 'v', 1000, meta)
    with raw(srv, 'me') as ws:
        ws.send(hello(token, 'me', 'ru', 'en', voice=protocol.voice_ref(v)))
        assert answer(ws)['type'] == 'send_voice'
        ws.send(v)
        a = answer(ws)
        assert (a['type'], a['voice']) == ('ready', 'stock')
        states = [answer(ws)['text'] for _ in srv.slots['me'].notes]
        assert any('cannot speak for you' in s for s in states), states
        assert (made[0].voice_dir, made[0].clone) == ('', 'piper') and not any(srv.voices.iterdir())


def test_must_pairing_files_off_the_loop(srv, made, monkeypatch):
    """A slow paired.json (Windows: held by `serve --revoke` from another process, check up to 2 s, add ~10 s) never
    stops the server's loop: Pairing.check and handle (-> add) run on threads; another connection is answered at once."""
    check, handle = pairing.Pairing.check, pairing.Pairing.handle
    monkeypatch.setattr(pairing.Pairing, 'check', lambda self, t: time.sleep(1.5) or check(self, t))
    monkeypatch.setattr(pairing.Pairing, 'handle', lambda self, m: time.sleep(1.5) or handle(self, m))
    token = srv.pairing.add('laptop')
    code = srv.pairing.new_code()
    for slow in (hello(token), protocol.pair(code, srv.fp, 'ab' * 32, 'other')):
        with raw(srv, path=protocol.PAIR_PATH if 'pair' in slow else '') as ws, raw(srv, 'me') as other:
            ws.send(slow)
            time.sleep(0.2)  # it is in check / handle now
            t = time.monotonic()
            other.send(hello('not a token', 'me', 'ru', 'en'))  # refused by the loop itself (no file)
            assert answer(other)['code'] == 'auth' and time.monotonic() - t < 0.8
            assert answer(ws)['type'] in ('ready', 'paired')



# --- FIXES-rs-stage8, round 1 ------------------------------------------------------------------------------------

def test_one_address_cannot_lock_the_laptop_out(tmp_path, made, monkeypatch):
    """1: silent connections from one address (anyone on the LAN, the tailnet, the same CGNAT) hold PER_IP places at most;
    the paired laptop from another address still gets in."""
    from websockets.exceptions import ConnectionClosed
    try:
        probe = socket.create_server(('::1', 0), family=socket.AF_INET6)
    except OSError:
        pytest.skip('no IPv6 loopback here')
    port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setattr(server, 'HELLO_S', 10.0)
    srv = start(tmp_path / 'server', host=['127.0.0.1', '::1'], port=port)
    token, silent, stack = srv.pairing.add('laptop'), [], contextlib.ExitStack()
    try:
        for _ in range(8):
            silent.append(stack.enter_context(raw(srv)))  # all from 127.0.0.1, none says anything
        until(lambda: srv.unauth == {'127.0.0.1': server.PER_IP})
        for ws in silent[server.PER_IP:]:
            with pytest.raises(ConnectionClosed) as e:
                ws.recv(timeout=5)
            assert getattr(e.value.rcvd, 'code', 1013) == 1013
        with raw(srv, addr=f'[::1]:{port}') as ws:
            ws.send(hello(token))
            assert answer(ws)['type'] == 'ready'
    finally:
        stack.close()
        srv.stop()
        srv.thread.join(10)


def test_garbage_before_the_token_leaves_no_trace(srv, made, caplog):
    """2: a nesting bomb (RecursionError, not a ValueError) as hello or as pair: refused with 'hello', no traceback in the
    log, nothing of the server's insides in the answer."""
    caplog.set_level(logging.INFO)
    bomb = '[' * 200_000
    with raw(srv) as ws:
        ws.send(bomb)
        a = answer(ws)
        assert (a['code'], a['text']) == ('hello', 'the first message is not a hello')
    srv.pairing.new_code()
    with raw(srv, path=protocol.PAIR_PATH) as ws:
        ws.send(bomb)
        assert answer(ws)['code'] == 'hello'
    assert not [r for r in caplog.records if r.exc_info] and 'Traceback' not in caplog.text


def test_the_voice_goes_first_when_the_server_stops(tmp_path, made, monkeypatch):
    """4: stopped (docker stop, Ctrl+C) while a side loads its models: its voice is off the disk at once, not after the
    load; its engines are stopped once they are up."""
    gate, local = threading.Event(), app.Side._local
    monkeypatch.setattr(app.Side, '_local', lambda side, s: (gate.wait(5), local(side, s))[1])
    srv = start(tmp_path / 'server')
    token, v = srv.pairing.add('laptop'), profile(tmp_path / 'v', 1000)
    with raw(srv, 'me') as ws:
        ws.send(hello(token, 'me', 'ru', 'en', voice=protocol.voice_ref(v)))
        assert answer(ws)['type'] == 'send_voice'
        ws.send(v)
        until(lambda: 'me' in srv.slots and srv.slots['me'].task is not None)  # unpacked; the engines load
        assert any(srv.voices.iterdir())
        srv.stop()
        until(lambda: not any(srv.voices.iterdir()), 1.0)
        assert not made  # still loading
        gate.set()
        srv.thread.join(10)
    assert not srv.thread.is_alive() and made[0].closed


def test_no_voice_asked_for_when_it_is_not_spoken(srv, made, tmp_path, monkeypatch):
    """5: me -> them where this server cannot speak my clone (no NVIDIA, a language without one), and them -> me (no voice
    of theirs is spoken from hello): the voice is not asked for; me -> them hears why."""
    from peresmeshnik import languages
    monkeypatch.setattr(languages, 'clone_why', lambda *a, **k: 'your cloned voice needs an NVIDIA GPU')
    token, ref = srv.pairing.add('laptop'), protocol.voice_ref(profile(tmp_path / 'v', 1000))
    with raw(srv, 'me') as ws:
        ws.send(hello(token, 'me', 'ru', 'en', voice=ref))
        a = answer(ws)
        assert (a['type'], a['voice']) == ('ready', 'stock')
        assert any('NVIDIA' in answer(ws)['text'] for _ in srv.slots['me'].notes)
        with raw(srv) as them:
            them.send(hello(token, voice=ref))
            assert answer(them)['type'] == 'ready'
    assert not any(srv.voices.iterdir()) and all(f.voice_dir == '' for f in made)


def test_the_pairing_lines(tmp_path, monkeypatch, capsys):
    """7: the loopback line says ssh (in Docker the container's own address comes first); every address with --public
    and no private one: the outward address, else a placeholder; no private address at all: said at the start; a name
    for --host: said that it is not an address."""
    from peresmeshnik.__main__ import main
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(config, 'load', lambda: config.Settings())

    async def serve(self):
        if self.on_start:
            self.on_start()
    monkeypatch.setattr(server.Server, 'serve', serve)
    monkeypatch.setattr(pairing, 'addresses', lambda: ['172.17.0.2'])
    assert main(['serve']) == 0
    out = capsys.readouterr().out
    assert 'https://172.17.0.2:47000/#' in out
    assert re.search(r'https://127\.0\.0\.1:47000/#\S+ +\(on this computer, or from the laptop through ssh -tt -L\)', out)
    monkeypatch.setattr(pairing, 'addresses', lambda: [])
    for seen, line in [('', 'https://<the address of this server>:47000/#'), ('203.0.113.5', 'https://203.0.113.5:47000/#')]:
        monkeypatch.setattr(server, 'outward', lambda seen=seen: seen)
        assert main(['serve', '--host', '0.0.0.0', '--public', '--code']) == 0 and line in capsys.readouterr().out
    assert main(['serve', '--code']) == 0
    c = capsys.readouterr()
    assert 'No home network or Tailscale address' in c.err and 'https://127.0.0.1:47000/#' in c.out
    assert main(['serve', '--host', 'myserver.lan']) == 2 and 'not a name' in capsys.readouterr().err


def test_a_failed_load_is_heard_by_the_connection_that_took_over(srv, made, monkeypatch):
    """8: the side's new connection takes over while its engines load, and the load fails: that connection hears why."""
    gate = threading.Event()

    def build(self, slot, h):
        gate.wait(5)
        raise server.ProtocolError('lang', 'the server is missing a model')
    monkeypatch.setattr(server.Server, '_build', build)
    token = srv.pairing.add('laptop')
    with raw(srv) as old, raw(srv) as new:
        old.send(hello(token))
        until(lambda: 'them' in srv.slots and srv.slots['them'].task is not None)
        new.send(hello(token))  # the same side, languages and voice: it takes the loading slot over
        until(lambda: srv.slots['them'].conn is not None and srv.slots['them'].conn.ws.remote_address[1]
              == new.socket.getsockname()[1])
        gate.set()
        a = answer(new)
        assert (a['code'], a['text']) == ('lang', 'the server is missing a model')


def test_one_computer_is_one_token(srv, made, monkeypatch):
    """8: who has the server is its device token, not the name it paired with (two computers may say the same)."""
    a, b = srv.pairing.add('laptop'), srv.pairing.add('other')
    check = pairing.Pairing.check
    monkeypatch.setattr(pairing.Pairing, 'check', lambda self, t: check(self, t) and 'laptop')
    with raw(srv) as x, raw(srv, 'me') as y:
        x.send(hello(a))
        assert answer(x)['type'] == 'ready'
        y.send(hello(b, 'me', 'ru', 'en'))
        assert answer(y)['code'] == 'busy'


def test_the_pause_and_a_slow_reader(monkeypatch):
    """8: the pause that closes a cut clause is the VAD's own minimum and two frames (0.5 s without a VAD); a client that
    does not read what waits for it (QUEUE_MAX) is cut off, not kept in memory."""
    from types import SimpleNamespace as NS

    from peresmeshnik.local.vad import FRAME
    assert len(server.pause(NS(vad=NS(min_silence_frames=10)))) == 12 * FRAME and len(server.pause(object())) == 8000
    monkeypatch.setattr(server, 'QUEUE_MAX', 3)
    cut = []
    ws = NS(transport=NS(is_closing=lambda: bool(cut), abort=lambda: cut.append(1)))

    async def go():
        c = server.Conn(ws, asyncio.get_running_loop())
        for _ in range(6):
            c.put(b'x', speech=True)
        await asyncio.sleep(0.05)
        return c.q.qsize()
    assert asyncio.run(go()) == 3 and cut == [1]

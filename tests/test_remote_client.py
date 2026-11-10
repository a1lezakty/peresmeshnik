"""RemoteDirection (peresmeshnik/remote/client.py) against a fake server (tests/fake_remote.py: the real protocol over a real
TLS WebSocket on 127.0.0.1): the session, the voice upload and a warm session, a certificate that is not the paired one,
refusals, a reset and a dead link (one new connection, the cut clause heard again), too slow, barge-in, close."""
import os
import queue
import threading
import time

import numpy as np
import pytest
from fake_remote import FakeServer

from peresmeshnik.cloud.gemini import CloudDown
from peresmeshnik.engines import Caption, Latency, Speech
from peresmeshnik.remote import client, protocol
from peresmeshnik.remote.client import RemoteDirection, ServerError

TOKEN = 'ab' * 32
FRAME = 512


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, 'timed out'
        time.sleep(0.01)


class Events:
    """The direction's events on a thread, as the side consumes them."""

    def __init__(self, d):
        self.q, self.error = queue.Queue(), None
        self.t = threading.Thread(target=self._run, args=(d,), daemon=True)
        self.t.start()

    def _run(self, d):
        try:
            for e in d.events():
                self.q.put(e)
        except Exception as e:
            self.error = e
        self.q.put(None)

    def get(self, timeout=5.0):
        return self.q.get(timeout=timeout)

    def end(self, timeout=8.0):
        self.t.join(timeout)
        assert not self.t.is_alive()
        return self.error


def frame(v):
    return np.full(FRAME, v, np.float32)


def values(audio: bytes):
    """The constant value of each frame the server got."""
    x = protocol.pcm_from(bytes(audio)).reshape(-1, FRAME)
    return [round(float(f[0]), 3) for f in x]


def direction(server, side='them', **kw):
    return RemoteDirection(server.address, side, 'en', 'ru', TOKEN, server.fp, **kw)


def test_a_session_with_the_voice_then_a_warm_one(server, tmp_path):
    (tmp_path / 'voice.json').write_text('{"kind": "me"}')
    (tmp_path / 'state.bin').write_bytes(os.urandom(int(2.5 * protocol.CHUNK)))  # 3 binary messages up
    voice = protocol.pack_voice(tmp_path)
    d = direction(server, 'me', voice=voice)
    ev = Events(d)
    assert d.ready.wait(5)
    h = server.hellos[0]
    assert (h['token'], h['side'], h['src'], h['tgt'], h['voice']) == (TOKEN, 'me', 'en', 'ru', protocol.voice_ref(voice))
    assert server.voices == [voice] and d.server['device'] == 'cpu'
    for i in range(10):
        d.feed(frame(i / 100))
    until(lambda: len(server.conns[0].audio) == 10 * FRAME * 2)
    assert values(server.conns[0].audio) == [i / 100 for i in range(10)]

    pcm = np.full(int(24_000 * 0.25), 0.3, np.float32)
    server.send(protocol.event(Caption('hel')), protocol.event(Caption('hello', 'привет', True, 0)),
                protocol.event(Latency(0, 'voice', 0.3)), *protocol.speech(Speech(pcm, 24_000)))
    assert ev.get() == Caption('hel')
    assert ev.get() == Caption('hello', 'привет', True, 0)
    lat = ev.get()
    assert lat.what == 'voice' and 0.3 <= lat.s < 0.35  # + the RTT (on loopback it may round to nothing)
    assert ('remote.voice', lat.s) in d.latency and d.said_until > 0
    pieces = [ev.get() for _ in range(3)]
    hold = 24_000 * client.JITTER_MS // 1000
    assert [len(p.pcm) for p in pieces] == [hold + 2400, 2400, 1200]  # only the first waits: the playback had run dry
    assert not pieces[0].pcm[:hold].any() and np.allclose(pieces[0].pcm[hold:], 0.3, atol=1e-4)
    t = time.monotonic()
    d.close()
    assert time.monotonic() - t < 2 and ev.end() is None

    d = direction(server, 'me', voice=voice)  # the same voice: the server keeps it, nothing goes up again
    assert d.ready.wait(5)
    d.close()
    assert len(server.voices) == 1 and len(server.hellos) == 2


def test_not_the_paired_certificate_gets_nothing(server):
    d = RemoteDirection(server.address, 'them', 'en', 'ru', TOKEN, '0' * 64)
    e = Events(d).end()
    assert isinstance(e, ServerError) and e.code == 'auth' and e.permanent and 'pair again' in str(e)
    assert server.hellos == []  # not even the token went out
    d.close()


@pytest.mark.parametrize('code,permanent', [('auth', True), ('lang', True), ('busy', False)])
def test_refused(server, code, permanent):
    server.refuse = (code, 'no')
    d = direction(server)
    e = Events(d).end()
    assert isinstance(e, ServerError) and isinstance(e, CloudDown) and e.code == code and e.permanent == permanent
    assert not d.ready.is_set()
    d.close()


def test_unreachable_is_down_fast(server):
    port = server.port
    server.close()
    t = time.monotonic()
    d = RemoteDirection(f'127.0.0.1:{port}', 'them', 'en', 'ru', TOKEN, 'f' * 64)
    e = Events(d).end()
    assert isinstance(e, CloudDown) and 'cannot reach the server' in str(e)
    assert time.monotonic() - t < client.CONNECT_S + 1
    d.close()


def feed_paced(d, vals, step=0.02):
    for v in vals:
        d.feed(frame(v))
        time.sleep(step)


def test_a_reset_reconnects_and_the_cut_clause_is_heard_again(server):
    d = direction(server)
    ev = Events(d)
    assert d.ready.wait(5)
    feed_paced(d, [0.1] * 10)  # a clause the server finished
    time.sleep(client.SETTLE_S + 0.1)
    server.send(protocol.event(Caption('one', 'один', True, 0)))
    assert ev.get().final
    feed_paced(d, [0.2] * 10)  # the clause the drop cuts
    until(lambda: len(server.conns[0].audio) == 20 * FRAME * 2)
    server.drop()
    until(lambda: len(server.conns) == 2)
    until(lambda: len(server.conns[1].audio) >= 10 * FRAME * 2)
    d.feed(frame(0.3))
    until(lambda: values(server.conns[1].audio)[-1:] == [0.3])
    assert values(server.conns[1].audio) == [0.2] * 10 + [0.3]  # the cut clause whole, not the finished one
    server.send(protocol.event(Caption('two', 'два', True, 1)))
    assert ev.get() == Caption('two', 'два', True, 1) and ev.error is None
    d.close()
    assert ev.end() is None


def test_a_reset_and_no_server_is_down(server):
    d = direction(server)
    ev = Events(d)
    assert d.ready.wait(5)
    server.refuse = ('internal', 'restarting')
    server.drop()
    e = ev.end()
    assert isinstance(e, CloudDown) and 'lost' in str(e) and 'a new one failed' in str(e)
    d.close()


def test_a_dead_link_without_a_reset(server, monkeypatch):
    monkeypatch.setattr(protocol, 'PING_S', 0.1)
    monkeypatch.setattr(protocol, 'PONG_S', 0.3)
    d = direction(server)
    ev = Events(d)
    assert d.ready.wait(5)
    t = time.monotonic()
    server.deafen()  # no pongs from now on
    until(lambda: len(server.conns) == 2)  # noticed in PONG_S, one new connection
    assert time.monotonic() - t < 1.5 and ev.error is None
    d.close()
    assert ev.end() is None


def test_a_server_that_stops_translating_mid_phrase(server, monkeypatch):
    monkeypatch.setattr(protocol, 'PING_S', 0.1)
    d = direction(server, stall_s=0.5)
    ev = Events(d)
    assert d.ready.wait(5)
    time.sleep(0.7)  # quiet with no phrase open: fine
    assert ev.q.empty()
    server.send(protocol.event(Caption('hel')))
    assert ev.get() == Caption('hel')
    t = time.monotonic()
    e = ev.end()
    assert isinstance(e, CloudDown) and 'stopped translating' in str(e) and time.monotonic() - t < 1.5
    d.close()


def test_an_uplink_too_slow_for_the_call(server):
    d = direction(server, stall_s=0.4)
    ev = Events(d)
    assert d.ready.wait(5)
    server.deafen()
    big = np.zeros(16_000, np.float32)  # 1 s blocks, faster than real time: the socket's buffers fill up
    for _ in range(400):
        d.feed(big)
        if ev.error or not ev.t.is_alive():
            break
        time.sleep(0.005)
    e = ev.end()
    assert isinstance(e, CloudDown) and 'too slow' in str(e)
    d.close()


def test_barge_in_drops_the_voice_on_its_way(server):
    d = direction(server, jitter_ms=0)
    ev = Events(d)
    assert d.ready.wait(5)
    d.interrupt()
    until(lambda: {'type': 'interrupt'} in server.conns[0].texts)
    old = protocol.speech(Speech(np.full(2400, 0.1, np.float32), 24_000))
    new = protocol.speech(Speech(np.full(2400, 0.2, np.float32), 24_000))
    server.send(*old, protocol.message('interrupted'), *new)
    s = ev.get()
    assert np.allclose(s.pcm, 0.2, atol=1e-4)  # the old voice never came out
    d.close()


def test_close_while_the_server_loads(server):
    server.ready_delay = None
    d = direction(server)
    ev = Events(d)
    until(lambda: server.hellos)
    t = time.monotonic()
    d.close()
    assert time.monotonic() - t < 1.5 and ev.end() is None and not d.ready.is_set()


def test_input_before_ready_keeps_only_the_last_seconds(server, monkeypatch):
    monkeypatch.setattr(client, 'KEEP_S', 0.3)
    server.ready_delay = 0.8
    d = direction(server)
    fed = []  # feed time of frame i (value i / 1000)
    while not d.ready.is_set() or len(fed) < 60:  # the capture goes on while the server loads its models
        fed.append(time.monotonic())
        d.feed(frame(len(fed) / 1000))
        time.sleep(0.02)
    until(lambda: values(server.conns[0].audio)[-1:] == [len(fed) / 1000])
    got = values(server.conns[0].audio)
    assert got == [i / 1000 for i in range(round(got[0] * 1000), len(fed) + 1)]  # in order, none lost after the first
    before = [t for t in fed[round(got[0] * 1000) - 1:] if t < d.ready_at]
    assert 0.2 <= before[-1] - before[0] <= 0.35  # of the 0.8 s fed while loading, the last KEEP_S went up
    d.close()

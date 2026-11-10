"""Pairing (peresmeshnik/remote/pairing.py): a real TLS WebSocket on 127.0.0.1, the code's limits, a man in the middle,
the token files, private addresses and the pairing line."""
import asyncio
import contextlib
import json
import os
import ssl
import threading

import pytest
from websockets.asyncio.server import serve

from peresmeshnik.remote import pairing, protocol
from peresmeshnik.remote.protocol import ProtocolError


@pytest.fixture(scope='module')
def tls(tmp_path_factory):
    crt, key, fp = pairing.cert(tmp_path_factory.mktemp('cert'))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    return ctx, fp


@contextlib.contextmanager
def server(ctx, respond):
    """A TLS WebSocket server on 127.0.0.1 in its own thread; `respond`: the first message -> the answer."""
    loop = asyncio.new_event_loop()
    started, stop = threading.Event(), asyncio.Event()
    port = []

    async def handler(ws):
        await ws.send(respond(await ws.recv()))

    async def main():
        async with serve(handler, '127.0.0.1', 0, ssl=ctx) as s:
            port.append(s.sockets[0].getsockname()[1])
            started.set()
            await stop.wait()
    t = threading.Thread(target=loop.run_until_complete, args=(main(),))
    t.start()
    started.wait(5)
    try:
        yield f'https://127.0.0.1:{port[0]}'
    finally:
        loop.call_soon_threadsafe(stop.set)
        t.join(5)
        loop.close()


def refused(fn) -> str:
    with pytest.raises(ProtocolError) as e:
        fn()
    return e.value.code


def test_pairing_over_tls_then_the_token_works(tls, tmp_path):
    ctx, fp = tls
    p = pairing.Pairing(tmp_path / 'srv', fp)
    code = p.new_code()
    assert len(code) == 14 and code[4] == code[9] == '-'
    remote = tmp_path / 'cli' / 'remote'
    with server(ctx, p.handle) as url:
        got = pairing.pair(f'{url}/#{code.lower()}', 'laptop', remote)
        assert got['fp'] == fp and got['name'] == 'laptop' and got['address'] == url.removeprefix('https://')
        assert pairing.load(remote) == got
        assert p.check(got['token']) == 'laptop' and p.check('ab' * 32) is None
        assert got['token'] not in (tmp_path / 'srv' / 'paired.json').read_text()  # only its sha256
        assert refused(lambda: pairing.pair(f'{url}/#{code}', 'laptop', tmp_path / 'again')) == 'code'  # used once
    assert not (tmp_path / 'again').exists()
    assert [d['name'] for d in p.devices()] == ['laptop']
    if os.name == 'posix':
        for f in (remote, tmp_path / 'srv' / 'paired.json'):
            assert f.stat().st_mode & 0o777 == 0o600, f
    assert p.revoke('laptop') and p.check(got['token']) is None and not p.revoke('laptop')
    pairing.forget(remote)
    assert pairing.load(remote) is None
    pairing.forget(remote)  # twice is fine


def test_wrong_codes_burn_the_code(tls, tmp_path):
    ctx, fp = tls
    p = pairing.Pairing(tmp_path, fp)
    good = p.new_code()
    wrong = ('2' if good[0] != '2' else '3') + good[1:]
    with server(ctx, p.handle) as url:
        for _ in range(pairing.CODE_TRIES - 1):
            assert refused(lambda: pairing.pair(f'{url}/#{wrong}', 'x', tmp_path / 'r')) == 'code'
        assert refused(lambda: pairing.pair(f'{url}/#{wrong}', 'x', tmp_path / 'r')) == 'attempts'
        assert refused(lambda: pairing.pair(f'{url}/#{good}', 'x', tmp_path / 'r')) == 'attempts'
        good = p.new_code()  # serve --pair again
        assert pairing.pair(f'{url}/#{good}', 'x', tmp_path / 'r')['token']


def test_expired_code_and_bad_messages(tmp_path):
    now = [100.0]
    p = pairing.Pairing(tmp_path, 'f' * 64, clock=lambda: now[0])
    assert json.loads(p.handle(protocol.pair('A' * 12, p.fp, '0a', 'x')))['code'] == 'code'  # no code yet
    code = p.new_code()
    now[0] += pairing.CODE_S + 1
    assert json.loads(p.handle(protocol.pair(code, p.fp, '0a', 'x')))['code'] == 'code'
    code = p.new_code()
    msg = json.loads(protocol.pair(code, p.fp, '0a', 'x'))
    for bad, why in (('not json', 'hello'), (json.dumps(msg | {'type': 'hello'}), 'hello'),
                     (json.dumps(msg | {'version': protocol.VERSION + 1}), 'version'), (json.dumps(msg | {'name': ' '}), 'hello'),
                     (json.dumps(msg | {'mac': None}), 'hello')):
        assert json.loads(p.handle(bad))['code'] == why
    assert p.tries == 0  # none of these was a guess
    assert json.loads(p.handle(json.dumps(msg)))['type'] == 'paired'


def test_man_in_the_middle(tls, tmp_path):
    ctx, fp = tls
    p = pairing.Pairing(tmp_path / 'srv', fp)
    code = p.new_code()
    # the client talked TLS to a man in the middle with his own certificate: its proof is bound to his fingerprint
    assert json.loads(p.handle(protocol.pair(code, 'e' * 64, '0a', 'x')))['code'] == 'code'
    # a server (or a man in the middle) that answers with a token but not the proof for this certificate and code
    remote = tmp_path / 'remote'
    with server(ctx, lambda m: protocol.paired(code, 'e' * 64, json.loads(m)['nonce'], 'ab' * 32)) as url:
        assert refused(lambda: pairing.pair(f'{url}/#{code}', 'x', remote)) == 'auth'
    assert not remote.exists() and not remote.parent.joinpath('srv').exists()
    with server(ctx, lambda m: protocol.paired('B' * 12, fp, json.loads(m)['nonce'], 'ab' * 32)) as url:  # other code
        assert refused(lambda: pairing.pair(f'{url}/#{code}', 'x', remote)) == 'auth'
    assert not remote.exists()


def test_no_server(tmp_path):
    with server(ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER), str) as url:  # a server without a certificate: TLS fails
        assert refused(lambda: pairing.pair(f'{url}/#{"A" * 12}', 'x', tmp_path / 'r')) == 'internal'
    assert not (tmp_path / 'r').exists()


def test_cert_is_kept(tmp_path):
    crt, key, fp = pairing.cert(tmp_path)
    assert pairing.cert(tmp_path) == (crt, key, fp) and len(fp) == 64
    if os.name == 'posix':
        assert key.stat().st_mode & 0o777 == 0o600


def test_private_addresses():
    for host in ('192.168.1.20', '10.1.2.3', '172.16.0.1', '172.31.255.254', '100.64.0.1', '100.127.1.1', '127.0.0.1',
                 '::1', 'fd12:3456::1', 'fe80::1', 'fe80::1%eth0', '[fd00::1]', '::ffff:192.168.1.1', 'localhost'):
        assert pairing.private(host), host
    for host in ('8.8.8.8', '172.32.0.1', '100.128.0.1', '2001:4860::1', '0.0.0.0', '::', 'example.com', ''):
        assert not pairing.private(host), host
    pairing.check_bind('192.168.1.20')
    pairing.check_bind('8.8.8.8', public=True)
    for host in ('8.8.8.8', '0.0.0.0'):
        with pytest.raises(ValueError, match='--public'):
            pairing.check_bind(host)
    assert all(pairing.private(a) and a not in ('127.0.0.1', '::1') for a in pairing.addresses())


def test_parse():
    code = 'K7QMXPAZDFEB'
    for url, want in (('https://192.168.1.20:47000/#K7QM-XPAZ-DFEB', ('192.168.1.20', 47000, code)),
                      ('192.168.1.20/#k7qm xpaz dfeb', ('192.168.1.20', protocol.PORT, code)),
                      ('  box.tailnet.ts.net:5000#K7QMXPAZDFEB ', ('box.tailnet.ts.net', 5000, code)),
                      ('https://[fd00::20]:47000/#K7QM-XPAZ-DFEB', ('fd00::20', 47000, code)),
                      ('[::1]#K7QM-XPAZ-DFEB', ('::1', protocol.PORT, code))):
        assert pairing.parse(url) == want, url
    for bad in ('', 'https://192.168.1.20:47000/', 'https://192.168.1.20:47000/#K7QM', 'http://192.168.1.20/#K7QM-XPAZ-DFEB',
                'https://192.168.1.20:99999/#K7QM-XPAZ-DFEB', 'https://192.168.1.20/x#K7QM-XPAZ-DFEB', 'fd00::20#K7QMXPAZDFEB',
                'https://:47000/#K7QM-XPAZ-DFEB', 'https://10.0.0.1/#K7QM-XPAZ-DFE8', 'garbage'):
        with pytest.raises(ValueError):
            pairing.parse(bad)
    assert pairing.parse('192.168.1.20#K7QM-XPAZ-DFE0')[2] == 'K7QMXPAZDFEO'  # 0 reads as O


def test_revoke_from_another_process_is_not_undone(tmp_path):
    srv = pairing.Pairing(tmp_path)  # the running server
    token, other = srv.add('laptop'), srv.add('desk')
    paired = tmp_path / 'paired.json'
    before = paired.stat().st_mtime_ns, paired.read_bytes()
    for _ in range(50):
        assert srv.check(other) == 'desk'
    assert (paired.stat().st_mtime_ns, paired.read_bytes()) == before  # check() writes nothing
    assert [d['last'] for d in srv.devices() if d['name'] == 'desk'] == [srv.seen[next(iter(srv.seen))]]  # from memory
    stop, checks, after = threading.Event(), [0], []

    def hellos():
        while not stop.is_set():
            got = srv.check(token)
            checks[0] += 1
            if got is None:
                after.append(checks[0])
    t = threading.Thread(target=hellos)
    t.start()
    try:
        while checks[0] < 20 and t.is_alive():
            pass
        assert pairing.Pairing(tmp_path).revoke('laptop')  # serve --revoke: another process, its own Pairing
        n = checks[0]
        while checks[0] < n + 50 and t.is_alive():
            pass
    finally:
        stop.set()
        t.join(5)
    assert after  # the server saw the revoke
    assert [d['name'] for d in json.loads(paired.read_text())] == ['desk']  # and no check brought it back
    assert srv.check(token) is None and srv.check(other) == 'desk'


def test_adds_from_two_processes_lose_nothing(tmp_path):
    def adds(who):
        p = pairing.Pairing(tmp_path)  # each its own object: only the file lock keeps them apart
        for i in range(20):
            p.add(f'{who}{i}')
    ts = [threading.Thread(target=adds, args=(w,)) for w in 'ab']
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert sorted(d['name'] for d in pairing.Pairing(tmp_path).devices()) == sorted(f'{w}{i}' for w in 'ab'
                                                                                    for i in range(20))


def test_same_name_gets_a_suffix(tmp_path):
    p = pairing.Pairing(tmp_path)
    tokens = [p.add('laptop') for _ in range(3)]
    assert [d['name'] for d in p.devices()] == ['laptop', 'laptop (2)', 'laptop (3)']
    assert [p.check(t) for t in tokens] == ['laptop', 'laptop (2)', 'laptop (3)']
    assert p.revoke('laptop (2)') and not p.revoke('laptop (2)')  # exactly that one
    assert [p.check(t) for t in tokens] == ['laptop', None, 'laptop (3)']
    p.add('laptop')
    assert [d['name'] for d in p.devices()] == ['laptop', 'laptop (3)', 'laptop (2)']


def test_garbage_does_not_raise(tmp_path):
    p = pairing.Pairing(tmp_path, 'f' * 64)
    code = p.new_code()
    msg = json.loads(protocol.pair(code, p.fp, '0a', 'x'))
    good_mac = msg['mac']
    bad = [{'nonce': n} for n in ('é' * 4, '\ud800', 'AB', '', 'a' * 129, 'g', 7)]
    bad += [{'mac': m} for m in ('é' * 64, '\ud800' * 64, good_mac.upper(), good_mac[:-1], good_mac + 'a', 7)]
    bad += [{'name': n} for n in ('\ud800', 'a\nb')]
    for b in bad:
        assert json.loads(p.handle(json.dumps(msg | b)))['code'] == 'hello', b  # json.dumps escapes the surrogate
    assert json.loads(p.handle(b'\xff\xfe'))['code'] == 'hello'
    assert json.loads(p.handle(json.dumps(msg | {'version': '\ud800'})))['code'] == 'version'
    assert p.tries == 0 and not p.used
    reply = json.loads(p.handle(json.dumps(msg)))
    token = reply['token']
    assert reply['type'] == 'paired' and p.check(token) == 'x'
    for t in ('é' * 64, '\ud800', '\ud800' * 64, token.upper(), token[:-1], token + 'a', '', None, 7):
        assert p.check(t) is None, t


def test_client_refuses_a_garbage_answer(tls, tmp_path):
    ctx, fp = tls
    code = 'K7QMXPAZDFEB'
    good = 'ab' * 32

    def answers(token, mac=None):
        def respond(m):
            nonce = json.loads(m)['nonce']
            if mac:
                return json.dumps({'type': 'paired', 'token': token, 'mac': mac(nonce)})
            try:  # the right proof for this token, where it can be made at all
                right = protocol.pair_mac(code, fp, nonce, token)
            except (AttributeError, UnicodeEncodeError):
                right = 'ab' * 32
            return json.dumps({'type': 'paired', 'token': token, 'mac': right})
        return respond
    remote = tmp_path / 'remote'
    for respond in (answers(good.upper()), answers('é' * 64), answers('\ud800'), answers(good[:-2]), answers(7),
                    answers(good, lambda n: protocol.pair_mac(code, fp, n, good).upper()),
                    answers(good, lambda n: 'é' * 64), answers(good, lambda n: '\ud800' * 64),
                    answers(good, lambda n: protocol.pair_mac(code, fp, n, good)[:-1]), answers(good, lambda n: None)):
        with server(ctx, respond) as url:
            assert refused(lambda: pairing.pair(f'{url}/#{code}', 'x', remote)) == 'auth'
        assert not remote.exists()
    with server(ctx, answers(good)) as url:  # the same server, a proper answer
        assert pairing.pair(f'{url}/#{code}', 'x', remote)['token'] == good

"""The remote server's wire protocol (peresmeshnik/remote/protocol.py): what the client and `serve` both rely on."""
import io
import json
import zipfile

import numpy as np
import pytest

from peresmeshnik.engines import Caption, Latency, Speech
from peresmeshnik.remote import protocol as p

TOKEN = 'ab' * 32


def test_hello_round_trip_and_rejections():
    voice = {'sha256': 'c' * 64, 'bytes': 1000}
    h = p.parse_hello(p.hello(TOKEN, 'me', 'ru', 'en', voice, app='0.3.0'), 'me')
    assert (h['token'], h['side'], h['src'], h['tgt'], h['voice']) == (TOKEN, 'me', 'ru', 'en', voice)
    assert p.parse_hello(p.hello(TOKEN, 'them', 'en', 'ru'), 'them')['voice'] is None

    def code(text, side='me'):
        with pytest.raises(p.ProtocolError) as e:
            p.parse_hello(text, side)
        return e.value.code
    good = json.loads(p.hello(TOKEN, 'me', 'ru', 'en'))
    assert code('not json') == 'hello'
    assert code(json.dumps(good | {'type': 'pair'})) == 'hello'
    assert code(json.dumps(good | {'version': p.VERSION + 1, 'token': ''})) == 'version'  # version before the token
    assert code(json.dumps(good | {'token': ''})) == 'auth'
    assert code(p.hello(TOKEN, 'me', 'ru', 'en'), 'them') == 'hello'
    assert code(p.hello(TOKEN, 'me', 'ru', 'xx')) == 'lang'
    assert code(p.hello(TOKEN, 'me', 'ru', 'ru')) == 'lang'
    for bad in ({'sha256': 'c' * 64, 'bytes': p.VOICE_MAX + 1}, {'sha256': 'c', 'bytes': 1}, {'bytes': 1}, 'x'):
        assert code(p.hello(TOKEN, 'me', 'ru', 'en', bad)) == 'hello'


def test_events_and_speech_round_trip():
    for e in (Caption('привет', 'hello', True, 3), Caption('при'), Latency(3, 'voice', 0.25)):
        assert p.decode(p.event(e)) == e
    pcm = np.sin(np.arange(24_000 * 35 // 100) / 5).astype(np.float32) * 0.5  # 350 ms at 24 kHz
    pieces = p.speech(Speech(pcm, 24_000))
    assert len(pieces) == 4  # 100 + 100 + 100 + 50 ms
    out = [p.decode(m) for m in pieces]
    assert all(s.sample_rate == 24_000 and len(s.pcm) <= 2400 for s in out)
    assert np.abs(np.concatenate([s.pcm for s in out]) - pcm).max() < 1e-4
    assert p.decode(p.message('ready', warm=True)) == {'type': 'ready', 'warm': True}
    with pytest.raises(p.ProtocolError):
        p.decode(b'\x01\x02\x03')
    up = np.array([0.0, 0.5, -1.0, 2.0], np.float32)  # clipped
    assert np.allclose(p.pcm_from(p.pcm_bytes(up)), [0, 0.5, -1, 1], atol=1e-4)


def test_voice_pack_unpack(tmp_path):
    src = tmp_path / 'me'
    (src / 'cache').mkdir(parents=True)
    (src / 'voice.json').write_text('{"name": "me", "kind": "me"}')
    (src / 'reference.wav').write_bytes(b'RIFF' + bytes(5000))
    (src / 'cache' / 'state.bin').write_bytes(b'x' * 10)
    data = p.pack_voice(src)
    assert p.voice_ref(data) == p.voice_ref(p.pack_voice(src))  # the same profile, the same sha256 (a warm session)
    dest = tmp_path / 'out'
    dest.mkdir()
    p.unpack_voice(data, dest)
    assert sorted(x.relative_to(dest).as_posix() for x in dest.rglob('*') if x.is_file()) == \
        ['cache/state.bin', 'reference.wav', 'voice.json']
    assert (dest / 'reference.wav').read_bytes() == (src / 'reference.wav').read_bytes()


@pytest.mark.parametrize('name', ['../evil', '/abs', 'a/../../evil', 'c:/x', 'a\\..\\evil'])
def test_voice_unpack_refuses_names_outside(tmp_path, name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('voice.json', '{}')
        z.writestr(name, 'x')
    with pytest.raises(p.ProtocolError) as e:
        p.unpack_voice(buf.getvalue(), tmp_path)
    assert e.value.code == 'voice'
    assert not (tmp_path.parent / 'evil').exists() and not list(tmp_path.iterdir())


def test_voice_unpack_refuses_a_bomb_garbage_and_no_profile(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('voice.json', '{}')
        z.writestr('big', bytes(p.VOICE_MAX))  # 64 MB of zeros, ~64 KB zipped
    assert len(buf.getvalue()) < 1 << 20
    for data in (buf.getvalue(), b'not a zip'):
        with pytest.raises(p.ProtocolError):
            p.unpack_voice(data, tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('reference.wav', 'x')
    with pytest.raises(p.ProtocolError, match='voice.json'):
        p.unpack_voice(buf.getvalue(), tmp_path)


def test_pairing_macs():
    fp, nonce, code = 'f' * 64, '0' * 64, 'K7QM-2XPA-9DFE'
    m = json.loads(p.pair(code, fp, nonce, 'laptop'))
    assert m['mac'] == p.pair_mac('k7qm 2xpa 9dfe', fp, nonce)  # as typed
    assert p.pair_mac('B0I1', fp, nonce) == p.pair_mac('BOII', fp, nonce)  # base32 has no 0 / 1
    assert m['mac'] != p.pair_mac(code, 'e' * 64, nonce)  # another certificate (a man in the middle): another mac
    assert m['mac'] != p.pair_mac('K7QM-2XPA-9DFF', fp, nonce)
    r = json.loads(p.paired(code, fp, nonce, TOKEN))
    assert r['mac'] == p.pair_mac(code, fp, nonce, TOKEN) != m['mac']  # the server's proof is not the client's echoed
    assert p.fingerprint(b'der') == p.fingerprint(b'der') != p.fingerprint(b'der2')

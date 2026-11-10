"""Pairing and the server's own safety: who may use `peresmeshnik serve`, and which server the app talks to.

The server: `cert` makes a self-signed certificate once and keeps it (a new one would unpair every client); `Pairing`
shows a one-time code (12 base32 characters, 10 minutes, one use, burnt after 5 wrong tries), answers /v1/pair
(`handle`: the client's message -> the answer) and keeps the paired devices in `paired.json` (0600): the sha256 of each
token, never the token, so the file alone lets nobody in. Every call reads the file again, so `serve --revoke` in another
process takes effect on the running server's next `check`. `check_bind` refuses a public address without `public=True`
(an open port on the internet with a 60-bit code in front of it); `addresses` finds this machine's private ones for the
line `https://<ip>:47000/#<code>`.

The client: `pair` connects without checking the certificate's chain (it is self-signed), takes its fingerprint from the
TLS session, proves it knows the code for that fingerprint and checks the server's proof back (a man in the middle with
his own certificate fails both, protocol.py), then pins the fingerprint and keeps the token in `<config dir>/remote`
(0600). Every failure is a `protocol.ProtocolError` whose `code` is the server's, 'auth' (a wrong proof: not that
server) or 'internal' (no connection).
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
import socket
import ssl
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from peresmeshnik.remote import protocol
from peresmeshnik.remote.protocol import ProtocolError

log = logging.getLogger(__name__)
CODE_LEN = 12
CODE_S = 600  # a pairing code lives this long
CODE_TRIES = 5  # wrong codes before it burns
B32 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
NAME_MAX = 64
PRIVATE = [ipaddress.ip_network(n) for n in (
    '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16',  # RFC 1918
    '100.64.0.0/10',  # CGNAT: Tailscale
    '127.0.0.0/8', '::1/128',
    'fc00::/7', 'fe80::/10',  # IPv6 ULA, link-local
)]


def _write(path: Path, data: bytes) -> None:
    """Whole or not at all, 0600 (mkstemp's own mode): a crash mid-write leaves the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')


# --- the server ----------------------------------------------------------------------------------------------------

def cert(folder: Path) -> tuple[Path, Path, str]:
    """`folder`/server.crt and server.key (PEM, the key 0600), made on the first call and kept: (crt, key, fingerprint)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    crt, key = folder / 'server.crt', folder / 'server.key'
    if not (crt.exists() and key.exists()):
        k = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Peresmeshnik server')])
        now = datetime.datetime.now(datetime.UTC)
        c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
             .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
             .not_valid_after(now + datetime.timedelta(days=3650)).sign(k, hashes.SHA256()))
        _write(key, k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
        _write(crt, c.public_bytes(serialization.Encoding.PEM))
    der = x509.load_pem_x509_certificate(crt.read_bytes()).public_bytes(serialization.Encoding.DER)
    return crt, key, protocol.fingerprint(der)


class Pairing:
    """The server's pairing code and paired devices. `fp`: `cert`'s fingerprint (only `handle` needs it, not
    `serve --list / --revoke`); `clock`: time.monotonic, a test's own in tests."""

    def __init__(self, folder: Path, fp: str = '', clock: Callable[[], float] = time.monotonic):
        self.path, self.fp, self.clock = folder / 'paired.json', fp, clock
        self.code, self.until, self.tries, self.used = '', 0.0, 0, False
        self.lock = threading.Lock()  # handle() and check() may come from different threads of the server

    def new_code(self) -> str:
        """A new code (the old one stops working), shown as K7QM-2XPA-9DFE."""
        self.code = ''.join(secrets.choice(B32) for _ in range(CODE_LEN))
        self.until, self.tries, self.used = self.clock() + CODE_S, 0, False
        return '-'.join(self.code[i:i + 4] for i in range(0, CODE_LEN, 4))

    # devices: {'name', 'sha256', 'paired', 'last'}, the times in UTC ISO

    def devices(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text('utf-8'))
        except FileNotFoundError:
            return []

    def _save(self, devices: list[dict]) -> None:
        _write(self.path, json.dumps(devices, ensure_ascii=False, indent=1).encode('utf-8'))

    def add(self, name: str) -> str:
        """A new device token (32 random bytes, hex); a device of the same name is replaced (this computer paired again)."""
        token = secrets.token_hex(32)
        with self.lock:
            devices = [d for d in self.devices() if d['name'] != name]
            now = _now()
            self._save(devices + [{'name': name, 'sha256': hashlib.sha256(token.encode()).hexdigest(), 'paired': now,
                                   'last': now}])
        return token

    def check(self, token: str) -> str | None:
        """The device's name for a token from hello, or None; marks when it was last seen."""
        h = hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            devices = self.devices()
            found = None
            for d in devices:  # every one compared, in constant time each
                if hmac.compare_digest(d['sha256'], h):
                    found = d
            if found is None:
                return None
            found['last'] = _now()
            self._save(devices)  # ponytail: read-modify-write; a `serve --revoke` from another process in the same
            return found['name']  # millisecond can be undone, a file lock if that ever matters

    def revoke(self, name: str) -> bool:
        with self.lock:
            devices = self.devices()
            left = [d for d in devices if d['name'] != name]
            if len(left) < len(devices):
                self._save(left)
            return len(left) < len(devices)

    def handle(self, text: str | bytes) -> str:
        """The first and only message on /v1/pair -> the answer: `protocol.paired` or `protocol.error`."""
        try:
            m = json.loads(text)
        except (TypeError, ValueError):
            m = None
        if not isinstance(m, dict) or m.get('type') != 'pair':
            return protocol.error('hello', 'the first message on /v1/pair is not a pair')
        if m.get('version') != protocol.VERSION:
            return protocol.error('version', f'protocol {m.get("version")} here, the server speaks {protocol.VERSION}: '
                                             'update the app on both computers to the same version')
        name, nonce, mac = m.get('name'), m.get('nonce'), m.get('mac')
        if not (isinstance(name, str) and name.strip() and isinstance(nonce, str) and 0 < len(nonce) <= 128
                and isinstance(mac, str)):
            return protocol.error('hello', 'a bad pair message')
        with self.lock:
            if self.tries >= CODE_TRIES:
                return protocol.error('attempts', 'too many wrong codes: start peresmeshnik serve --pair for a new one')
            if not self.code or self.used or self.clock() > self.until:
                return protocol.error('code', 'the code was used or has expired: start peresmeshnik serve --pair for a '
                                              'new one')
            if not hmac.compare_digest(protocol.pair_mac(self.code, self.fp, nonce), mac):
                self.tries += 1
                if self.tries >= CODE_TRIES:
                    return protocol.error('attempts', 'too many wrong codes: start peresmeshnik serve --pair for a new '
                                                      'one')
                return protocol.error('code', 'a wrong code, or something between the two computers is not the server')
            self.used = True
        token = self.add(name.strip()[:NAME_MAX])
        return protocol.paired(self.code, self.fp, nonce, token)


def private(host: str) -> bool:
    """A LAN, Tailscale, loopback or link-local address (or 'localhost'); a name or 0.0.0.0 / :: (every address) is not."""
    if host == 'localhost':
        return True
    try:
        ip = ipaddress.ip_address(host.strip('[]').split('%')[0])
    except ValueError:
        return False
    ip = getattr(ip, 'ipv4_mapped', None) or ip
    return any(ip in n for n in PRIVATE if n.version == ip.version)


def check_bind(host: str, public: bool = False) -> None:
    """Raises ValueError for an address `serve` must not listen on without `--public`."""
    if not public and not private(host):
        raise ValueError(f'{host} is not a private address: the server would be open to the internet. Listen on a LAN '
                         'or Tailscale address (or 127.0.0.1 behind ssh -L), or add --public if you mean it')


def addresses() -> list[str]:
    """This machine's private addresses, loopback aside: for each private range, the source address the system would
    send from (a UDP connect sends nothing). Finds the LAN and Tailscale ones with no new dependency."""
    found = []
    for family, probe in ((socket.AF_INET, '192.168.255.255'), (socket.AF_INET, '10.255.255.255'),
                          (socket.AF_INET, '172.31.255.255'), (socket.AF_INET, '100.100.100.100'),
                          (socket.AF_INET6, 'fd00::1')):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))
                ip = s.getsockname()[0]
        except OSError:
            continue
        if private(ip) and not ipaddress.ip_address(ip).is_loopback and ip not in found:
            found.append(ip)
    return found  # ponytail: one address per range (the routed one); a second LAN card on the same range is not listed


# --- the client ----------------------------------------------------------------------------------------------------

def parse(url: str) -> tuple[str, int, str]:
    """`https://192.168.1.20:47000/#K7QM-2XPA-9DFE` (no scheme, no port, [IPv6] all fine) -> (host, port, code)."""
    s = url.strip()
    try:
        u = urlsplit(s if '://' in s else 'https://' + s)
        port = u.port or protocol.PORT
    except ValueError:
        u = None
    if u is None or u.scheme not in ('https', 'wss') or not u.hostname or u.path not in ('', '/') or u.query:
        raise ValueError(f'{url!r} is not a server address: copy the whole line peresmeshnik serve shows, like '
                         f'https://192.168.1.20:{protocol.PORT}/#K7QM-2XPA-9DFE')
    code = protocol.norm_code(u.fragment)
    if len(code) != CODE_LEN or set(code) - set(B32):
        raise ValueError(f'no pairing code after # in {url!r}: it is 12 letters and digits, like K7QM-2XPA-9DFE')
    return u.hostname, port, code


def remote_path() -> Path:
    from peresmeshnik import config
    return config.config_dir() / 'remote'


def pair(url: str, name: str, path: Path | None = None) -> dict:
    """Pairs with the server at `url` (`parse`) as `name`, saves and returns {address, fp, token, name}. Blocks (seconds)."""
    host, port, code = parse(url)
    address = f'[{host}]:{port}' if ':' in host else f'{host}:{port}'
    try:
        fp, nonce, answer = asyncio.run(_pair(address, code, name))
    except ProtocolError:
        raise
    except Exception as e:  # OSError, TimeoutError, websockets' handshake and close errors
        raise ProtocolError('internal', f'cannot pair with {address}: {e}') from None
    if answer.get('type') == 'error':
        raise ProtocolError(answer.get('code', 'internal'), answer.get('text', 'the server refused'))
    token = answer.get('token')
    if (answer.get('type') != 'paired' or not isinstance(token, str) or not token
            or not hmac.compare_digest(protocol.pair_mac(code, fp, nonce, token), str(answer.get('mac')))):
        raise ProtocolError('auth', f'{address} does not prove it knows the code: it is not that server, or something '
                                    'between the two computers is. Nothing was saved')
    remote = {'address': address, 'fp': fp, 'token': token, 'name': name}
    _write(path or remote_path(), json.dumps(remote, ensure_ascii=False).encode('utf-8'))
    return remote


async def _pair(address: str, code: str, name: str) -> tuple[str, str, dict]:
    from websockets.asyncio.client import connect

    from peresmeshnik import __version__
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE  # self-signed: the fingerprint is checked by the code
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    async with connect(f'wss://{address}{protocol.PAIR_PATH}', ssl=ctx, proxy=None, compression=None,
                       open_timeout=5, close_timeout=1) as ws:
        fp = protocol.fingerprint(ws.transport.get_extra_info('ssl_object').getpeercert(binary_form=True))
        nonce = secrets.token_hex(32)
        await ws.send(protocol.pair(code, fp, nonce, name, app=__version__))
        answer = json.loads(await asyncio.wait_for(ws.recv(), 10))
    return fp, nonce, answer if isinstance(answer, dict) else {}


def load(path: Path | None = None) -> dict | None:
    """The saved pairing, or None (not paired, or the file is broken: pair again)."""
    path = path or remote_path()
    try:
        d = json.loads(path.read_text('utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        log.warning('%s cannot be read (%s): pair this computer again', path, e)
        return None
    return d if isinstance(d, dict) and all(isinstance(d.get(k), str) for k in ('address', 'fp', 'token')) else None


def forget(path: Path | None = None) -> None:
    """Drops the pairing here: the file goes, not aside (it holds a live token). The server keeps the device until
    `serve --revoke`."""
    (path or remote_path()).unlink(missing_ok=True)

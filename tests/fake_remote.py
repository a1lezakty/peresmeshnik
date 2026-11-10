"""A `peresmeshnik serve` stand-in for the client's tests: the real protocol (peresmeshnik/remote/protocol.py) over a real
WebSocket and TLS on 127.0.0.1, on its own thread; the test says what it answers and when, and can drop the connection
(a reset) or stop reading it (a Wi-Fi gone without a reset: no pongs)."""
import asyncio
import datetime
import json
import ssl
import tempfile
import threading
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from peresmeshnik.remote import protocol

_CERT: tuple[Path, Path, str] | None = None


def cert() -> tuple[Path, Path, str]:
    """A self-signed certificate for the tests: (cert.pem, key.pem, its fingerprint)."""
    global _CERT
    if _CERT is None:
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'test server')])
        now = datetime.datetime.now(datetime.UTC)
        c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
             .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
             .not_valid_after(now + datetime.timedelta(days=30)).sign(key, hashes.SHA256()))
        d = Path(tempfile.mkdtemp(prefix='fake-remote-'))
        (d / 'cert.pem').write_bytes(c.public_bytes(serialization.Encoding.PEM))
        (d / 'key.pem').write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                      serialization.NoEncryption()))
        _CERT = d / 'cert.pem', d / 'key.pem', protocol.fingerprint(c.public_bytes(serialization.Encoding.DER))
    return _CERT


class Conn:
    def __init__(self, ws, hello):
        self.ws, self.hello, self.audio, self.texts = ws, hello, bytearray(), []


class FakeServer:
    """`refuse`: (code, text) answers every hello with that error; `ready_delay`: seconds before `ready`, None: never;
    `after_ready`: messages sent right after each `ready`, then the connection closes (a server that crashes at once);
    or a function of the connection's number (1, 2, ...) giving them, None: it lives."""

    def __init__(self):
        self.refuse: tuple[str, str] | None = None
        self.ready_delay: float | None = 0.0
        self.after_ready: list | None = None
        self.voices: list[bytes] = []
        self.have: set[str] = set()  # voice sha256s it keeps (a warm session)
        self.conns: list[Conn] = []
        self.hellos: list[dict] = []
        self.loop = asyncio.new_event_loop()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.server = self.call(self._start())
        self.port = self.server.sockets[0].getsockname()[1]
        self.address, self.fp = f'127.0.0.1:{self.port}', cert()[2]

    def call(self, coro, timeout=5):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _start(self):
        from websockets.asyncio.server import serve
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(*map(str, cert()[:2]))
        return await serve(self._handler, '127.0.0.1', 0, ssl=ctx, compression=None, max_size=protocol.CHUNK,
                           ping_interval=None)

    async def _handler(self, ws):
        side = ws.request.path.rsplit('/', 1)[-1]
        try:
            h = protocol.parse_hello(await ws.recv(), side)
        except protocol.ProtocolError as e:
            await ws.send(protocol.error(e.code, str(e)))
            return
        self.hellos.append(h)
        if self.refuse:
            await ws.send(protocol.error(*self.refuse))
            return
        if (v := h['voice']) and v['sha256'] not in self.have:
            await ws.send(protocol.message('send_voice'))
            data = b''
            while len(data) < v['bytes']:
                data += await ws.recv()
            self.voices.append(data)
            self.have.add(protocol.voice_ref(data)['sha256'])
        if self.ready_delay is None:
            await ws.wait_closed()
            return
        await asyncio.sleep(self.ready_delay)
        c = Conn(ws, h)
        self.conns.append(c)
        await ws.send(protocol.message('ready', warm=False, device='cpu'))
        after = self.after_ready(len(self.hellos)) if callable(self.after_ready) else self.after_ready
        if after is not None:
            for m in after:
                await ws.send(m)
            return
        async for m in ws:
            if isinstance(m, bytes):
                c.audio += m
            else:
                c.texts.append(json.loads(m))

    # from the test's thread
    def send(self, *msgs, conn=-1):
        async def go():
            for m in msgs:
                await self.conns[conn].ws.send(m)
        self.call(go())

    def drop(self, conn=-1):
        """A reset: the socket goes without a close frame."""
        self.loop.call_soon_threadsafe(self.conns[conn].ws.transport.abort)

    def deafen(self, conn=-1):
        """Stops reading the connection: no pongs, the client's audio piles up (a network gone without a reset)."""
        self.loop.call_soon_threadsafe(self.conns[conn].ws.transport.pause_reading)

    def close(self):
        if not self.loop.is_running():
            return

        async def go():
            self.server.close()
            for c in self.conns:
                c.ws.transport.abort()
            await self.server.wait_closed()
        self.call(go())
        self.loop.call_soon_threadsafe(self.loop.stop)

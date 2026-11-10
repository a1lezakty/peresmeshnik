"""`peresmeshnik serve` without models, for tests: stub engines behind the server's sides (the app's own Side; the fixture
`made` is in conftest.py), the server on 127.0.0.1 on its own thread, my confirmed voice profile. Used by
test_remote_server.py and test_remote_e2e.py."""
import asyncio
import json
import os
import queue
import threading

from peresmeshnik import config
from peresmeshnik.remote import protocol, server


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


def start(folder, host='127.0.0.1', port=0, **kw):
    srv = server.Server(folder, config.Settings(), host, port, **kw)
    srv.thread = threading.Thread(target=asyncio.run, args=(srv.serve(),), daemon=True)
    srv.thread.start()
    assert srv.started.wait(10)
    srv.address = f'127.0.0.1:{srv.ports[0]}'
    return srv


CONFIRMED = {'version': 2, 'kind': 'me', 'name': 'me', 'consent': {'how': 'mic-live'}}  # voice.json v2 (S3); its
# consent names the recording it checked (ref_sha256): profile() fills it in


def profile(folder, size=int(2.5 * protocol.CHUNK), meta=CONFIRMED):
    """My confirmed voice (a recording as big as `size`), and engines' caches the server must not keep: a pickle
    (xtts.pt) and a state nothing on the server reads (state.npy)."""
    import hashlib
    folder.mkdir()
    (folder / 'reference.wav').write_bytes(wav := os.urandom(size))
    if isinstance(meta, dict) and isinstance(c := meta.get('consent'), dict) and 'ref_sha256' not in c:
        meta = {**meta, 'consent': {**c, 'ref_sha256': hashlib.sha256(wav).hexdigest()}}
    (folder / 'voice.json').write_text(json.dumps(meta))
    (folder / 'state.npy').write_bytes(b'a state')
    (folder / 'xtts.pt').write_bytes(b'a pickle')
    return protocol.pack_voice(folder)

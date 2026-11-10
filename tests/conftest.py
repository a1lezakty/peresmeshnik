"""Shared fixtures: a local HTTP server with Range support (model downloads)."""
import http.server
import os
import sys
import threading
import time

import pytest

from peresmeshnik import models

if sys.platform == 'darwin':
    # the macOS runner has no microphone: its tests record BlackHole 16ch, which the app refuses as the microphone
    # (a virtual device); audio/macos.stand_in_mic() lets this one through, and only under pytest
    os.environ.setdefault('PERESMESHNIK_TEST_MIC', 'BlackHole16ch_UID')


class Range(http.server.BaseHTTPRequestHandler):
    """Static files with HTTP Range, like Hugging Face and GitHub release downloads. Records every
    request's Range header; a path in `cut` gets half of its body once, then the connection closes; a path in `whole`
    ignores Range (200 and the whole file, as some proxies and mirrors do); `slow`: seconds before each body."""
    def do_GET(self):
        srv = self.server
        body = (srv.root / self.path.lstrip('/')).read_bytes()
        rng = self.headers.get('Range')
        srv.ranges.append((self.path, rng))
        if self.path in srv.whole:
            rng = None
        start = int(rng.split('=')[1].rstrip('-')) if rng else 0
        self.send_response(206 if rng else 200)
        self.send_header('Content-Length', str(len(body) - start))
        self.end_headers()
        out = body[start:]
        time.sleep(srv.slow)  # a slow server: two downloads of one model overlap
        if self.path in srv.cut:
            srv.cut.discard(self.path)
            out = out[:len(out) // 2]  # HTTP/1.0: the socket closes after this, short of Content-Length
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(models, 'RETRY_PAUSE_S', 0.0)
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Range)
    httpd.root, httpd.ranges, httpd.cut, httpd.whole, httpd.slow = tmp_path / 'srv', [], set(), set(), 0.0
    httpd.root.mkdir()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, f'http://127.0.0.1:{httpd.server_port}/'
    httpd.shutdown()


@pytest.fixture
def made(monkeypatch):
    """Every stub engine the remote server's sides made, in order (app.Side._local, so the server's Side runs as in the
    app); tests/remote_stubs.py."""
    from remote_stubs import Fake

    from peresmeshnik import app
    out = []

    def local(side, s):
        out.append(Fake(side, s))
        return out[-1]
    monkeypatch.setattr(app.Side, '_local', local)
    return out

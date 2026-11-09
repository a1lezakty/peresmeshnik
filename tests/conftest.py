"""Shared fixtures: a local HTTP server with Range support (model downloads)."""
import http.server
import os
import threading

import pytest

from peresmeshnik import models

# the macOS runner's stand-in microphone: a virtual device, refused as the microphone otherwise (audio/macos.py)
os.environ.setdefault('PERESMESHNIK_TEST_MIC', 'BlackHole16ch_UID')


class Range(http.server.BaseHTTPRequestHandler):
    """Static files with HTTP Range, like Hugging Face and GitHub release downloads. Records every
    request's Range header; a path in `cut` gets half of its body once, then the connection closes."""
    def do_GET(self):
        srv = self.server
        body = (srv.root / self.path.lstrip('/')).read_bytes()
        rng = self.headers.get('Range')
        srv.ranges.append((self.path, rng))
        start = int(rng.split('=')[1].rstrip('-')) if rng else 0
        self.send_response(206 if rng else 200)
        self.send_header('Content-Length', str(len(body) - start))
        self.end_headers()
        out = body[start:]
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
    httpd.root, httpd.ranges, httpd.cut = tmp_path / 'srv', [], set()
    httpd.root.mkdir()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, f'http://127.0.0.1:{httpd.server_port}/'
    httpd.shutdown()

"""GitHub's REST API as the updater uses it (peresmeshnik/update.py), on 127.0.0.1 on its own thread: the releases list,
their files with a .sha256 each as release.yml writes it, and the attestations by digest (their bundles at a
`bundle_url`, Snappy, as GitHub serves them). The app is pointed at it with
$PERESMESHNIK_UPDATE_API. `requests` lists every path asked for.

    with FakeGitHub() as gh:
        gh.publish('0.3.4', {'Peresmeshnik-0.3.4-macos-arm64.dmg': data})
        os.environ['PERESMESHNIK_UPDATE_API'] = gh.url
"""
import base64
import hashlib
import http.server
import json
import threading
import urllib.parse

from peresmeshnik.update import REPO, WORKFLOW


def snappy(data: bytes) -> bytes:
    """Snappy's raw block format, literals only: how `bundle_url` serves a bundle (there compressed)."""
    out, n = bytearray(), len(data)
    while n > 0x7f:
        out.append(n & 0x7f | 0x80)
        n >>= 7
    out.append(n)
    for i in range(0, len(data), 1 << 16):
        chunk = data[i:i + (1 << 16)]
        out += bytes([61 << 2]) + (len(chunk) - 1).to_bytes(2, 'little') + chunk  # 61: the length in 2 bytes
    return bytes(out)


def statement(name: str, digest: str, version: str, repo: str = REPO, workflow: str = WORKFLOW) -> dict:
    """An attestation's Sigstore bundle, made by `workflow` of `repo` from the tag v<version> (its signature is no
    signature: the app does not verify it)."""
    st = {'_type': 'https://in-toto.io/Statement/v1', 'subject': [{'name': name, 'digest': {'sha256': digest}}],
          'predicateType': 'https://slsa.dev/provenance/v1',
          'predicate': {'buildDefinition': {'externalParameters': {'workflow': {
              'ref': f'refs/tags/v{version}', 'repository': f'https://github.com/{repo}', 'path': workflow}}}}}
    return {'mediaType': 'application/vnd.dev.sigstore.bundle.v0.3+json', 'dsseEnvelope': {
        'payloadType': 'application/vnd.in-toto+json', 'payload': base64.b64encode(json.dumps(st).encode()).decode()}}


class FakeGitHub(http.server.ThreadingHTTPServer):
    def __init__(self):
        super().__init__(('127.0.0.1', 0), _Handler)
        self.releases: list[dict] = []
        self.files: dict[str, bytes] = {}
        self.attestations: dict[str, list] = {}  # digest -> the API's entries
        self.bundles: dict[str, bytes] = {}  # served at /bundles/<n>, Snappy
        self.requests: list[str] = []
        self.url = f'http://127.0.0.1:{self.server_address[1]}'

    def publish(self, version: str, files: dict[str, bytes], draft: bool = False, prerelease: bool = False,
                attest: bool = True, sha: str | None = None, tag: str | None = None) -> dict:
        """A release with `files` and their .sha256 (`sha`: another digest in them), attested unless `attest` is False."""
        assets = []
        for name, data in files.items():
            digest = hashlib.sha256(data).hexdigest()
            for n, d in ((name, data), (f'{name}.sha256', f'{sha or digest}  {name}\n'.encode())):
                self.files[f'{version}/{n}'] = d
                assets.append({'name': n, 'size': len(d), 'browser_download_url': f'{self.url}/download/{version}/{n}'})
            if attest:
                self.attest(digest, statement(name, digest, version))
        rel = {'tag_name': tag or f'v{version}', 'draft': draft, 'prerelease': prerelease, 'assets': assets,
               'body': f'## {version}\n\nWhat is new.', 'html_url': f'https://github.com/{REPO}/releases/tag/v{version}'}
        self.releases.append(rel)
        return rel

    def attest(self, digest: str, bundle: dict, inline: bool = False) -> None:
        """An attestation of `digest`: as GitHub gives it now (no bundle inline, its `bundle_url`), or `inline`."""
        if inline:
            entry = {'bundle': bundle}
        else:
            n = str(len(self.bundles))
            self.bundles[n] = snappy(json.dumps(bundle).encode())
            entry = {'bundle': None, 'bundle_url': f'{self.url}/bundles/{n}'}
        self.attestations.setdefault(digest, []).append({'repository_id': 1, **entry})

    def __enter__(self):
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server's name)
        gh, path = self.server, urllib.parse.urlsplit(self.path).path
        gh.requests.append(path)
        if path == f'/repos/{REPO}/releases':
            return self._send(json.dumps(gh.releases).encode())
        if path.startswith(f'/repos/{REPO}/attestations/sha256:'):
            found = gh.attestations.get(path.rsplit(':', 1)[1])
            return self._send(json.dumps({'attestations': found}).encode()) if found else self._send(b'{}', 404)
        if path.startswith('/download/') and (data := gh.files.get(path.removeprefix('/download/'))) is not None:
            return self._send(data)
        if path.startswith('/bundles/') and (data := gh.bundles.get(path.removeprefix('/bundles/'))) is not None:
            return self._send(data)
        self._send(b'{"message": "Not Found"}', 404)

    def _send(self, body: bytes, code: int = 200):
        self.send_response(code)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

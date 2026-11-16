"""CA certificates for HTTPS (peresmeshnik/__init__.py): a Python whose OpenSSL finds none where it was built to look
(the .app on a Mac without python.org's Python) gets certifi's bundle through SSL_CERT_FILE; one the user set wins."""
import os
import shutil
import ssl
import subprocess
import sys
import types

import certifi
import pytest

import peresmeshnik

NONE = ssl.DefaultVerifyPaths(None, None, 'SSL_CERT_FILE', '/nowhere/cert.pem', 'SSL_CERT_DIR', '/nowhere/certs')
EMPTY = types.SimpleNamespace(cert_store_stats=lambda: {'x509': 0, 'crl': 0, 'x509_ca': 0})


@pytest.fixture
def bare(monkeypatch):
    """No SSL_CERT_* from the user; OpenSSL finds no CA certificates of its own (and no Windows store)."""
    monkeypatch.delenv('SSL_CERT_FILE', raising=False)
    monkeypatch.delenv('SSL_CERT_DIR', raising=False)
    monkeypatch.setattr(ssl, 'get_default_verify_paths', lambda: NONE)
    monkeypatch.setattr(ssl, 'create_default_context', lambda *a, **k: EMPTY)


def test_no_system_certificates_take_certifis(bare):
    peresmeshnik._ca_bundle()
    assert os.environ['SSL_CERT_FILE'] == certifi.where() and os.path.isfile(certifi.where())


@pytest.mark.parametrize('var', ['SSL_CERT_FILE', 'SSL_CERT_DIR'])
def test_the_users_setting_wins(bare, monkeypatch, tmp_path, var):
    monkeypatch.setenv(var, str(tmp_path / 'mine'))
    peresmeshnik._ca_bundle()
    assert os.environ[var] == str(tmp_path / 'mine')
    assert os.environ.get({'SSL_CERT_FILE': 'SSL_CERT_DIR', 'SSL_CERT_DIR': 'SSL_CERT_FILE'}[var]) is None


def test_system_certificates_stay(bare, monkeypatch, tmp_path):
    (tmp_path / 'cert.pem').write_text('')
    monkeypatch.setattr(ssl, 'get_default_verify_paths', lambda: NONE._replace(cafile=str(tmp_path / 'cert.pem')))
    peresmeshnik._ca_bundle()
    assert 'SSL_CERT_FILE' not in os.environ
    hashed = tmp_path / 'certs'  # Debian's /etc/ssl/certs: no cert.pem, a folder of c_rehash links
    hashed.mkdir()
    (hashed / '0123abcd.0').write_text('')
    monkeypatch.setattr(ssl, 'get_default_verify_paths', lambda: NONE._replace(capath=str(hashed)))
    peresmeshnik._ca_bundle()
    assert 'SSL_CERT_FILE' not in os.environ
    monkeypatch.setattr(ssl, 'get_default_verify_paths', lambda: NONE)  # Windows: its system store, loaded by the context
    monkeypatch.setattr(ssl, 'create_default_context', lambda *a, **k: types.SimpleNamespace(
        cert_store_stats=lambda: {'x509': 140, 'crl': 0, 'x509_ca': 140}))
    peresmeshnik._ca_bundle()
    assert 'SSL_CERT_FILE' not in os.environ


HIDE = ('for d in /etc/ssl /usr/lib/ssl /etc/pki /usr/share/ca-certificates; do [ -d $d ] && mount -t tmpfs none $d; done; '
        'exec "$@"')
GET = 'import urllib.request; print(urllib.request.urlopen("https://huggingface.co/", timeout=30).status)'


@pytest.mark.skipif(not sys.platform.startswith('linux') or not shutil.which('unshare'), reason='Linux: unshare')
def test_a_real_host_verifies_with_the_system_certificates_hidden():
    """This Python with the box's CA folders hidden (an empty tmpfs over them, a private mount namespace): plain
    urllib fails as on the Mac; after `import peresmeshnik` the same request verifies huggingface.co's certificate."""
    env = {k: v for k, v in os.environ.items() if k not in ('SSL_CERT_FILE', 'SSL_CERT_DIR')}
    unshare = ['unshare', '-m', '--propagation', 'private'] if os.geteuid() == 0 else ['unshare', '-r', '-m']

    def run(code):
        return subprocess.run([*unshare, 'sh', '-c', HIDE, '_', sys.executable, '-c', code], capture_output=True,
                              text=True, timeout=120, env=env)

    before = run(GET)
    if 'CERTIFICATE_VERIFY_FAILED' not in before.stderr:  # no namespace here, or no network: nothing shown
        pytest.skip(f'could not hide the certificates or reach the host: {(before.stdout + before.stderr)[-300:]}')
    after = run('import peresmeshnik; ' + GET)
    assert after.returncode == 0 and after.stdout.strip() == '200', after.stderr[-2000:]

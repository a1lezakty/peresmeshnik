"""Peresmeshnik: real-time call translator."""
import os

# Before anything imports numpy: its BLAS pool would otherwise take every core (the app imports numpy before
# the pipeline: 3.0 instead of 0.8 CPU s per audio s for them -> me on a 16-core box). Runtimes get their
# thread counts explicitly; OpenMP pools sleep between calls.
for _k, _v in {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'VECLIB_MAXIMUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE'}.items():
    os.environ.setdefault(_k, _v)
# onnxruntime's own telemetry (Linux, macOS: events to Microsoft at every session) is on by default: off before its
# first import, here and in the helpers that inherit this environment (the Piper runner). Windows: local/vad.ort_options
os.environ.setdefault('ORT_DISABLE_TELEMETRY', '1')


def _ca_bundle() -> None:
    """Certificates for HTTPS (model downloads, the cloud, the voice pack): when this Python's OpenSSL finds no CA
    certificates where it was built to look, it gets certifi's bundle through SSL_CERT_FILE, which every default
    context reads (urllib, websockets, httpx, requests) here and in the helpers that inherit the environment.
    The .app is built with python.org's Python, whose OpenSSL looks in its own framework folder: a Mac without that
    Python had none, and every download failed with CERTIFICATE_VERIFY_FAILED. Windows' own store and a Linux /etc/ssl
    count as found; SSL_CERT_FILE or SSL_CERT_DIR set by the user win. Verification itself is never turned off; the
    remote server's contexts (pinned self-signed certificate) load no CA certificates and are not touched."""
    if os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR'):
        return
    import ssl
    p = ssl.get_default_verify_paths()
    try:
        hashed = bool(p.capath) and any(n.endswith('.0') for n in os.listdir(p.capath))  # OpenSSL's c_rehash names
    except OSError:
        hashed = False
    if p.cafile or hashed or ssl.create_default_context().cert_store_stats()['x509_ca']:  # Windows: its system store
        return
    try:
        import certifi
    except ImportError:  # a build without it: started all the same, HTTPS fails as before and says why
        return
    os.environ['SSL_CERT_FILE'] = certifi.where()


_ca_bundle()

__version__ = '0.3.3'

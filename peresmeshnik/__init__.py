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

__version__ = '0.3.0'

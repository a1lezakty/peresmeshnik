"""Device choice for them -> me (local/device.py): forcing CUDA on a machine without it falls back to CPU with a
logged message (the exit criterion's CUDA -> CPU fallback), and auto is CPU there. Runs on the CPU CI runners (the
onnxruntime build has no CUDAExecutionProvider available), so the fallback is observable without a GPU.
"""
import logging

import pytest

from peresmeshnik.local import device


def _has_cuda() -> bool:
    try:
        import onnxruntime as rt
        return 'CUDAExecutionProvider' in rt.get_available_providers()
    except Exception:
        return False


def test_forced_cuda_without_a_gpu_falls_back_to_cpu(monkeypatch, caplog):
    pytest.importorskip('onnxruntime')
    if _has_cuda():
        pytest.skip('this runner has CUDA: the fallback is not observable here')
    monkeypatch.setenv('PERESMESHNIK_DEVICE', 'cuda')
    with caplog.at_level(logging.WARNING, logger='peresmeshnik.device'):
        assert device.onnx_cuda() is False
        assert device.onnx_providers() == ['CPUExecutionProvider']
    assert any('CPU' in r.message or 'cpu' in r.message for r in caplog.records), 'no CPU-fallback message'


def test_auto_without_a_gpu_is_cpu(monkeypatch):
    pytest.importorskip('onnxruntime')
    if _has_cuda():
        pytest.skip('this runner has CUDA')
    monkeypatch.setenv('PERESMESHNIK_DEVICE', 'auto')
    assert device.onnx_providers() == ['CPUExecutionProvider']


def test_cpu_forced_is_cpu(monkeypatch):
    monkeypatch.setenv('PERESMESHNIK_DEVICE', 'cpu')
    assert device.onnx_cuda() is False
    assert device.ct2_cuda() is False
    assert device.onnx_providers() == ['CPUExecutionProvider']


def test_on_cuda_reflects_the_sessions_real_providers():
    """asr._on_cuda reports the truth from the session's get_providers(): a CPU session is not CUDA, a CUDA session
    is, and a model with no discoverable session is not claimed as CUDA (do not trust what cannot be confirmed)."""
    from peresmeshnik.local.asr import _on_cuda

    class Holder:
        def __init__(self, providers):
            self.sess = type('S', (), {'get_providers': lambda self: providers})()
    assert _on_cuda(Holder(['CPUExecutionProvider'])) is False
    assert _on_cuda(Holder(['CUDAExecutionProvider', 'CPUExecutionProvider'])) is True
    assert _on_cuda(object()) is False


def test_mt_falls_back_when_cuda_fails_at_the_first_translation(monkeypatch, tmp_path):
    """A driver without the CUDA libraries: CTranslate2 sees the GPU and loads the model, but cuBLAS fails at the
    first translation (libcublas.so.12 not found). The engine finds that while it is built and goes to the CPU,
    instead of the pipeline's MT worker dying at the first clause."""
    ctranslate2 = pytest.importorskip('ctranslate2')
    spm = pytest.importorskip('sentencepiece')
    from peresmeshnik import models
    from peresmeshnik.local import mt

    class Translator:
        def __init__(self, path, device, **kw):
            self.device = device

        def translate_batch(self, batch, **kw):
            if self.device == 'cuda':
                raise RuntimeError('Library libcublas.so.12 is not found or cannot be loaded')
            return [type('R', (), {'hypotheses': [['▁ok']]})()]

    class Sp:
        def __init__(self, path):
            pass

        def encode(self, text, out_type=str):
            return ['▁hi']

        def decode(self, toks):
            return 'ok'
    monkeypatch.setattr(device, 'ct2_cuda', lambda: True)
    monkeypatch.setattr(ctranslate2, 'Translator', Translator)
    monkeypatch.setattr(spm, 'SentencePieceProcessor', Sp)
    monkeypatch.setattr(models, 'path', lambda model_id: tmp_path)
    e = mt.OpusMt('en', 'ru')
    assert e.device == 'cpu' and e.translate('Hello.', 'en', 'ru') == 'ok'

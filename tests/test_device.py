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

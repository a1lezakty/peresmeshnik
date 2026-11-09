"""Pick CPU or CUDA for the local them -> me engines (ASR on onnxruntime, MT on CTranslate2).

Auto by default: CUDA when the GPU build and a device are both there, else CPU. `PERESMESHNIK_DEVICE=cpu|cuda|auto`
overrides. CUDA is used only for them -> me (ASR, MT); the clone (me -> them) is S2's. A CUDA that fails to start
falls back to CPU with one logged message (the caller rebuilds on CPU). On Windows the CUDA runtime comes with the
voice pack (S2) and is unproven there -> USER-CHECKS.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger('peresmeshnik.device')


def want() -> str:
    """'cpu' | 'cuda' | 'auto' from the environment (default auto)."""
    v = os.environ.get('PERESMESHNIK_DEVICE', 'auto').lower()
    return v if v in ('cpu', 'cuda', 'auto') else 'auto'


def onnx_cuda() -> bool:
    """True when onnxruntime can run on CUDA now (the gpu build and a usable device)."""
    if want() == 'cpu':
        return False
    try:
        import onnxruntime as rt
        if 'CUDAExecutionProvider' not in rt.get_available_providers():
            return want() == 'cuda' and _forced('onnxruntime has no CUDAExecutionProvider (install onnxruntime-gpu)')
        return rt.get_device() == 'GPU' or want() == 'cuda'
    except Exception as e:
        log.warning('device: onnxruntime CUDA check failed: %s', e)
        return False


def ct2_cuda() -> bool:
    """True when CTranslate2 sees a CUDA device."""
    if want() == 'cpu':
        return False
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception as e:
        log.warning('device: CTranslate2 CUDA check failed: %s', e)
        return want() == 'cuda' and _forced(str(e))


def _forced(why: str) -> bool:
    log.warning('device: PERESMESHNIK_DEVICE=cuda but %s; using CPU', why)
    return False


def onnx_providers() -> list:
    """Execution providers for an onnxruntime session: CUDA first when available, CPU always as the fallback."""
    return (['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnx_cuda() else ['CPUExecutionProvider'])

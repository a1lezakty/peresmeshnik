"""Does a loaded model keep burning CPU after an inference? (NOTE: idle must be ~0 CPU.)

Loads a small model of each runtime, runs it once, then sleeps and measures process CPU time.

    python bench/idle_loaded.py ort  [--no-spin]   # onnxruntime (onnx-asr GigaAM v3 CTC)
    OMP_WAIT_POLICY=PASSIVE python bench/idle_loaded.py ct2   # CTranslate2 (faster-whisper small)
    python bench/idle_loaded.py torch               # torch CPU ops
"""
import argparse
import os
import time

import numpy as np

from common import record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runtime', choices=['ort', 'ct2', 'torch'])
    ap.add_argument('--no-spin', action='store_true', help='onnxruntime: session.intra_op.allow_spinning=0')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--idle', type=float, default=20)
    a = ap.parse_args()
    x = np.random.default_rng(0).standard_normal(16000 * 5).astype(np.float32) * 0.1
    if a.runtime == 'ort':
        import onnx_asr
        import onnxruntime as rt
        so = rt.SessionOptions()
        so.intra_op_num_threads = a.threads
        if a.no_spin:
            so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        m = onnx_asr.load_model('gigaam-v3-ctc', sess_options=so, providers=['CPUExecutionProvider'])

        def run():
            m.recognize(x)
    elif a.runtime == 'ct2':
        from faster_whisper import WhisperModel
        m = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=a.threads)

        def run():
            list(m.transcribe(x, language='en', beam_size=1)[0])
    else:
        import torch
        torch.set_num_threads(a.threads)
        lin = torch.nn.Linear(2048, 2048)

        def run():
            with torch.no_grad():
                for _ in range(50):
                    lin(torch.randn(64, 2048))
    run()
    t, c = time.perf_counter(), time.process_time()
    run()
    busy_cpu, busy_wall = time.process_time() - c, time.perf_counter() - t
    c = time.process_time()
    time.sleep(a.idle)
    idle_cpu = time.process_time() - c
    record('idle_loaded', a.runtime, threads=a.threads, no_spin=a.no_spin, omp_wait_policy=os.environ.get('OMP_WAIT_POLICY', ''),
           busy_cpu_s=round(busy_cpu, 3), busy_wall_s=round(busy_wall, 3), idle_s=a.idle, idle_cpu_s=round(idle_cpu, 4))


if __name__ == '__main__':
    main()

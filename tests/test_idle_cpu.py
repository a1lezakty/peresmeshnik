"""Idle-CPU watchdog: with translation off the app must not burn CPU.

Same check as Zagon (0.13 s CPU per 60 s idle). Window and limit can be overridden for
quick local runs: IDLE_WINDOW_S=5 pytest tests/test_idle_cpu.py
"""
import os
import subprocess
import sys
import time

import psutil

WINDOW_S = float(os.environ.get('IDLE_WINDOW_S', 60))
LIMIT_S = float(os.environ.get('IDLE_CPU_LIMIT_S', 0.13)) * WINDOW_S / 60
# macOS: AppKit does one-off work in the first seconds after the windows are first shown (0.07-0.09 s, BENCH);
# the gate measures the settled idle after it, the first seconds are printed, not gated
SETTLE_S = 15.0 if sys.platform == 'darwin' else 0.0


def cpu_s(proc):
    """CPU seconds of the process and all its children."""
    total = 0.0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.NoSuchProcess:
            pass
    return total


def test_idle_cpu():
    app = subprocess.Popen([sys.executable, '-m', 'peresmeshnik'], stdout=subprocess.PIPE, text=True)
    try:
        assert 'idle' in app.stdout.readline()  # started; startup cost is not counted
        proc = psutil.Process(app.pid)
        before = cpu_s(proc)
        if SETTLE_S:
            time.sleep(SETTLE_S)
            settle, before = cpu_s(proc) - before, cpu_s(proc)
            print(f'idle CPU in the first {SETTLE_S:.0f} s after "idle" (not gated): {settle:.3f} s')
        time.sleep(WINDOW_S)
        used = cpu_s(proc) - before
        assert app.poll() is None, 'app exited while idle'
    finally:
        app.terminate()
        app.wait(10)
    print(f'idle CPU: {used:.3f} s in {WINDOW_S:.0f} s')
    assert used < LIMIT_S, f'{used:.3f} s CPU in {WINDOW_S:.0f} s idle, limit {LIMIT_S:.3f} s'

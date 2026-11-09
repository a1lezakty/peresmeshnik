"""A writer into BlackHole 2ch goes while the call app reads it and another writer goes on: nothing of the old sound may
stay. Found on the runner: a helper killed (SIGKILL) with its IOProc running sometimes left its last 10.7 ms buffer
in coreaudiod's mix, looping (harmonics of 93.75 Hz) with no process left; the app stops helpers with SIGTERM, and the
helper stops its IOProcs before it exits. SIGKILL is not tried here: when that buffer stays, it stays for the rest of
the machine's session, and every later test on BlackHole 2ch hears it (the runner showed it at 656 Hz = 7 x 93.75)."""
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from peresmeshnik.audio import macos

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS only')
SR = 16_000


def writer(f, cycle_loop=False):
    if cycle_loop:  # the loop: 1024-frame cycle; fed from BlackHole 16ch, where a player puts the tone
        feed = macos.spawn('play', '--to', 'BlackHole16ch_UID', '--rate', str(SR), stdout=subprocess.DEVNULL)
        loop = macos.spawn('loop', '--from', 'BlackHole16ch_UID', '--to', macos.BLACKHOLE_UID, '--always', stdout=subprocess.DEVNULL)
        procs = [feed, loop]
    else:
        feed = macos.spawn('play', '--to', macos.BLACKHOLE_UID, '--rate', str(SR), stdout=subprocess.DEVNULL)
        procs = [feed]
    stop = threading.Event()
    chunk = (0.3 * np.sin(2 * np.pi * f * np.arange(SR // 10) / SR)).astype('<f4').tobytes()

    def run():
        t0 = time.monotonic()
        for i in range(10 ** 6):
            if stop.wait(max(0.0, t0 + i * 0.1 - 0.2 - time.monotonic())):
                return
            try:
                feed.stdin.write(chunk)
                feed.stdin.flush()
            except (BrokenPipeError, ValueError):
                return
    threading.Thread(target=run, daemon=True).start()
    return procs, stop


W = 320  # 20 ms windows


def tone(y, f):
    """Per window: loud with its peak at f; loud."""
    wins = y[:len(y) // W * W].reshape(-1, W)
    peaks = np.fft.rfftfreq(W, 1 / SR)[np.argmax(np.abs(np.fft.rfft(wins, axis=1)), axis=1)]
    loud = np.sqrt(np.mean(wins ** 2, axis=1)) > 0.02
    return (np.abs(peaks - f) < 30) & loud, loud


@pytest.mark.parametrize('loop', [False, True])
def test_old_tone_after_its_writer_goes(loop):
    from peresmeshnik.ui.selftest import Recorder
    rec = Recorder(macos.BLACKHOLE_UID, SR)
    a, a_stop = writer(220.0)
    b, b_stop = [], threading.Event()
    try:
        time.sleep(2.0)
        a[0].terminate()
        a[0].wait(5)
        a_stop.set()
        b, b_stop = writer(300.0, cycle_loop=loop)
        n0 = sum(len(c) for c in rec.chunks)
        time.sleep(3.0)
        y = rec.stop()[n0 + SR // 2:]  # from 0.5 s after the switch
    finally:
        rec.p.poll() is None and rec.stop()
        a_stop.set()
        b_stop.set()
        for p in a + b:
            if p.poll() is None:
                p.terminate()
                p.wait(5)
    old_w, loud_w = tone(y, 220.0)
    new_w, _ = tone(y, 300.0)
    # the new tone from its first window on: how fast its helpers start is not what this checks (a slow runner: new 89 %
    # with loud 89 % from 0.5 s after the switch, the rest silence, not another tone); a gap after it still fails
    first = int(np.argmax(new_w)) if new_w.any() else len(new_w)
    old, loud, new = float(np.mean(old_w)), float(np.mean(loud_w)), float(np.mean(new_w[first:])) if new_w.any() else 0.0
    heard = 0.5 + first * W / SR  # s after the switch
    runs = np.diff(np.flatnonzero(np.diff(np.r_[1, loud_w[first:].astype(int), 1]) != 0))[::2]  # silent runs after it
    gap = int(runs.max(initial=0)) * W * 1000 // SR  # ms
    time.sleep(2.0)  # everything stopped: a new reader alone
    r2 = Recorder(macos.BLACKHOLE_UID, SR)
    time.sleep(1.0)
    z = r2.stop()[SR // 4:]
    alone = float(np.sqrt(np.mean(z ** 2)))
    print(f'\nterminate, then {"the loop" if loop else "a player"}: old tone in {old:.0%} of the windows, new {new:.0%} '
          f'from {heard:.2f} s after the switch on (longest silence after it {gap} ms), loud {loud:.0%}; then a reader '
          f'alone: rms {alone:.4f}')
    assert old < 0.05 and heard < 1.5 and new > 0.9 and alone < 0.001

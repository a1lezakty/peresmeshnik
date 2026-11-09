"""taskkill (a posted WM_CLOSE) of the installed Windows app right after its start, N times: how many runs end with
exit code 0 within 20 s (tests/test_app_windows.py::test_started_from_the_start_menu_shortcut, repeated).

    python bench/windows_close.py --runs 20 --mode shortcut            # as the test: the Start menu shortcut
    python bench/windows_close.py --runs 30 --mode direct --poll 0.05 --delays 0,0.05,0.1,0.2,0.4,0.8

`direct` starts Peresmeshnik.exe with no arguments (as the shortcut does) but with stderr in a file. One JSON line per run
(the windows on screen and all of the app's at taskkill), then the count. Stage 5 (BENCH): the app as on main ended in 80
of 105 runs as the test does it (taskkill's WM_CLOSE went to Qt's hidden '_q_titlebar' before the wizard showed); fixed:
120 of 120.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_app_windows import DIR, SHORTCUT, _running  # noqa: E402
from test_windows import describe, pid_of, toplevel, tray_icon_rect, user32  # noqa: E402


def visible(pid: int) -> list[str]:
    return [describe(h) for h in toplevel() if pid_of(h) == pid and user32.IsWindowVisible(h)]


def one(mode: str, poll: float, delay: float, work: Path) -> dict:
    cfg = work / 'settings.toml'  # absent: a first run, the wizard opens
    os.environ['PERESMESHNIK_CONFIG'] = str(cfg)
    err = work / 'stderr.txt'
    t0 = time.monotonic()
    if mode == 'shortcut':
        os.startfile(SHORTCUT)
        proc = _running(DIR / 'Peresmeshnik.exe')
    else:
        with open(err, 'wb') as f:
            p = subprocess.Popen([str(DIR / 'Peresmeshnik.exe')], stderr=f, stdout=subprocess.DEVNULL,
                                 env=os.environ.copy())
        proc = psutil.Process(p.pid)
    r = {'mode': mode, 'delay': delay}
    try:
        end, where = time.monotonic() + 60, None
        while not where and time.monotonic() < end:
            time.sleep(poll)
            where = tray_icon_rect(proc.pid)
        r['tray_s'] = round(time.monotonic() - t0, 2)
        time.sleep(delay)
        r['visible_at_kill'] = visible(proc.pid)
        r['all_at_kill'] = [describe(h) for h in toplevel() if pid_of(h) == proc.pid]
        k = subprocess.run(['taskkill', '/PID', str(proc.pid)], capture_output=True, text=True)
        r['taskkill'] = [k.returncode, (k.stdout + k.stderr).strip()[:160]]
        try:
            r['code'] = proc.wait(20)
        except psutil.TimeoutExpired:
            r['code'] = 'timeout'
            r['visible_after'] = visible(proc.pid)
    finally:
        if proc.is_running():
            proc.kill()
            proc.wait(10)
    if err.exists():
        r['stderr'] = err.read_text('utf-8', 'replace').splitlines()[-30:]
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--mode', choices=['shortcut', 'direct'], default='shortcut')
    ap.add_argument('--poll', type=float, default=0.5, help='seconds between looks for the tray icon (the test: 0.5)')
    ap.add_argument('--delays', default='0', help='comma list: seconds after the tray icon before taskkill, cycled')
    a = ap.parse_args()
    delays = [float(x) for x in a.delays.split(',')]
    ok = 0
    for i in range(a.runs):
        with tempfile.TemporaryDirectory(prefix='pm-close-') as tmp:
            r = one(a.mode, a.poll, delays[i % len(delays)], Path(tmp))
        ok += r['code'] == 0
        print(json.dumps(r, ensure_ascii=False), flush=True)
        time.sleep(1)  # the tray icon of the last run leaves the notification area
    print(f'{a.mode}: {ok} of {a.runs} runs ended with exit code 0 within 20 s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

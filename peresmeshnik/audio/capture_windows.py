"""Windows capture of the call sound ("them -> me"): a WASAPI process loopback of the call app (Windows 10 2004+).

The native helper `peresmeshnik-loopback` (native/windows/loopback) captures only what the call app's process tree
plays (`ActivateAudioInterfaceAsync`, `AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`, mono float32 on stdout): the
counterpart of the macOS process tap. The call app is `Ui.call_app` on Windows = its exe name (e.g. `Telegram.exe`);
its process tree (a browser's tab processes, a call app's helpers) is captured with it. The loopback takes the sound
BEFORE the app's audio session volume, so the mute spike (set that volume to 0) silences the real speakers but not
the loopback. The helper is $PERESMESHNIK_LOOPBACK (the installed app sets it), else native/windows/loopback/build of
this checkout.

Unlike the macOS tap, process loopback needs the target process id, not a name, so the exe name is resolved to the
root pid of its process tree here (psutil); an app not running has nothing to tap.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from peresmeshnik.audio.resample import Resampler
from peresmeshnik.engines import SAMPLE_RATE

log = logging.getLogger('peresmeshnik.call')
# the app is a windowed .exe; the helper is a console .exe. Without this every spawn flashes a console window, and the
# long-lived ones (tap, watch, mute) keep it open for the whole translation — the user would close it and kill them.
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
BUILT = Path(__file__).resolve().parents[2] / 'native' / 'windows' / 'loopback' / 'build' / 'peresmeshnik-loopback.exe'
BUILD = 'native\\windows\\loopback\\build.cmd'
MIN_BUILD = 19041  # Windows 10 2004: process loopback
RATE = 48000  # the helper requests this mix format (AUTOCONVERTPCM), so the header rate is fixed
EXIT = {2: 'bad arguments to the loopback helper', 3: 'the call app is not running',
        4: 'access denied to the call app\'s loopback', 5: 'Windows older than 10 2004 (build 19041) has no process '
        'loopback: update Windows, or route the call app through a second virtual cable',
        6: 'the process loopback failed (a Windows audio error)', 7: 'the loopback\'s mix format changed'}
RESTART = 7
RESTART_S, RESTARTS = 1.0, 5


def helper() -> str:
    path = os.environ.get('PERESMESHNIK_LOOPBACK') or str(BUILT)
    if not os.access(path, os.X_OK):
        raise RuntimeError(f'no loopback helper at {path}: build it with `{BUILD}`')
    return path


def _run(args: list[str], timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run([helper(), *args], capture_output=True, text=True, encoding='utf-8', timeout=timeout,
                          creationflags=CREATE_NO_WINDOW)


def loopback_supported() -> bool:
    try:
        r = _run(['check'])
    except Exception:
        return False
    return r.returncode == 0 and 'loopback=yes' in r.stdout


def path() -> str:
    """'loopback' when process loopback can work, else '' (too old / no helper): the only call path on Windows."""
    return 'loopback' if loopback_supported() else ''


def check() -> tuple[int, str]:
    """(0, '') when the loopback can work now; else the helper's exit code and the reason to show."""
    r = _run(['check'])
    log.info('loopback helper: %s', r.stdout.strip())
    return r.returncode, '' if r.returncode == 0 else EXIT.get(r.returncode, r.stderr.strip() or f'loopback helper exit {r.returncode}')


def sessions() -> list[dict]:
    """The audio sessions the helper sees: [{'pid', 'name', 'exe', 'playing'}]."""
    r = _run(['list'])
    if r.returncode:
        raise RuntimeError(EXIT.get(r.returncode, r.stderr.strip()))
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]


def players() -> list[dict]:
    """The apps playing now, one per app, ours left out: [{'key' (exe name), 'name'}] (capture_pipewire.apps shape)."""
    import psutil
    me = {psutil.Process().name().lower()}
    seen = {}
    for s in sessions():
        if s['playing'] and s['name'] and s['name'].lower() not in me:
            key = s['name']  # the exe name, e.g. Telegram.exe
            seen.setdefault(key.lower(), {'key': key, 'name': key.removesuffix('.exe')})
    return list(seen.values())


def apps() -> list[tuple[str, str]]:
    """[(exe name, label)] for the tray's "Call app" menu (a browser's label says all of it is taken)."""
    from peresmeshnik.audio.capture_pipewire import apps as label
    return label(players())


def guess_call_app() -> str:
    from peresmeshnik.audio.capture_pipewire import guess_call_app as guess
    return guess(players())


def tree_root(exe: str) -> int | None:
    """The root pid of the process tree whose exe is `exe` (case-insensitive): the loopback captures the target and
    its children, so the topmost same-exe ancestor takes every part of the app (a browser's tab processes, a call
    app's helpers). The tree whose sound is playing now is preferred when several run (the session usually sits in a
    child process, so a root is matched by a playing session anywhere in its tree, not by the root pid itself).
    ponytail: one instance; a second window of the same app is not tapped."""
    import psutil
    want = exe.lower()
    procs = {p.pid: p for p in psutil.process_iter(['name', 'ppid']) if (p.info['name'] or '').lower() == want}
    if not procs:
        return None
    roots = []
    for pid, p in procs.items():
        cur, seen = pid, set()
        while True:
            ppid = procs[cur].info['ppid'] if cur in procs else None
            if ppid not in procs or ppid in seen:  # parent is not the same exe: cur is a tree root
                break
            seen.add(cur)
            cur = ppid
        roots.append(cur)
    roots = list(dict.fromkeys(roots))
    if len(roots) == 1:
        return roots[0]
    playing = [s['pid'] for s in sessions() if s['playing']]

    def plays(root: int) -> bool:  # a playing session is this root or a descendant of it
        for spid in playing:
            try:
                if spid == root or root in {a.pid for a in psutil.Process(spid).parents()}:
                    return True
            except psutil.Error:
                pass
        return False
    return next((r for r in roots if plays(r)), roots[0])


class ProcessLoopback:
    """`AudioIn` from a process loopback of the app `app` (exe name; `pid` for tests), 16 kHz mono. `retarget(app)`
    taps another app ('' = none: the helper goes, the call is heard as usual) without a new start(). `on_end(why)`
    runs once when the helper ends by itself (the app ended, access denied, a Windows audio error). The same shape as
    capture_macos.ProcessTap, so app.py wires them the same way."""

    def __init__(self, app: str = '', pid: int | None = None, on_end: Callable[[str], None] = lambda why: None):
        self.app, self.pid, self.on_end = app, pid, on_end
        self.rate = 0
        self.status = ''
        self._on_frames = self._proc = None
        self._lock = threading.Lock()
        self._restarts: list[float] = []

    def start(self, on_frames: Callable[[np.ndarray], None]) -> None:
        self._on_frames = on_frames
        if self.app or self.pid:
            self._spawn()

    def retarget(self, app: str) -> None:
        with self._lock:
            if app and app == self.app and self._proc is not None and self._proc.poll() is None:
                return  # tapping it already
            old, self._proc, self.app = self._proc, None, app
        if old:
            self._end(old)
        if app and self._on_frames:
            self._spawn()

    def stop(self) -> None:
        self.retarget('')
        self._on_frames = None

    def _spawn(self) -> None:
        with self._lock:
            self._spawn_locked()

    def _spawn_locked(self) -> None:
        pid = self.pid
        if pid is None:
            pid = tree_root(self.app)
            if pid is None:
                self.status = f'{self.app} is not running'
                log.warning('loopback: %s', self.status)
                threading.Thread(target=self.on_end, args=(f'{self.app} is not running',), daemon=True).start()
                return
        self._proc = proc = subprocess.Popen([helper(), 'tap', '--pid', str(pid)], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             creationflags=CREATE_NO_WINDOW)
        threading.Thread(target=self._messages, args=(proc,), name='loopback-messages', daemon=True).start()
        threading.Thread(target=self._read, args=(proc, self._on_frames), name='capture', daemon=True).start()

    def _messages(self, proc) -> None:
        for line in proc.stderr:
            self.status = line.decode(errors='replace').strip().removeprefix('peresmeshnik-loopback: ')
            log.info('call loopback: %s', self.status)

    def _read(self, proc, on_frames) -> None:
        head = proc.stdout.readline().decode(errors='replace')
        import re
        if m := re.match(r'peresmeshnik-loopback 1 rate=(\d+) channels=1 format=f32le', head):
            self.rate = int(m[1])
            rs, n = Resampler(self.rate, SAMPLE_RATE), self.rate * 32 // 1000 * 4  # 32 ms, as the Linux capture
            while len(b := proc.stdout.read(n)) == n:
                on_frames(rs(np.frombuffer(b, '<f4')))
        proc.wait()
        why = EXIT.get(proc.returncode) or f'the loopback helper ended ({proc.returncode}): {self.status}'
        if proc.returncode == 3:
            why = f'{self.app or self.pid} is not running'
        again = proc.returncode == RESTART
        if again:
            now = time.monotonic()
            self._restarts = [t for t in self._restarts if now - t < 60]
            if len(self._restarts) >= RESTARTS:
                again, why = False, f'the loopback mix format keeps changing ({len(self._restarts)} restarts in a minute)'
            elif self._restarts:
                time.sleep(max(0.0, self._restarts[-1] + RESTART_S - now))
        with self._lock:
            if proc is not self._proc:
                return
            self._proc = None
            if again and on_frames is self._on_frames:
                log.info('call loopback: %s; starting it again', self.status)
                self._restarts.append(time.monotonic())
                self._spawn_locked()
                return
        self.on_end(why)

    @staticmethod
    def _end(proc) -> None:
        proc.stdin.close()  # the helper's sign to drop the loopback: the app is heard as usual again
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---- the mute spike: hold the call app's sessions at 0 (variant A), new ones too -------------------------------
# A persistent `mute` helper sets every session of the call app's tree to 0 and mutes new ones as they appear (a call
# that starts, a new browser tab), each session's original level saved. On a clean stop (restore_call) it restores
# them and exits. A crash leaves the saved levels in a file, so the next start (or the next translation) puts them
# back: the file is kept until the app is there to restore, and a fresh mute restores a leftover first (so it never
# saves the 0 a previous crash left as the "original").

_muter: subprocess.Popen | None = None
_muter_lock = threading.Lock()


def _mute_state() -> Path:
    from peresmeshnik import config
    return config.config_dir() / 'muted-session.json'


def _save_state(app: str, levels: dict[int, float]) -> None:
    try:
        _mute_state().parent.mkdir(parents=True, exist_ok=True)
        _mute_state().write_text(json.dumps({'app': app, 'levels': {str(p): v for p, v in levels.items()}}))
    except OSError as e:
        log.warning('mute: could not save the call volume: %s', e)


def _clear_state() -> None:
    try:
        _mute_state().unlink()
    except OSError:
        pass


def session_volume(app: str) -> float | None:
    """The call app's current session volume (0..1), the max over its sessions; None when it has none."""
    pid = tree_root(app)
    if pid is None:
        return None
    r = _run(['volume', '--pid', str(pid), '--tree'])
    vols = [json.loads(line)['volume'] for line in r.stdout.splitlines() if line.startswith('{')]
    return max(vols) if vols else None


def set_session_volume(app: str, level: float) -> bool:
    pid = tree_root(app)
    if pid is None:
        return False
    r = _run(['volume', '--pid', str(pid), '--tree', '--set', f'{level:.4f}'])
    return r.returncode == 0


def mute_call(app: str) -> None:
    """Start the mute helper: hold the call app's sessions (and new ones) at 0, each original level saved so
    restore_call / the next start can put it back. A leftover crash mute of any app is restored first."""
    global _muter
    restore_call()  # clear a leftover crash mute first, so the helper reads the true original levels, not a saved 0
    pid = tree_root(app)
    if pid is None:
        log.warning('mute: %s is not running', app)
        return
    proc = subprocess.Popen([helper(), 'mute', '--pid', str(pid), '--tree'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    levels: dict[int, float] = {}

    def pump() -> None:
        for line in proc.stdout:  # one {"event":"muted","pid","level"} per session, as they are muted
            try:
                ev = json.loads(line.decode('utf-8', 'replace'))
            except ValueError:
                continue
            if ev.get('event') == 'muted':
                levels[int(ev['pid'])] = float(ev['level'])
                _save_state(app, levels)
                log.info('mute: %s session %s volume %.2f -> 0', app, ev['pid'], ev['level'])

    with _muter_lock:
        _muter = proc
    threading.Thread(target=pump, name='mute-pump', daemon=True).start()


def restore_call() -> None:
    """Put the call app's sessions back: stop the live mute helper (it restores each session to its own level), or,
    after a crash, the saved levels from the file. The file is kept if the app is not running yet, so a later
    restore can still apply it."""
    global _muter
    with _muter_lock:
        proc, _muter = _muter, None
    if proc is not None:
        try:
            proc.stdin.close()  # the helper restores every session it muted and exits
            proc.wait(5)
        except Exception:
            proc.kill()
        _clear_state()
        return
    try:
        state = json.loads(_mute_state().read_text())
    except (OSError, ValueError):
        return
    app = state.get('app', '')
    levels = [float(v) for v in state.get('levels', {}).values()]
    if not (app and levels):
        _clear_state()
        return
    # ponytail: crash recovery puts the max original back on the whole tree (per-session levels only on the live
    # path), so a second window's tab levels are flattened. Keep the file until the app is back to restore.
    if set_session_volume(app, max(levels)):
        _clear_state()
        log.info('mute: %s sessions restored to %.2f after a crash', app, max(levels))

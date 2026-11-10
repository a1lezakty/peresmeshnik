"""Windows capture of the call sound ("them -> me"): a WASAPI process loopback of the call app (Windows 10 2004+).

The native helper `peresmeshnik-loopback` (native/windows/loopback) captures only what the call app's process tree
plays (`ActivateAudioInterfaceAsync`, `AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`, mono float32 on stdout): the
counterpart of the macOS process tap. The call app is `Ui.call_app` on Windows = its exe name (e.g. `Telegram.exe`);
its process tree (a browser's tab processes, a call app's helpers) is captured with it. The loopback takes the sound
AFTER the app's audio session volume (measured on the runner, stage 7: a session at 0 gives silence), so the call
app is not muted but ducked to DUCK (-46 dB: near silent under the translation) and the loopback brings its sound
back up by 1/DUCK. The helper is $PERESMESHNIK_LOOPBACK (the installed app sets it), else native/windows/loopback/build
of this checkout.

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
        6: 'the process loopback failed (a Windows audio error)',
        7: 'the loopback stream was reset (a Windows audio change)'}
RESTART = 7
RESTART_S, RESTARTS = 1.0, 5
DUCK = 0.005  # the call app's session volume while translating; below 0.01, so a saved level never takes it for one
_gain = 1.0  # 1/DUCK once the mute helper said a session is ducked, until restore_call (a dead helper leaves it ducked)
HOLD_S = 2.0  # after a block too loud to be ducked (the session raised by hand or by the app), this long as it is before
# quiet blocks are brought up again: a raised session's pauses, breath and call noise are as quiet as a ducked one's
# speech and would come out 200 times up, a noise for speech recognition that never lets its VAD close
KEEP_S = 14 * 24 * 3600  # how long a keep helper waits for the call app's next sound, the app quit or not


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
    """The apps playing now, one per app, ours left out: [{'key' (exe name), 'name'}] (capture_pipewire.apps shape).
    Ours: this process (python.exe from a checkout) and every Peresmeshnik program (peresmeshnik-audio plays the
    translation into the virtual cable: picked as the call app, its voice would be muted)."""
    import psutil
    me = {psutil.Process().name().lower()}
    seen = {}
    for s in sessions():
        if s['playing'] and s['name'] and s['name'].lower() not in me and not s['name'].lower().startswith('peresmeshnik'):
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
            hold_n = round(HOLD_S * 1000 / 32)
            hold, armed = 0, False  # blocks left as they are after a loud one; a ducked block seen since the gain came
            while len(b := proc.stdout.read(n)) == n:
                x = np.frombuffer(b, '<f4')
                # ducked: brought back up. A block louder than a ducked one can be is not ducked: as it is, not 200
                # times into a clip. Once the duck had taken (armed), that is a raised session: HOLD_S as it is; the
                # sound from before the duck (the first ~0.1-0.4 s) holds nothing
                if _gain == 1.0:
                    hold, armed = 0, False
                elif float(np.abs(x).max(initial=0.0)) > 2 * DUCK:
                    hold = hold_n if armed else 0
                elif hold:
                    hold -= 1
                else:
                    armed = True
                    x = np.clip(x * _gain, -1.0, 1.0)
                on_frames(rs(x))
        proc.wait()
        why = EXIT.get(proc.returncode) or f'the loopback helper ended ({proc.returncode}): {self.status}'
        if proc.returncode == 3:
            why = f'{self.app or self.pid} is not running'
        again = proc.returncode == RESTART
        if again:
            now = time.monotonic()
            self._restarts = [t for t in self._restarts if now - t < 60]
            if len(self._restarts) >= RESTARTS:
                again, why = False, f'the loopback keeps being reset ({len(self._restarts)} restarts in a minute): {self.status}'
            elif self._restarts:
                time.sleep(max(0.0, self._restarts[-1] + RESTART_S - now))
        with self._lock:
            if proc is not self._proc:
                return
            self._proc = None
            if again and on_frames is self._on_frames:
                log.info('call loopback: %s; starting it again', self.status)
                self._restarts.append(time.monotonic())
                try:
                    self._spawn_locked()
                    return
                except Exception as e:  # not a silent capture thread: the side ends with the reason (and restarts)
                    log.exception('call loopback: starting it again')
                    why = f'the loopback helper did not start again: {e}'
        self.on_end(why)

    @staticmethod
    def _end(proc) -> None:
        proc.stdin.close()  # the helper's sign to drop the loopback: the app is heard as usual again
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---- the mute spike: hold the call app's sessions at DUCK (variant A), new ones too ----------------------------
# A `mute` helper sets every session of the call app's tree to DUCK and ducks new ones as they appear (a call that
# starts, a new browser tab, a headset plugged in); the loopback is brought back up by 1/DUCK (_gain). It is
# mute-only: Python owns the restore, from a per-app original level in a file. So a crash (either side) never loses
# the original: the helper dying leaves the sessions ducked, the file still has the level, and the next restore_call
# (translation off, loopback end, or the next start) puts it back. The saved level is the app's original (>= 0.01)
# and is never overwritten by a near-zero (Windows persists a ducked level across a restart, and a fresh helper would
# read it — the file, not the live reading, is the source of truth). Windows keeps an app's volume per device, so the
# file also has the devices it was ducked on; an entry goes only when every one of them got its level back. What
# cannot be given back now (the call app closed before translation was turned off, a headset unplugged) is watched
# for by a `keep` helper: on the app's next session there (opened at the persisted level, below 0.01) it sets the
# level and says so. The keep helper reads no stdin: it outlives the app (up to KEEP_S), event-driven, and ends when
# every device is done; the next run ends a leftover one first (its pid is in the file).
# ponytail: one level per app restored to the whole tree, so a second window's tab volumes are flattened on restore.
# ponytail: one _gain for the module: one call app is translated at a time.

_muter: subprocess.Popen | None = None
_keepers: dict[str, subprocess.Popen] = {}  # app -> its keep helper
_muter_lock = threading.Lock()  # _muter, _keepers
_state_lock = threading.RLock()  # the file: read-modify-write from restore_call and the helpers' pumps


def _mute_state() -> Path:
    from peresmeshnik import config
    return config.config_dir() / 'muted-session.json'


def _entry(v) -> dict:
    """An app's saved state {'level': its original, 'devices': endpoint ids still at 0, 'keeper': the pid of its keep
    helper, which outlives the app}. An older file has the bare level: no devices, any session given back clears it."""
    if isinstance(v, dict):
        e = {'level': float(v.get('level', 0.0)), 'devices': [str(x) for x in v.get('devices', [])]}
        if isinstance(v.get('keeper'), int):
            e['keeper'] = v['keeper']
        return e
    return {'level': float(v), 'devices': []}


def _load_state() -> dict:
    try:
        d = json.loads(_mute_state().read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for app, v in (d.items() if isinstance(d, dict) else ()):
        try:
            out[app] = _entry(v)
        except (TypeError, ValueError, AttributeError):
            log.warning('mute: a bad entry for %s in %s', app, _mute_state())
    return out


def _write_state(d: dict) -> None:
    """Whole or not at all (a temporary file, then os.replace): a crash or a reader mid-write never sees half a file,
    which would read as no saved level at all."""
    try:
        if d:
            _mute_state().parent.mkdir(parents=True, exist_ok=True)
            tmp = _mute_state().with_suffix('.tmp')
            tmp.write_text(json.dumps(d))
            for i in range(3):  # Windows: an antivirus or the indexer may hold the file a moment
                try:
                    os.replace(tmp, _mute_state())
                    break
                except PermissionError:
                    if i == 2:
                        raise
                    time.sleep(0.05)
        elif _mute_state().exists():
            _mute_state().unlink()
    except OSError as e:
        log.warning('mute: could not save the call volume: %s', e)


def _remember(app: str, level: float, device: str = '') -> None:
    """Record the call app's original level and the device it was muted on. Never overwrite a saved non-zero with a
    near-zero: a crash leaves the app at 0 and Windows persists it, so a fresh helper reads 0 — the saved non-zero is
    the true original (the device is recorded all the same: it is at 0 now)."""
    with _state_lock:
        d = _load_state()
        e = d.get(app, {'level': 0.0, 'devices': []})
        changed = level >= 0.01 and level > e['level']  # first non-zero, or a louder original (keep the max)
        if changed:
            e['level'] = level
        if device and device not in e['devices']:
            e['devices'].append(device)
            changed = True
        if changed:
            d[app] = e
            _write_state(d)


def session_volume(app: str) -> float | None:
    """The call app's current session volume (0..1), the max over its sessions; None when it has none."""
    pid = tree_root(app)
    if pid is None:
        return None
    r = _run(['volume', '--pid', str(pid), '--tree'])
    vols = [json.loads(line)['volume'] for line in r.stdout.splitlines() if line.startswith('{')]
    return max(vols) if vols else None


def set_session_volume(app: str, level: float) -> list[str]:
    """Give the call app's sessions (its tree, every device there now) that are still down (below 0.01, as `keep`)
    their `level`; one already up (given back by a keep helper, set by the user since) is left as it is. The endpoint
    ids of every session found, [] when there is none (the app is not running, or has no session)."""
    pid = tree_root(app)
    if pid is None:
        return []
    r = _run(['volume', '--pid', str(pid), '--tree', '--set', f'{level:.4f}', '--below', '0.01'])
    if r.returncode:
        return []
    return [json.loads(line).get('device', '') for line in r.stdout.splitlines() if line.startswith('{')]


def _stop(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.close()  # the mute helper's sign to stop (it restores nothing on its way out)
        proc.wait(5)
    except Exception:
        proc.kill()


def _kill(proc: subprocess.Popen) -> None:
    """A keep helper reads no stdin (it outlives the app): it is ended. It holds nothing to clean up."""
    try:
        proc.kill()
        proc.wait(5)
    except Exception:
        pass


def _kill_orphan(pid: int) -> None:
    """The keep helper a previous run left (the app quit or crashed while it waited): it would undo the next duck.
    The pid may have been reused since it ended: by another program, or by a helper of this run (watch, tap)."""
    import psutil
    try:
        p = psutil.Process(pid)
        if p.name().lower().startswith('peresmeshnik-loopback') and 'keep' in p.cmdline()[1:2]:
            p.kill()
    except psutil.Error:
        pass


def mute_call(app: str) -> None:
    """Start the mute helper on the call app's tree (and new sessions), remembering each session's original level
    and device. A leftover (another app, or a crash) is restored first, so the helper reads true originals and other
    apps keep theirs; the app's own keep helper is not started (it would undo the mute)."""
    global _muter, _gain
    restore_call(muting=app)  # restore any leftover first: the helper then reads real levels, not a crash's persisted 0
    pid = tree_root(app)
    if pid is None:
        log.warning('mute: %s is not running', app)
        return
    proc = subprocess.Popen([helper(), 'mute', '--pid', str(pid), '--tree', '--level', f'{DUCK:.4f}'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=CREATE_NO_WINDOW)

    def pump() -> None:
        global _gain
        for line in proc.stdout:  # one {"event":"muted","pid","level","device"} per session, as they are ducked
            try:
                ev = json.loads(line.decode('utf-8', 'replace'))
            except ValueError:
                continue
            if ev.get('event') == 'muted':
                _remember(app, float(ev['level']), ev.get('device', ''))
                with _muter_lock:  # a session is ducked now: its sound is brought back up (not after restore_call)
                    if _muter is proc:
                        _gain = 1 / DUCK
                log.info('mute: %s session %s volume %.2f -> %g', app, ev.get('pid'), ev['level'], DUCK)
            elif ev.get('event') == 'failed':
                log.warning('mute: %s session %s could not be ducked (0x%x)', app, ev.get('pid'), ev.get('hr', 0))
        with _muter_lock:
            died = _muter is proc  # not stopped by restore_call
        if died:  # its sessions stay ducked (it gives nothing back), so the gain stays too, or the call is heard
            # nowhere; restore_call gives the levels back as usual
            log.warning('mute: the helper for %s ended (%s): new sessions are not ducked; translation off gives '
                        'the volume back', app, proc.poll())

    with _muter_lock:
        _muter = proc
    threading.Thread(target=pump, name='mute-pump', daemon=True).start()


def restore_call(muting: str = '') -> list[str]:
    """Stop the mute helper (mute-only; Python owns restore) and put each saved app's level back, clearing an entry
    only where it was actually applied: on every device it was muted on. What cannot be applied yet (the app is not
    running, its device is unplugged) is kept and watched for by a keep helper (but not for `muting`, the app about
    to be muted), so nothing is lost if the helper died or the app is gone. Returns those apps: still quiet somewhere."""
    global _muter, _gain
    with _muter_lock:
        _gain = 1.0
        proc, _muter = _muter, None
        keepers = list(_keepers.values())
        _keepers.clear()
    if proc is not None:
        _stop(proc)
    for p in keepers:
        _kill(p)
    with _state_lock:
        d = _load_state()
        for app, e in list(d.items()):
            if 'keeper' in e:  # a previous run's keep helper, if it still waits
                _kill_orphan(e.pop('keeper'))
            if e['level'] < 0.01:  # nothing known to put back
                d.pop(app)
                continue
            got = set(set_session_volume(app, e['level']))
            e['devices'] = [x for x in e['devices'] if x not in got]
            if got and not e['devices']:  # applied everywhere it was muted: drop it
                d.pop(app)
                log.info('mute: %s sessions restored to %.2f', app, e['level'])
        _write_state(d)
    waiting = [app for app in d if app != muting]
    for app in waiting:
        _keep(app, d[app])
    return waiting


def quiet_note(apps: list[str]) -> str:
    """The tray's word for restore_call's apps still quiet somewhere ('' for none)."""
    if not apps:
        return ''
    names = ', '.join(a[:-4] if a.lower().endswith('.exe') else a for a in apps)
    return (f'{names} stays quiet where it played during translation (it closed first, or a headset was off): its '
            f'volume comes back when it plays sound there again (within {KEEP_S // 86400} days), even without '
            'Peresmeshnik, or raise it in the Windows Volume mixer.')


def _keep(app: str, e: dict) -> None:
    """A keep helper for the level of `app` not given back yet: on the app's next session on one of e['devices'] (any
    device for an entry of 0.3.0) it sets the level if the session is at 0, and says "restored" per device."""
    args = [helper(), 'keep', '--exe', app, '--level', f'{e["level"]:.4f}', '--for', str(KEEP_S)]
    for dev in e['devices']:
        args += ['--device', dev]
    try:  # no stdin: it outlives the app (the call app's next call without Peresmeshnik) and ends when done
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=CREATE_NO_WINDOW)
    except (OSError, RuntimeError) as ex:
        log.warning('mute: cannot watch for %s to give its volume back: %s', app, ex)
        return
    with _state_lock:  # its pid: the next run ends it before it ducks or gives back anything itself
        d = _load_state()
        if app in d:
            d[app]['keeper'] = proc.pid
            _write_state(d)
    log.info('mute: %s is still down on %s: its volume goes back when it plays there', app, e['devices'] or 'a device')

    def pump() -> None:
        for line in proc.stdout:  # {"event":"restored","pid","device"} per session given back (or found not at 0)
            try:
                ev = json.loads(line.decode('utf-8', 'replace'))
            except ValueError:
                continue
            if ev.get('event') == 'restored' and _restored(app, ev.get('device', '')):
                break
        with _muter_lock:
            if _keepers.get(app) is proc:
                _keepers.pop(app)
        _kill(proc)  # done (it ends by itself too)

    with _muter_lock:
        _keepers[app] = proc
    threading.Thread(target=pump, name='mute-keep', daemon=True).start()


def _restored(app: str, device: str) -> bool:
    """The keep helper gave `app` its level back on `device`: True when the app's entry is done."""
    with _state_lock:
        d = _load_state()
        e = d.get(app)
        if e is not None:
            e['devices'] = [x for x in e['devices'] if x != device]
            if e['devices']:
                _write_state(d)
                return False
            d.pop(app)
            _write_state(d)
            log.info('mute: %s volume given back (%.2f)', app, e['level'])
        return True

"""The app: tray, subtitle overlay, global hotkey, and the two directions wired to the devices.

    peresmeshnik app [--on] [--them-file call.wav] [--me-file me.wav] [--show-overlay] [-v]
    peresmeshnik                     # the same when a display is there (else the headless idle)

Each side ("them -> me", "me -> them") gets a local or cloud `Direction` from the settings. Turning
translation on loads the engines on a control thread (the tray says "warming up"), then opens the
inputs; turning it off closes them again, so the idle app holds no stream, no engine thread and no
timer. Linux devices live as long as the app: the call sink "Peresmeshnik Call" (the call's sound is
moved there once) and the virtual mic "Peresmeshnik Mic"; while translation is off PipeWire loops the
real microphone into the virtual mic, so the call still hears the user. The hotkey switches
"translation <-> live voice": a direction that supports `set_live` is told so; for any other the input
goes straight to the output instead. One app per sound server (a lock file).
`--them-file` / `--me-file` feed recordings at their own pace instead of the devices (tests, demos).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from peresmeshnik import __version__, config
from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech

log = logging.getLogger('peresmeshnik.app')
OURS = 'peresmeshnik_'  # every PipeWire device the app makes starts with this
MIC_SINK = 'peresmeshnik_mic_sink'  # = audio/virtual_mic_linux.SINK, without importing it on other OSes


def pactl(*args: str) -> str:
    return subprocess.run(['pactl', *args], check=True, capture_output=True, text=True, timeout=10).stdout


def not_ours(name: str, what: str) -> str:
    """A default device that is one of ours would make a loop (the translation translated again, or sent to the call)."""
    if name.startswith(OURS):
        raise RuntimeError(f'the {what} is our own device {name}: choose your real device in the settings or the system')
    return name


class Resampler:
    """Linear resampling of a stream in chunks, keeping the fractional position and the last sample between
    chunks: per-chunk rounding would add ~34 ms per minute at 16 -> 22.05 kHz.
    ponytail: linear interpolation, fine for speech and the live pass-through; soxr if quality matters."""

    def __init__(self, sr: int, to: int):
        self.step, self.pos, self.prev = sr / to, 1.0, 0.0  # pos counts from the previous chunk's last sample

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        if self.step == 1.0:
            return pcm
        x = np.concatenate([[self.prev], pcm])
        t = np.arange(self.pos, len(x) - 1, self.step)
        self.pos = (t[-1] + self.step if len(t) else self.pos) - (len(x) - 1)
        self.prev = x[-1]
        return np.interp(t, np.arange(len(x)), x).astype(np.float32)


class NoTts:
    """TTS that says nothing: a local pipeline that only makes subtitles (cloud voice, local captions)."""
    sample_rate = SAMPLE_RATE

    def synth(self, text, lang, voice=None):
        return iter(())


class FileIn:
    """`AudioIn` from a recording, at its own pace (32 ms frames), then 1.5 s of silence to close the clause."""

    def __init__(self, path: str):
        from peresmeshnik.pipeline import read_audio
        self.pcm = np.concatenate([read_audio(path), np.zeros(SAMPLE_RATE * 3 // 2, np.float32)])
        self._stop = threading.Event()
        self.done = threading.Event()
        self._thread = None

    def start(self, on_frames) -> None:
        def run():
            t0 = time.perf_counter()
            for i in range(0, len(self.pcm), 512):
                if self._stop.wait(max(0.0, t0 + i / SAMPLE_RATE - time.perf_counter())):
                    break
                on_frames(self.pcm[i:i + 512])
            self.done.set()
        self._thread = threading.Thread(target=run, name='file-in', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()


class Headphones:
    """`AudioOut` to a PipeWire/PulseAudio sink at the first chunk's rate (Piper 22.05 kHz, Gemini 24 kHz);
    chunks at another rate (the live pass-through, 16 kHz) are resampled. `sink` '' = the default.
    pacat ends when its device goes (Bluetooth profile switch, USB unplugged: it does not follow another
    device): the next chunk opens the saved sink again, else the current default, and the tray says so."""
    retry_s = 2.0

    def __init__(self, sink: str = '', on_state=lambda text: None):
        self.sink, self.on_state = sink, on_state
        self._out, self._rs, self._retry_at, self._failing = None, {}, 0.0, False
        self._lock = threading.Lock()  # capture (live), events (speech) and control (switch, stop) threads

    def resolve(self) -> str:
        names = [line.split('\t')[1] for line in pactl('list', 'short', 'sinks').splitlines() if '\t' in line]
        if self.sink in names:
            return self.sink
        return not_ours(pactl('get-default-sink').strip(), 'default output')

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            out = self._out
            # ponytail: PulseOut (S1) has no end-of-stream callback yet; a dead pacat or writer thread is seen here
            if out is not None and (out._proc.poll() is not None or not out._thread.is_alive()):
                out.flush()
                threading.Thread(target=out.stop, daemon=True).start()  # its writer thread ends (pacat is gone)
                self._out = out = None
                self._fail('the headphones went away; reopening')
            if out is None:
                if time.monotonic() < self._retry_at:
                    return  # no device for now: this audio is dropped
                try:
                    from peresmeshnik.audio.out import PulseOut
                    sink = self.resolve()
                    self._out = out = PulseOut(sample_rate, sink)
                    if self._failing:  # the tray still shows the error: put it back
                        self.on_state(f'translating (the headphones are back: {sink})')
                    self._rs, self._failing = {}, False
                except Exception as e:
                    self._retry_at = time.monotonic() + self.retry_s
                    self._fail(f'no headphones: {e}')
                    return
            rs = self._rs.get(sample_rate) or self._rs.setdefault(sample_rate, Resampler(sample_rate, out.sample_rate))
            out.play(rs(pcm), out.sample_rate)

    def _fail(self, why: str) -> None:
        if not self._failing:  # once per outage, not per chunk
            self._failing = True
            self.on_state(f'error: {why}')

    def flush(self) -> None:
        with self._lock:
            if self._out:
                self._out.flush()

    def stop(self) -> None:
        with self._lock:
            out, self._out = self._out, None
        if out:
            out.flush()
            out.stop()  # plays out what pacat holds: outside the lock


class CallSink:
    """"Peresmeshnik Call": the null sink the call's sound is moved into (S1, once), kept for the app's run."""

    def __init__(self):
        from peresmeshnik.audio.capture_pipewire import CALL_SINK
        self.name = CALL_SINK
        # priority 0: never preferred over a real output as the default sink (as in S1's capture)
        self.module = pactl('load-module', 'module-null-sink', f'sink_name={CALL_SINK}',
                            'sink_properties=\'device.description="Peresmeshnik Call" priority.session=0 priority.driver=0\'').strip()

    def remove(self) -> None:
        subprocess.run(['pactl', 'unload-module', self.module], capture_output=True, timeout=10)


class CallCapture:
    """`AudioIn` from the call sink's monitor (parec, 20 ms), read on a thread blocked on the pipe.
    ponytail: S1's PipeWireCapture makes and removes its own sink on every start / stop; until it can use
    the app's sink, the app records the monitor itself with the same flags."""

    def __init__(self, sink: str, frame: int = 512):
        self.sink, self.frame, self._proc, self._thread = sink, frame, None, None

    def start(self, on_frames) -> None:
        self._proc = subprocess.Popen(['parec', f'--device={self.sink}.monitor', '--raw', '--format=float32le', f'--rate={SAMPLE_RATE}',
                                       '--channels=1', '--latency-msec=20', '--property=node.dont-reconnect=true'], stdout=subprocess.PIPE)

        def read():
            n = self.frame * 4
            while len(b := self._proc.stdout.read(n)) == n:
                on_frames(np.frombuffer(b, '<f4'))
        self._thread = threading.Thread(target=read, name='call-capture', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._thread.join(5)
            self._proc = None


def remove_stale() -> None:
    """Our devices left by a run that was killed (SIGKILL, a crash): loopbacks first, then the rest, newest first."""
    mods = [m.split('\t') for m in pactl('list', 'short', 'modules').splitlines()]
    mods = [m for m in mods if len(m) > 2 and OURS in m[2]]
    for m in sorted(mods, key=lambda m: (m[1] != 'module-loopback', -int(m[0]))):
        subprocess.run(['pactl', 'unload-module', m[0]], capture_output=True, timeout=10)


class Side:
    """One direction with its input, output and subtitles. `name` is 'them' (them -> me) or 'me'."""

    def __init__(self, name: str, s: config.Settings, inp, out, overlay, on_error=lambda text: None):
        self.name, self.inp, self.out, self.overlay, self.on_error = name, inp, out, overlay, on_error
        self.src, self.tgt = (s.their_lang, s.my_lang) if name == 'them' else (s.my_lang, s.their_lang)
        self.engine = s.them_to_me if name == 'them' else s.me_to_them
        self.live = False
        self.directions = []  # (direction, takes its captions, takes its speech)
        self._threads: list[threading.Thread] = []
        self._live_q: queue.Queue = queue.Queue()  # the live pass-through, played on its own thread
        if self.engine == 'local':
            self.directions.append((self._local(s), True, True))
        else:
            voice = self._cloud(s)
            local_captions = None
            if not s.cloud.subtitles:  # subtitles from the local ASR (faster than the cloud's, ARCHITECTURE §4)
                from peresmeshnik import models
                from peresmeshnik.pipeline import make_local
                try:
                    local_captions = make_local(self.src, self.tgt, tts=NoTts(), settings=s)
                    local_captions.warm_up()
                except models.ModelMissing as e:
                    log.warning('%s: local subtitles need %s; the cloud transcripts are shown instead', name, e)
                except BaseException:
                    voice.close()  # its Live session is already open
                    raise
            self.directions.append((voice, local_captions is None, True))
            if local_captions:
                self.directions.append((local_captions, True, False))

    def _local(self, s: config.Settings):
        from peresmeshnik.pipeline import make_local
        if self.name == 'them':
            d = make_local(self.src, self.tgt, settings=s)
        else:
            if not s.local.voice:
                raise RuntimeError('me -> them local needs your voice: enroll it first (python -m peresmeshnik.voice enroll), '
                                   'or switch "Me -> them" to cloud')
            from peresmeshnik import models, voice
            try:
                from peresmeshnik.local.clone import XttsClone
            except ImportError as e:
                raise RuntimeError(f'me -> them local needs the voice pack (torch, coqui-tts): {e}; '
                                   'or switch "Me -> them" to cloud') from e
            d = make_local(self.src, self.tgt, settings=s, tts=XttsClone(models.path(s.local.clone), threads=s.local.threads),
                           voice=voice.load(s.local.voice))
        d.warm_up()
        return d

    def _cloud(self, s: config.Settings):
        from peresmeshnik.cloud import api_key
        from peresmeshnik.cloud.gemini import GeminiDirection
        try:
            key = api_key(s.cloud)
        except SystemExit as e:  # the CLI's way to say "no key"
            raise RuntimeError(str(e)) from None
        return GeminiDirection(self.src, self.tgt, key, s.cloud)

    def start(self) -> None:
        for target, args in [(self._consume, d) for d in self.directions] + [(self._pass_through, ())]:
            t = threading.Thread(target=target, args=args, name=f'{self.name}-{target.__name__[1:]}', daemon=True)
            t.start()
            self._threads.append(t)
        self.inp.start(self._on_frames)

    def _on_frames(self, pcm: np.ndarray) -> None:  # capture thread or PortAudio callback: only hand on
        if self.live and not self._supports_live:
            self._live_q.put(pcm)  # ponytail: a hard switch, no crossfade; S2's set_live crossfades
            return
        for d, _, _ in self.directions:
            d.feed(pcm)

    def _pass_through(self) -> None:
        while (pcm := self._live_q.get()) is not None:
            if self.live:  # frames queued just before switching back are dropped
                self.out.play(pcm, SAMPLE_RATE)

    @property
    def _supports_live(self) -> bool:
        return getattr(self.directions[0][0], 'supports_live', False)

    def _consume(self, d, captions: bool, speech: bool) -> None:
        try:
            for e in d.events():
                if isinstance(e, Caption) and captions:
                    self.overlay.show(e)
                elif isinstance(e, Speech) and speech and (not self.live or self._supports_live):
                    self.out.play(e.pcm, e.sample_rate)
        except Exception as e:  # a dead session (key revoked, quota, network) must show in the tray, not only in the log
            log.exception('%s: direction failed', self.name)
            self.on_error(f'error: {self.name}: {e}')

    def set_live(self, on: bool) -> None:
        if self._supports_live:
            self.directions[0][0].set_live(on)
        elif on != self.live:
            self.out.flush()  # drop translated speech still queued (going live) or the live tail (back)
        self.live = on

    def stop(self) -> None:
        self.inp.stop()
        self._live_q.put(None)
        for d, _, _ in self.directions:
            d.close()
        for t in self._threads:
            t.join(15)
        self.out.stop()


class Controller:
    """Turns translation on and off on one control thread; reports states through `on_state(str)`."""

    def __init__(self, overlay, on_state, them_file: str | None = None, me_file: str | None = None,
                 settings: config.Settings | None = None, test: bool = False):
        """`settings`: use these instead of the saved ones; `test`: both directions speak into the headphones
        (the wizard's test phrase) and the devices are left alone."""
        self.overlay, self.on_state, self.them_file, self.me_file = overlay, on_state, them_file, me_file
        self.settings, self.test = settings, test
        self.devices = not (test or them_file or me_file)  # this controller owns the app's Linux devices
        self.sides: list[Side] = []
        self.live = False
        self.on = False
        self.startup_error = ''
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='control')  # no thread until first use
        self._vmic = self._call = None
        self._mic = ''  # the default input as it was before our devices existed

    def submit(self, fn, *args):
        return self._pool.submit(self._guard, fn, *args)

    def _guard(self, fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            log.exception('control')
            self._off()
            self.on_state(f'error: {e}')

    def virtual_mic(self):
        if self._vmic is None and sys.platform.startswith('linux'):
            from peresmeshnik.audio.virtual_mic_linux import VirtualMic
            self._vmic = VirtualMic()
        return self._vmic

    def set_translation(self, on: bool) -> concurrent.futures.Future:
        return self.submit(self._on if on else self._off)

    def startup(self) -> None:
        """The devices are there from the start, so the call app can pick the microphone and S1 can move the
        call's sound once; until translation is on, the real microphone is looped into the virtual one."""
        if not (self.devices and sys.platform.startswith('linux')):
            return
        errors = []

        def step(what, fn):  # each device on its own: no call sink still leaves the microphone working
            try:
                return fn()
            except Exception as e:  # no sound server or no pactl: the app still runs; the tray says why
                log.warning('%s: %s', what, e)
                errors.append(f'{what}: {e}')
        step('old devices', remove_stale)  # first: a stale sink of ours may be the default input now
        self._mic = step('default input', lambda: pactl('get-default-source').strip()) or ''  # before ours exist
        self._call = step('call sink', CallSink)
        step('virtual mic', self.virtual_mic)
        self._loopback(True)
        self.startup_error = '; '.join(errors)

    def _mic_source(self) -> str:
        s = self.settings or config.load()
        if s.ui.mic:
            return not_ours(s.ui.mic, 'microphone')
        now = pactl('get-default-source').strip()
        return now if not now.startswith(OURS) else not_ours(self._mic, 'default input')

    def _loopback(self, on: bool) -> None:
        """Translation off: PipeWire loops the real microphone into "Peresmeshnik Mic" (none of our CPU), so the
        call hears the user as is; translation on: our me -> them direction speaks there instead."""
        if not self.devices or self._vmic is None:  # only the app's own devices (startup), not files or tests
            return
        try:  # found by its arguments, not a kept id: it goes away with its microphone, and PipeWire reuses ids
            loops = [m.split('\t')[0] for m in pactl('list', 'short', 'modules').splitlines()
                     if '\tmodule-loopback\t' in m and f'sink={MIC_SINK}' in m]
            for m in loops if not on else ():
                subprocess.run(['pactl', 'unload-module', m], capture_output=True, timeout=10)
            if on and not loops:
                pactl('load-module', 'module-loopback', f'source={self._mic_source()}', f'sink={MIC_SINK}',
                      'latency_msec=20', 'source_dont_move=true', 'sink_dont_move=true')
        except Exception as e:
            log.warning('microphone loopback: %s', e)
            if on:
                self.on_state(f'error: the call will not hear your microphone while translation is off: {e}')

    def _on(self) -> None:
        if self.on:
            return
        self.on_state('warming up')
        s = self.settings or config.load()
        files = {'them': self.them_file, 'me': self.me_file}
        t, failed = time.perf_counter(), []
        for name in ('them', 'me'):  # a side that cannot start (no key, no voice pack, no device) does not stop the other
            if self.devices or files[name]:
                try:
                    inp = FileIn(files[name]) if files[name] else self._input(name)
                    self.sides.append(Side(name, s, inp, self._output(name, s), self.overlay.side(name), self.on_state))
                except Exception as e:
                    log.exception('%s', name)
                    failed.append(f'{name}: {e}')
        if any(x.name == 'me' for x in self.sides):
            self._loopback(False)  # the call hears the user until the translation is ready
        for side in self.sides:
            side.set_live(self.live)
            side.start()
        self.on = bool(self.sides)
        for f in failed:
            self.on_state(f'error: {f}')
        if self.sides:
            self.on_state(f'translating ({", ".join(f"{x.name}: {x.engine}" for x in self.sides)}; '
                          f'ready in {time.perf_counter() - t:.1f} s)')

    def _input(self, name: str):
        if name == 'them':
            if self._call is None:
                raise RuntimeError(f'no call sink ({self.startup_error or "no sound server"})')
            return CallCapture(self._call.name)
        from peresmeshnik.audio.mic import Mic
        not_ours(pactl('get-default-source').strip(), 'default input')  # Mic(None) records it
        return Mic(None)  # S2's Mic takes the PipeWire source (s.ui.mic) in stage 2

    def _output(self, name: str, s: config.Settings):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('audio routing on this OS comes in a later version (macOS: stage 3, Windows: stage 4)')
        if name == 'them' or self.test:
            out = Headphones(s.ui.headphones, self.on_state)
            out.resolve()  # a missing device or our own as the default: say so now, not at the first word
            return out
        return self.virtual_mic()

    def _off(self) -> None:
        sides, self.sides = self.sides, []
        for side in sides:
            try:
                side.stop()
            except Exception:
                log.exception('stopping %s', side.name)
        self.live = False
        if not self.test:  # the test phrase's subtitles stay until the wizard closes
            self.overlay.side('them').hide()
        self._loopback(True)
        if self.on or sides:
            self.on = False
            self.on_state('idle, translation off')

    def set_live(self, on: bool) -> None:
        self.live = on  # the menu shows it at once; the sides switch on the control thread, after a start in progress
        self.submit(self._apply_live)

    def _apply_live(self) -> None:
        for side in self.sides:
            side.set_live(self.live)

    def quit(self) -> None:
        self.submit(self._off).result()
        self._pool.shutdown()
        self._loopback(False)
        if self._vmic:
            self._vmic.remove()  # the devices go with the app
        if self._call:
            self._call.remove()


def gui_available() -> bool:
    if sys.platform.startswith('linux'):
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or os.environ.get('QT_QPA_PLATFORM')):
            return False
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return True


def use_xwayland() -> bool:
    """On a Wayland session Qt would take its wayland plugin, where a window can neither stay on top nor
    choose its place (xdg-shell): the subtitles would end up under the browser. XWayland (xcb) keeps both;
    the hotkey still goes through the portal (the same `hotkey.wayland()` test)."""
    from peresmeshnik.ui.hotkey import wayland
    if sys.platform.startswith('linux') and wayland() and os.environ.get('DISPLAY') and not os.environ.get('QT_QPA_PLATFORM'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        return True
    return False


def lock_path() -> Path:
    """One app per sound server: two would remove each other's devices. XDG_RUNTIME_DIR is per user, like
    the user's PipeWire (its socket is $XDG_RUNTIME_DIR/pulse/native)."""
    return Path(os.environ.get('XDG_RUNTIME_DIR') or config.config_dir()) / 'peresmeshnik.lock'


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='peresmeshnik app', description='Tray, subtitles, hotkey and both translation directions.')
    ap.add_argument('--on', action='store_true', help='turn translation on at start')
    ap.add_argument('--them-file', help='feed this recording as the call sound (instead of the devices)')
    ap.add_argument('--me-file', help='feed this recording as my microphone (instead of the devices)')
    ap.add_argument('--show-overlay', action='store_true', help='show a sample subtitle at start (to check the overlay)')
    ap.add_argument('--no-wizard', action='store_true', help='do not open the setup wizard on the first run')
    ap.add_argument('-v', '--verbose', action='store_true', help='log subtitles, states and latency to stderr')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, stream=sys.stderr,
                        format='%(asctime)s %(name)s %(message)s')
    sys.stdout.reconfigure(encoding='utf-8')
    if use_xwayland():
        log.info('Wayland session: the overlay runs through XWayland')

    from PySide6.QtCore import QAbstractEventDispatcher, QLockFile, QObject, QTimer, Signal, Slot
    from PySide6.QtWidgets import QApplication, QMessageBox

    from peresmeshnik.ui import hotkey
    from peresmeshnik.ui.overlay import SubtitleOverlay
    from peresmeshnik.ui.tray import Tray

    qapp = QApplication(sys.argv[:1])
    qapp.setApplicationName('Peresmeshnik')
    qapp.setQuitOnLastWindowClosed(False)
    qapp.setCursorFlashTime(0)  # a steady text cursor: a blinking one repaints its field twice a second (idle watchdog)
    lock_path().parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(lock_path()))
    lock.setStaleLockTime(0)  # stale only when its process is gone, however old
    if not lock.tryLock(200):
        print('Peresmeshnik is already running: its icon is in the tray', file=sys.stderr, flush=True)
        box = QMessageBox(QMessageBox.Information, 'Peresmeshnik', 'Peresmeshnik is already running: its icon is in the tray.')
        QTimer.singleShot(5000, box.close)
        box.exec()
        return 3

    class Bridge(QObject):
        state = Signal(str)  # from the control thread to the GUI thread
        started = Signal()

        @Slot()
        def wake(self):  # a queued call: the GUI loop runs once more
            pass
    bridge = Bridge()
    overlay = SubtitleOverlay()
    ctl = Controller(overlay, bridge.state.emit, a.them_file, a.me_file)
    tray = Tray(ctl)
    bridge.state.connect(tray.set_state)
    bridge.state.connect(lambda st: print(st, flush=True))  # tests and the idle watchdog read these lines

    def toggle_live():
        if not ctl.on:  # the call hears the microphone as is already (the loopback)
            print('mode: translation is off', flush=True)
            tray.message('Translation is off: the call hears your own voice.')
            return
        ctl.set_live(not ctl.live)
        tray.sync()
        print(f'mode: {"live voice" if ctl.live else "translation"}', flush=True)

    s = config.load()
    backend, why = hotkey.make()
    binding = hotkey.Binding(backend, toggle_live, why)
    binding.done.connect(lambda combo, why: tray.hotkey_note(combo, why, announce=bool(why)))
    tray.toggle_live, tray.reload_hotkey = toggle_live, binding.bind
    tray.hotkey_note('', why, announce=sys.platform.startswith('linux'))  # macOS / Windows: no hotkey before stages 3-4
    if why:
        print(f'warning: {why}', file=sys.stderr)
    binding.bind(s.ui.hotkey)

    if a.show_overlay:
        overlay.side('them').show(Caption('Peresmeshnik subtitles: the original in grey', 'and the translation in white', True, 0))
    if not config.settings_path().exists() and not a.no_wizard:
        tray.open_wizard()  # first run; built before the "idle" line: it is part of the start
    started = ctl.submit(ctl.startup)
    bridge.started.connect(bridge.wake)
    started.add_done_callback(lambda _: bridge.started.emit())
    dispatcher = QAbstractEventDispatcher.instance()

    def announce():  # started (devices made) and nothing left to do (first paints done): idle from here on
        if started.done():
            dispatcher.aboutToBlock.disconnect(announce)
            bridge.state.emit(f'Peresmeshnik {__version__}: idle, translation off')
            if ctl.startup_error:
                tray.message(ctl.startup_error)
    dispatcher.aboutToBlock.connect(announce)
    if a.on:
        QTimer.singleShot(0, lambda: tray.set_translation(True))
    qapp.aboutToQuit.connect(ctl.quit)
    qapp.aboutToQuit.connect(lambda: log.info('overlay: %s', overlay.stats()))
    qapp.aboutToQuit.connect(binding.close)
    qapp.aboutToQuit.connect(lock.unlock)
    _quit_on_signals(qapp)
    return qapp.exec()


def _quit_on_signals(qapp) -> None:
    """SIGTERM, SIGHUP (the terminal closed) and Ctrl+C quit cleanly: the devices are removed. Python runs a
    signal handler only between bytecodes and the Qt loop sits in C++: the C handler writes to a socket
    that wakes the loop."""
    import signal
    import socket

    from PySide6.QtCore import QSocketNotifier, QTimer
    r, w = socket.socketpair()
    r.setblocking(False)
    w.setblocking(False)
    signal.set_wakeup_fd(w.fileno())
    notifier = QSocketNotifier(r.fileno(), QSocketNotifier.Read, qapp)
    notifier.activated.connect(lambda *_: r.recv(64))  # Python code runs: the handlers below fire
    qapp._signal_sockets = (r, w, notifier)
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP'):
        if hasattr(signal, name):  # no SIGHUP on Windows
            # queued: the handler runs between any two bytecodes of the GUI thread, e.g. inside the hotkey's
            # X read, and a direct quit() (aboutToQuit: hotkey.close) would close its display under it
            signal.signal(getattr(signal, name), lambda *_: QTimer.singleShot(0, qapp.quit))

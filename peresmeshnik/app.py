"""The app: tray, subtitle overlay, global hotkey, and the two directions wired to the devices.

    peresmeshnik app [--on] [--them-file call.wav] [--me-file me.wav] [--show-overlay] [-v]
    peresmeshnik                     # the same when a display is there (else the headless idle)

Each side ("them -> me", "me -> them") gets a local or cloud `Direction` from the settings. Turning
translation on loads the engines on a control thread (the tray says "warming up"), then opens the
inputs; turning it off closes everything again, so the idle app holds no stream, no engine thread
and no timer. The hotkey switches "translation <-> live voice": a direction that supports
`set_live` is told so; for any other the input goes straight to the output instead.
`--them-file` / `--me-file` feed recordings at their own pace instead of the devices (tests, demos).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import threading
import time

import numpy as np

from peresmeshnik import __version__, config
from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech

log = logging.getLogger('peresmeshnik.app')


def resample(pcm: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return pcm
    # ponytail: linear interpolation, fine for speech and for the live pass-through; soxr if quality matters
    return np.interp(np.arange(0, len(pcm), sr / to), np.arange(len(pcm)), pcm).astype(np.float32)


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
    """`AudioOut` to a PipeWire/PulseAudio sink at whatever rate the first chunk has (Piper 22.05 kHz,
    Gemini 24 kHz); later chunks at another rate (the live pass-through, 16 kHz) are resampled."""

    def __init__(self, sink: str):
        self.sink, self._out = sink, None
        self._lock = threading.Lock()  # capture (live), events (speech) and control (switch, stop) threads

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            if self._out is None:
                from peresmeshnik.audio.out import PulseOut
                self._out = PulseOut(sample_rate, self.sink)
            self._out.play(resample(pcm, sample_rate, self._out.sample_rate), self._out.sample_rate)

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


class Side:
    """One direction with its input, output and subtitles. `name` is 'them' (them -> me) or 'me'."""

    def __init__(self, name: str, s: config.Settings, inp, out, overlay, on_error=lambda text: None):
        self.name, self.inp, self.out, self.overlay, self.on_error = name, inp, out, overlay, on_error
        self.src, self.tgt = (s.their_lang, s.my_lang) if name == 'them' else (s.my_lang, s.their_lang)
        self.engine = s.them_to_me if name == 'them' else s.me_to_them
        self.live = False
        self.directions = []  # (direction, takes its captions, takes its speech)
        self._consumers: list[threading.Thread] = []
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
        for d, captions, speech in self.directions:
            t = threading.Thread(target=self._consume, args=(d, captions, speech), name=f'{self.name}-events', daemon=True)
            t.start()
            self._consumers.append(t)
        self.inp.start(self._on_frames)

    def _on_frames(self, pcm: np.ndarray) -> None:  # capture thread: only hand on
        if self.live and not self._supports_live:
            self.out.play(pcm, SAMPLE_RATE)  # ponytail: hard switch, no crossfade; S2's set_live crossfades
            return
        for d, _, _ in self.directions:
            d.feed(pcm)

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
        for d, _, _ in self.directions:
            d.close()
        for t in self._consumers:
            t.join(15)
        self.out.stop()


class Controller:
    """Turns translation on and off on one control thread; reports states through `on_state(str)`."""

    def __init__(self, overlay, on_state, them_file: str | None = None, me_file: str | None = None,
                 settings: config.Settings | None = None, test: bool = False):
        """`settings`: use these instead of the saved ones; `test`: both directions speak into the headphones
        (the wizard's test phrase)."""
        self.overlay, self.on_state, self.them_file, self.me_file = overlay, on_state, them_file, me_file
        self.settings, self.test = settings, test
        self.sides: list[Side] = []
        self.live = False
        self.on = False
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='control')  # no thread until first use
        self._vmic = None  # the virtual microphone lives while the app runs (ARCHITECTURE §7)

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
        """The virtual mic is there from the start, so the call app can pick it before translation is on."""
        if self.them_file is None and self.me_file is None and sys.platform.startswith('linux'):
            try:
                self.virtual_mic()
            except Exception as e:  # no sound server or no pactl: say so, the app still runs
                log.warning('no virtual microphone: %s', e)

    def _on(self) -> None:
        if self.on:
            return
        self.on_state('warming up')
        s = self.settings or config.load()
        outs = self._outputs(s)
        if self.them_file or self.me_file:  # recordings instead of devices
            ins = {k: FileIn(v) for k, v in (('them', self.them_file), ('me', self.me_file)) if v}
        else:
            ins = self._inputs(s)
        t, failed = time.perf_counter(), []
        for name in ('them', 'me'):  # a side that cannot start (no key, no voice pack) does not stop the other
            if name in ins:
                try:
                    self.sides.append(Side(name, s, ins[name], outs[name], self.overlay.side(name), self.on_state))
                except Exception as e:
                    log.exception('%s', name)
                    failed.append(f'{name}: {e}')
        for side in self.sides:
            side.set_live(self.live)
            side.start()
        self.on = bool(self.sides)
        for f in failed:
            self.on_state(f'error: {f}')
        if self.sides:
            self.on_state(f'translating ({", ".join(f"{x.name}: {x.engine}" for x in self.sides)}; '
                          f'ready in {time.perf_counter() - t:.1f} s)')

    def _outputs(self, s: config.Settings) -> dict:
        if not sys.platform.startswith('linux'):
            raise RuntimeError('audio routing on this OS comes in a later version (macOS: stage 3, Windows: stage 4)')
        from peresmeshnik.audio.out import default_sink
        headphones = s.ui.headphones or default_sink()
        if headphones.startswith('peresmeshnik_'):  # no real output: the default fell to our own device
            raise RuntimeError(f'the default output is our own device {headphones}: choose the headphones in the settings')
        return {'them': Headphones(headphones), 'me': Headphones(headphones) if self.test else self.virtual_mic()}

    def _inputs(self, s: config.Settings) -> dict:
        from peresmeshnik.audio.capture_pipewire import PipeWireCapture
        from peresmeshnik.audio.mic import Mic
        return {'them': PipeWireCapture(), 'me': Mic(None)}  # Mic takes a PipeWire source name in S2's stage 2 (s.ui.mic)

    def _off(self) -> None:
        sides, self.sides = self.sides, []
        for side in sides:
            try:
                side.stop()
            except Exception:
                log.exception('stopping %s', side.name)
        if not self.test:  # the test phrase's subtitles stay until the wizard closes
            self.overlay.side('them').hide()
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
        if self._vmic:
            self._vmic.remove()  # the device goes with the app


def gui_available() -> bool:
    if sys.platform.startswith('linux'):
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or os.environ.get('QT_QPA_PLATFORM')):
            return False
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return True


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

    from PySide6.QtCore import QAbstractEventDispatcher, QObject, QTimer, Signal, Slot
    from PySide6.QtWidgets import QApplication

    from peresmeshnik.ui import hotkey
    from peresmeshnik.ui.overlay import SubtitleOverlay
    from peresmeshnik.ui.tray import Tray

    qapp = QApplication(sys.argv[:1])
    qapp.setApplicationName('Peresmeshnik')
    qapp.setQuitOnLastWindowClosed(False)
    qapp.setCursorFlashTime(0)  # a steady text cursor: a blinking one repaints its field twice a second (idle watchdog)

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
        ctl.set_live(not ctl.live)
        tray.sync()
        print(f'mode: {"live voice" if ctl.live else "translation"}', flush=True)

    s = config.load()
    hk, why = hotkey.make()
    if hk is not None:
        try:
            hk.register(s.ui.hotkey, toggle_live)
        except hotkey.HotkeyError as e:
            hk, why = None, f'no global hotkey: {e}. Use the tray menu item "Live voice"'
    tray.toggle_live = toggle_live
    tray.hotkey_note(s.ui.hotkey if hk else '', why)
    if why:
        print(f'warning: {why}', file=sys.stderr)
    bound = [s.ui.hotkey if hk else '']

    def rebind(combo: str) -> str:
        """New hotkey from the settings or the wizard: '' when it works, else why not."""
        if hk is None:
            return ''  # nothing to bind here (the tray says why); the combo is kept for where it works
        if bound[0]:
            hk.unregister(bound[0])
        try:
            hk.register(combo, toggle_live)
        except hotkey.HotkeyError as e:
            if bound[0]:
                hk.register(bound[0], toggle_live)
            return str(e)
        bound[0] = combo
        tray.hotkey_note(combo, '')
        return ''
    tray.reload_hotkey = rebind

    if a.show_overlay:
        overlay.side('them').show(Caption('Peresmeshnik subtitles: the original in grey', 'and the translation in white', True, 0))
    if not config.settings_path().exists() and not a.no_wizard:
        tray.open_wizard()  # first run; built before the "idle" line: it is part of the start
    started = ctl.submit(ctl.startup)
    bridge.started.connect(bridge.wake)
    started.add_done_callback(lambda _: bridge.started.emit())
    dispatcher = QAbstractEventDispatcher.instance()

    def announce():  # started (virtual mic made) and nothing left to do (first paints done): idle from here on
        if started.done():
            dispatcher.aboutToBlock.disconnect(announce)
            bridge.state.emit(f'Peresmeshnik {__version__}: idle, translation off')
    dispatcher.aboutToBlock.connect(announce)
    if a.on:
        QTimer.singleShot(0, lambda: tray.set_translation(True))
    qapp.aboutToQuit.connect(ctl.quit)
    qapp.aboutToQuit.connect(lambda: log.info('overlay: %s', overlay.stats()))
    if hk is not None:
        qapp.aboutToQuit.connect(hk.close)
    _quit_on_signals(qapp)
    return qapp.exec()


def _quit_on_signals(qapp) -> None:
    """SIGTERM / Ctrl+C quit cleanly (the virtual mic is removed). Python runs a signal handler only
    between bytecodes, and the Qt loop sits in C++: the C handler writes to a socket that wakes the loop."""
    import signal
    import socket

    from PySide6.QtCore import QSocketNotifier
    r, w = socket.socketpair()
    r.setblocking(False)
    w.setblocking(False)
    signal.set_wakeup_fd(w.fileno())
    notifier = QSocketNotifier(r.fileno(), QSocketNotifier.Read, qapp)
    notifier.activated.connect(lambda *_: r.recv(64))  # Python code runs: the handlers below fire
    qapp._signal_sockets = (r, w, notifier)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: qapp.quit())

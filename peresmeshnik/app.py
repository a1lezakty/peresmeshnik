"""The app: tray, subtitle overlay, global hotkey, and the two directions wired to the devices.

    peresmeshnik app [--on] [--them-file call.wav] [--me-file me.wav] [--show-overlay] [-v]
    peresmeshnik                     # the same when a display is there (else the headless idle)

Each side ("them -> me", "me -> them") gets a local or cloud `Direction` from the settings. Turning
translation on loads the engines on a control thread (the tray says "warming up"), then opens the
inputs; turning it off closes them again, so the idle app holds no stream, no engine thread and no
timer. Linux devices live as long as the app: the call sink "Peresmeshnik Call" (while translating the call
app's sound is moved there, and back when off) and the virtual mic "Peresmeshnik Mic"; while translation is off PipeWire loops the
real microphone into the virtual mic, so the call still hears the user. The hotkey switches
"translation <-> live voice": a direction that supports `set_live` is told so (me -> them: wrapped in
audio/mic.LiveSwitch, a crossfade); for any other the input goes straight to the output instead. The
microphone is the PipeWire source of the settings, reopened when it goes away. One app per sound server
(a lock file).
`--them-file` / `--me-file` feed recordings at their own pace instead of the devices (tests, demos).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from peresmeshnik import __version__, config
from peresmeshnik.audio import c_locale
from peresmeshnik.audio.resample import Resampler
from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech

log = logging.getLogger('peresmeshnik.app')
OURS = 'peresmeshnik_'  # every PipeWire device the app makes starts with this
MIC_SINK = 'peresmeshnik_mic_sink'  # = audio/virtual_mic_linux.SINK, without importing it on other OSes
BARGE_IN_WORDS = 3  # my clause has this many words: I talk over their voice (fewer: "yeah", "ага" while listening)


def pactl(*args: str) -> str:
    return subprocess.run(['pactl', *args], check=True, capture_output=True, text=True, timeout=10, env=c_locale()).stdout


def not_ours(name: str, what: str) -> str:
    """A default device that is one of ours would make a loop (the translation translated again, or sent to the call)."""
    if name.startswith(OURS):
        raise RuntimeError(f'the {what} is our own device {name}: choose your real device in the settings or the system')
    return name


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
    """`AudioOut` to a PipeWire/PulseAudio sink (Linux) or a Core Audio device (macOS) at the first chunk's rate
    (Piper 22.05 kHz, Gemini 24 kHz); chunks at another rate (the live pass-through, 16 kHz) are resampled. `sink`
    '' = the default. The output ends when its device goes (Bluetooth profile switch, USB unplugged: it does not
    follow another device): its `on_end` tells the tray at once, the next chunk opens the saved device again, else
    the current default. macOS: PortAudio opens only the devices of its last refresh, so `on_back()` (the controller
    refreshes it) is called once when the saved device is back, or there but unknown to PortAudio; suspend() and
    resume() close and reopen around that refresh, the speech not played yet plays on the new device."""
    retry_s = 2.0

    def __init__(self, sink: str = '', on_state=lambda text: None, on_back=None):
        self.sink, self.on_state, self.on_back = sink, on_state, on_back
        self._out, self._rs, self._retry_at, self._failing = None, {}, 0.0, False
        self._arrival = None  # macOS: waiting for the saved device while another one plays
        self._suspended, self._pending, self._asked = False, [], set()  # _asked: refreshes asked for, per device
        self._lock = threading.Lock()  # capture (live), events (speech) and control (switch, stop) threads

    def resolve(self) -> str:
        if sys.platform == 'darwin':
            from peresmeshnik.audio import capture_macos, coreaudio
            name = self.sink if self.sink in {d['name'] for d in coreaudio.devices() if d['outputs']} else coreaudio.default()
            if not name:
                raise RuntimeError('no output device')
            if name == capture_macos.CALL_DEVICE and capture_macos.path() == 'blackhole':  # the call would hear itself
                raise RuntimeError(f'the headphones are {name}, the call app\'s speaker: choose your real headphones in the '
                                   'settings or the system')
            return name
        names = [line.split('\t')[1] for line in pactl('list', 'short', 'sinks').splitlines() if '\t' in line]
        if self.sink in names:
            return self.sink
        return not_ours(pactl('get-default-sink').strip(), 'default output')

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            if self._suspended:  # PortAudio is being refreshed: played after it
                self._pending.append((pcm, sample_rate))
                return
            out = self._out
            if out is not None and not out.alive:  # died just now (its on_end may come before or after this)
                self._drop()
                out = None
            if out is None:
                if time.monotonic() < self._retry_at:
                    return  # no device for now: this audio is dropped
                try:
                    from peresmeshnik.audio.out import Output
                    sink = self.resolve()
                    self._out = out = Output(sample_rate, sink, on_end=self._gone)
                    if self._failing:  # the tray still shows the error: put it back
                        self.on_state(f'translating (the headphones are back: {sink})')
                    self._rs, self._failing = {}, False
                    self._await(sink)
                except Exception as e:
                    from peresmeshnik.audio.out import NotInPortAudio
                    if isinstance(e, NotInPortAudio):  # there, but plugged in after PortAudio's last refresh
                        self._ask_back(sink)
                    self._retry_at = time.monotonic() + self.retry_s
                    self._fail(f'no headphones: {e}')
                    return
            rs = self._rs.get(sample_rate) or self._rs.setdefault(sample_rate, Resampler(sample_rate, out.sample_rate))
            out.play(rs(pcm), out.sample_rate)

    def _await(self, opened: str) -> None:  # under the lock
        """macOS, another device than the saved one opened: wait for the saved one to come back."""
        if sys.platform == 'darwin' and self.on_back and self.sink and opened != self.sink and self._arrival is None:
            from peresmeshnik.audio.coreaudio import Arrival
            self._asked.discard(self.sink)  # it went away since: its coming back is worth a refresh again
            self._arrival = Arrival(self.sink, lambda: self._ask_back(self.sink))

    def _ask_back(self, name: str) -> None:
        """Once per device: a refresh that did not make PortAudio see it would only be repeated every retry."""
        if self.on_back and name not in self._asked:
            self._asked.add(name)
            log.info('headphones: %s is there: refreshing the devices', name)
            self.on_back()

    def suspend(self) -> None:
        """macOS, before PortAudio's refresh: close the output; what it has not played yet waits for resume()."""
        with self._lock:
            self._suspended = True
            out, self._out = self._out, None
            arrival, self._arrival = self._arrival, None
        if arrival:
            arrival.close()
        if out:
            self._pending[:0] = [(c, out.rate) for c in out.take()]
            out.stop(drain=False)

    def resume(self) -> None:
        with self._lock:
            self._suspended, self._retry_at = False, 0.0
            pending, self._pending = self._pending, []
        for pcm, sr in pending:  # the first chunk opens the device (the saved one, if it is back)
            self.play(pcm, sr)

    def _gone(self, why: str) -> None:
        """PulseOut's on_end, on its own thread: the tray hears it now; the next chunk reopens."""
        log.warning('headphones: %s', why)
        with self._lock:
            if self._out is not None and not self._out.alive:  # else play() has already reopened
                self._drop()

    def _drop(self) -> None:  # under the lock: the current output is dead
        out, self._out = self._out, None
        out.flush()
        threading.Thread(target=out.stop, daemon=True).start()  # its writer thread ends (pacat is gone)
        self._fail('the headphones went away; reopening')

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
            arrival, self._arrival = self._arrival, None
        if arrival:
            arrival.close()
        if out:
            out.flush()
            out.stop()  # plays out what pacat holds: outside the lock


class NoMicrophone(RuntimeError):
    """No microphone at all (none plugged in, or only our devices and monitors of sinks)."""


def mic_source(saved: str, fallback: str = '') -> str:
    """The microphone of me -> them, the same for the translation and for the loopback while it is off: the saved
    PipeWire source when it is there, else the default input, else `fallback` (the default from before our devices
    existed); never one of ours (the app would hear itself)."""
    names = [line.split('\t')[1] for line in pactl('list', 'short', 'sources').splitlines() if '\t' in line]
    if saved and (saved in names or saved.startswith(OURS)):
        return not_ours(saved, 'microphone')
    now = pactl('get-default-source').strip()
    # then any real source: WirePlumber 0.5 may keep naming a default that is gone; a monitor of a sink is no
    # microphone (the headphones' monitor would echo into the call)
    for name in (now, fallback, *names):
        if name and name in names and not name.startswith(OURS) and not name.endswith('.monitor'):
            if saved or name != now:
                log.warning('microphone %s is not there: using %s', saved or now or 'the default', name)
            return name
    missing = f'{saved} is not there and ' if saved else ''
    if now.startswith(OURS):
        raise NoMicrophone(f'no microphone: {missing}the default input is our own device {now}: choose your real '
                           'microphone in the settings or the system')
    raise NoMicrophone(f'no microphone: {missing}the default input is {now or "none"}; plug one in or choose it in the settings')


class Microphone:
    """`AudioIn` for me -> them from `mic_source`. While it runs one `pactl subscribe` watches PipeWire's events (no
    polling): when its source is removed (a headset unplugged, a Bluetooth profile switch) or the stream ends by
    itself, it is reopened on the saved source if it is back, else on the default, with one tray message per outage;
    with no microphone at all the next new source is taken. The removal event matters: since WirePlumber 0.5 a
    `node.dont-reconnect` stream may outlive its device without an end. When the event stream itself ends (the sound
    server restarted) it subscribes again after a pause, longer while the server stays away, then reopens."""
    RESUBSCRIBE_S, RESUBSCRIBE_MAX_S = 2.0, 30.0

    def __init__(self, saved: str, on_state=lambda text: None, fallback: str = ''):
        self.saved, self.on_state, self.fallback = saved, on_state, fallback
        self._mic = self._index = self._watch = self._on_frames = None
        self._stopped = False
        self._wake = threading.Event()  # stop() ends a pause before resubscribing
        self._lock = threading.Lock()  # the current stream: control thread (stop), events thread, a dead stream's reader
        self._opening = threading.RLock()  # one reopen at a time (a source gone while opening retries from inside)

    def start(self, on_frames) -> None:
        self._on_frames, self._stopped = on_frames, False
        self._wake.clear()
        # subscribed before opening: a removal right after the start is still seen
        self._watch = watch = self._subscribe()
        try:
            self._open()
        except BaseException:
            watch.kill()
            watch.wait()
            raise
        threading.Thread(target=self._events, args=(watch,), name='mic-events', daemon=True).start()

    @staticmethod
    def _subscribe():
        return subprocess.Popen(  # c_locale: the events are parsed, in English whatever the desktop's language
            ['pactl', 'subscribe'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=c_locale())

    def _index_of(self, name: str) -> str | None:
        return next((line.split('\t')[0] for line in pactl('list', 'short', 'sources').splitlines()
                     if line.split('\t')[1:2] == [name]), None)

    def _open(self) -> str:
        from peresmeshnik.audio.mic import Mic
        name = mic_source(self.saved, self.fallback)
        index = self._index_of(name)  # before the start: its removal event can come any time after
        m = Mic(name)
        m.on_end = lambda why: self._lost(m, why)
        m.start(self._on_frames)
        with self._lock:
            if self._stopped:
                m.stop()
                return ''
            self._mic, self._index = m, index
        if self._index_of(name) != index:  # gone (or replaced) while opening: its event may have come before us
            self._lost(m, f'{name} went away while opening')
            return ''
        return m.source

    def _lost(self, m, why: str) -> None:
        with self._lock:
            if self._stopped or m is not self._mic:  # stopping, or a stream already given up
                return
            self._mic = self._index = None
        m.stop()
        log.warning('microphone: %s', why)
        self.on_state(f'error: the microphone went away ({why}); reopening')
        self._retry()  # the default input may be there already

    def _retry(self) -> None:
        with self._opening:
            if self._mic is not None or self._stopped:
                return
            try:
                name = self._open()
            except Exception as e:
                log.info('microphone: not yet (%s)', e)
                return
        if name:
            self.on_state(f'translating (the microphone is back: {name})')

    def _events(self, watch) -> None:
        pause = self.RESUBSCRIBE_S
        while True:
            t0 = time.monotonic()
            for line in watch.stdout:
                ev = re.match(r"Event '(\w+)' on (\w+) #(\d+)", line)
                if self._stopped:
                    return
                if not ev:
                    continue
                what, kind, index = ev.groups()
                m = self._mic
                if m is not None and kind == 'source' and what == 'remove' and index == self._index:
                    self._lost(m, f'{m.source} was removed')
                elif m is None and (kind == 'server' or (kind == 'source' and what == 'new')):
                    self._retry()
            watch.wait()
            if self._stopped:
                return
            # the event stream ended: the sound server restarted or is gone. A repeat after a pause, not polling;
            # the pause grows while the server stays away
            pause = self.RESUBSCRIBE_S if time.monotonic() - t0 > self.RESUBSCRIBE_MAX_S else pause
            log.warning('microphone: PipeWire events ended (pactl exit %s); subscribing again in %g s', watch.returncode, pause)
            if pause == self.RESUBSCRIBE_S:  # once per outage
                self.on_state(f'error: lost the sound server; reconnecting the microphone in {pause:g} s')
            if self._wake.wait(pause):
                return
            with self._lock:
                if self._stopped:
                    return
                self._watch = watch = self._subscribe()
            pause = min(pause * 2, self.RESUBSCRIBE_MAX_S)
            m = self._mic
            if m is None:
                self._retry()
            elif m._proc is None or m._proc.poll() is not None:  # went down with the server, its end not reported
                self._lost(m, f'{m.source} ended with the sound server')
            else:  # only the events were gone
                self.on_state('translating (the sound server is back)')

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            mic, self._mic = self._mic, None
        self._wake.set()
        if self._watch:
            self._watch.kill()
            self._watch.wait()
        if mic:
            mic.stop()


class CallSink:
    """"Peresmeshnik Call": the null sink the call's sound is moved into while translating, kept for the app's run."""

    def __init__(self):
        from peresmeshnik.audio.capture_pipewire import CALL_SINK, LOW_PRIORITY
        self.name = CALL_SINK
        self.module = pactl('load-module', 'module-null-sink', f'sink_name={CALL_SINK}',
                            f'sink_properties=\'device.description="Peresmeshnik Call" {LOW_PRIORITY}\'').strip()

    def remove(self) -> None:
        subprocess.run(['pactl', 'unload-module', self.module], capture_output=True, timeout=10)


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
        self.on_talk = lambda: None  # 'me': I am talking (a partial caption of BARGE_IN_WORDS): barge-in on the other side
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
        d, captions, speech = self.directions[0]
        if name == 'me' and not getattr(d, 'supports_live', False):  # "translation <-> live voice" crossfaded (S2)
            from peresmeshnik.audio.mic import LiveSwitch
            self.directions[0] = (LiveSwitch(d), captions, speech)

    def _local(self, s: config.Settings):
        from peresmeshnik.pipeline import make_local
        if self.name == 'them':
            from peresmeshnik.pipeline import MAX_LAG_S
            d = make_local(self.src, self.tgt, settings=s, max_lag_s=MAX_LAG_S)  # the queue policy: subtitles keep the rest
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
        for d, _, _ in self.directions[:1] if self.live else self.directions:  # live: only the switch, nothing translates
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
                    if not e.final and len(e.original.split()) >= BARGE_IN_WORDS:
                        self.on_talk()
                elif isinstance(e, Speech) and speech and (not self.live or self._supports_live):
                    self.out.play(e.pcm, e.sample_rate)
        except Exception as e:  # a dead session (key revoked, quota, network) must show in the tray, not only in the log
            log.exception('%s: direction failed', self.name)
            self.on_error(f'error: {self.name}: {e}')

    def interrupt(self) -> None:
        """Barge-in: the user talks over this side's voice. What is not heard yet goes; the subtitles stay."""
        if self.live:  # the original voice passes through: the user chose to hear it
            return
        for d, _, _ in self.directions:
            if hasattr(d, 'interrupt'):
                d.interrupt()
        self.out.flush()

    def set_live(self, on: bool) -> None:
        if self._supports_live:
            self.directions[0][0].set_live(on)
            if on and not self.live:  # the local subtitles get no input while live: a pause closes their open clause
                for d, _, _ in self.directions[1:]:
                    d.feed(np.zeros(SAMPLE_RATE // 2, np.float32))
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
        self._vmic = self._call = self._router = None
        self._through = self._through_arrival = None  # macOS < 14.4, translation off: the call app's speaker on to the headphones
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
        if self.devices and sys.platform == 'darwin':
            from peresmeshnik.audio import capture_macos
            if capture_macos.path() == 'tap':  # no permission yet: say so now, not at the first call
                try:
                    code, why = capture_macos.check()
                except Exception as e:
                    code, why = 1, str(e)
                self.startup_error = why if code else ''
            else:
                self._through_start()
            return
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
        if self._call:  # a killed run, or WirePlumber restoring an old move: the call must not go silent while off
            from peresmeshnik.audio.capture_pipewire import return_strays
            if back := step('call streams', lambda: return_strays(self._call.name)):
                log.warning('gave the call sound back to the default output: %s', ', '.join(back))
        step('virtual mic', self.virtual_mic)
        self._loopback(True)
        self.startup_error = '; '.join(errors)

    def _mic_source(self) -> str:
        return mic_source((self.settings or config.load()).ui.mic, self._mic)

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
        except NoMicrophone as e:  # nothing to loop; me -> them says so when translation is turned on
            log.warning('microphone loopback: %s', e)
        except Exception as e:
            log.warning('microphone loopback: %s', e)
            if on:
                self.on_state(f'error: the call will not hear your microphone while translation is off: {e}')

    def _on(self) -> None:
        if self.on:
            return
        self.on_state('warming up')
        self._refresh_devices()  # macOS: devices plugged in since (AirPods back) become visible to PortAudio
        s = self.settings or config.load()
        files = {'them': self.them_file, 'me': self.me_file}
        t, failed = time.perf_counter(), []
        for name in ('them', 'me'):  # a side that cannot start (no key, no voice pack, no device) does not stop the other
            if self.devices or files[name]:
                try:
                    inp = FileIn(files[name]) if files[name] else self._input(name, s)
                    self.sides.append(Side(name, s, inp, self._output(name, s), self.overlay.side(name), self.on_state))
                except Exception as e:
                    log.exception('%s', name)
                    failed.append(f'{name}: {e}')
        if any(x.name == 'me' for x in self.sides):
            self._loopback(False)  # the call hears the user until the translation is ready
        them = next((x for x in self.sides if x.name == 'them'), None)
        for x in self.sides:
            if x.name == 'me' and them is not None:
                x.on_talk = them.interrupt
        self._through_stop()  # BlackHole 16ch goes to the translation now
        for side in self.sides:
            side.set_live(self.live)
            side.start()
        self.on = bool(self.sides)
        call = self._route(s) if self.devices and any(x.name == 'them' for x in self.sides) else ''
        for f in failed:
            self.on_state(f'error: {f}')
        if self.sides:
            self.on_state(f'translating ({", ".join(f"{x.name}: {x.engine}" for x in self.sides)}; {call}'
                          f'ready in {time.perf_counter() - t:.1f} s)')
        if call.startswith('call app: none'):
            self.on_state('error: choose the call app in the tray menu (Call app): until then the call is not translated')

    def _route(self, s: config.Settings) -> str:
        """The call app's sound into the call sink (Linux) or a tap of it (macOS 14.4+) while translating: the app
        chosen, else the one known call app playing now (remembered; a browser only by hand: all of its tabs would
        move)."""
        from peresmeshnik.audio.capture_pipewire import CallRouter, guess_call_app
        try:
            tap = None
            if sys.platform == 'darwin':
                from peresmeshnik.audio import capture_macos
                tap = next(x for x in self.sides if x.name == 'them').inp
                if not isinstance(tap, capture_macos.ProcessTap):  # BlackHole 16ch: the call app's own speaker setting
                    return f'call sound: {tap.device} (the call app\'s speaker); '
                guess_call_app = capture_macos.guess_call_app
            app = s.ui.call_app
            if not app and (app := guess_call_app()):
                self._save_call_app(app)
            if not app:
                return 'call app: none; '
            if tap is not None:
                tap.retarget(app)
                self._router = SimpleNamespace(stop=lambda: tap.retarget(''))  # off: no tap, the call is heard as usual
                return f'call sound: {app}; '
            self._router = CallRouter(app, self._call.name)
            self._router.start()
            return f'call sound: {app}{"" if self._router.moved else " (not playing yet)"}; '
        except Exception as e:
            log.exception('call routing')
            self._router = None
            self.on_state(f'error: cannot move the call sound: {e}')
            return ''

    def _unroute(self) -> None:
        router, self._router = self._router, None
        if router:
            try:
                router.stop()  # the call app's sound goes back where it was
            except Exception as e:
                log.exception('call routing')
                self.on_state(f'error: give the call sound back by hand (pavucontrol): {e}')

    def _save_call_app(self, app: str) -> None:
        if self.settings is not None:
            self.settings.ui.call_app = app
        else:
            s = config.load()
            s.ui.call_app = app
            config.save(s)

    def set_call_app(self, app: str) -> concurrent.futures.Future:
        """From the tray: remember the call app; while translating its sound moves now (the old one's goes back)."""
        return self.submit(self._set_call_app, app)

    def _set_call_app(self, app: str) -> None:
        self._save_call_app(app)
        if self.on and (self._call or sys.platform == 'darwin') and any(x.name == 'them' for x in self.sides):
            self._unroute()
            self.on_state(f'translating ({self._route(self.settings or config.load()).rstrip("; ")})')

    def _input(self, name: str, s: config.Settings | None = None):
        s = s or self.settings or config.load()
        if name == 'them' and sys.platform == 'darwin':
            from peresmeshnik.audio import capture_macos
            on_end = lambda why: self.on_state(f'error: the call sound capture stopped: {why}')  # noqa: E731
            if capture_macos.path() == 'blackhole':
                return capture_macos.BlackHoleIn(on_end=on_end)
            code, why = capture_macos.check()
            if code:
                raise RuntimeError(why)
            return capture_macos.ProcessTap(on_end=on_end)  # the app is chosen in _route, after the start
        if name == 'them':
            if self._call is None:
                raise RuntimeError(f'no call sink ({self.startup_error or "no sound server"})')
            from peresmeshnik.audio.capture_pipewire import PipeWireCapture
            return PipeWireCapture(self._call.name, make_sink=False,  # the app's sink, kept for its whole run
                                   on_end=lambda why: self.on_state(f'error: the call sound capture stopped: {why}'))
        mic_source(s.ui.mic, self._mic)  # no microphone now: this side says why, the other one still starts
        return Microphone(s.ui.mic, self.on_state, self._mic)

    def _output(self, name: str, s: config.Settings):
        if sys.platform == 'win32':
            raise RuntimeError('audio routing on Windows comes in a later version (stage 4)')
        if name == 'them' or self.test:
            back = (lambda: self.submit(self._refresh_devices)) if self.devices and sys.platform == 'darwin' else None
            out = Headphones(s.ui.headphones, self.on_state, on_back=back)
            out.resolve()  # a missing device or our own as the default: say so now, not at the first word
            return out
        if not sys.platform.startswith('linux'):
            raise RuntimeError('the virtual microphone on macOS comes in a later version (stage 3)')
        return self.virtual_mic()

    def _off(self) -> None:
        self._unroute()  # first: the user hears the call as is again at once
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
        self._through_start()
        if self.on or sides:
            self.on = False
            self.on_state('idle, translation off')

    def _refresh_devices(self) -> dict:
        """macOS: PortAudio opens only the devices of its last refresh, and a refresh closes every PortAudio stream of
        the process. So the streams that hold PortAudio (`suspend()` / `resume()`) are closed around it; the engines and
        the process tap stay (BENCH: the gap). The control thread; at translation on (nothing open yet) and when the
        saved headphones are back."""
        if not (self.devices and sys.platform == 'darwin'):
            return {}
        t = [time.perf_counter()]
        held = [x for side in self.sides for x in (side.inp, side.out) if hasattr(x, 'suspend')]
        # at once: PortAudio stops each stream in 0.1-0.25 s (pa_mac_core polls the unit with Pa_Sleep(100))
        stops = [threading.Thread(target=self._quietly, args=(x.suspend,), name='suspend') for x in held]
        for th in stops:
            th.start()
        self._through_stop()
        for th in stops:
            th.join()
        t.append(time.perf_counter())
        from peresmeshnik.audio.coreaudio import refresh_portaudio
        refresh_portaudio()
        t.append(time.perf_counter())
        for x in held:
            x.resume()
        self._through_start()  # translation off (or still warming up): the call heard on macOS < 14.4
        t.append(time.perf_counter())
        ms = {k: round(1000 * (b - a)) for k, a, b in zip(('suspend', 'refresh', 'resume'), t, t[1:])}
        log.info('devices refreshed: %s ms (%d streams reopened)', ms, len(held))
        if held:
            self.on_state(f'translating (devices changed: reopened in {sum(ms.values())} ms)')
        return ms

    @staticmethod
    def _quietly(fn) -> None:
        try:
            fn()
        except Exception:
            log.exception('suspending a stream')

    def _through_start(self) -> None:
        """macOS < 14.4 with translation off: the call app's speaker (BlackHole 16ch) on to the headphones, else the
        call is not heard. Reopened on the saved headphones or the default when the device goes."""
        from peresmeshnik.audio import capture_macos
        if not (self.devices and sys.platform == 'darwin') or capture_macos.path() != 'blackhole' or self._through or self.sides:
            return
        try:
            saved = (self.settings or config.load()).ui.headphones
            out = Headphones(saved).resolve()
            if saved and out != saved:  # the saved headphones are gone: back to them when they are back
                from peresmeshnik.audio.coreaudio import Arrival
                self._through_arrival = Arrival(saved, lambda: self.submit(self._refresh_devices))
            self._through = through = capture_macos.PassThrough(out, on_end=lambda why: self.submit(self._through_lost, through, why))
            through.start()
            log.info('call sound: %s -> %s while translation is off', capture_macos.CALL_DEVICE, out)
        except Exception as e:
            self._through = None
            log.warning('pass-through: %s', e)
            self.on_state(f'error: the call is not heard while translation is off: {e}')

    def _through_lost(self, through, why: str) -> None:  # the control thread
        if through is not self._through:
            return
        log.warning('pass-through: %s', why)
        self._through_stop()
        if not self.on:
            self.on_state(f'error: the headphones went away ({why}); reopening')  # once: a failed reopen says its own
            self._through_start()
            if self._through:
                self.on_state(f'idle, translation off (the call plays on {self._through.output})')

    def _through_stop(self) -> None:
        arrival, self._through_arrival = self._through_arrival, None
        if arrival:
            arrival.close()
        through, self._through = self._through, None
        if through:
            through.stop()

    def set_live(self, on: bool) -> None:
        self.live = on  # the menu shows it at once; the sides switch on the control thread, after a start in progress
        self.submit(self._apply_live)

    def _apply_live(self) -> None:
        for side in self.sides:
            side.set_live(self.live)

    def quit(self) -> None:
        self.submit(self._off).result()
        self._pool.shutdown()
        self._through_stop()
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
    if sys.platform == 'darwin':
        from peresmeshnik.ui import macos
        macos.accessory()  # a menu-bar app: no Dock icon (the .app says so in its Info.plist too, LSUIElement)
    lock_path().parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(lock_path()))
    lock.setStaleLockTime(0)  # stale only when its process is gone, however old
    if not lock.tryLock(200):
        where = 'the menu bar' if sys.platform == 'darwin' else 'the tray'
        print(f'Peresmeshnik is already running: its icon is in {where}', file=sys.stderr, flush=True)
        box = QMessageBox(QMessageBox.Information, 'Peresmeshnik', f'Peresmeshnik is already running: its icon is in {where}.')
        if sys.platform == 'darwin':
            macos.activate()  # an accessory app's box would open behind the other windows
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
    tray.hotkey_note('', why, announce=sys.platform.startswith(('linux', 'darwin')))  # Windows: no hotkey before stage 4
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

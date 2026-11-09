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
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from peresmeshnik import __version__, config, crash
from peresmeshnik.audio import c_locale
from peresmeshnik.audio.echo import EchoGate, Reference, Referenced, Threaded
from peresmeshnik.audio.resample import Resampler
from peresmeshnik.engines import SAMPLE_RATE, Caption, Latency, Speech

log = logging.getLogger('peresmeshnik.app')
ARROW = {'them': 'them → me', 'me': 'me → them'}
RESTARTS = 3  # a side that fails this often within a minute stays off, with the reason
MAX_CLONE_RTF = 1.0  # seconds of synthesis per second of speech, warm: above, the clone falls behind for good (me -> them
#                      has no lag limit) and the stock voice speaks; the RTX 3060 runs it at 0.4-0.5 (BENCH), a GTX 10xx
#                      is not measured
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
    the current default. macOS: PortAudio opens only the devices of its last refresh. `on_back()` (the controller
    refreshes PortAudio; it returns a future) is asked once when the saved device is there but unknown to PortAudio,
    and when the saved device comes back while another one plays (coreaudio.Arrival): after that refresh the output
    moves to it, with the speech not played yet. A device PortAudio still does not see after its refresh counts as
    gone: another one plays meanwhile."""
    retry_s = 2.0

    def __init__(self, sink: str = '', on_state=lambda text: None, on_back=None):
        self.sink, self.on_state, self.on_back = sink, on_state, on_back
        self._out, self._rs, self._retry_at, self._failing = None, {}, 0.0, False
        self._arrival = None  # macOS: waiting for the saved device while another one plays
        self._asked: set[str] = set()  # macOS: devices a refresh was asked for (once each)
        self._lock = threading.Lock()  # capture (live), events (speech) and control (switch, stop) threads

    def resolve(self, exclude: str = '') -> str:
        if sys.platform == 'win32':
            from peresmeshnik.audio import winaudio
            names = winaudio.output_names()
            crash.private(*names)
            is_cable = lambda n: 'CABLE' in n.upper()  # noqa: E731  VB-Cable is the call's mic side, not headphones
            if self.sink and self.sink != exclude and self.sink in names:
                name = self.sink
            else:
                name = winaudio.default_output()
                if not name or name == exclude or is_cable(name):
                    name = next((n for n in names if n != exclude and not is_cable(n)), '')
                if not name:
                    raise RuntimeError('no output device')
            if is_cable(name):
                raise RuntimeError(f'the output is {name}, a virtual cable (the call app\'s microphone): choose your '
                                   'headphones in the settings or the system')
            return name
        if sys.platform == 'darwin':
            from peresmeshnik.audio import capture_macos, coreaudio
            outs = [d for d in coreaudio.devices() if d['outputs']]
            if self.sink and self.sink != exclude and any(d['name'] == self.sink for d in outs):
                name = self.sink
            else:
                name = coreaudio.default()
                if name == exclude:  # it is there, PortAudio does not see it: any other real output meanwhile
                    from peresmeshnik.audio.out import portaudio_names
                    known = portaudio_names()
                    name = next((d['name'] for d in outs if d['name'] in known and d['name'] != exclude and not d['virtual']), '')
                if not name:
                    raise RuntimeError('no output device')
                if name.startswith('BlackHole'):  # 2ch is the call's microphone, 16ch the call's speaker (< 14.4)
                    raise RuntimeError(f'the default output is {name}, a device of the call: choose your headphones in the '
                                       'settings or the system')
            if name == capture_macos.CALL_DEVICE and capture_macos.path() == 'blackhole':  # the call would hear itself
                raise RuntimeError(f'the headphones are {name}, the call app\'s speaker: choose your real headphones in the '
                                   'settings or the system')
            crash.private(name)
            return name
        names = [line.split('\t')[1] for line in pactl('list', 'short', 'sinks').splitlines() if '\t' in line]
        crash.private(*names)
        if self.sink in names:
            return self.sink
        return not_ours(pactl('get-default-sink').strip(), 'default output')

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            out = self._out
            if out is not None and not out.alive:  # died just now (its on_end may come before or after this)
                self._drop()
                out = None
            if out is None:
                if time.monotonic() < self._retry_at:
                    return  # no device for now: this audio is dropped
                try:
                    sink, out = self._open(sample_rate)
                    self._out = out
                    if self._failing:  # the tray still shows the error: put it back
                        self.on_state(f'translating (the headphones are back: {sink})')
                    self._rs, self._failing = {}, False
                except Exception as e:
                    self._retry_at = time.monotonic() + self.retry_s
                    self._fail(f'no headphones: {e}')
                    return
            rs = self._rs.get(sample_rate) or self._rs.setdefault(sample_rate, Resampler(sample_rate, out.sample_rate))
            out.play(rs(pcm), out.sample_rate)

    def _open(self, sample_rate: int):  # under the lock
        from peresmeshnik.audio.out import NotInPortAudio, Output
        sink = self.resolve()
        try:
            out = Output(sample_rate, sink, on_end=self._gone)
        except NotInPortAudio:
            if self.on_back and sink not in self._asked:  # a refresh makes PortAudio see it: the next chunk plays there
                self._asked.add(sink)
                log.info('headphones: %s is there, unknown to PortAudio: refreshing the devices', sink)
                self.on_back()
                raise
            # still unknown after its refresh (a Bluetooth profile settling), or nobody can refresh: another device
            # for now, back to this one when it changes again
            log.warning('headphones: PortAudio does not see %s: another output meanwhile', sink)
            sink = self.resolve(exclude=sink)
            out = Output(sample_rate, sink, on_end=self._gone)
        if sys.platform == 'darwin' and self.on_back and self.sink and sink != self.sink:
            self._wait_for_saved()
        return sink, out

    def _wait_for_saved(self) -> None:  # under the lock
        if self._arrival is None:
            from peresmeshnik.audio.coreaudio import Arrival
            self._arrival = Arrival(self.sink, lambda: threading.Thread(target=self._back, daemon=True).start())

    def _back(self) -> None:
        """macOS: the saved device is back while another one plays: PortAudio is refreshed, then the output moves."""
        with self._lock:
            arrival, self._arrival = self._arrival, None
            self._asked.discard(self.sink)
        if arrival:
            arrival.close()
        done = self.on_back()
        try:
            if done is not None:
                done.result(15)
        except Exception as e:
            log.warning('headphones: refreshing the devices: %s', e)
        from peresmeshnik.audio.out import portaudio_names
        with self._lock:
            out = self._out
            if out is None or not out.alive or out.device == self.sink:
                return
            if self.sink not in portaudio_names():  # PortAudio still does not see it: stay, wait for its next change
                log.info('headphones: %s is not there for PortAudio yet', self.sink)
                self._wait_for_saved()
                return
            self._out = None  # the next chunk opens the saved device
            pending = [(c, out.rate) for c in out.take()]
        threading.Thread(target=out.stop, kwargs={'drain': False}, daemon=True).start()
        for pcm, sr in pending:
            self.play(pcm, sr)
        log.info('headphones: back on %s', self.sink)

    def _gone(self, why: str) -> None:
        """The output's on_end, on its own thread: the tray hears it now; the next chunk reopens."""
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
    crash.private(*names)
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


def clone_rtf(tts, lang: str, voice) -> float:
    """The clone's real-time factor, warm (after warm_up): one ~4 s test phrase, synthesis time over its length."""
    from peresmeshnik.voice import TEST_PHRASES
    lang = lang if lang in TEST_PHRASES else 'en'
    t0, n = time.perf_counter(), 0
    for chunk in tts.synth(TEST_PHRASES[lang][0], lang, voice):
        n += len(chunk) if chunk.any() else 0  # not the pause after the sentence (xtts.shape): it costs nothing
    return (time.perf_counter() - t0) / max(n / tts.sample_rate, 1e-3)


def spoke(e: BaseException | None) -> bool:
    """The error came from the pipeline's speak worker (the TTS: the clone), not from ASR or MT."""
    return e is not None and any(f.name == '_speak' for f in traceback.extract_tb(e.__traceback__))


class Side:
    """One direction with its input, output and subtitles. `name` is 'them' (them -> me) or 'me'."""
    RETRY_S = 2.0  # the live voice's output failed: tried again at most this often (on the frames that come anyway)
    on_failure = None  # (why, exception): the controller restarts the side; None: the tray hears it
    _stopping = False

    def __init__(self, name: str, s: config.Settings, inp, out, overlay, on_error=lambda text: None, echo=None):
        """`echo`: an audio.echo.EchoGate on the input (our own speech coming back from the other side is not translated)."""
        self.name, self.inp, self.out, self.overlay, self.on_error = name, inp, out, overlay, on_error
        self.echo, self._gate = echo, None
        self.src, self.tgt = (s.their_lang, s.my_lang) if name == 'them' else (s.my_lang, s.their_lang)
        self.engine = s.them_to_me if name == 'them' else s.me_to_them
        self.live, self.settings = False, s
        self._lock = threading.Lock()  # stop() vs a fallback voice being made mid-call
        self.on_talk = lambda: None  # 'me': I am talking (a partial caption of BARGE_IN_WORDS): barge-in on the other side
        self.directions = []  # (direction, takes its captions, takes its speech)
        self._threads: list[threading.Thread] = []
        self._live_q: queue.Queue = queue.Queue()  # the live pass-through, played on its own thread
        if self.engine == 'local':
            self.directions.append((self._local(s), True, True))
        else:  # Gemini speaks; the local path makes the subtitles and speaks when the cloud fails (cloud/fallback.py)
            self.directions.append((self._cloud(s), True, True))
        d, captions, speech = self.directions[0]
        if name == 'me' and not getattr(d, 'supports_live', False):  # "translation <-> live voice" crossfaded (S2)
            from peresmeshnik.audio.mic import LiveSwitch
            self.directions[0] = (LiveSwitch(d), captions, speech)

    def _local(self, s: config.Settings):
        from peresmeshnik import languages
        from peresmeshnik.pipeline import make_local
        pair = languages.direction(s, 'them_to_me' if self.name == 'them' else 'me_to_them')
        if not pair.local:  # the wizard and the tray say so before; settings edited by hand land here
            raise RuntimeError(f'{"; ".join(pair.why)}: switch "{ARROW[self.name].capitalize()}" to cloud')
        kw = {'asr': pair.asr, 'mt': pair.mt, 'settings': s}  # languages.direction's ids: its '*' and NVIDIA GPU rules
        if self.name == 'them':
            from peresmeshnik.pipeline import MAX_LAG_S
            d = make_local(self.src, self.tgt, tts=pair.tts, max_lag_s=MAX_LAG_S, **kw)  # the queue policy: subtitles keep the rest
        elif pair.voice == 'stock':  # a stock voice: chosen where the clone falls behind (an 8 GB Mac, BENCH), or the
            d = make_local(self.src, self.tgt, tts=pair.tts, **kw)  # clone cannot speak the language
        else:
            if not s.local.voice:
                raise RuntimeError('me -> them local needs your voice: enroll it first (python -m peresmeshnik.voice enroll), '
                                   'or switch "Me -> them" to cloud')
            from peresmeshnik import models, voice
            from peresmeshnik.local.clone import make_clone
            model_dir, v = models.path(s.local.clone), voice.load(s.local.voice)  # missing: said as usual, no fallback
            try:
                tts = make_clone(model_dir, threads=s.local.threads)
            except RuntimeError as e:  # no voice pack, no CUDA, its runner did not start
                return self._fallback(s, e)
            d = make_local(self.src, self.tgt, tts=tts, voice=v, **kw)
            try:
                d.warm_up()
                rtf = clone_rtf(tts, self.tgt, v)
            except RuntimeError as e:  # out of VRAM on the first phrase, the runner died
                d.close()
                return self._fallback(s, e)
            if rtf > MAX_CLONE_RTF:
                d.close()
                return self._fallback(s, f'the GPU is too slow for the clone: {rtf:.1f} s of synthesis per second of speech')
            return d
        return self._stock(d, s)

    def _stock(self, d, s: config.Settings):
        """A side that speaks with Piper, warmed up. The Piper runner dies: only it starts again."""
        if (tts := getattr(d, 'tts', None)) is not None:
            from peresmeshnik import languages
            from peresmeshnik.local.tts import PiperTts
            vid = languages.pick(s.local.tts, self.tgt)  # the language's stock voice (its '*' too)
            d.tts = crash.Respawn(lambda: PiperTts(vid, threads=s.local.threads), self._tts_respawned, tts)
        d.warm_up()
        return d

    def _fallback(self, s: config.Settings, why):
        """The clone cannot speak: the stock voice (voice.STOCK), still local; said once. Not the cloud, even with a
        key: local was chosen, so the user's voice does not leave the machine unasked."""
        from peresmeshnik.cloud import api_key
        from peresmeshnik.pipeline import make_local
        d = self._stock(make_local(self.src, self.tgt, settings=s), s)
        try:
            cloud = bool(api_key(s.cloud))
        except SystemExit:  # no key
            cloud = False
        self.on_error(f'error: {self.name}: the voice clone is off ({why}); a stock voice speaks instead'
                      + ('. With your Gemini key you can switch "Me → them" to cloud in the settings' if cloud else ''))
        return d

    def _clone_died(self, dead):
        """Mid-call the clone's worker died (its runner crashed, out of VRAM): the rest of the call in the fallback voice.
        None: translation went off meanwhile; no fallback voice raises (the side restarts or stops, and says why)."""
        dead.close()
        try:
            d = self._fallback(self.settings, dead._error)
        except Exception as e:
            raise RuntimeError(f'the voice clone stopped ({dead._error}) and the stock voice does not start: {e}') from e
        if not getattr(d, 'supports_live', False):
            from peresmeshnik.audio.mic import LiveSwitch
            d = LiveSwitch(d)
        with self._lock:
            if self._stopping:  # translation went off meanwhile
                d.close()
                return None
            self.directions[0] = (d, True, True)
        d.set_live(self.live)
        return d

    def _cloud(self, s: config.Settings):
        from peresmeshnik.cloud import api_key, fallback
        try:
            key = api_key(s.cloud)
        except SystemExit as e:  # the CLI's way to say "no key"
            raise RuntimeError(str(e)) from None
        return fallback.make(self.name, s, key, on_state=self.on_error, on_respawn=self._tts_respawned)

    def _tts_respawned(self, e: BaseException) -> None:  # the speak worker's thread: the voice goes on, the side too
        crash.report(f'{ARROW[self.name]}: the voice helper died', e, quiet=True)
        self.on_error(f'note: {ARROW[self.name]}: the voice helper stopped ({e}) and was started again; a report '
                      'without your data is in the tray menu (Crash report…)')

    def start(self) -> None:
        for target, args in [(self._consume, d) for d in self.directions] + [(self._pass_through, ())]:
            t = threading.Thread(target=target, args=args, name=f'{self.name}-{target.__name__[1:]}', daemon=True)
            t.start()
            self._threads.append(t)
        if self.echo is not None:  # on its own thread: _on_frames runs in the capture's callback
            self._gate = Threaded(self.echo, self._on_frames)
        self.inp.start(self._gate or self._on_frames)

    def _on_frames(self, pcm: np.ndarray) -> None:  # capture thread or PortAudio callback: only hand on
        if self.live and not self._supports_live:
            self._live_q.put(pcm)  # ponytail: a hard switch, no crossfade; S2's set_live crossfades
            return
        for d, _, _ in self.directions[:1] if self.live else self.directions:  # live: only the switch, nothing translates
            d.feed(pcm)

    def _pass_through(self) -> None:
        failed = 0.0  # when the output last failed (0: it works); the tray hears it once per outage
        while (pcm := self._live_q.get()) is not None:
            if not self.live:  # frames queued just before switching back are dropped
                continue
            now = time.monotonic()
            if failed and now - failed < self.RETRY_S:  # still out: the frame goes, the queue does not grow
                continue
            try:
                self.out.play(pcm, SAMPLE_RATE)  # after a failure: a new stream, so a device that is back plays again
            except Exception as e:  # BlackHole 2ch gone (VirtualMic says why): the tray, not only a traceback
                if not failed:
                    log.exception('%s: live voice failed', self.name)
                    self.on_error(f'error: {self.name}: {e}')
                failed = now
                continue
            if failed and now - failed > 2 * self.RETRY_S:  # it has played for a while: the outage is over
                log.info('%s: the live voice plays again', self.name)
                failed = 0.0

    @property
    def _supports_live(self) -> bool:
        return getattr(self.directions[0][0], 'supports_live', False)

    def _consume(self, d, captions: bool, speech: bool) -> None:
        try:
            for e in d.events():
                if isinstance(e, Caption) and captions:
                    crash.said(e.original, e.translation, final=e.final)  # kept out of any crash report
                    self.overlay.show(e)
                    if not e.final and len(e.original.split()) >= BARGE_IN_WORDS:
                        self.on_talk()
                elif isinstance(e, Speech) and speech and (not self.live or self._supports_live):
                    self.out.play(e.pcm, e.sample_rate)
                elif isinstance(e, Latency) and captions and hasattr(self.overlay, 'latency'):
                    self.overlay.latency(e)
            if self._stopping:
                return
            inner = getattr(d, 'inner', d)  # LiveSwitch's
            if self.name == 'me' and self.engine == 'local' and getattr(inner, 'voice', None) is not None \
                    and spoke(getattr(inner, '_error', None)):  # the clone died: the rest of the call in a stock voice
                if (d := self._clone_died(inner)) is not None:
                    self._consume(d, True, True)
                return
            if hasattr(inner, 'finish'):
                inner.finish()  # raises with the error of a worker that died (the Piper runner gone, a bug)
            raise RuntimeError('its events ended by themselves')
        except Exception as e:  # a dead direction must show in the tray, not only in the log
            if self._stopping:
                return
            log.exception('%s: direction failed', self.name)
            if self.on_failure:
                self.on_failure(f'the translation failed ({type(e).__name__}: {e})', e)
            else:
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
        with self._lock:
            self._stopping = True
        self.inp.stop()
        if self._gate:
            self._gate.close()
        self._live_q.put(None)
        for d, _, _ in self.directions:
            d.close()
        for t in self._threads:
            t.join(15)
        self.out.stop()


class CallWatch:
    """Windows, translation on: does the call app record CABLE Output (the helper's `watch`: "call" / "idle")? If not
    while the user talks, the tray says once which microphone to choose there: else the other side hears nothing."""
    HINT_S = 10.0  # nobody recorded the cable this long, and the user talks
    HINT = ('note: the call app does not record CABLE Output: choose "CABLE Output (VB-Audio Virtual Cable)" as the '
            'microphone in the call app, else the other side does not hear the translation')

    def __init__(self, uid: str, on_state, spawn):
        self.on_state, self.reading, self.since, self.hinted = on_state, True, time.monotonic(), False  # until it says
        self.p = spawn('watch', '--watch', uid, stdout=subprocess.DEVNULL)
        threading.Thread(target=self._read, name='call-watch', daemon=True).start()

    def _read(self) -> None:
        for line in self.p.stderr:
            word = line.decode(errors='replace').strip()
            if word in ('call', 'idle'):
                self.reading, self.since = word == 'call', time.monotonic()
            elif word:
                log.info('call watch: %s', word)

    def talked(self) -> None:  # the me side's thread: a partial caption of the user's speech
        if not self.reading and not self.hinted and time.monotonic() - self.since >= self.HINT_S:
            self.hinted = True
            self.on_state(self.HINT)

    def stop(self) -> None:
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()


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
        self._vmic = self._call = self._router = self._loop = self._call_watch = None
        self._through = self._through_arrival = None  # macOS < 14.4, translation off: the call app's speaker on to the headphones
        self._mic = ''  # the default input as it was before our devices existed
        self._restarts: dict[str, list[float]] = {}  # side -> times of its restarts in the last minute
        self.call_echo = Reference()  # what we said into the call: its echo from a far end on a loudspeaker is not ours to translate
        self.mic_echo = Reference()  # what the headphones (or speakers) played: for the microphone's echo gate
        self._fonts_said = ()

    def submit(self, fn, *args):
        """None once quit() shut the control thread down (a Core Audio notification can still come then)."""
        try:
            return self._pool.submit(self._guard, fn, *args)
        except RuntimeError:  # cannot schedule new futures after shutdown
            log.info('quitting: %s not run', getattr(fn, '__name__', fn))
            return None

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
        elif self._vmic is None and sys.platform == 'darwin':  # BlackHole 2ch, the user's (BlackHoleMissing says how)
            from peresmeshnik.audio.virtual_mic_macos import VirtualMic
            self._vmic = VirtualMic()
        elif self._vmic is None and sys.platform == 'win32':  # VB-Cable, the user's (CableMissing says how)
            from peresmeshnik.audio.virtual_mic_windows import VirtualMic
            self._vmic = VirtualMic()
        return self._vmic

    def set_translation(self, on: bool) -> concurrent.futures.Future:
        return self.submit(self._on if on else self._off)

    def startup(self) -> None:
        """The devices are there from the start, so the call app can pick the microphone and S1 can move the
        call's sound once; until translation is on, the real microphone is looped into the virtual one."""
        if self.devices and sys.platform == 'darwin':
            self._startup_call_macos()
            self._startup_vmic()
        if self.devices and sys.platform == 'win32':
            self._startup_call_windows()
            self._startup_vmic()
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

    def _startup_call_macos(self) -> None:
        """macOS, the call's sound (S1): 14.4+ asks the tap helper about the permission now, not at the first call;
        older: the call app's speaker (BlackHole 16ch) is passed on to the headphones while translation is off."""
        from peresmeshnik.audio import capture_macos
        if capture_macos.path() != 'tap':
            self._through_start()
            return
        try:
            code, why = capture_macos.check()
        except Exception as e:
            code, why = 1, str(e)
        if code:
            self.startup_error = '; '.join(x for x in (self.startup_error, why) if x)

    def _startup_vmic(self) -> None:
        """macOS, Windows: BlackHole 2ch / VB-Cable is the user's (nothing to make); the loop starts. Missing: the tray
        says how to install."""
        try:
            self.virtual_mic()
        except Exception as e:
            log.warning('virtual mic: %s', e)
            self.startup_error = '; '.join(x for x in (self.startup_error, str(e)) if x)  # next to S1's, not over it
            return
        self._loopback(True)

    def _startup_call_windows(self) -> None:
        """Windows, the call's sound (S1): a previous run that crashed while muting the call app left its session
        volume at 0 -> put it back now; then check the loopback helper so a missing one is said at start, not at the
        first call."""
        from peresmeshnik.audio import capture_windows
        try:
            capture_windows.restore_call()
            code, why = capture_windows.check()
        except Exception as e:
            code, why = 1, str(e)
        if code:
            self.startup_error = '; '.join(x for x in (self.startup_error, why) if x)

    def _mic_source(self) -> str:
        return mic_source((self.settings or config.load()).ui.mic, self._mic)

    def _loopback(self, on: bool) -> None:
        """Translation off: PipeWire loops the real microphone into "Peresmeshnik Mic" (none of our CPU), so the
        call hears the user as is; translation on: our me -> them direction speaks there instead."""
        if not self.devices or self._vmic is None:  # only the app's own devices (startup), not files or tests
            return
        if sys.platform in ('darwin', 'win32'):
            return self._loop_helper(on)
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

    def _loop_helper(self, on: bool) -> None:
        """macOS, Windows: the helper's loop "microphone -> BlackHole 2ch / CABLE Input" (none of our CPU). It opens the
        microphone only while a call app records the virtual microphone (macOS 14+; Windows: an active audio session on
        CABLE Output), so no orange dot or microphone icon and no AirPods / headset call profile outside a call."""
        from peresmeshnik.audio.mic import native
        helper = native()
        loop = self._loop
        if on and loop and loop.poll() is None:
            return
        self._loop = None  # before it ends: its watcher then knows it was stopped
        if loop:
            loop.stdin.close()  # it closes the microphone and exits
            try:
                loop.wait(5)
            except subprocess.TimeoutExpired:
                loop.kill()
                loop.wait()
        if not on:
            return
        why = 'error: the call will not hear your microphone while translation is off: '
        try:
            mic = helper.not_virtual((self.settings or config.load()).ui.mic)
            watch = ['--watch', self._vmic.call_mic['uid']] if sys.platform == 'win32' else []  # macOS: the same device
            self._loop = p = helper.spawn('loop', '--from', mic, '--to', self._vmic.device['uid'], *watch, stdout=subprocess.DEVNULL)
        except Exception as e:
            log.warning('microphone loop: %s', e)
            self.on_state(why + str(e))
            return

        ready = threading.Event()

        def watch():
            for line in p.stderr:
                ready.set()
                log.info('microphone loop: %s', line.decode(errors='replace').strip())
            ready.set()
            if p.wait() and p is self._loop:  # ended by itself: started again as a side is, at most RESTARTS a minute
                now = time.monotonic()
                times = self._restarts['loop'] = [x for x in self._restarts.get('loop', ()) if now - x < 60]
                if p.returncode != 6 and len(times) < RESTARTS:
                    times.append(now)
                    log.warning('microphone loop ended (%s): starting it again', p.returncode)
                    self.submit(lambda: self._loop is p and self._loop_helper(True))
                else:
                    self.on_state(why + (helper.NO_PERMISSION if p.returncode == 6 else f'the loop ended ({p.returncode})'))
        threading.Thread(target=watch, name='mic-loop', daemon=True).start()
        ready.wait(3)  # its first word ("idle": set up, waiting for a call): its start belongs to ours, not to the idle

    def _watch_call(self, on: bool) -> None:
        """Windows: CallWatch while me -> them translates into CABLE Input (only the app's own devices, not tests)."""
        if self._call_watch:
            self._call_watch.stop()
            self._call_watch = None
        if on and self.devices and sys.platform == 'win32' and self._vmic is not None:
            from peresmeshnik.audio import windows
            try:
                self._call_watch = CallWatch(self._vmic.call_mic['uid'], self.on_state, windows.spawn)
            except Exception as e:  # no hint, nothing else lost
                log.warning('call watch: %s', e)

    def _on(self) -> None:
        if self.on:
            return
        self.on_state('warming up')
        self._refresh_devices()  # macOS: devices plugged in since (AirPods back) become visible to PortAudio
        s = self.settings or config.load()
        self._pair_to_overlay(s)
        files = {'them': self.them_file, 'me': self.me_file}
        from peresmeshnik.voice import STOCK
        t, failed, call = time.perf_counter(), [], ''
        for name in ('them', 'me'):  # a side that cannot start (no key, no voice pack, no device) does not stop the other
            if not (self.devices or files[name]):
                continue
            try:
                side = self._side(name, s)
            except Exception as e:
                log.exception('%s', name)
                failed.append(f'{name}: {e}')
                continue
            self.sides.append(side)
            if name == 'me':
                self._loopback(False)  # the call hears the user until the translation is ready
                self._watch_call(True)
            else:
                self._through_stop()  # BlackHole 16ch goes to the translation now
            side.set_live(self.live)
            side.start()  # them -> me runs while the clone loads (its runner: up to ~30 s)
            if name == 'them':
                call = self._route(s) if self.devices else ''
                if s.me_to_them == 'local' and s.local.clone != STOCK and (self.devices or files['me']):
                    self.on_state('warming up: them → me translates, your voice is loading')  # not ready yet
        self.on = bool(self.sides)
        for f in failed:
            self.on_state(f'error: {f}')
        if self.sides:
            self.on_state(f'translating ({", ".join(f"{x.name}: {x.engine}" for x in self.sides)}; {call}'
                          f'ready in {time.perf_counter() - t:.1f} s)')
        if call.startswith('call app: none'):
            self.on_state('error: choose the call app in the tray menu (Call app): until then the call is not translated')
        self._stock_note(s)
        self._fonts(s)

    def _fonts(self, s: config.Settings) -> None:
        """No font for a language of the pair (Linux without CJK / Arabic fonts): said once per pair. Only with the app's
        GUI running (a headless controller may have no Qt libraries at all); never in the way of translating."""
        gui = sys.modules.get('PySide6.QtGui')
        if self.test or not self.sides or self._fonts_said == (s.my_lang, s.their_lang) or not (
                gui and gui.QGuiApplication.instance()):
            return
        self._fonts_said = s.my_lang, s.their_lang
        try:
            from peresmeshnik.ui.overlay import font_note
            if note := font_note([s.my_lang, s.their_lang]):
                self.on_state(f'note: {note}')
        except Exception:
            log.exception('fonts')

    def _side(self, name: str, s: config.Settings) -> Side:
        inp = FileIn(f) if (f := {'them': self.them_file, 'me': self.me_file}[name]) else self._input(name, s)
        # our speech coming back: 'them' hears what we said into the call, 'me' what the headphones played (no echo
        # path from a file, nor in the wizard's test, where both sides speak into the headphones)
        gate = None if f or self.test else EchoGate(self.call_echo if name == 'them' else self.mic_echo,
                                                    max_delay_s=s.local.echo_max_delay_s,
                                                    **({'rho_on': s.local.mic_echo_rho_on} if name == 'me' else {}))
        side = Side(name, s, inp, self._output(name, s), self.overlay.side(name), self.on_state, echo=gate)
        side.on_failure = lambda why, e: self.submit(self._restart, side, why, e)
        if name == 'me':  # I talk over their voice: barge-in on the them -> me side of now; does the call hear me?
            side.on_talk = lambda: ([x.interrupt() for x in self.sides if x.name == 'them'],
                                    self._call_watch and self._call_watch.talked())
        return side

    def failed(self, inp, why: str, restart: bool = True) -> None:
        """An input ended under its side (the call sound helper gone; any thread): that side restarts; a late end of an
        input whose side was restarted since does nothing. `restart` False (no helper died: the call app is not
        running): the route is undone, so the call is heard as usual, and the tray says why."""
        def run():
            side = next((x for x in self.sides if x.inp is inp), None)
            if side is None:
                return
            if restart:
                self._restart(side, why, None)
            else:
                self._unroute()
                self.on_state(f'error: {why}')
        self.submit(run)

    def _restart(self, side: Side, why: str, e: BaseException | None) -> None:
        """The control thread: a side whose direction died or whose input ended starts again, with one message and a
        report without the user's data; after RESTARTS within a minute it stays off and says why."""
        if side not in self.sides or not self.on:
            return
        crash.report(f'{ARROW[side.name]}: {why}', e, self.settings, quiet=True)  # the message below names it
        self._drop(side)
        now = time.monotonic()
        times = self._restarts[side.name] = [t for t in self._restarts.get(side.name, ()) if now - t < 60]
        if len(times) >= RESTARTS:
            self.on_state(f'error: {ARROW[side.name]} stopped: it failed {RESTARTS + 1} times within a minute ({why}). '
                          'A report without your data is in the tray menu (Crash report…)')
            if not self.sides:
                self._off()
            elif side.name == 'me':
                self._loopback(True)  # the call hears the user as is again
            return
        times.append(now)
        try:
            self._make(side.name, self.settings or config.load())
        except Exception as ex:
            log.exception('%s', side.name)
            self.on_state(f'error: {ARROW[side.name]} stopped: {why}; it cannot start again: {ex}')
            if not self.sides:
                self._off()
            elif side.name == 'me':
                self._loopback(True)  # the call hears the user as is again
            return
        self.on_state(f'note: {ARROW[side.name]} restarted after an error ({why}); a report without your data is in '
                      'the tray menu (Crash report…)')

    def _drop(self, side: Side) -> None:
        """A side goes (to be made again): meanwhile the call is heard as it is and hears me as I am, as with translation
        off — them: the route undone (Windows: un-muted), macOS < 14.4 the pass-through; me: the microphone loop."""
        self.sides.remove(side)
        if side.name == 'them':  # before its input stops (the route holds it); the new side routes it again
            self._unroute()
        else:
            self._loopback(True)
            self._watch_call(False)  # Windows: no translation into CABLE Input meanwhile
        try:
            side.stop()
        except Exception:
            log.exception('stopping %s', side.name)
        if side.name == 'them':
            self._through_start()  # after its input let BlackHole 16ch go

    def _make(self, name: str, s: config.Settings) -> Side:
        new = self._side(name, s)  # raises: the caller says why (the loop / the pass-through then stay)
        new.set_live(self.live)
        try:
            new.start()
        except Exception:
            try:
                new.stop()
            except Exception:
                log.exception('stopping %s', name)
            raise
        if name == 'me':
            self._loopback(False)  # the call heard me through the loop until the translation was ready (as in _on)
            self._watch_call(True)
        else:
            self._through_stop()  # BlackHole 16ch goes to the translation now
        self.sides.append(new)
        if new.name == 'them' and self.devices:  # the call app's sound to the new input (a tap, a loopback, the sink)
            self._route(s)
        return new

    def set_languages(self) -> concurrent.futures.Future | None:
        """The pair changed (tray): while translating, each side is made again for it, as after an error (`_drop`,
        `_make`: meanwhile the call and I hear each other as with translation off); off, the next start takes it."""
        return self.submit(self._relanguage)

    def _pair_to_overlay(self, s: config.Settings, side: str | None = None) -> None:
        if hasattr(self.overlay, 'set_pair'):  # the rows' fonts (Han in Chinese, Japanese, Korean forms)
            self.overlay.set_pair(s.my_lang, s.their_lang, side)

    def _relanguage(self) -> None:
        s = self.settings or config.load()
        if not self.on:
            self.on_state(f'idle, translation off (languages {s.my_lang} ⇄ {s.their_lang})')
            return
        files = {'them': self.them_file, 'me': self.me_file}
        for name in ('them', 'me'):
            if not (self.devices or files[name]):
                continue
            if old := next((x for x in self.sides if x.name == name), None):
                self._drop(old)
            try:
                self._make(name, s)
            except Exception as e:
                log.exception('%s', name)
                self.on_state(f'error: {ARROW[name]} cannot start for {s.my_lang} ⇄ {s.their_lang}: {e}')
            self._pair_to_overlay(s, name)  # its rows in the new languages now (the other side's still in the old)
        if not self.sides:
            self._off()
            return
        self.on_state(f'translating ({", ".join(f"{x.name}: {x.engine}" for x in self.sides)}; '
                      f'languages {s.my_lang} ⇄ {s.their_lang})')
        self._fonts(s)  # a stock voice instead of the clone: the tray said it with the pair (switch_pair)

    def _stock_note(self, s: config.Settings) -> None:
        """Me -> them local speaks in a stock voice though the clone is chosen (it cannot speak the language, or needs an
        NVIDIA GPU here): said at each start (a new pair: by the tray)."""
        from peresmeshnik import languages
        from peresmeshnik.voice import STOCK
        if s.local.clone != STOCK and any(x.name == 'me' and x.engine == 'local' for x in self.sides) and (
                d := languages.direction(s, 'me_to_them')).voice == 'stock':
            self.on_state(f'note: {ARROW["me"]}: {"; ".join(d.notes)}')

    def _route(self, s: config.Settings) -> str:
        """The call app's sound into the call sink (Linux) or a tap of it (macOS 14.4+) while translating: the app
        chosen, else the one known call app playing now (remembered; a browser only by hand: all of its tabs would
        move)."""
        from peresmeshnik.audio.capture_pipewire import CallRouter, guess_call_app
        try:
            tap = None  # a process tap (macOS) or loopback (Windows): retarget it to the app, no sink move
            mute = False  # Windows: the loopback does not mute, so lower the call app's session volume (variant A)
            if sys.platform == 'darwin':
                from peresmeshnik.audio import capture_macos
                tap = next(x for x in self.sides if x.name == 'them').inp
                if not isinstance(tap, capture_macos.ProcessTap):  # BlackHole 16ch: the call app's own speaker setting
                    return f'call sound: {tap.device} (the call app\'s speaker); '
                guess_call_app = capture_macos.guess_call_app
            elif sys.platform == 'win32':
                from peresmeshnik.audio import capture_windows
                tap = next(x for x in self.sides if x.name == 'them').inp
                guess_call_app = capture_windows.guess_call_app
                mute = True
            app = s.ui.call_app
            if not app and (app := guess_call_app()):
                self._save_call_app(app)
            crash.private(app)
            if not app:
                return 'call app: none; '
            if tap is not None:
                tap.retarget(app)
                if mute:  # the translation is heard, the original is not; restored on off and after a crash
                    from peresmeshnik.audio import capture_windows
                    capture_windows.mute_call(app)
                    self._router = SimpleNamespace(stop=lambda: (tap.retarget(''), capture_windows.restore_call()))
                else:
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
        if self.on and (self._call or sys.platform in ('darwin', 'win32')) and any(x.name == 'them' for x in self.sides):
            self._unroute()
            self.on_state(f'translating ({self._route(self.settings or config.load()).rstrip("; ")})')

    def _input(self, name: str, s: config.Settings | None = None):
        s = s or self.settings or config.load()
        if name == 'them' and sys.platform == 'win32':
            from peresmeshnik.audio import capture_windows
            code, why = capture_windows.check()
            if code:
                raise RuntimeError(why)
            # the loopback ended: its helper died -> the side restarts (un-muted meanwhile, _restart); the call app is
            # not running (not yet, or closed: "<app> is not running", capture_windows) -> no restart, only un-muted
            tap = capture_windows.ProcessLoopback(on_end=lambda why: self.failed(
                tap, f'the call sound capture stopped: {why}', restart=not why.endswith(' is not running')))
            return tap  # the app is chosen in _route, after the start
        if name == 'them' and sys.platform == 'darwin':
            from peresmeshnik.audio import capture_macos
            on_end = lambda why: self.failed(inp, f'the call sound capture stopped: {why}')  # noqa: E731
            if capture_macos.path() == 'blackhole':
                inp = capture_macos.BlackHoleIn(on_end=on_end)
                return inp
            code, why = capture_macos.check()
            if code:
                raise RuntimeError(why)
            inp = capture_macos.ProcessTap(on_end=on_end)  # the app is chosen in _route, after the start
            return inp
        if name == 'them':
            if self._call is None:
                raise RuntimeError(f'no call sink ({self.startup_error or "no sound server"})')
            from peresmeshnik.audio.capture_pipewire import PipeWireCapture
            inp = PipeWireCapture(self._call.name, make_sink=False,  # the app's sink, kept for its whole run
                                  on_end=lambda why: self.failed(inp, f'the call sound capture stopped: {why}'))
            return inp
        if sys.platform in ('darwin', 'win32'):  # the helper keeps to the saved microphone or the default one by itself
            from peresmeshnik.audio.mic import followed
            return followed(s.ui.mic, self.on_state)
        mic_source(s.ui.mic, self._mic)  # no microphone now: this side says why, the other one still starts
        return Microphone(s.ui.mic, self.on_state, self._mic)

    def _output(self, name: str, s: config.Settings):
        if name == 'them' or self.test:
            back = (lambda: self.submit(self._refresh_devices)) if self.devices and sys.platform in ('darwin', 'win32') else None
            out = Headphones(s.ui.headphones, self.on_state, on_back=back)
            out.resolve()  # a missing device or our own as the default: say so now, not at the first word
            return Referenced(out, self.mic_echo)
        return Referenced(self.virtual_mic(), self.call_echo)

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
        self._watch_call(False)
        self._loopback(True)
        self._through_start()
        if self.on or sides:
            self.on = False
            self.on_state('idle, translation off')

    def _refresh_devices(self) -> dict:
        """macOS: PortAudio opens only the devices of its last refresh; refreshing frees every PortAudio stream of the
        process, so coreaudio.refresh_portaudio closes and opens again every registered one (these sides', the
        pass-through's, the wizard's test phrase's), all at once; the engines and the process tap stay (BENCH: the
        gap). The control thread: at translation on and when the saved headphones are back. A stream that cannot open
        again is reported and its owner reopens elsewhere; translation stays on."""
        if not (self.devices and sys.platform in ('darwin', 'win32')):
            return {}
        from peresmeshnik.audio.coreaudio import refresh_portaudio
        r = refresh_portaudio()
        log.info('devices refreshed: %s', r)
        for why in r['failed']:
            self.on_state(f'error: after the device refresh: {why}')
        saved = (self.settings or config.load()).ui.headphones
        if sys.platform == 'darwin' and self._through and saved:
            from peresmeshnik.audio.out import portaudio_names
            if saved not in portaudio_names():  # still not there for PortAudio: the next change of it, not a loop
                self._through_wait(saved)
            elif self._through.output and self._through.output != saved:
                self._through_stop()  # the pass-through played elsewhere meanwhile: to the saved headphones now
                self._through_start()
        if self.on and r['streams']:
            self.on_state(f'translating (devices changed: reopened in {r["suspend"] + r["refresh"] + r["resume"]} ms)')
        return r

    def _through_start(self) -> None:
        """macOS < 14.4 with translation off (or them -> me being made again): the call app's speaker (BlackHole 16ch)
        on to the headphones while a call app plays into it (capture_macos.PassThrough), else the call is not heard. The
        saved headphones, else the default; back to the saved ones when they are back."""
        from peresmeshnik.audio import capture_macos
        if not (self.devices and sys.platform == 'darwin') or capture_macos.path() != 'blackhole' or self._through or any(
                x.name == 'them' for x in self.sides):  # them -> me reads BlackHole 16ch itself
            return
        try:
            saved = (self.settings or config.load()).ui.headphones
            pick = Headphones(saved).resolve
            from peresmeshnik.audio.out import portaudio_names
            if saved and (pick() != saved or saved not in portaudio_names()):  # gone: back to them when they are back
                self._through_wait(saved)
            self._through = through = capture_macos.PassThrough(
                pick, on_end=lambda why: self.submit(self._through_lost, through, why),
                on_unknown=lambda name: self.submit(self._refresh_devices))
            through.start()
        except Exception as e:
            self._through = None
            log.warning('pass-through: %s', e)
            self.on_state(f'error: the call is not heard while translation is off: {e}')

    def _through_wait(self, saved: str) -> None:
        """The pass-through plays elsewhere (or will): when the saved headphones change (come, a Bluetooth profile
        settles), PortAudio is refreshed and the pass-through moves to them (_refresh_devices)."""
        from peresmeshnik.audio.coreaudio import Arrival
        old, self._through_arrival = self._through_arrival, Arrival(saved, lambda: self.submit(self._refresh_devices))
        if old:
            old.close()

    def _through_lost(self, through, why: str) -> None:  # the control thread
        if through is not self._through:
            return
        log.warning('pass-through: %s', why)
        self._through_stop()
        if not self.on:
            self.on_state(f'error: the headphones went away ({why}); reopening')  # once: a failed reopen says its own
            self._through_start()
            if self._through:
                self.on_state('idle, translation off')

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
        self._through_stop()  # its Core Audio listeners go before the control thread does
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
    try:
        settings = config.load()
    except Exception as e:  # the app still starts; the settings window says it again
        log.warning('settings: %s', e)
        settings = None
    earlier = crash.install(settings, a.verbose)  # the log file, the hooks; hard crashes of the last run as reports
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
        report = Signal(str, bool)  # a crash report was written (any thread): its path, quiet

        @Slot()
        def wake(self):  # a queued call: the GUI loop runs once more
            pass
    bridge = Bridge()
    if sys.platform == 'win32':  # WM_CLOSE from outside (taskkill, an installer): quit as on SIGTERM
        from peresmeshnik.ui import windows
        windows.host().closing.connect(lambda: QTimer.singleShot(0, qapp.quit))
    overlay = SubtitleOverlay()
    ctl = Controller(overlay, bridge.state.emit, a.them_file, a.me_file)
    tray = Tray(ctl)
    bridge.state.connect(tray.set_state)
    bridge.report.connect(tray.report_ready)
    crash.on_report.append(lambda p, quiet: bridge.report.emit(str(p), quiet))
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
    tray.hotkey_note('', why, announce=True)
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
            for p in earlier:  # the last run crashed
                tray.report_ready(str(p), why='Peresmeshnik stopped unexpectedly last time')
    dispatcher.aboutToBlock.connect(announce)
    if a.on:
        QTimer.singleShot(0, lambda: tray.set_translation(True))
    qapp.aboutToQuit.connect(ctl.quit)
    qapp.aboutToQuit.connect(lambda: log.info('overlay: %s', overlay.stats()))
    qapp.aboutToQuit.connect(binding.close)
    qapp.aboutToQuit.connect(lock.unlock)
    qapp.aboutToQuit.connect(crash.uninstall)  # a clean exit: no fault file for the next start
    _quit_on_signals(qapp)
    return qapp.exec()


def _quit_on_signals(qapp) -> None:
    """SIGTERM, SIGHUP (the terminal closed), Ctrl+C and Ctrl+Break (Windows) quit cleanly: the devices are removed. Python runs a
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
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP', 'SIGBREAK'):
        if hasattr(signal, name):  # no SIGHUP on Windows, SIGBREAK only there
            # queued: the handler runs between any two bytecodes of the GUI thread, e.g. inside the hotkey's
            # X read, and a direct quit() (aboutToQuit: hotkey.close) would close its display under it
            signal.signal(getattr(signal, name), lambda *_: QTimer.singleShot(0, qapp.quit))

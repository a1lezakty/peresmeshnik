"""Local pipeline behind `Direction`: VAD -> streaming ASR -> clauses -> MT -> TTS.

Threads (ARCHITECTURE §2): feed() only enqueues; the "listen" worker runs VAD, ASR partials and
clause segmentation, the "translate" worker runs MT and puts out the caption, the "speak" worker
runs TTS, so captions never wait for speech (a clone TTS on CPU is slower than real time) and a
clause is spoken while the next one is recognised. Generic over `Tts` and `Voice`: the stock Piper
voice (them -> me) and the clone (me -> them) plug in the same way. On silence only the VAD runs.

A clause ends at a VAD pause (>= 300 ms), at sentence punctuation from the ASR with a word after
it, or at `max_clause_s` (then at a comma or between words).

The voice must not fall ever further behind on continuous speech (a translation is often longer than
its clause, and the speech plays back to back). When audio arrives in real time, the speak worker
knows when the clause's voice would start: from `FAST_AFTER_S` behind its clause end it speaks faster
(linearly up to `MAX_SPEED` at `max_lag_s`, where the TTS supports `speed`), and a clause whose voice
would start more than `max_lag_s` after its end is not said at all (its caption stays). `interrupt()`
is barge-in: the user talks over the voice, so what is not heard yet is dropped.

Per clause one JSON line goes to the
`peresmeshnik.latency` logger: time of each link and, when audio arrives in real time, the lag
from the end of the clause's speech to its caption, to its synthesized audio and to the estimated
start of its playback (the output plays speech back to back, so a clause can wait behind the last).

    peresmeshnik models fetch silero-vad parakeet-tdt-0.6b-v3 opus-mt-en-ru piper-ru_RU-dmitri-medium
    peresmeshnik run --in call.wav [--from en] [--to ru] [--out speech.wav] [--log latency.jsonl]
    peresmeshnik run --in call.wav --pipewire --play   # Linux: file -> "Peresmeshnik Call" sink -> monitor
    peresmeshnik run --in call.wav --blackhole --play "BlackHole 2ch"   # macOS: file -> BlackHole 16ch -> capture
    peresmeshnik run --in call.wav --tap --play        # macOS 14.4+: afplay plays the file, a process tap of it
"""
import argparse
import inspect
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict

import numpy as np  # BLAS threads: pinned in peresmeshnik/__init__.py, which runs first

from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech, Voice

log = logging.getLogger('peresmeshnik.latency')
OTHER = {'en': 'ru', 'ru': 'en'}
HELLO = {'en': 'Hello.', 'ru': 'Привет.'}
MAX_BEHIND_S = 0.15
MAX_LAG_S = 4.0  # them -> me: voice starting later than this after its clause ended is dropped (BENCH stage 2: queue policy)
FAST_AFTER_S = 0.5  # voice this far behind its clause end is sped up ...
MAX_SPEED = 1.4  # ... linearly up to this at MAX_LAG_S (4 s, 0.5 s, 1.4x: 1-3 % of the words dropped in conversation, BENCH)


class LocalDirection:
    """`asr.stream()` objects also take `feed(pcm, partial=False)` and expose `duration` (s) and
    `cut(force) -> str | None` (local/asr.py). A `tts.synth` with a `speed` keyword is sped up when
    behind. `max_lag_s` None: no queue policy, every clause is said however late — me -> them, where the other
side has no subtitles of what was dropped; them -> me takes MAX_LAG_S."""

    def __init__(self, vad, asr, mt, tts, src: str, tgt: str, voice: Voice | None = None, max_clause_s: float = 5.0,
                 max_lag_s: float | None = None):
        self.vad, self.asr, self.mt, self.tts, self.voice = vad, asr, mt, tts, voice
        self.src, self.tgt, self.max_clause_s, self.max_lag_s = src, tgt, max_clause_s, max_lag_s
        self._speeds = 'speed' in inspect.signature(tts.synth).parameters
        self._ttfa = 0.0  # the last TTS first-chunk time: a clause's voice starts at least that late
        self._cut = 0.0  # interrupt(): speech of clauses that ended before this is not said
        self.realtime = True  # audio arrives at its own pace, so lags to the clause end are real
        self.stats: dict[str, float] = defaultdict(float)  # busy seconds per link, counts; see summary()
        self._in: queue.Queue = queue.Queue()
        self._clauses: queue.Queue = queue.Queue()
        self._spoken: queue.Queue = queue.Queue()  # translated clauses waiting for TTS
        self._events: queue.Queue = queue.Queue()
        self._fed, self._fed_wall = 0, 0.0  # samples fed so far and when the last chunk arrived
        self._start = 0  # where the current clause starts, samples
        self._play_end = 0.0  # when the speech put out so far has played, if played back to back
        self._play_lock = threading.Lock()  # _play_end: the speech thread adds to it, interrupt() resets it
        self._stop = threading.Event()  # close(): drop what is queued instead of saying it
        self._error: BaseException | None = None  # the first worker that died
        self._threads = [threading.Thread(target=self._run, args=(work, out), name=work.__name__[1:], daemon=True)
                         for work, out in ((self._listen, self._clauses), (self._translate, self._spoken), (self._speak, self._events))]
        for t in self._threads:
            t.start()

    # Direction
    def feed(self, pcm: np.ndarray) -> None:
        if self._error is None:  # a dead pipeline drops audio; finish() reports why
            self._in.put((time.perf_counter(), pcm))

    def events(self):
        while (e := self._events.get()) is not None:
            yield e

    def interrupt(self) -> None:
        """Barge-in (the user talks over the voice): speech not heard yet is dropped, captions stay; the app
        flushes its output too. Clauses that end from now on are spoken as usual."""
        with self._play_lock:  # not lost under the speech thread's update of _play_end
            self._cut = self._play_end = time.perf_counter()

    def warm_up(self) -> None:
        """Run every engine once before audio comes (first calls allocate); the app shows "warming up"."""
        s = self.asr.stream(self.src)
        s.feed(np.zeros(SAMPLE_RATE, np.float32))
        s.finish()
        for _ in self.tts.synth(self.mt.translate(HELLO[self.src], self.src, self.tgt), self.tgt, self.voice):
            pass

    def finish(self) -> None:
        """End of input: recognise, translate and speak what is left, then stop the workers.
        Raises if a worker died (its captions and speech are lost)."""
        self._join()
        if self._error is not None:
            raise RuntimeError(f'pipeline worker died: {self._error!r}') from self._error

    def close(self) -> None:
        """Translation off: drop queued clauses and speech (nothing more is said), stop the workers,
        free the engines (the TTS process too)."""
        self._stop.set()
        self._join()
        for engine in (self.tts, self.mt, self.asr, self.vad):
            if hasattr(engine, 'close'):
                engine.close()

    def _join(self) -> None:
        self._in.put(None)
        for t in self._threads:
            t.join()

    # workers
    def _run(self, work, out: queue.Queue) -> None:
        """A worker that dies records why and still ends its output stream, so nothing hangs."""
        try:
            work()
        except Exception as e:
            self._error = self._error or e
            logging.getLogger('peresmeshnik').exception('pipeline worker %s died', work.__name__[1:])
        finally:
            out.put(None)

    def _wall(self, pos: int) -> float:
        """Arrival time of audio sample `pos` (exact when audio arrives in real time)."""
        return self._fed_wall - (self._fed - pos) / SAMPLE_RATE

    def _listen(self):
        stream, n, pos, seg0, partial_s = None, 0, 0, 0, 0.0
        while (item := self._in.get()) is not None:
            if self._stop.is_set():
                continue
            self._fed_wall, pcm = item
            self._fed += len(pcm)
            # behind the audio (slow or busy machine): skip partials until caught up; clause ends still count
            catch_up = self.realtime and time.perf_counter() - self._fed_wall > MAX_BEHIND_S
            for kind, x in self.vad.feed(pcm):  # lazy: vad.pos is the sample right after x
                pos = self.vad.pos
                if kind == 'start':
                    stream, seg0, partial_s = self.asr.stream(self.src), pos - len(x), 0.0
                    self._start = seg0
                t = time.perf_counter()
                partial = stream.feed(x, partial=not catch_up)
                partial_s += time.perf_counter() - t
                if partial:
                    self._events.put(Caption(partial, clause=n))
                if kind == 'end':
                    silence = self.vad.min_silence_frames * len(x)
                    n = self._clause(n, stream, None, 'pause', pos - silence, partial_s, settled_s=silence / SAMPLE_RATE)
                    stream = None
                elif (force := stream.duration >= self.max_clause_s) or partial is not None:
                    t = time.perf_counter()
                    text = stream.cut(force)
                    if text is not None:
                        n = self._clause(n, stream, text, 'max' if force else 'punct', seg0 + stream.offset, partial_s,
                                         time.perf_counter() - t)
                        partial_s = 0.0
        if stream is not None and not self._stop.is_set():
            self._clause(n, stream, None, 'eof', pos, partial_s)
        self.stats['audio_s'] = self._fed / SAMPLE_RATE
        self.stats['vad_s'] = self.vad.busy_s

    def _clause(self, n, stream, text, reason, end, partial_s, final_s=0.0, settled_s=0.0):
        start, self._start = self._start, end  # a cut's audio after `end` starts the next clause
        if text is None:
            t = time.perf_counter()
            text = stream.finish(settled_s)
            final_s = time.perf_counter() - t
        self.stats['asr_partial_s'] += partial_s
        self.stats['asr_final_s'] += final_s
        if not text:
            return n
        self._clauses.put({'clause': n, 'reason': reason, 'start_s': round(start / SAMPLE_RATE, 2), 'end_s': round(end / SAMPLE_RATE, 2),
                           'end_wall': self._wall(end),
                           'asr_partial_ms': round(1000 * partial_s), 'asr_final_ms': round(1000 * final_s), 'text': text})
        return n + 1

    def _translate(self):
        while (c := self._clauses.get()) is not None:
            if self._stop.is_set():
                continue
            t = time.perf_counter()
            c['translation'] = self.mt.translate(c['text'], self.src, self.tgt)
            self.stats['mt_s'] += (dt := time.perf_counter() - t)
            self._events.put(Caption(c['text'], c['translation'], final=True, clause=c['clause']))
            c['mt_ms'] = round(1000 * dt)
            if self.realtime:
                c['caption_lag_s'] = round(time.perf_counter() - c['end_wall'], 3)
            self._spoken.put(c)

    def _speak(self):
        st, sr = self.stats, self.tts.sample_rate
        while (c := self._spoken.get()) is not None:
            if self._stop.is_set():
                continue
            end_wall, translation = c.pop('end_wall'), c.pop('translation')
            t, samples, start = time.perf_counter(), 0, 0.0
            speed = 1.0
            if self.realtime:  # lags are real: the queue policy applies
                lag = max(self._play_end, t + self._ttfa) - end_wall  # when this clause's voice would start
                if end_wall < self._cut or (self.max_lag_s is not None and lag > self.max_lag_s):
                    c['dropped'] = 'barge-in' if end_wall < self._cut else round(lag, 3)
                    st['dropped'] += 1
                    log.info(json.dumps(c | {'translation': translation}, ensure_ascii=False))
                    continue
                if self.max_lag_s is not None and self._speeds and lag > FAST_AFTER_S:
                    speed = round(1 + (MAX_SPEED - 1) * min(1.0, (lag - FAST_AFTER_S) / max(self.max_lag_s - FAST_AFTER_S, 1e-3)), 3)
                    c['speed'] = speed
            cut = self._cut
            for chunk in self.tts.synth(translation, self.tgt, self.voice, **({'speed': speed} if speed != 1.0 else {})):
                if self._stop.is_set() or self._cut != cut:
                    break  # closes synth(), which reads away the rest of the request
                now = time.perf_counter()
                if not samples:
                    start = max(self._play_end, now)  # its playback starts after the speech before it
                    c['tts_first_ms'] = round(1000 * (now - t))
                    self._ttfa = now - t
                    if self.realtime:
                        c |= {'synth_lag_s': round(now - end_wall, 3), 'voice_lag_s': round(start - end_wall, 3),
                              'play_queue_s': round(start - now, 3)}
                with self._play_lock:
                    if self._cut != cut:  # interrupted since this chunk came: it is not played
                        break
                    self._play_end = max(self._play_end, now) + len(chunk) / sr
                samples += len(chunk)
                self._events.put(Speech(chunk, sr))
            st['tts_s'] += (dt := time.perf_counter() - t)
            st['tts_audio_s'] += samples / sr
            st['clauses'] += 1
            log.info(json.dumps(c | {'tts_ms': round(1000 * dt), 'speech_s': round(samples / sr, 2),
                                     'translation': translation}, ensure_ascii=False))

    def summary(self) -> dict:
        """Busy time of each link per second of input audio (link RTF)."""
        s, a = self.stats, max(self.stats['audio_s'], 1e-9)
        return {'audio_s': round(a, 2), 'clauses': int(s['clauses']), 'dropped': int(s['dropped']), 'speech_s': round(s['tts_audio_s'], 2),
                **{f'{k[:-2]}_rtf': round(s[k] / a, 4) for k in ('vad_s', 'asr_partial_s', 'asr_final_s', 'mt_s', 'tts_s')}}


def make_local(src: str, tgt: str, asr: str | None = None, threads: int | None = None, tts=None, voice: Voice | None = None,
               settings=None, max_lag_s: float | None = None) -> LocalDirection:
    """Engines and threads from the settings (`[local]`, ARCHITECTURE §6), models from the registry
    (raises models.ModelMissing until fetched); S2 passes its clone `tts` and `voice`."""
    from peresmeshnik import config
    from peresmeshnik.local.asr import OnnxAsr
    from peresmeshnik.local.mt import OpusMt
    from peresmeshnik.local.tts import PiperTts
    from peresmeshnik.local.vad import SileroVad
    loc = (settings or config.load()).local
    threads = threads or loc.threads
    return LocalDirection(SileroVad(), OnnxAsr(asr or loc.asr[src], threads=threads),
                          OpusMt(src, tgt, loc.mt[f'{src}-{tgt}'], threads=threads),
                          tts or PiperTts(loc.tts[tgt], threads=threads), src, tgt, voice, max_lag_s=max_lag_s)


def cpu_s(direction: LocalDirection | None = None) -> float:
    """CPU seconds of this process plus the TTS process tree (Piper runs in its own process)."""
    total = time.process_time()
    proc = getattr(direction and direction.tts, '_p', None)
    if proc is not None:
        import psutil
        try:
            p = psutil.Process(proc.pid)
            for q in [p, *p.children(recursive=True)]:
                t = q.cpu_times()
                total += t.user + t.system
        except psutil.NoSuchProcess:
            pass
    return total


def read_audio(path: str) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:  # ponytail: linear resampling, fine for speech; use soxr if quality matters
        x = np.interp(np.arange(0, len(x), sr / SAMPLE_RATE), np.arange(len(x)), x).astype(np.float32)
    return x


def run(direction: LocalDirection, pcm: np.ndarray, realtime: bool = False, frame: int = 512) -> None:
    """Feed a recording as a device would: 32 ms frames, at its own pace if `realtime`."""
    t0 = time.perf_counter()
    for i in range(0, len(pcm), frame):
        if realtime and (dt := t0 + i / SAMPLE_RATE - time.perf_counter()) > 0:
            time.sleep(dt)
        direction.feed(pcm[i:i + frame])


def cli(argv=None) -> int:
    from peresmeshnik import models
    from peresmeshnik.local.asr import ONNX_ASR
    ap = argparse.ArgumentParser(prog='peresmeshnik run', description='Translate a recording through the local pipeline.')
    ap.add_argument('--in', dest='inp', required=True, help='audio file (any rate, mixed to mono)')
    ap.add_argument('--from', dest='src', default='en', choices=sorted(OTHER))
    ap.add_argument('--to', dest='tgt', choices=sorted(OTHER), help='default: the other language')
    ap.add_argument('--asr', choices=sorted(ONNX_ASR), help='ASR model (default: settings [local] asr for the language)')
    ap.add_argument('--threads', type=int, help='threads per engine (default: settings [local] threads)')
    ap.add_argument('--realtime', action='store_true', help='feed the file at its own pace (real lags in the log)')
    ap.add_argument('--pipewire', action='store_true', help='Linux: play the file into the "Peresmeshnik Call" sink, capture its monitor')
    ap.add_argument('--blackhole', action='store_true', help='macOS: play the file into BlackHole 16ch, capture it back')
    ap.add_argument('--tap', action='store_true', help='macOS 14.4+: afplay plays the file, a process tap of it is the input')
    ap.add_argument('--play', nargs='?', const='', metavar='SINK',
                    help='play the translation (Linux: a sink, macOS: an output device; the default if no name)')
    ap.add_argument('--out', help='write the translated speech here (wav)')
    ap.add_argument('--log', help='per-clause latency, JSON lines')
    ap.add_argument('--partials', action='store_true', help='print ASR partials too')
    ap.add_argument('--max-lag', type=float, default=MAX_LAG_S,
                    help=f'with --realtime / --pipewire: drop speech that would start later than this after its clause, s '
                         f'(0 = say everything; default {MAX_LAG_S:g})')
    a = ap.parse_args(argv)
    tgt = a.tgt or OTHER[a.src]
    if tgt == a.src:
        ap.error('--from and --to are the same language')
    if not os.path.isfile(a.inp):
        ap.error(f'no such file: {a.inp}')
    sink = None
    mac = sys.platform == 'darwin'
    if (a.blackhole or a.tap) and not mac:
        ap.error('--blackhole and --tap need macOS')
    if mac and a.play is not None:
        from peresmeshnik.audio import coreaudio
        from peresmeshnik.audio.capture_macos import CALL_DEVICE
        sink = a.play or coreaudio.default()
        if a.blackhole and sink == CALL_DEVICE:
            ap.error(f'the output would be {CALL_DEVICE}, the capture: the translation would be translated again; pass --play <device>')
    elif a.pipewire or a.play is not None:
        if not sys.platform.startswith('linux'):
            ap.error('--pipewire and --play need Linux (PipeWire or PulseAudio) for now')
        from peresmeshnik.audio.capture_pipewire import CALL_SINK
        from peresmeshnik.audio.out import default_sink
        try:
            default = default_sink()  # also checks that a sound server is up, before the engines load
        except (OSError, subprocess.CalledProcessError) as e:
            ap.error(f'no sound server (pactl: {e})')
        if a.play is not None:
            sink = a.play or default
        if sink == CALL_SINK:  # e.g. left default by a crashed run on a box without a real output
            ap.error(f'the output would be our own capture sink {CALL_SINK}: the translation would be translated again; '
                     'pass --play <sink>')
    sys.stdout.reconfigure(encoding='utf-8')  # captions are Cyrillic; a Windows pipe defaults to the ANSI code page
    if a.log:
        log.addHandler(logging.FileHandler(a.log, 'w', encoding='utf-8'))
        log.setLevel(logging.INFO)

    t = time.perf_counter()
    try:
        d = make_local(a.src, tgt, a.asr, a.threads, max_lag_s=a.max_lag or None)
    except models.ModelMissing as e:
        print(e, file=sys.stderr)
        return 2
    d.warm_up()
    live = a.pipewire or a.blackhole or a.tap  # audio from a device, at its own pace
    d.realtime = a.realtime or live
    print(f'engines loaded in {time.perf_counter() - t:.1f} s', file=sys.stderr)
    player = None
    if sink:
        from peresmeshnik.audio.out import Output
        player = Output(d.tts.sample_rate, sink)  # a named device: it must not follow a new default
    speech, t_start = [], time.perf_counter()

    def consume():
        at = 0  # write cursor of --out, samples
        for e in d.events():
            if isinstance(e, Speech):
                if player:
                    player.play(e.pcm, e.sample_rate)
                if a.out:  # real-time runs keep the timeline: a chunk starts when it was produced
                    gap = int((time.perf_counter() - t_start) * e.sample_rate) - at if d.realtime else 0
                    if gap > 0:
                        speech.append(np.zeros(gap, np.float32))
                    speech.append(e.pcm)
                    at += max(gap, 0) + len(e.pcm)
            elif e.final:
                print(f'{a.src}: {e.original}\n{tgt}: {e.translation}', flush=True)
            elif a.partials:
                print(f'  … {e.original}', flush=True)
    consumer = threading.Thread(target=consume, name='consumer')
    consumer.start()

    cpu0, wall0 = cpu_s(d), time.perf_counter()
    try:
        try:
            if a.pipewire:
                from peresmeshnik.audio.capture_pipewire import PipeWireCapture
                cap = PipeWireCapture()
                cap.start(d.feed)
                try:
                    subprocess.run(['paplay', f'--device={cap.sink}', a.inp], check=True)  # stands in for the call app
                    time.sleep(1.0)  # trailing silence closes the last clause
                finally:
                    cap.stop()  # removes the sink
            elif a.blackhole:
                import sounddevice as sd

                from peresmeshnik.audio.capture_macos import CALL_DEVICE, BlackHoleIn
                from peresmeshnik.audio.coreaudio import stream_args
                from peresmeshnik.audio.resample import Resampler
                cap = BlackHoleIn()
                cap.start(d.feed)
                try:
                    args = stream_args(CALL_DEVICE)  # as the capture's: another buffer size on one device stops the other
                    sr = args.pop('samplerate')
                    sd.play(Resampler(SAMPLE_RATE, sr)(read_audio(a.inp)), sr, device=CALL_DEVICE, blocking=True, **args)
                    time.sleep(1.0)
                finally:
                    cap.stop()
            elif a.tap:
                import tempfile

                import soundfile as sf

                from peresmeshnik.audio.capture_macos import ProcessTap
                wav = os.path.join(tempfile.mkdtemp(prefix='peresmeshnik-'), 'call.wav')  # afplay reads no Ogg
                sf.write(wav, read_audio(a.inp), SAMPLE_RATE)
                call = subprocess.Popen(['afplay', wav])  # stands in for the call app, on the default output
                ended = []
                cap = ProcessTap(pid=call.pid, on_end=ended.append)
                cap.start(d.feed)
                try:
                    call.wait()
                    time.sleep(1.0)
                finally:
                    cap.stop()
                    call.kill()
                    os.remove(wav)
                print(f'tap: {cap.rate} Hz, {cap.status}', file=sys.stderr)
                if ended and not ended[0].endswith('is not running'):  # the player's end ends the tap too
                    raise RuntimeError(ended[0])
            else:
                run(d, read_audio(a.inp), a.realtime)
        finally:
            d.finish()  # always: it ends the event stream, so the consumer and the process can exit
        cpu, wall = cpu_s(d), time.perf_counter() - wall0  # before close(): the TTS process is still alive
    except Exception as e:  # a failed device, file or pipeline worker: report it, exit non-zero
        print(f'error: {e}', file=sys.stderr)
        return 1
    finally:
        d.close()
        consumer.join()
        if player:
            player.stop()
    if a.out:
        import soundfile as sf
        sf.write(a.out, np.concatenate(speech) if speech else np.zeros(1, np.float32), d.tts.sample_rate)
    s = d.summary() | {'rtf': round(wall / d.stats['audio_s'], 3), 'cpu_s_per_audio_s': round((cpu - cpu0) / d.stats['audio_s'], 3)}
    print(json.dumps(s), file=sys.stderr)
    return 0

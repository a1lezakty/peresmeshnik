"""Local pipeline behind `Direction`: VAD -> streaming ASR -> clauses -> MT -> TTS.

Threads (ARCHITECTURE §2): feed() only enqueues; the "listen" worker runs VAD, ASR partials and
clause segmentation, the "translate" worker runs MT and puts out the caption, the "speak" worker
runs TTS, so captions never wait for speech (a clone TTS on CPU is slower than real time) and a
clause is spoken while the next one is recognised. Generic over `Tts` and `Voice`: the stock Piper voice (them -> me) and the clone
(me -> them) plug in the same way. On silence only the VAD runs.

A clause ends at a VAD pause (>= 300 ms), at sentence punctuation from the ASR with a word after
it, or at `max_clause_s` (then at a comma or before the last word). Per clause one JSON line goes
to the `peresmeshnik.latency` logger: time of each link and, when audio arrives in real time,
the lag from the end of the clause to its translated caption and to its first translated audio.

    peresmeshnik models fetch silero-vad parakeet-tdt-0.6b-v3 opus-mt-en-ru piper-ru_RU-dmitri-medium
    peresmeshnik run --in call.wav [--from en] [--to ru] [--out speech.wav] [--log latency.jsonl]
    peresmeshnik run --in call.wav --pipewire --play   # file -> "Peresmeshnik Call" sink -> monitor capture
"""
import os

for _k, _v in {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'VECLIB_MAXIMUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE'}.items():
    os.environ.setdefault(_k, _v)  # before numpy: its BLAS pool would otherwise take every core

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import queue  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech, Voice  # noqa: E402

log = logging.getLogger('peresmeshnik.latency')
OTHER = {'en': 'ru', 'ru': 'en'}
HELLO = {'en': 'Hello.', 'ru': 'Привет.'}
MAX_BEHIND_S = 0.15


class LocalDirection:
    """`asr.stream()` objects also take `feed(pcm, partial=False)` and expose `duration` (s) and
    `cut(force) -> str | None` (local/asr.py)."""

    def __init__(self, vad, asr, mt, tts, src: str, tgt: str, voice: Voice | None = None, max_clause_s: float = 5.0):
        self.vad, self.asr, self.mt, self.tts, self.voice = vad, asr, mt, tts, voice
        self.src, self.tgt, self.max_clause_s = src, tgt, max_clause_s
        self.realtime = True  # audio arrives at its own pace, so lags to the clause end are real
        self.stats: dict[str, float] = defaultdict(float)  # busy seconds per link, counts; see summary()
        self._in: queue.Queue = queue.Queue()
        self._clauses: queue.Queue = queue.Queue()
        self._spoken: queue.Queue = queue.Queue()  # translated clauses waiting for TTS
        self._events: queue.Queue = queue.Queue()
        self._fed, self._fed_wall = 0, 0.0  # samples fed so far and when the last chunk arrived
        self._threads = [threading.Thread(target=self._run, args=(work, out), name=work.__name__[1:], daemon=True)
                         for work, out in ((self._listen, self._clauses), (self._translate, self._spoken), (self._speak, self._events))]
        for t in self._threads:
            t.start()

    # Direction
    def feed(self, pcm: np.ndarray) -> None:
        self._in.put((time.perf_counter(), pcm))

    def events(self):
        while (e := self._events.get()) is not None:
            yield e

    def warm_up(self) -> None:
        """Run every engine once before audio comes (first calls allocate); the app shows "warming up"."""
        s = self.asr.stream(self.src)
        s.feed(np.zeros(SAMPLE_RATE, np.float32))
        s.finish()
        for _ in self.tts.synth(self.mt.translate(HELLO[self.src], self.src, self.tgt), self.tgt, self.voice):
            pass

    def finish(self) -> None:
        """End of input: recognise, translate and speak what is left, then stop the workers."""
        self._in.put(None)
        for t in self._threads:
            t.join()

    def close(self) -> None:
        """finish(), then free the engines (the TTS process too)."""
        self.finish()
        for engine in (self.tts, self.mt, self.asr, self.vad):
            if hasattr(engine, 'close'):
                engine.close()

    # workers
    @staticmethod
    def _run(work, out: queue.Queue) -> None:
        """A worker that dies still ends its output stream, so finish() and events() never hang."""
        try:
            work()
        finally:
            out.put(None)

    def _wall(self, pos: int) -> float:
        """Arrival time of audio sample `pos` (exact when audio arrives in real time)."""
        return self._fed_wall - (self._fed - pos) / SAMPLE_RATE

    def _listen(self):
        stream, n, pos, seg0, partial_s = None, 0, 0, 0, 0.0
        while (item := self._in.get()) is not None:
            self._fed_wall, pcm = item
            self._fed += len(pcm)
            # behind the audio (slow or busy machine): skip partials until caught up; clause ends still count
            catch_up = self.realtime and time.perf_counter() - self._fed_wall > MAX_BEHIND_S
            for kind, x in self.vad.feed(pcm):  # lazy: vad.pos is the sample right after x
                pos = self.vad.pos
                if kind == 'start':
                    stream, seg0, partial_s = self.asr.stream(self.src), pos - len(x), 0.0
                t = time.perf_counter()
                partial = stream.feed(x, partial=not catch_up)
                partial_s += time.perf_counter() - t
                if partial:
                    self._events.put(Caption(partial))
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
        if stream is not None:
            self._clause(n, stream, None, 'eof', pos, partial_s)
        self.stats['audio_s'] = self._fed / SAMPLE_RATE
        self.stats['vad_s'] = self.vad.busy_s

    def _clause(self, n, stream, text, reason, end, partial_s, final_s=0.0, settled_s=0.0):
        if text is None:
            t = time.perf_counter()
            text = stream.finish(settled_s)
            final_s = time.perf_counter() - t
        self.stats['asr_partial_s'] += partial_s
        self.stats['asr_final_s'] += final_s
        if not text:
            return n
        self._clauses.put({'clause': n, 'reason': reason, 'end_s': round(end / SAMPLE_RATE, 2), 'end_wall': self._wall(end),
                           'asr_partial_ms': round(1000 * partial_s), 'asr_final_ms': round(1000 * final_s), 'text': text})
        return n + 1

    def _translate(self):
        while (c := self._clauses.get()) is not None:
            t = time.perf_counter()
            c['translation'] = self.mt.translate(c['text'], self.src, self.tgt)
            self.stats['mt_s'] += (dt := time.perf_counter() - t)
            self._events.put(Caption(c['text'], c['translation'], final=True))
            c['mt_ms'] = round(1000 * dt)
            if self.realtime:
                c['caption_lag_s'] = round(time.perf_counter() - c['end_wall'], 3)
            self._spoken.put(c)

    def _speak(self):
        st = self.stats
        while (c := self._spoken.get()) is not None:
            end_wall, translation = c.pop('end_wall'), c.pop('translation')
            t, samples = time.perf_counter(), 0
            for chunk in self.tts.synth(translation, self.tgt, self.voice):
                if not samples:
                    c['tts_first_ms'] = round(1000 * (time.perf_counter() - t))
                    if self.realtime:
                        c['voice_lag_s'] = round(time.perf_counter() - end_wall, 3)
                samples += len(chunk)
                self._events.put(Speech(chunk, self.tts.sample_rate))
            st['tts_s'] += (dt := time.perf_counter() - t)
            st['tts_audio_s'] += samples / self.tts.sample_rate
            st['clauses'] += 1
            log.info(json.dumps(c | {'tts_ms': round(1000 * dt), 'speech_s': round(samples / self.tts.sample_rate, 2),
                                     'translation': translation}, ensure_ascii=False))

    def summary(self) -> dict:
        """Busy time of each link per second of input audio (link RTF)."""
        s, a = self.stats, max(self.stats['audio_s'], 1e-9)
        return {'audio_s': round(a, 2), 'clauses': int(s['clauses']), 'speech_s': round(s['tts_audio_s'], 2),
                **{f'{k[:-2]}_rtf': round(s[k] / a, 4) for k in ('vad_s', 'asr_partial_s', 'asr_final_s', 'mt_s', 'tts_s')}}


def make_local(src: str, tgt: str, asr: str | None = None, threads: int | None = None, tts=None, voice: Voice | None = None,
               settings=None) -> LocalDirection:
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
                          tts or PiperTts(loc.tts[tgt], threads=threads), src, tgt, voice)


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
    ap = argparse.ArgumentParser(prog='peresmeshnik run', description='Translate a recording through the local pipeline.')
    ap.add_argument('--in', dest='inp', required=True, help='audio file (any rate, mixed to mono)')
    ap.add_argument('--from', dest='src', default='en', choices=sorted(OTHER))
    ap.add_argument('--to', dest='tgt', choices=sorted(OTHER), help='default: the other language')
    ap.add_argument('--asr', help='ASR model id (default: settings [local] asr; gigaam-v3-e2e-ctc for Russian)')
    ap.add_argument('--threads', type=int, help='threads per engine (default: settings [local] threads)')
    ap.add_argument('--realtime', action='store_true', help='feed the file at its own pace (real lags in the log)')
    ap.add_argument('--pipewire', action='store_true', help='play the file into the "Peresmeshnik Call" sink and capture its monitor')
    ap.add_argument('--play', nargs='?', const='', metavar='SINK', help='play the translation (default sink if no name)')
    ap.add_argument('--out', help='write the translated speech here (wav)')
    ap.add_argument('--log', help='per-clause latency, JSON lines')
    ap.add_argument('--partials', action='store_true', help='print ASR partials too')
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding='utf-8')  # captions are Cyrillic; a Windows pipe defaults to the ANSI code page
    tgt = a.tgt or OTHER[a.src]
    if a.log:
        log.addHandler(logging.FileHandler(a.log, 'w', encoding='utf-8'))
        log.setLevel(logging.INFO)

    from peresmeshnik import models
    t = time.perf_counter()
    try:
        d = make_local(a.src, tgt, a.asr, a.threads)
    except models.ModelMissing as e:
        print(e, file=sys.stderr)
        return 2
    d.warm_up()
    d.realtime = a.realtime or a.pipewire
    print(f'engines loaded in {time.perf_counter() - t:.1f} s', file=sys.stderr)
    player = None
    if a.play is not None:
        from peresmeshnik.audio.out import PulseOut, default_sink
        player = PulseOut(d.tts.sample_rate, a.play or default_sink())  # a named sink: it must not follow a new default
    speech, t_start = [], time.perf_counter()

    def consume():
        at = 0  # write cursor of --out, samples
        for e in d.events():
            if isinstance(e, Speech):
                if player:
                    player.play(e.pcm, e.sample_rate)
                if a.out:  # real-time runs keep the timeline: a chunk starts when it was produced
                    gap = int((time.perf_counter() - t_start) * e.sample_rate) - at if (a.realtime or a.pipewire) else 0
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
    if a.pipewire:
        from peresmeshnik.audio.capture_pipewire import PipeWireCapture
        cap = PipeWireCapture()
        cap.start(d.feed)
        try:
            subprocess.run(['paplay', f'--device={cap.sink}', a.inp], check=True)  # stands in for the call app
            time.sleep(1.0)  # trailing silence closes the last clause
        finally:
            cap.stop()  # removes the sink
    else:
        run(d, read_audio(a.inp), a.realtime)
    d.finish()
    cpu, wall = cpu_s(d), time.perf_counter() - wall0  # before close(): the TTS process is still alive
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

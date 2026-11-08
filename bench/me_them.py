"""End to end "me -> them" on Linux: my recording -> LocalDirection (VAD, ASR, MT, clone TTS)
-> "Peresmeshnik Mic" -> recorded back from the source the way a call app records it -> ASR check.

    python bench/me_them.py input                       # 30 s of the enrolled reader, other lines than the reference
    python bench/me_them.py run --voice DIR --model-dir DIR    # clone in this process (GPU box; on CPU it falls behind)
    CLONE_SERVER='<command>' python bench/me_them.py run --remote   # clone on a GPU box, audio back over a pipe
    python bench/me_them.py serve --voice DIR --model-dir DIR  # what $CLONE_SERVER starts on the GPU box

`$CLONE_SERVER` is any command that starts `me_them.py serve` elsewhere (for example through ssh);
requests go to its stdin as JSON lines, audio comes back on its stdout. Results: one `e2e_me_them`
row in bench/results/<host>.jsonl, per-clause JSON lines and the wavs in --out-dir.
"""
import argparse
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the package from this checkout
from common import ROOT, record  # noqa: E402

SR_REC = 16_000  # what the call app records
LOUD = 0.02


# --- the clone on another machine

def serve(a):
    out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)  # library prints go to stderr: stdout carries only audio frames
    from peresmeshnik import voice
    from peresmeshnik.local.clone import XttsClone
    eng = XttsClone(a.model_dir, a.device, a.threads)
    if eng.device == 'cuda':
        import torch
        torch.cuda.set_per_process_memory_fraction(0.6)  # a shared GPU: stay inside the cap
    v = voice.load(a.voice.name, a.voice.parent)
    for _ in eng.synth('Hello there.', 'en', v):  # warm-up
        pass
    out.write(f'READY {eng.sample_rate} {eng.device}{"" if eng.device == "cuda" else a.threads}\n'.encode())
    out.flush()
    for line in sys.stdin:
        req = json.loads(line)
        t0, c0, n, ttfa = time.perf_counter(), time.process_time(), 0, None
        for c in eng.synth(req['text'], req['lang'], v):
            ttfa = ttfa or time.perf_counter() - t0
            out.write(struct.pack('<I', len(c)) + c.astype('<f4').tobytes())
            out.flush()
            n += len(c)
        stats = {'ttfa_s': round(ttfa or 0, 3), 'synth_s': round(time.perf_counter() - t0, 3),
                 'cpu_s': round(time.process_time() - c0, 3), 'audio_s': round(n / eng.sample_rate, 3)}
        out.write(struct.pack('<I', 0) + (json.dumps(stats) + '\n').encode())
        out.flush()


class RemoteTts:
    """`Tts` over a pipe to `serve`; it speaks in the server's voice profile (`voice` is ignored).
    ponytail: a caller that stops reading mid-utterance desyncs the pipe; fine for file runs."""

    def __init__(self, cmd: str):
        self.p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        ready = self.p.stdout.readline().split()
        if ready[:1] != [b'READY']:
            raise RuntimeError(f'clone server did not start: {ready}')
        self.sample_rate, self.device, self.stats = int(ready[1]), ready[2].decode(), []

    def _read(self, n):
        b = self.p.stdout.read(n)
        if len(b) < n:
            raise EOFError('clone server closed the pipe')
        return b

    def synth(self, text, lang, voice=None):
        self.p.stdin.write((json.dumps({'text': text, 'lang': lang}) + '\n').encode())
        self.p.stdin.flush()
        while n := struct.unpack('<I', self._read(4))[0]:
            yield np.frombuffer(self._read(4 * n), '<f4')
        self.stats.append(json.loads(self.p.stdout.readline()))

    def close(self):
        self.p.stdin.close()
        self.p.wait(60)


class Marked:
    """Remembers the first chunk of each utterance: the k-th synth() speaks the k-th logged clause
    (Speech events carry no clause number, and captions no longer wait for speech)."""

    def __init__(self, tts):
        self.tts, self.sample_rate, self.firsts = tts, tts.sample_rate, []

    def __getattr__(self, name):  # close() and the rest: the wrapped engine's
        return getattr(self.tts, name)

    def synth(self, text, lang, voice=None):
        self.firsts.append(None)
        for c in self.tts.synth(text, lang, voice):
            if self.firsts[-1] is None:
                self.firsts[-1] = c
            yield c


# --- the call app's side

class Recorder:
    """Records a PulseAudio/PipeWire source; remembers when each piece arrived."""

    def __init__(self, source: str):
        self.p = subprocess.Popen(['parec', f'--device={source}', '--format=float32le', f'--rate={SR_REC}',
                                   '--channels=1', '--raw', '--latency-msec=20', '--property=node.dont-reconnect=true'],
                                  stdout=subprocess.PIPE)
        self.chunks, self.marks, self.n = [], [(0, time.perf_counter())], 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        rest = b''
        while b := os.read(self.p.stdout.fileno(), 6400):
            b, rest = rest + b, b''
            if len(b) % 4:
                b, rest = b[:len(b) // 4 * 4], b[len(b) // 4 * 4:]
            x = np.frombuffer(b, '<f4')
            self.chunks.append(x)
            self.n += len(x)
            self.marks.append((self.n, time.perf_counter()))

    def wall(self, i):
        """Arrival time of sample i."""
        n, w = zip(*self.marks)
        return float(np.interp(i, n, w))

    def index(self, wall):
        n, w = zip(*self.marks)
        return int(np.interp(wall, w, n))

    def stop(self) -> np.ndarray:
        self.p.terminate()
        self.t.join()
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


def onset_after(x, i, quiet=int(0.1 * SR_REC)):
    """First sample from i on where sound starts after >= 100 ms of quiet."""
    loud = np.abs(x) > LOUD
    for j in np.nonzero(loud[i:])[0] + i:
        if not loud[max(0, j - quiet):j].any():
            return int(j)
    return None


def probe(mic, rec, n=5):
    """Virtual mic link alone: play short beeps, time them at the recorder."""
    sr, lags = 24_000, []
    beep = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr // 10) / sr)).astype(np.float32)
    for _ in range(n):
        time.sleep(0.4)
        t = time.perf_counter()
        mic.play(beep, sr)
        time.sleep(0.4)
        j = onset_after(np.concatenate(rec.chunks), rec.index(t) - SR_REC // 20)
        if j is not None:
            lags.append(rec.wall(j) - t)
    mic.drain()
    return lags


# --- checks

def words(t):
    return re.sub(r"[^\w\s']", ' ', t.lower().replace('ё', 'е')).split()


def wer(ref, hyp):
    import jiwer
    return round(jiwer.wer(' '.join(words(ref)), ' '.join(words(hyp))), 4)


def transcribe(x, sr, lang='en'):
    from faster_whisper import WhisperModel
    m = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=2)
    if sr != 16_000:
        x = np.interp(np.arange(0, len(x), sr / 16_000), np.arange(len(x)), x).astype(np.float32)
    return ' '.join(s.text.strip() for s in m.transcribe(x, language=lang, beam_size=5)[0])


def per_audio_s(stats, key):
    audio = sum(s['audio_s'] for s in stats)
    return round(sum(s[key] for s in stats) / audio, 3) if audio else None


# --- commands

def make_input(a):
    """~30 s of the same LibriVox reader as ref_ru.wav (lines after the reference), 0.6 s pauses."""
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    t = pq.read_table(ROOT / 'refs' / 'ru.parquet')
    audio, text = t.column('audio').to_pylist(), t.column('transcription').to_pylist()
    keep = [i for i, x in enumerate(audio) if re.match(r'poemi_01_', x['path'])][a.skip:]
    parts, said, total = [np.zeros(SR_REC // 2, np.float32)], [], 0.0
    for i in keep:
        x, sr = sf.read(io.BytesIO(audio[i]['bytes']), dtype='float32')
        parts += [x, np.zeros(int(0.6 * sr), np.float32)]
        said.append(text[i].strip())
        total += len(x) / sr + 0.6
        if total >= a.seconds:
            break
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / 'ru_30s.wav', np.concatenate(parts + [np.zeros(SR_REC, np.float32)]), SR_REC)
    (out / 'ru_30s.txt').write_text(' '.join(said), encoding='utf-8')
    print(out / 'ru_30s.wav', round(total, 1), 's')


def run(a):
    import logging

    import soundfile as sf

    from peresmeshnik import pipeline, voice
    from peresmeshnik.audio.virtual_mic_linux import SOURCE, VirtualMic
    from peresmeshnik.engines import Speech
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clauses = []

    class Keep(logging.Handler):
        def emit(self, r):
            clauses.append(json.loads(r.getMessage()))
    pipeline.log.addHandler(Keep())
    pipeline.log.setLevel(logging.INFO)

    t = time.perf_counter()
    if a.remote:
        tts, v = RemoteTts(os.environ['CLONE_SERVER']), None
    else:
        from peresmeshnik.local.clone import XttsClone
        tts, v = XttsClone(a.model_dir, a.device, a.threads), voice.load(a.voice.name, a.voice.parent)
    marked = Marked(tts)
    d = pipeline.make_local('ru', 'en', threads=a.threads, tts=marked, voice=v)
    d.realtime = True
    load_s = time.perf_counter() - t
    mic = VirtualMic()
    rec = Recorder(SOURCE)
    time.sleep(0.5)
    link = probe(mic, rec)
    print(f'engines {load_s:.1f} s; virtual mic link {[round(x * 1000) for x in link]} ms', flush=True)

    starts, captions, direct, said = {}, {}, [], []  # utterance k -> its first sample due at the mic; clause -> caption

    def consume():
        due = 0.0  # when what is queued in the mic has played (it plays back to back)
        for e in d.events():
            if isinstance(e, Speech):
                now = time.perf_counter()
                k = next((k for k, f in enumerate(marked.firsts) if f is e.pcm), None)
                if k is not None:
                    starts[k] = max(due, now)
                due = max(due, now) + len(e.pcm) / e.sample_rate
                mic.play(e.pcm, e.sample_rate)
                direct.append(e.pcm)
            elif e.final:
                captions[e.clause] = time.perf_counter()
                said.append(e.translation)
                print(f'ru: {e.original}\nen: {e.translation}', flush=True)
    th = threading.Thread(target=consume)
    th.start()
    pcm = pipeline.read_audio(str(a.inp))
    i0 = rec.n
    cpu0, wall0 = time.process_time(), time.perf_counter()
    pipeline.run(d, pcm, realtime=True)
    feed_end = time.perf_counter()
    d.finish()
    th.join()
    mic.drain()
    done = time.perf_counter()
    wall = done - wall0
    cpu = time.process_time() - cpu0
    time.sleep(0.5)
    x = rec.stop()[i0:]
    d.close()
    mic.remove()

    for k, c in enumerate(clauses):  # clause end -> its translated audio at the call app's mic
        s, cap = starts.get(k), captions.get(c['clause'])
        j = onset_after(x, rec.index(s) - i0 - SR_REC // 50) if s is not None else None
        if j is None or cap is None:
            continue
        heard = rec.wall(j + i0)
        c['mic_lag_s'] = round(heard - s, 3)  # virtual mic link, from when the audio was due to play
        c['call_lag_s'] = round(heard - (cap - c['caption_lag_s']), 3)  # from the clause end
        c['after_caption_s'] = round(heard - cap, 3)  # TTS + playback queue + virtual mic
    for c in clauses:
        print(json.dumps(c, ensure_ascii=False))
    sr = tts.sample_rate
    sf.write(out / 'recorded.wav', x, SR_REC)
    sf.write(out / 'direct.wav', np.concatenate(direct) if direct else np.zeros(1, np.float32), sr)
    (out / 'clauses.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in clauses), encoding='utf-8')
    mt = ' '.join(said)
    hyp_rec = transcribe(x, SR_REC)
    hyp_dir = transcribe(np.concatenate(direct), sr) if direct else ''
    lags = [c['call_lag_s'] for c in clauses if c.get('call_lag_s') is not None]
    remote = getattr(tts, 'stats', [])
    row = record('e2e_me_them', 'xtts-remote' if a.remote else 'xtts', device=f'remote-{tts.device}' if a.remote else (a.device or 'auto'),
                 threads=a.threads, audio_s=round(len(pcm) / pipeline.SAMPLE_RATE, 2), clauses=len(clauses),
                 load_s=round(load_s, 1), wall_s=round(wall, 1), tail_after_input_s=round(done - feed_end, 1),
                 cpu_s_per_audio_s=round(cpu / (len(pcm) / pipeline.SAMPLE_RATE), 3),
                 mic_link_ms=[round(v * 1000) for v in link],
                 voice_lag_s=[c.get('voice_lag_s') for c in clauses], call_lag_s=lags,
                 call_lag_median_s=round(float(np.median(lags)), 2) if lags else None,
                 caption_lag_s=[c.get('caption_lag_s') for c in clauses],
                 after_caption_s=[c.get('after_caption_s') for c in clauses],
                 tts_first_ms=[c.get('tts_first_ms') for c in clauses],
                 server_ttfa_s=[s['ttfa_s'] for s in remote], server_rtf=per_audio_s(remote, 'synth_s'),
                 server_cpu_per_audio_s=per_audio_s(remote, 'cpu_s'),
                 wer_recorded=wer(mt, hyp_rec), wer_direct=wer(mt, hyp_dir), summary=d.summary(),
                 recorded_peak=round(float(np.max(np.abs(x))), 3) if len(x) else 0.0)
    (out / 'summary.json').write_text(json.dumps(row | {'mt': mt, 'asr_recorded': hyp_rec, 'asr_direct': hyp_dir},
                                                 ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('input')
    i.add_argument('--out', default=str(ROOT / 'me_them'))
    i.add_argument('--skip', type=int, default=40, help='reader lines to skip (the reference uses the first ones)')
    i.add_argument('--seconds', type=float, default=30)
    for name in ('run', 'serve'):
        p = sub.add_parser(name)
        p.add_argument('--voice', type=Path, help='voice profile folder')
        p.add_argument('--model-dir', type=Path)
        p.add_argument('--device')
        p.add_argument('--threads', type=int, default=2)
    r = sub.choices['run']
    r.add_argument('--in', dest='inp', type=Path, default=ROOT / 'me_them' / 'ru_30s.wav')
    r.add_argument('--remote', action='store_true', help='clone through $CLONE_SERVER')
    r.add_argument('--out-dir', default=str(ROOT.parent / 'out' / 'me_them'))
    a = ap.parse_args()
    {'input': make_input, 'run': run, 'serve': serve}[a.cmd](a)


if __name__ == '__main__':
    main()

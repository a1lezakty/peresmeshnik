"""The local pipeline behind a WebSocket: what the network adds to "voice after the clause end" (remote server, package 0).

    python bench/remote_rt.py serve --port 47000 [--stub] [--threads 2]   # on the strong box; listens on 127.0.0.1 only
    python bench/remote_rt.py run --url ws://127.0.0.1:47000 [--lang en]  # the client: a call recording, live pace
    python bench/remote_rt.py local [--stub] [--lang en]                  # the reference: the same, in process, no network

Up: binary frames of 512 int16 samples at 16 kHz (32 ms) behind a u32 frame number; the server answers each with a
short `ack` (the frame RTT). Down: `Caption` / `Latency` and each clause's log line as JSON text, `Speech` binary
(sample_rate u32 + int16). Over an ssh tunnel (`ssh -N -L 47000:127.0.0.1:47000 host`) the server port stays closed.

`voice_lag` is taken on the client's clock: from the clause end (when its last sample was sent, at the live pace)
to the start of its voice in a player that plays speech back to back and holds the first chunk of a clause for
`jitter_ms` (the planned `remote.jitter_ms`); a later chunk that comes after the playback ran dry is an underrun.
Several jitter values are replayed from the same arrival times. `local` runs the same player on the in-process events.
`--stub`: no models, fixed delays (energy VAD, ASR 0.25 s, MT 0.15 s, first TTS chunk 0.08 s, TTS RTF 0.2): transport only.
One row per run in bench/results/<host>.jsonl, per-clause lines and the frame RTTs in --out-dir.
"""
import argparse
import asyncio
import json
import logging
import socket
import struct
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package from this checkout
# isort: off
from peresmeshnik import pipeline  # noqa: E402  first: it pins BLAS threads before numpy loads
import numpy as np  # noqa: E402
# isort: on
from peresmeshnik.engines import SAMPLE_RATE, Caption, Latency, Speech  # noqa: E402

from common import ROOT, record  # noqa: E402

FRAME = 512  # 32 ms, as the app's capture feeds the pipeline
DATA = Path(__file__).resolve().parents[1] / 'tests' / 'data'
PAIRS = {'en': 'ru', 'ru': 'en'}  # call-<lang>.ogg -> the other language
JITTERS_MS = (0, 40, 80, 120, 150, 200)
GAP_S = 0.1


# --- no models: the pipeline's own threads and events, engines that only wait

class StubVad:
    """Energy VAD with Silero's interface: a frame is speech above RMS 0.01, 9 quiet frames (~300 ms) end a segment."""
    min_silence_frames = 9

    def __init__(self):
        self.pos, self.busy_s, self.speaking, self._silent = 0, 0.0, False, 0

    def reset(self):
        self.__init__()

    def feed(self, pcm):
        loud = float(np.sqrt(np.mean(pcm ** 2))) > 0.01 if len(pcm) else False
        self.pos += len(pcm)
        if not self.speaking:
            if loud:
                self.speaking, self._silent = True, 0
                yield 'start', pcm
            return
        self._silent = 0 if loud else self._silent + 1
        if self._silent >= self.min_silence_frames:
            self.speaking = False
            yield 'end', pcm
        else:
            yield 'speech', pcm


class StubStream:
    cut_at = 'word'

    def __init__(self, asr_s):
        self.asr_s, self.pcm, self.offset = asr_s, np.zeros(0, np.float32), 0

    @property
    def duration(self):
        return len(self.pcm) / SAMPLE_RATE

    def feed(self, pcm, partial=True):
        self.pcm = np.concatenate([self.pcm, pcm])  # no partials: only final captions

    def cut(self, force=False, comma=False):
        return self.finish() if force else None

    def finish(self, settled_s=0.0):
        time.sleep(self.asr_s)
        n = max(1, round(self.duration * 2.5))  # ~2.5 words a second
        self.offset += len(self.pcm)
        self.pcm = np.zeros(0, np.float32)
        return ' '.join(['word'] * n)


class StubAsr:
    def __init__(self, asr_s=0.25):
        self.asr_s = asr_s

    def stream(self, lang):
        return StubStream(self.asr_s)


class StubMt:
    def __init__(self, mt_s=0.15):
        self.mt_s = mt_s

    def translate(self, text, src, tgt, context=()):
        time.sleep(self.mt_s)
        return text


class StubTts:
    """0.3 s of tone a word, in 0.5 s chunks: the first after `ttfa_s`, then at `rtf` of real time."""
    sample_rate = 24_000

    def __init__(self, ttfa_s=0.08, rtf=0.2):
        self.ttfa_s, self.rtf = ttfa_s, rtf

    def synth(self, text, lang, voice=None):
        sr, left = self.sample_rate, 0.3 * len(text.split())
        time.sleep(self.ttfa_s)
        while left > 0:
            dur = min(0.5, left)
            yield (0.1 * np.sin(2 * np.pi * 220 * np.arange(int(dur * sr)) / sr)).astype(np.float32)
            left -= dur
            if left > 0:
                time.sleep(dur * self.rtf)


def make(src, tgt, stub, threads, max_lag=0.0):
    """The direction a server connection (or `local`) runs. max_lag 0 (default): no queue policy, every clause is said
    and none sped up, so remote and local runs have the same clauses; pipeline.MAX_LAG_S is them -> me in the app."""
    if stub:
        return pipeline.LocalDirection(StubVad(), StubAsr(), StubMt(), StubTts(), src, tgt, partial_mt=False,
                                       max_lag_s=max_lag or None)
    return pipeline.make_local(src, tgt, threads=threads, max_lag_s=max_lag or None)


def devices(d):
    return {k: getattr(getattr(d, k), 'device', 'cpu') for k in ('asr', 'mt')}


def nodelay(transport):
    s = transport.get_extra_info('socket')
    return bool(s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)) if s is not None else None


class Clauses(logging.Handler):
    """The pipeline's per-clause log lines (clause number, end_s, its own voice_lag_s) to a callback."""

    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    def emit(self, r):
        self.cb(json.loads(r.getMessage()))


def encode(e):
    if isinstance(e, Speech):
        return struct.pack('<I', e.sample_rate) + (np.clip(e.pcm, -1, 1) * 32767).astype('<i2').tobytes()
    if isinstance(e, Latency):
        return json.dumps({'type': 'latency', 'clause': e.clause, 'what': e.what, 's': e.s})
    return json.dumps({'type': 'caption', 'clause': e.clause, 'final': e.final, 'original': e.original,
                       'translation': e.translation}, ensure_ascii=False)


# --- server

def serve(a):
    from websockets.asyncio.server import serve as ws_serve
    busy = asyncio.Lock()  # ponytail: one session at a time (the clause log handler is process-wide)

    async def handler(ws):
        if busy.locked():
            await ws.send(json.dumps({'type': 'busy'}))
            return
        async with busy:
            hello = json.loads(await ws.recv())
            loop = asyncio.get_running_loop()
            t = time.perf_counter()
            d = await loop.run_in_executor(None, make, hello['src'], hello['tgt'], a.stub, a.threads, a.max_lag)
            await loop.run_in_executor(None, d.warm_up)
            d.realtime = True
            load_s = time.perf_counter() - t
            out: asyncio.Queue = asyncio.Queue()
            put = lambda m: loop.call_soon_threadsafe(out.put_nowait, m)  # noqa: E731
            h = Clauses(lambda c: put(json.dumps({'type': 'clause', **c}, ensure_ascii=False)))
            pipeline.log.addHandler(h)
            pipeline.log.setLevel(logging.INFO)

            def pump():
                try:
                    for e in d.events():
                        put(encode(e))
                finally:
                    put(None)
            pumper = threading.Thread(target=pump, daemon=True)
            pumper.start()

            async def sender():
                while (m := await out.get()) is not None:
                    await ws.send(m)
            down = asyncio.create_task(sender())
            await ws.send(json.dumps({'type': 'ready', 'load_s': round(load_s, 2), 'stub': a.stub, 'threads': a.threads,
                                      'max_lag': a.max_lag, 'nodelay': nodelay(ws.transport), **devices(d)}))
            arrived = []
            try:
                async for m in ws:
                    if isinstance(m, bytes):
                        arrived.append(time.perf_counter())
                        seq = struct.unpack_from('<I', m)[0]
                        await ws.send(json.dumps({'type': 'ack', 'seq': seq}))
                        d.feed(np.frombuffer(m, '<i2', offset=4).astype(np.float32) / 32768)
                    elif json.loads(m).get('type') == 'end':
                        break
                await loop.run_in_executor(None, d.finish)
                await down
                gaps = np.diff(arrived) if len(arrived) > 1 else np.zeros(0)
                await ws.send(json.dumps({'type': 'done', 'summary': d.summary(), 'frames': len(arrived),
                                          'up_gaps_over_100ms': int((gaps > GAP_S).sum()),
                                          'up_gap_max_ms': round(1000 * float(gaps.max()), 1) if len(gaps) else None}))
            finally:
                pipeline.log.removeHandler(h)
                down.cancel()
                await loop.run_in_executor(None, d.close)

    async def main():
        async with ws_serve(handler, a.host, a.port, compression=None, max_size=None) as srv:
            print(f'listening on ws://{a.host}:{a.port} ({"stub" if a.stub else "models"}, threads {a.threads})', flush=True)
            await srv.serve_forever()
    asyncio.run(main())


# --- what the client gets, and the player replayed on it

class Trace:
    def __init__(self):
        self.cur = None  # the clause of the last 'voice' latency: its Speech follows (one speak thread, in order)
        self.chunks = defaultdict(list)  # clause -> [(arrival, seconds of audio)]
        self.server_voice, self.server_first, self.clauses, self.captions = {}, {}, {}, {}
        self.down = []  # (arrival, bytes)

    def text(self, t, m):
        if m['type'] == 'latency' and m['what'] == 'tts_first':
            self.server_first[m['clause']] = m['s']
        elif m['type'] == 'latency' and m['what'] == 'voice':
            self.cur = m['clause']
            self.server_voice[m['clause']] = m['s']
        elif m['type'] == 'caption' and m['final']:
            self.captions[m['clause']] = t
        elif m['type'] == 'clause':
            self.clauses[m['clause']] = m

    def speech(self, t, seconds):
        self.chunks[self.cur].append((t, seconds))


def clause_ends(trace, sent):
    """Client time of each clause's end: the send time of the frame with its last sample, back to that sample."""
    ends = {}
    for k, c in trace.clauses.items():
        if c.get('dropped') is not None:
            continue
        end = round(c['end_s'] * SAMPLE_RATE)  # ponytail: end_s is rounded to 10 ms in the log
        f = min(end // FRAME, len(sent) - 1)
        ends[k] = sent[f] - ((f + 1) * FRAME - end) / SAMPLE_RATE
    return ends


def play(trace, ends, jitter_s):
    """Back-to-back player: per clause its voice lag; underruns (a chunk late after the playback ran dry)."""
    play_end, lags, under, under_s = float('-inf'), {}, 0, 0.0
    for k in sorted(k for k in trace.chunks if k in ends):
        for i, (t, dur) in enumerate(trace.chunks[k]):
            if i == 0:
                start = lags[k] = max(t + jitter_s, play_end)
            else:
                if t > play_end:
                    under, under_s = under + 1, under_s + t - play_end
                start = max(t, play_end)
            play_end = start + dur
        lags[k] -= ends[k]
    return lags, under, under_s


def pct(x, q):
    return round(float(np.percentile(x, q)), 3) if len(x) else None


def summarize(trace, sent, due, a, label, extra):
    ends = clause_ends(trace, sent)
    by_jitter, per_clause = {}, {k: {'clause': k, 'end_s': trace.clauses[k]['end_s']} for k in ends}
    for j in a.jitters:
        lags, under, under_s = play(trace, ends, j / 1000)
        v = list(lags.values())
        by_jitter[j] = {'voice_lag_p50_s': pct(v, 50), 'voice_lag_p95_s': pct(v, 95), 'voice_lag_max_s': pct(v, 100),
                        'underruns': under, 'underrun_ms': round(1000 * under_s)}
        for k, s in lags.items():
            per_clause[k][f'voice_lag_j{j}_s'] = round(s, 3)
    for k, c in per_clause.items():
        c['server_voice_lag_s'] = trace.server_voice.get(k)
        c['caption_lag_s'] = round(trace.captions[k] - ends[k], 3) if k in trace.captions else None
        if c['server_voice_lag_s'] is not None and 'voice_lag_j0_s' in c:  # what the transport added on top
            c['net_extra_s'] = round(c['voice_lag_j0_s'] - c['server_voice_lag_s'], 3)
        if trace.chunks.get(k):  # its first audio here, not waiting behind earlier speech: pipeline + network alone
            c['first_audio_s'] = round(trace.chunks[k][0][0] - ends[k], 3)
            if k in trace.server_first:  # minus the server's own clause end -> first chunk: the transport alone
                c['net_first_s'] = round(c['first_audio_s'] - trace.server_first[k], 3)
    late = np.array(sent) - np.array(due)
    net = [c['net_extra_s'] for c in per_clause.values() if 'net_extra_s' in c]
    first = [c['first_audio_s'] for c in per_clause.values() if 'first_audio_s' in c]
    net_first = [c['net_first_s'] for c in per_clause.values() if 'net_first_s' in c]
    cap = [c['caption_lag_s'] for c in per_clause.values() if c['caption_lag_s'] is not None]
    audio_s, raw = len(sent) * FRAME / SAMPLE_RATE, extra.pop('_raw', {})
    row = record('remote_rt', 'stub' if a.stub else 'local-pipeline', mode=a.cmd, label=label, lang=a.lang,
                 audio_s=round(audio_s, 2), clauses=len(ends), spoken=len([k for k in ends if k in trace.chunks]),
                 voice_lag=by_jitter, caption_lag_p50_s=pct(cap, 50), caption_lag_p95_s=pct(cap, 95),
                 net_extra_p50_s=pct(net, 50), net_extra_p95_s=pct(net, 95), max_lag=a.max_lag,
                 first_audio_p50_s=pct(first, 50), first_audio_p95_s=pct(first, 95),
                 net_first_p50_s=pct(net_first, 50), net_first_p95_s=pct(net_first, 95),
                 send_late_p99_ms=round(1000 * pct(late, 99), 1), send_late_max_ms=round(1000 * float(late.max()), 1),
                 **extra)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{a.cmd}-{label}-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
    (out / f'{stem}.clauses.jsonl').write_text(''.join(json.dumps(per_clause[k] | {'log': trace.clauses.get(k)}, ensure_ascii=False)
                                                       + '\n' for k in sorted(per_clause)), encoding='utf-8')
    (out / f'{stem}.json').write_text(json.dumps(row | raw, ensure_ascii=False), encoding='utf-8')
    return row


def frames(lang):
    pcm = pipeline.read_audio(str(DATA / f'call-{lang}.ogg'))
    pcm = np.concatenate([pcm, np.zeros(SAMPLE_RATE, np.float32)])  # a trailing second closes the last clause
    pcm = np.pad(pcm, (0, -len(pcm) % FRAME))
    return (np.clip(pcm, -1, 1) * 32767).astype('<i2').reshape(-1, FRAME)


# --- client

def run(a):
    asyncio.run(_run(a))


async def _run(a):
    from websockets.asyncio.client import connect
    x = frames(a.lang)
    trace, sent, due, rtt_at = Trace(), [], [], {}
    up = down = 0
    async with connect(a.url, compression=None, max_size=None, open_timeout=30) as ws:
        client_nodelay = nodelay(ws.transport)
        await ws.send(json.dumps({'type': 'hello', 'src': a.lang, 'tgt': PAIRS[a.lang]}))
        ready = json.loads(await asyncio.wait_for(ws.recv(), 600))  # the server loads its models first
        if ready['type'] != 'ready':
            raise SystemExit(f'server: {ready}')
        print(f'server ready: {ready}', flush=True)
        a.stub, a.max_lag = ready['stub'], ready['max_lag']
        done = {}

        async def receive():
            nonlocal down
            async for m in ws:
                t = time.perf_counter()
                down += len(m)
                trace.down.append((t, len(m)))
                if isinstance(m, bytes):
                    sr = struct.unpack_from('<I', m)[0]
                    trace.speech(t, (len(m) - 4) / 2 / sr)
                    continue
                msg = json.loads(m)
                if msg['type'] == 'ack':
                    rtt_at[msg['seq']] = t
                elif msg['type'] == 'done':
                    done.update(msg)
                    return
                else:
                    trace.text(t, msg)
        rx = asyncio.create_task(receive())
        t0 = time.perf_counter() + 0.05
        for i, f in enumerate(x):
            due.append(t0 + i * FRAME / SAMPLE_RATE)
            if (dt := due[-1] - time.perf_counter()) > 0:
                await asyncio.sleep(dt)
            sent.append(time.perf_counter())
            b = struct.pack('<I', i) + f.tobytes()
            up += len(b)
            await ws.send(b)
        await ws.send(json.dumps({'type': 'end'}))
        await asyncio.wait_for(rx, 600)
        wall = time.perf_counter() - t0
    rtt = np.array([rtt_at[i] - sent[i] for i in range(len(sent)) if i in rtt_at]) * 1000
    acks = np.diff([rtt_at[i] for i in sorted(rtt_at)])
    secs = defaultdict(int)
    for t, n in trace.down:
        secs[int(t - t0)] += n
    audio_s = len(sent) * FRAME / SAMPLE_RATE
    extra = {'url_host': a.url.split('//')[1].split(':')[0], 'server': {k: v for k, v in ready.items() if k != 'type'},
             'nodelay_client': client_nodelay, 'nodelay_server': ready.get('nodelay'),
             'rtt_ms': {q: round(float(np.percentile(rtt, q)), 2) for q in (50, 95, 99, 100)} if len(rtt) else None,
             'acks_lost': len(sent) - len(rtt), 'ack_gaps_over_100ms': int((acks > GAP_S).sum()),
             'ack_gap_max_ms': round(1000 * float(acks.max()), 1) if len(acks) else None,
             'up_gaps_over_100ms': done.get('up_gaps_over_100ms'), 'up_gap_max_ms': done.get('up_gap_max_ms'),
             'up_kbps': round(8 * up / audio_s / 1000, 1), 'down_kbps_avg': round(8 * down / wall / 1000, 1),
             'down_kbps_peak_1s': round(8 * max(secs.values(), default=0) / 1000, 1),
             'wall_s': round(wall, 1), 'summary': done.get('summary'),
             '_raw': {'rtt_ms_all': [round(float(v), 2) for v in rtt]}}
    summarize(trace, sent, due, a, a.label, extra)


# --- reference: the same in process

def local(a):
    x = frames(a.lang)
    trace, sent, due = Trace(), [], []
    t = time.perf_counter()
    d = make(a.lang, PAIRS[a.lang], a.stub, a.threads, a.max_lag)
    d.warm_up()
    d.realtime = True
    load_s = time.perf_counter() - t
    pipeline.log.addHandler(Clauses(lambda c: trace.text(time.perf_counter(), {'type': 'clause', **c})))
    pipeline.log.setLevel(logging.INFO)

    def consume():
        for e in d.events():
            now = time.perf_counter()
            if isinstance(e, Speech):
                trace.speech(now, len(e.pcm) / e.sample_rate)
            elif isinstance(e, Latency):
                trace.text(now, {'type': 'latency', 'clause': e.clause, 'what': e.what, 's': e.s})
            elif isinstance(e, Caption):
                trace.text(now, {'type': 'caption', 'clause': e.clause, 'final': e.final})
    th = threading.Thread(target=consume)
    th.start()
    t0 = time.perf_counter() + 0.05
    for i, f in enumerate(x):
        due.append(t0 + i * FRAME / SAMPLE_RATE)
        if (dt := due[-1] - time.perf_counter()) > 0:
            time.sleep(dt)
        sent.append(time.perf_counter())
        d.feed(f.astype(np.float32) / 32768)
    d.finish()
    th.join()
    wall = time.perf_counter() - t0
    summary = d.summary()
    d.close()
    summarize(trace, sent, due, a, a.label, {'load_s': round(load_s, 2), 'threads': a.threads, **devices(d),
                                             'wall_s': round(wall, 1), 'summary': summary})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--host', default='127.0.0.1', help='listen address (default loopback: reach it through ssh -L)')
    s.add_argument('--port', type=int, default=47000)
    for p in (s, sub.add_parser('local')):
        p.add_argument('--stub', action='store_true', help='no models: fixed delays, the transport alone')
        p.add_argument('--threads', type=int, default=2, help='threads per engine, as [local] threads')
        p.add_argument('--max-lag', type=float, default=0.0,
                       help=f'queue policy: drop voice later than this, speed up from 0.5 s (0 = off; the app\'s them -> me: '
                            f'{pipeline.MAX_LAG_S:g})')
    r = sub.add_parser('run')
    r.add_argument('--url', default='ws://127.0.0.1:47000')
    for p in (r, sub.choices['local']):
        p.add_argument('--lang', choices=sorted(PAIRS), default='en', help='tests/data/call-<lang>.ogg, to the other language')
        p.add_argument('--label', default='loop', help='where it ran, for the result row (loop, vm-C, lan, wifi, ...)')
        p.add_argument('--jitters', type=lambda v: [int(x) for x in v.split(',')], default=list(JITTERS_MS),
                       help='player hold of a clause\'s first chunk to replay, ms (comma list)')
        p.add_argument('--out-dir', default=str(ROOT.parent / 'out' / 'remote_rt'))
    r.set_defaults(stub=None, max_lag=None)
    a = ap.parse_args()
    {'serve': serve, 'run': run, 'local': local}[a.cmd](a)


if __name__ == '__main__':
    main()

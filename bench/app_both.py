"""Both directions at once in the running app (stage 2), through its Linux devices: a fake call app plays
the other side's recording (the app moves its stream into "Peresmeshnik Call" when it appears), a virtual
microphone plays mine; the app's headphones and "Peresmeshnik Mic" are recorded back. CPU and memory of
the app's process tree (the app, Piper, pacat/parec) while both talk.

    python bench/app_both.py --me cloud [--overlap 15]    # Linux, PipeWire (pactl), Qt offscreen; models fetched
    python bench/app_both.py --files --me local --voice me   # no sound server (the GPU box): app.Side from the files

`--overlap`: my recording starts this many seconds after theirs (30 s each: 0 = both at once, 30 = turns).
While I talk, their translated voice is cut (barge-in), so "dropped" counts those clauses.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
import peresmeshnik  # noqa: E402, F401  first: it pins BLAS threads before numpy loads
import numpy as np  # noqa: E402
# isort: on
import psutil  # noqa: E402

from common import record  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'tests' / 'data'
SR = 16000


def pactl(*a):
    return subprocess.run(['pactl', *a], check=True, capture_output=True, text=True).stdout.strip()


def pcm_of(path):
    from peresmeshnik.pipeline import read_audio
    return read_audio(str(path))


def play(pcm, *args):
    p = subprocess.Popen(['pacat', '--playback', '--raw', '--format=float32le', f'--rate={SR}', '--channels=1', '--latency-msec=40',
                          *args], stdin=subprocess.PIPE)

    def feed():  # the pipe blocks: real-time pace
        p.stdin.write(pcm.astype('<f4').tobytes())
        p.stdin.close()
    threading.Thread(target=feed, daemon=True).start()
    return p


class Rec:
    def __init__(self, device):
        self.p = subprocess.Popen(['parec', f'--device={device}', '--raw', '--format=float32le', f'--rate={SR}', '--channels=1',
                                   '--latency-msec=20', '--property=node.dont-reconnect=true'], stdout=subprocess.PIPE)
        self.chunks = []
        threading.Thread(target=lambda: self.chunks.extend(iter(lambda: self.p.stdout.read(6400), b'')), daemon=True).start()

    def speech_s(self):
        self.p.terminate()
        self.p.wait()
        x = np.frombuffer(b''.join(self.chunks), '<f4')
        frames = x[:len(x) // 320 * 320].reshape(-1, 320)  # 20 ms
        return round(float((np.sqrt((frames ** 2).mean(axis=1)) > 0.01).sum() * 0.02), 1)


def tree(p):
    try:
        return [p, *p.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def cpu(p):
    total = 0.0
    for q in tree(p):
        try:
            t = q.cpu_times()
            total += t.user + t.system
        except psutil.NoSuchProcess:
            pass
    return total


class Out:
    """Headphones / virtual mic stand-in: counts the speech seconds."""
    def __init__(self):
        self.s = 0.0

    def play(self, pcm, sr):
        self.s += len(pcm) / sr

    def flush(self):
        pass

    def stop(self):
        pass


class Overlay:
    def show(self, c):
        pass

    def hide(self):
        pass


def files(a):
    """The app's two sides (app.Side, the barge-in wiring as in app.Controller) fed from the recordings."""
    from peresmeshnik import config
    from peresmeshnik.app import FileIn, Side
    s = config.Settings()
    s.me_to_them, s.local.voice = a.me, a.voice
    proc = psutil.Process()
    t0 = time.monotonic()
    them = Side('them', s, FileIn(str(DATA / 'call-en.ogg')), Out(), Overlay()) if 'them' in a.sides else None
    me = Side('me', s, FileIn(str(DATA / 'call-ru.ogg')), Out(), Overlay()) if 'me' in a.sides else None
    if them and me:
        me.on_talk = them.interrupt
    warm, idle_rss = time.monotonic() - t0, sum(q.memory_info().rss for q in tree(proc)) / 2**20
    c0, w0, rss_tree, started = cpu(proc), time.monotonic(), 0, False
    if them:
        them.start()
    end = w0 + a.overlap + 30 + 10.0
    while time.monotonic() < end:
        if not started and time.monotonic() - w0 >= a.overlap:
            if me:
                me.start()
            started = True
        rss_tree = max(rss_tree, sum(q.memory_info().rss for q in tree(proc)))
        time.sleep(0.5)
    wall, used = time.monotonic() - w0, cpu(proc) - c0
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    for side in (them, me):
        if side:
            side.stop()
    audio = 30.0 * len(a.sides.split(','))
    name = ' + '.join({'them': 'them local', 'me': f'me {a.me}'}[x] for x in a.sides.split(','))
    record('app', f'{name} (app.Side, files)', mode='both', overlap_s=a.overlap, audio_s=audio, wall_s=round(wall, 1),
           warm_up_s=round(warm, 1), cpu_s=round(used, 1), cpu_s_per_audio_s=round(used / audio, 3), cpu_cores_mean=round(used / wall, 2),
           rss_idle_mb=round(idle_rss), rss_tree_peak_mb=round(rss_tree / 2**20), gpu_mem_mb=gpu.strip() or None,
           headphones_speech_s=round(them.out.s, 1) if them else None, virtual_mic_speech_s=round(me.out.s, 1) if me else None,
           dropped=int(them.directions[0][0].stats['dropped']) if them else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--me', choices=['local', 'cloud'], default='cloud')
    ap.add_argument('--overlap', type=float, default=15.0, help='my recording starts this many seconds after theirs')
    ap.add_argument('--files', action='store_true', help='no devices: the two sides in this process, fed from the files')
    ap.add_argument('--voice', default='me', help='--files --me local: the enrolled voice')
    ap.add_argument('--sides', default='them,me', help='--files: them, me or both')
    a = ap.parse_args()
    if a.files:
        return files(a)
    mods = []
    for line in pactl('list', 'short', 'modules').splitlines():
        if 'bench_' in line or 'peresmeshnik_' in line:
            pactl('unload-module', line.split()[0])
    before = pactl('get-default-sink'), pactl('get-default-source')
    mods.append(pactl('load-module', 'module-null-sink', 'sink_name=bench_speakers'))
    mods.append(pactl('load-module', 'module-null-sink', 'sink_name=bench_headphones'))
    mods.append(pactl('load-module', 'module-null-sink', 'sink_name=bench_mic_feed', 'channels=1'))
    mods.append(pactl('load-module', 'module-remap-source', 'master=bench_mic_feed.monitor', 'source_name=bench_mic', 'channels=1'))
    pactl('set-default-sink', 'bench_speakers')
    pactl('set-default-source', 'bench_mic')
    tmp = Path(tempfile.mkdtemp(prefix='pm-bench-', dir=os.environ.get('BENCH_TMP')))  # settings and the app's log
    cfg = tmp / 'settings.toml'
    cfg.write_text(f'them_to_me = "local"\nme_to_them = "{a.me}"\n[ui]\nheadphones = "bench_headphones"\nmic = "bench_mic"\n'
                   'call_app = "pm-bench-call"\n')
    env = {**os.environ, 'PERESMESHNIK_CONFIG': str(cfg), 'QT_QPA_PLATFORM': os.environ.get('QT_QPA_PLATFORM', 'offscreen')}
    err = (tmp / 'app.err').open('w')
    app = subprocess.Popen([sys.executable, '-m', 'peresmeshnik', 'app', '--no-wizard', '--on', '-v'], env=env, stdout=subprocess.PIPE,
                           stderr=err, text=True)
    lines = []
    threading.Thread(target=lambda: lines.extend(app.stdout), daemon=True).start()
    try:
        t0 = time.monotonic()
        while not any(x.startswith('translating') for x in lines):
            if app.poll() is not None or time.monotonic() - t0 > 180:
                raise SystemExit(f'the app did not start translating: {lines}\n{(tmp / "app.err").read_text()[-3000:]}')
            time.sleep(0.2)
        warm = time.monotonic() - t0
        proc = psutil.Process(app.pid)
        rss_app = rss_tree = 0
        idle_rss = sum(q.memory_info().rss for q in tree(proc)) / 2**20
        heard, mic = Rec('bench_headphones.monitor'), Rec('peresmeshnik_mic')
        them, me = pcm_of(DATA / 'call-en.ogg'), pcm_of(DATA / 'call-ru.ogg')
        call_sink = next(x.split('\t')[0] for x in pactl('list', 'short', 'sinks').splitlines() if '\tperesmeshnik_call\t' in x)
        call_sink = int(call_sink)
        c0, w0 = cpu(proc), time.monotonic()
        call = play(them, '--property=application.name=Bench call', '--property=application.process.binary=pm-bench-call')
        mine, moved_after = None, None
        end = w0 + a.overlap + len(me) / SR + 8.0  # + the last clauses' translation
        while time.monotonic() < end:
            if mine is None and time.monotonic() - w0 >= a.overlap:
                mine = play(me, '--device=bench_mic_feed')
            rss_app = max(rss_app, proc.memory_info().rss)
            rss_tree = max(rss_tree, sum(q.memory_info().rss for q in tree(proc)))
            if moved_after is None and any(s['properties'].get('application.process.binary') == 'pm-bench-call' and
                                           s['sink'] == call_sink for s in json.loads(pactl('-f', 'json', 'list', 'sink-inputs'))):
                moved_after = round(time.monotonic() - w0, 1)
            time.sleep(0.5)
        wall, used = time.monotonic() - w0, cpu(proc) - c0
        heard_s, mic_s = heard.speech_s(), mic.speech_s()
        call.wait(5)
        mine.wait(5)
    finally:
        app.terminate()
        code = app.wait(30)
        for m in reversed(mods):
            subprocess.run(['pactl', 'unload-module', m], capture_output=True)
        subprocess.run(['pactl', 'set-default-sink', before[0]], capture_output=True)
        subprocess.run(['pactl', 'set-default-source', before[1]], capture_output=True)
    log = (tmp / 'app.err').read_text()
    rows = [json.loads(x.split(' peresmeshnik.latency ', 1)[1]) for x in log.splitlines() if ' peresmeshnik.latency {' in x]
    audio = (len(them) + len(me)) / SR
    record('app', f'them local + me {a.me}', mode='both', overlap_s=a.overlap, audio_s=round(audio, 1), wall_s=round(wall, 1),
           warm_up_s=round(warm, 1), cpu_s=round(used, 1), cpu_s_per_audio_s=round(used / audio, 3), cpu_cores_mean=round(used / wall, 2),
           rss_idle_mb=round(idle_rss), rss_app_peak_mb=round(rss_app / 2**20), rss_tree_peak_mb=round(rss_tree / 2**20),
           clauses=len(rows), dropped_lag=sum(isinstance(r.get('dropped'), float) for r in rows),
           dropped_barge_in=sum(r.get('dropped') == 'barge-in' for r in rows), headphones_speech_s=heard_s, virtual_mic_speech_s=mic_s,
           call_moved_after_s=moved_after, app_exit=code,
           states=[x.strip() for x in lines][:6])


if __name__ == '__main__':
    main()

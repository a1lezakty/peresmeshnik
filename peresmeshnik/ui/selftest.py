"""Loopback self-test of the audio routing (Linux: PipeWire / PulseAudio; macOS: Core Audio), no call needed.

A short chirp goes out through each of our paths and is recorded back from the other end:
- call capture: chirp -> "Peresmeshnik Call" sink -> its monitor, read by the app's capture (S1);
- virtual mic: chirp -> the virtual mic (S2) -> recorded from "Peresmeshnik Mic", what the call app hears;
- headphones: chirp -> the chosen output sink -> its monitor;
- microphone: 3 s from the chosen input, level only (say something).
A path passes when the chirp is found by cross-correlation, clearly above the noise. macOS: the virtual mic is
BlackHole 2ch (chirp into its output, recorded from its input, as the call app does) and the microphone.

    python -m peresmeshnik.ui.selftest [--headphones SINK] [--mic SOURCE]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

MACOS = sys.platform == 'darwin'
SR = 48_000
MIN_SCORE = 0.5  # normalised correlation of the found chirp; noise and silence stay < 0.1
MIN_MIC_DB = -50.0


def pactl(*args: str) -> str:
    return subprocess.run(['pactl', *args], check=True, capture_output=True, text=True, timeout=10).stdout


def devices(kind: str) -> list[tuple[str, str]]:
    """[(name, description)] of 'sinks' or 'sources' (monitors of sinks left out); default first.
    macOS: [(Core Audio UID, name)] of outputs or inputs, BlackHole 2ch left out (it is ours: the virtual mic); as inputs
    no virtual device (BlackHole, Zoom's, Teams') and no aggregate either: none of them is the user's voice."""
    if MACOS:
        from peresmeshnik.audio import macos
        side, default = ('outputs', 'default_output') if kind == 'sinks' else ('inputs', 'default_input')
        out = [d for d in macos.devices() if d[side] and not macos.is_blackhole(d['uid']) and not macos.is_blackhole(d['name'])
               and (kind == 'sinks' or d['transport'] not in ('virt', 'grup') or d['uid'] == macos.stand_in_mic())]
        return [(d['uid'], d['name']) for d in sorted(out, key=lambda d: not d[default])]
    items = json.loads(pactl('-f', 'json', 'list', kind) or '[]')
    default = pactl('get-default-sink' if kind == 'sinks' else 'get-default-source').strip()
    out = [(d['name'], d.get('description') or d['name']) for d in items if not d['name'].endswith('.monitor')]
    return sorted(out, key=lambda d: d[0] != default)


def chirp(sr: int = SR, seconds: float = 0.4) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    f0, f1 = 400.0, 4000.0
    x = np.sin(2 * np.pi * f0 * seconds / np.log(f1 / f0) * (np.exp(t / seconds * np.log(f1 / f0)) - 1))
    fade = np.minimum(1, np.minimum(t, seconds - t) / 0.01)
    return (0.25 * x * fade).astype(np.float32)  # -12 dBFS peak


def find(rec: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(score, sample index) of ref inside rec: peak of the normalised cross-correlation (1.0 = exact copy)."""
    if len(rec) < len(ref):
        return 0.0, 0.0
    n = 1 << (len(rec) + len(ref)).bit_length()
    c = np.fft.irfft(np.fft.rfft(rec, n) * np.conj(np.fft.rfft(ref, n)), n)[:len(rec) - len(ref) + 1]
    floor = 1e-4 ** 2 * len(ref)  # windows below -80 dBFS RMS count as silence, not as a perfect match
    energy = np.sqrt(np.maximum(np.convolve(rec.astype(np.float64) ** 2, np.ones(len(ref)), 'valid'), floor)) * np.linalg.norm(ref)
    score = c / energy
    i = int(np.argmax(score))
    return float(score[i]), i


def play(sink: str, pcm: np.ndarray, sr: int) -> None:
    """Play and wait until played (+ 0.25 s of silence: pacat drops a tail at EOF)."""
    data = np.concatenate([pcm, np.zeros(sr // 4, np.float32)]).astype('<f4').tobytes()
    subprocess.run(['pacat', f'--device={sink}', '--raw', '--format=float32le', f'--rate={sr}', '--channels=1', '--latency-msec=20',
                    '--property=node.dont-reconnect=true'], input=data, check=True, timeout=10)


class Recorder:
    """parec of one source at SR (macOS: the helper's record of a device, '' = the default input), in a thread blocked
    on the pipe."""

    def __init__(self, source: str, rate: int = SR):
        self.rate, self.chunks = rate, []
        if MACOS:
            from peresmeshnik.audio import macos
            self.p = macos.spawn('record', '--from', source, '--rate', str(rate))
        else:
            self.p = subprocess.Popen(['parec', f'--device={source}', '--raw', '--format=float32le', f'--rate={rate}', '--channels=1',
                                       '--latency-msec=20', '--property=node.dont-reconnect=true'], stdout=subprocess.PIPE)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        while b := self.p.stdout.read(4096):
            self.chunks.append(np.frombuffer(b[:len(b) // 4 * 4], '<f4'))

    def stop(self) -> np.ndarray:
        self.p.terminate()
        try:
            self.p.wait(5)
        except subprocess.TimeoutExpired:  # the helper once hung stopping (1 in ~400 on the runner): a reader, kill is safe
            self.p.kill()
            self.p.wait()
        self.t.join(5)
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def _judge(name: str, rec: np.ndarray, ref: np.ndarray, sr: int) -> Result:
    score, at = find(rec, ref)
    level = 20 * np.log10(np.sqrt(np.mean(rec[at:at + len(ref)] ** 2)) + 1e-9) if len(rec) else -120
    ok = bool(score >= MIN_SCORE)
    return Result(name, ok, f'chirp {"found" if ok else "not found"}: match {score:.2f}, level {level:.0f} dBFS, '
                            f'{len(rec) / sr:.1f} s recorded')


def sinks() -> list[str]:
    return [line.split('\t')[1] for line in pactl('list', 'short', 'sinks').splitlines() if '\t' in line]


def check_call_capture() -> Result:
    """Through the app's call sink while it runs (never a second one: that would drop the call's stream), else
    a temporary one; recorded the way the app records it."""
    from peresmeshnik.app import CallSink
    from peresmeshnik.audio.capture_pipewire import CALL_SINK, PipeWireCapture
    from peresmeshnik.engines import SAMPLE_RATE
    own = None if CALL_SINK in sinks() else CallSink()
    frames: list[np.ndarray] = []
    cap = PipeWireCapture(CALL_SINK, make_sink=False)
    ref = chirp(SAMPLE_RATE)
    try:
        cap.start(frames.append)
        time.sleep(0.3)
        play(CALL_SINK, ref, SAMPLE_RATE)
        time.sleep(0.4)
    finally:
        cap.stop()
        if own:
            own.remove()
    return _judge('call capture', np.concatenate(frames) if frames else np.zeros(0, np.float32), ref, SAMPLE_RATE)


def check_virtual_mic(vmic=None) -> Result:
    """Through the app's virtual mic (`vmic`, or the device while the app runs), else a temporary one."""
    if MACOS:
        return _check_blackhole(vmic)
    from peresmeshnik.audio.virtual_mic_linux import SINK, SOURCE, VirtualMic, exists
    own = None if vmic is not None or exists() else VirtualMic()  # a new VirtualMic would replace the app's
    ref = chirp()
    try:
        rec = Recorder(SOURCE)
        try:
            time.sleep(0.3)
            if vmic or own:
                (vmic or own).play(ref, SR)
                (vmic or own).drain()
            else:
                play(SINK, ref, SR)
            time.sleep(0.4)
        finally:
            got = rec.stop()  # also when playing failed: no parec is left recording
    finally:
        if own:
            own.remove()
    return _judge('virtual mic', got, ref, SR)


def _check_blackhole(vmic=None) -> Result:
    """macOS: a chirp into BlackHole 2ch's output, recorded from its input like the call app (none to make or remove)."""
    from peresmeshnik.audio import macos
    from peresmeshnik.audio.virtual_mic_macos import VirtualMic
    out, ref = vmic or VirtualMic(), chirp()
    rec = Recorder(out.device['uid'])
    try:
        time.sleep(0.3)
        out.play(ref, SR)
        out.drain()
        time.sleep(0.4)
    finally:
        got = rec.stop()
        if vmic is None:
            out.remove()
    if rec.p.returncode == 6:
        return Result('virtual mic', False, macos.NO_PERMISSION)
    return _judge(f'virtual mic ({macos.BLACKHOLE})', got, ref, SR)


def check_headphones(sink: str) -> Result:
    from peresmeshnik.app import not_ours
    not_ours(sink, 'output')  # the chirp would go into the call
    rec = Recorder(f'{sink}.monitor')
    ref = chirp()
    try:
        time.sleep(0.3)
        play(sink, ref, SR)
        time.sleep(0.4)
    finally:
        got = rec.stop()
    return _judge(f'headphones ({sink})', got, ref, SR)


def check_mic(source: str, seconds: float = 3.0) -> Result:
    rec = Recorder(source, 16_000)
    try:
        time.sleep(seconds)
    finally:
        x = rec.stop()
    if MACOS and rec.p.returncode == 6:
        from peresmeshnik.audio import macos
        return Result('microphone', False, macos.NO_PERMISSION)
    db = 20 * np.log10(np.max(np.sqrt(np.convolve(x ** 2, np.ones(1600) / 1600, 'valid'))) + 1e-9) if len(x) > 1600 else -120.0
    ok = bool(db >= MIN_MIC_DB)
    return Result(f'microphone ({source or "the default input"})', ok,
                  f'loudest 0.1 s at {db:.0f} dBFS' + ('' if ok else ': say something, or check the input'))


def _mac_mic(mic: str | None) -> str:
    from peresmeshnik.audio import macos
    return macos.not_blackhole(mic or '')  # '' = the default input (never BlackHole: the helper skips virtual devices)


def run_all(headphones: str | None = None, mic: str | None = None, vmic=None, report=lambda r: None) -> list[Result]:
    """Every check, each reported as soon as it is done; a failing check does not stop the others."""
    if MACOS:  # ponytail: no call capture or headphones check here yet (S1: the process tap, the output device)
        checks = [('virtual mic', lambda: check_virtual_mic(vmic)), ('microphone', lambda: check_mic(_mac_mic(mic)))]
    else:
        checks = [('call capture', check_call_capture), ('virtual mic', lambda: check_virtual_mic(vmic)),
                  ('headphones', lambda: check_headphones(headphones or pactl('get-default-sink').strip())),
                  ('microphone', lambda: check_mic(mic or '@DEFAULT_SOURCE@'))]
    results = []
    for name, check in checks:
        try:
            r = check()
        except Exception as e:  # missing tools, no sound server, a device that went away, our own device as the default
            r = Result(name, False, f'error: {e}')
        results.append(r)
        report(r)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m peresmeshnik.ui.selftest', description='Loopback check of the audio routing.')
    ap.add_argument('--headphones', help='output sink (default: the default sink)')
    ap.add_argument('--mic', help='input source (default: the default source)')
    a = ap.parse_args(argv)
    rs = run_all(a.headphones, a.mic, report=lambda r: print(f'{"OK  " if r.ok else "FAIL"} {r.name}: {r.detail}', flush=True))
    return 0 if all(r.ok for r in rs) else 1


if __name__ == '__main__':
    raise SystemExit(main())

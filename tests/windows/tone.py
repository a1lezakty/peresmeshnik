"""A process that plays a continuous 440 Hz tone on an output device until killed (the Windows tests' tone source,
the counterpart of tests/macos/tone.swift). Its pid is what a process loopback captures.

    python tests/windows/tone.py DEVICE [HZ]   DEVICE: a WASAPI output name ('' = the default output)
    python tests/windows/tone.py DEVICE FILE   3 s of speech from FILE (from 1 s in) over and over, the same samples
"""
import sys
import time

import numpy as np
import sounddevice as sd


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else ''
    what = sys.argv[2] if len(sys.argv) > 2 else '440'
    hz = float(what) if what.replace('.', '', 1).isdigit() else 0.0
    if name:
        from peresmeshnik.audio import winaudio
        device = winaudio.device_index(name, output=True)  # pinned to WASAPI (a VB-Cable name repeats across host APIs)
    else:
        device = sd.default.device[1]
    info = sd.query_devices(device, 'output')
    sr = int(info['default_samplerate'])
    phase = 0.0
    step = 2 * np.pi * hz / sr
    clip, pos = None, 0
    if not hz:  # a speech loop: periodic, so a capture of it can be lined up with another one sample for sample
        import soundfile as sf
        x, fsr = sf.read(what, dtype='float32', always_2d=True)
        x = x.mean(1)[fsr:4 * fsr]
        clip = np.interp(np.arange(3 * sr) * fsr / sr, np.arange(len(x)), x).astype(np.float32)
        clip *= 0.5 / max(float(np.abs(clip).max()), 1e-6)

    def cb(outdata, frames, time_info, status):
        nonlocal phase, pos
        if clip is not None:
            idx = (pos + np.arange(frames)) % len(clip)
            pos = (pos + frames) % len(clip)
            outdata[:] = clip[idx][:, None]
            return
        t = phase + step * np.arange(frames)
        outdata[:, 0] = (0.3 * np.sin(t)).astype(np.float32)
        if outdata.shape[1] > 1:
            outdata[:, 1] = outdata[:, 0]
        phase = (t[-1] + step) % (2 * np.pi)

    with sd.OutputStream(device=device, channels=min(2, info['max_output_channels']), samplerate=sr,
                         dtype='float32', callback=cb):
        print(f'tone {what} on {info["name"]} at {sr} Hz', flush=True)
        while True:
            time.sleep(1.0)


if __name__ == '__main__':
    sys.exit(main())

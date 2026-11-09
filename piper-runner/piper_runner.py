# SPDX-License-Identifier: GPL-3.0-or-later
"""Piper TTS runner: a separate program under GPL-3.0 (it imports piper-tts, GPL-3.0).

Peresmeshnik never imports it: the app starts this script in its own process and talks to it
over stdin/stdout. Unlike the upstream Piper CLI it fixes the onnxruntime thread count and lets
the pool sleep between sentences.

    uv run --no-project --with piper-tts==1.8.0 python piper_runner.py VOICE.onnx [--threads 2]

Protocol: on start it writes the sample rate as uint32 LE. Then per request, one JSON line on
stdin {"text": "...", "speed": 1.0} (speed optional: > 1 speaks faster); per sentence a uint32 LE
byte count plus float32 LE mono PCM; 0 ends it.
"""
import argparse
import json
import os
import struct
import sys


def usable_stderr() -> None:
    """The app gives the runner a log file as stderr; started from a windowed program without one, stderr can be a pipe
    nobody reads (or None), and the first warning or error printed there would raise OSError and kill the runner or
    hide the reason. Then it goes nowhere instead."""
    try:
        sys.stderr.write('piper-runner: started\n')
        sys.stderr.flush()
    except AttributeError:  # no stderr at all
        sys.stderr = open(os.devnull, 'w')
    except (OSError, ValueError):  # its descriptor (and what the C libraries write there) to nowhere
        try:
            fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(fd, sys.stderr.fileno())
            os.close(fd)
            sys.stderr.flush()
        except (OSError, ValueError):
            sys.stderr = open(os.devnull, 'w')


def main():
    import onnxruntime as rt
    from piper import PiperVoice
    from piper.config import PiperConfig, SynthesisConfig
    ap = argparse.ArgumentParser()
    ap.add_argument('voice', help='path to a Piper .onnx voice; its .onnx.json sits next to it')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    so = rt.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = a.threads, 1
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    with open(a.voice + '.json', encoding='utf-8') as f:
        config = PiperConfig.from_dict(json.load(f))
    voice = PiperVoice(config=config, session=rt.InferenceSession(a.voice, so, providers=['CPUExecutionProvider']))
    out = sys.stdout.buffer
    out.write(struct.pack('<I', config.sample_rate))
    out.flush()
    for line in sys.stdin:
        req = json.loads(line)
        syn = SynthesisConfig(length_scale=config.length_scale / req.get('speed', 1.0))
        for chunk in voice.synthesize(req['text'], syn):
            pcm = chunk.audio_float_array.astype('<f4').tobytes()
            out.write(struct.pack('<I', len(pcm)) + pcm)
            out.flush()
        out.write(struct.pack('<I', 0))
        out.flush()


if __name__ == '__main__':
    usable_stderr()
    main()

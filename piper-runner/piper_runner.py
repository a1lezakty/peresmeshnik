# SPDX-License-Identifier: GPL-3.0-or-later
"""Piper TTS runner: a separate program under GPL-3.0 (it imports piper-tts, GPL-3.0).

Peresmeshnik never imports it: the app starts this script in its own process and talks to it
over stdin/stdout. Unlike the upstream Piper CLI it fixes the onnxruntime thread count and lets
the pool sleep between sentences.

    uv run --no-project --with piper-tts==1.8.0 python piper_runner.py VOICE.onnx [--threads 2]
    piper-runner --self-check   # a built runner has what its voices need (the release job): exit code 0

Protocol: on start it writes the sample rate as uint32 LE. Then per request, one JSON line on
stdin {"text": "...", "speed": 1.0} (speed optional: > 1 speaks faster); per sentence a uint32 LE
byte count plus float32 LE mono PCM; 0 ends it.
"""
import argparse
import functools
import json
import os
import struct
import sys
import traceback


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


def no_sudachi() -> None:
    """Japanese voices read kanji by OpenJTalk's dictionary alone: pyopenjtalk-plus would fix some readings with
    Sudachi, whose 192 MB dictionary the builds leave out (FLEURS ja test: 6 of 321 sentences, 0.05 % of the
    phonemes differ). Without it such a sentence would come out silent."""
    import pyopenjtalk
    pyopenjtalk.run_frontend = functools.partial(pyopenjtalk.run_frontend, use_sudachi_kanji_yomi=False)


def main():
    """Errors end it with exit code 1 and one line on stderr: their type and where, not their message, which can
    quote what was said (an espeak or phonemizer error); the app keeps stderr in a log file. Also the frozen
    Windows entry (packaging/windows/piper_entry.py) runs this."""
    usable_stderr()
    try:
        serve()
    except Exception as e:
        where = traceback.extract_tb(e.__traceback__)[-1]
        sys.stderr.write(f'piper-runner: {type(e).__name__} at {os.path.basename(where.filename)}:{where.lineno}\n')
        sys.stderr.flush()
        sys.exit(1)


def serve():
    import onnxruntime as rt
    from piper import PiperVoice
    from piper.config import PiperConfig, SynthesisConfig
    ap = argparse.ArgumentParser()
    ap.add_argument('voice', nargs='?', help='path to a Piper .onnx voice; its .onnx.json sits next to it')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--self-check', action='store_true', help="phonemize Japanese (OpenJTalk, Piper's [ja] extra) and exit")
    a = ap.parse_args()
    if a.self_check:
        from piper.phonemize_japanese import JapanesePhonemizer
        no_sudachi()
        assert JapanesePhonemizer().phonemize('今日は風が強い。'), 'no phonemes'  # 今日: a kanji Sudachi would read
        sys.stderr.write('piper-runner: self-check ok\n')
        return
    if not a.voice:
        ap.error('a voice is needed')
    so = rt.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = a.threads, 1
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    with open(a.voice + '.json', encoding='utf-8') as f:
        raw = json.load(f)
    if raw.get('phoneme_type') == 'japanese':
        no_sudachi()
    config = PiperConfig.from_dict(raw)
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
    main()

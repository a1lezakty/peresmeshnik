"""Voice-clone TTS for "me -> them": XTTS-v2 behind `Tts` and `VoiceCloner`.

Part of the optional voice pack (torch + coqui-tts; weights under the noncommercial Coqui Public
Model License, docs/MODELS.md): this module imports neither; `make_clone` picks where XTTS runs. With the voice
pack installed (Windows, NVIDIA) it is a process of its own on the pack's Python (VoicePackClone, voice-runner/);
in a source install with torch and coqui-tts it runs in this process (XttsClone); the model code of both is
local/xtts.py. On NVIDIA it streams faster than real time (docs/BENCH.md); on CPU it is slower than real time,
so the direction falls behind and its queue policy has to drop clauses. Nothing here pretends otherwise.

    python -m peresmeshnik.local.clone --voice ~/.local/share/Peresmeshnik/voices/me --model-dir DIR --lang en "Hello"

`model_dir` holds the XTTS-v2 files (config.json, model.pth, vocab.json, speakers_xtts.pth): the
registry's `xtts-v2` folder (`peresmeshnik models fetch xtts-v2`).
"""
from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import shlex
import struct
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from peresmeshnik import config
from peresmeshnik import voice as profiles
from peresmeshnik.engines import Voice
from peresmeshnik.local import voicepack, xtts
from peresmeshnik.local.xtts import SR, budget_samples, cap_pauses, guard, split_text  # noqa: F401  (re-exported)

RUNNER = voicepack.runner_dir() / 'voice_runner.py'  # ships with the app: it always matches it
ERR = 0xFFFFFFFF  # voice_runner.py: a synth reply that ends with an error


log = logging.getLogger(__name__)

class CloneUnavailable(RuntimeError):
    """The clone cannot speak here: no voice pack, no CUDA, the runner did not start or died. The app falls back."""


class XttsClone(xtts.Engine):
    """XTTS-v2 in this process: a source install with torch and coqui-tts in its own environment."""
    _errors = 0  # clauses in a row it could not say: MAX_ERRORS of them end the clone, as VoicePackClone (L-43)

    def synth(self, text: str, lang: str, voice=None):
        try:
            yield from super().synth(text, lang, voice)
        except Exception as e:  # CUDA out of memory on a long clause: this clause is lost, not the clone
            self._errors += 1
            if self._errors >= VoicePackClone.MAX_ERRORS:
                raise
            log.warning('the clone could not say a clause, it is left out: %s', e)
            return
        self._errors = 0

    def enroll(self, pcm: np.ndarray, sample_rate: int, transcript: str | None = None, name: str = 'me',
               lang: str = 'ru', root: Path | None = None) -> Voice:
        v = profiles.save(pcm, sample_rate, name, lang, transcript, root)
        self.forget(v)  # a new recording of the same profile: condition again
        self.latents(v)
        return Voice(name=v.name, engine='xtts', path=v.path, meta=v.meta)


def _b64(pcm: np.ndarray) -> str:
    return base64.b64encode(np.asarray(pcm, '<f4').tobytes()).decode()


def voice_cmd() -> list[str] | None:
    """How to start the voice pack's runner: $PERESMESHNIK_VOICE_CMD, else the installed pack of this app version in
    <data dir>/voice-pack (local/voicepack.py) with its own python.exe; None: no such pack here."""
    if cmd := os.environ.get('PERESMESHNIK_VOICE_CMD'):
        if os.name != 'nt':
            return shlex.split(cmd)
        # Windows: backslashes stay (posix=False), and so do the quotes around "C:\Program Files\...": one pair goes
        return [t[1:-1] if len(t) > 1 and t[0] == t[-1] == '"' else t for t in shlex.split(cmd, posix=False)]
    if voicepack.state() == 'ready':
        return [str(voicepack.python()), str(RUNNER)]
    return None


def make_clone(model_dir: Path | str, threads: int = 2):
    """The clone engine of this install (the app, voice.py and the wizard): the voice pack's runner where there is one,
    else XTTS in this process. Raises CloneUnavailable with the reason when neither can speak."""
    if cmd := voice_cmd():
        return VoicePackClone(model_dir, threads, cmd)
    if (st := voicepack.state()) != 'none':  # never an incompatible or half-installed pack
        raise CloneUnavailable({'outdated': 'the voice pack was installed for another version of the app',
                                'partial': 'the voice pack install did not finish'}[st] + ': install it again in the setup '
                               'wizard (Your voice)')
    try:
        return XttsClone(model_dir, threads=threads)
    except ImportError as e:
        if sys.platform == 'win32':  # the app's own place for it: the pack is the wizard's to install
            raise CloneUnavailable('the voice pack is not installed: install it in the setup wizard (Your voice)') from e
        raise CloneUnavailable(f'the voice clone needs the voice pack (torch, coqui-tts): {e}') from e


def make_engine(model_id: str, model_dir: Path | str, threads: int = 2):
    """The clone engine of the registry's `model_id` (`languages.clone_engine` picks it): Pocket TTS in this process on
    onnxruntime (local/pocket.py), XTTS-v2 through the voice pack (`make_clone`). Raises CloneUnavailable with the reason
    when it cannot speak here."""
    if model_id.startswith('pocket-tts'):
        from peresmeshnik.local.pocket import PocketTts
        try:
            # int8: 1.5x faster and 0.15 GB smaller than fp32 at the same similarity and WER (BENCH stage 8)
            return PocketTts(model_dir, threads=threads, model_id=model_id, int8=True)
        except Exception as e:  # a broken weights file; onnxruntime's own errors (Fail, InvalidGraph, out of memory)
            raise CloneUnavailable(f'Pocket TTS did not start: {e}') from e
    if model_id == 'qwen3-tts':  # the voice pack's runner with Qwen3, Chatterbox taking over a voice that breaks down
        from peresmeshnik import models
        if not (cmd := voice_cmd()):
            raise CloneUnavailable('Qwen3-TTS needs the voice pack: install it in the setup wizard (Your voice)')
        return VoicePackClone(model_dir, threads, [*cmd, '--kind', 'qwen3', '--fallback-dir',
                                                   str(models.path(models.get(model_id).fallback))])
    if model_id != 'xtts-v2':  # chatterbox-mtl: only Qwen3's runner speaks with it, never an engine of its own
        raise ValueError(f'{model_id} is not a clone engine')
    return make_clone(model_dir, threads)


class VoicePackClone:
    """XttsClone's interface over the voice pack's runner (voice-runner/voice_runner.py: the protocol is there).
    `info`: what the runner got (device, torch, VRAM). A runner that fails at start raises CloneUnavailable; one that
    dies later, or sends no reply chunk for `reply_s`, makes the call raise RuntimeError (a silent one is killed), so
    the direction ends with a message instead of hanging. Its stderr: `log` (<data dir>/logs/voice-runner.log, new
    at each start)."""

    MAX_ERRORS = 3  # clauses in a row the runner (alive) could not say: then the clone is given up (L-43)

    def __init__(self, model_dir: Path | str, threads: int = 2, cmd: list[str] | None = None, start_s: float = 120.0,
                 reply_s: float = 30.0, log: Path | None = None):
        self._errors = 0
        self._lock = threading.Lock()  # one request on the pipe at a time: a synth for its whole stream (FIXES addon 1)
        cmd = [*(cmd or voice_cmd() or []), '--model-dir', str(model_dir), '--threads', str(threads)]
        self._log, self.reply_s, self._hung = log or config.data_dir() / 'logs' / 'voice-runner.log', reply_s, False
        try:  # a file, not the app's stderr: a windowed .exe has none and the runner's first traceback would kill it
            self._log.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log, 'wb') as err:  # no console window from a windowed app (python.exe is a console program)
                self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as e:
            raise CloneUnavailable(f'the voice pack did not start ({cmd[0]}): {e}') from e
        head: list[bytes] = []
        t = threading.Thread(target=lambda: head.append(self._p.stdout.readline()), daemon=True)
        t.start()
        t.join(start_s)  # loading takes ~10 s; a runner stuck in CUDA init must not hold the app
        line = head[0] if head else None
        try:
            info = json.loads(line) if line and line.strip() else {}
        except ValueError:  # something else wrote to its stdout first
            info = {'error': f'not a voice pack runner: {line[:80]!r}'}
        if 'sample_rate' not in info:
            self._p.kill()
            code = self._p.wait()
            why = info.get('error') or (f'exited ({code})' if line is not None else f'no answer in {start_s:g} s')
            tail = [ln.strip() for ln in self._log.read_text('utf-8', 'replace').splitlines() if ln.strip()][-3:]
            if tail and 'error' not in info:  # no reason of its own: what it wrote last
                why += ': ' + ' / '.join(tail)
            raise CloneUnavailable(f'the voice pack did not start: {why}')
        self.sample_rate, self.info, self._id = info.pop('sample_rate'), info, 0

    def _wait(self, read):
        """`read()` within reply_s: a runner that sends nothing (a hung CUDA call) is killed, so the read ends."""
        t = threading.Timer(self.reply_s, self._kill_hung)
        t.daemon = True
        t.start()
        try:
            return read()
        finally:
            t.cancel()

    def _kill_hung(self) -> None:
        self._hung = True
        self._p.kill()

    def _exited(self) -> RuntimeError:
        code = self._p.wait()
        return RuntimeError(f'the voice pack did not answer in {self.reply_s:g} s (killed)' if self._hung
                            else f'the voice pack exited ({code})')

    def _send(self, req: dict) -> int:
        self._id += 1
        try:
            self._p.stdin.write(json.dumps({'id': self._id, **req}).encode() + b'\n')
            self._p.stdin.flush()
        except OSError:
            raise self._exited() from None
        return self._id

    def _read(self, n: int) -> bytes:
        b = self._wait(lambda: self._p.stdout.read(n))
        if len(b) < n:
            raise self._exited()
        return b

    def _u32(self) -> int:
        return struct.unpack('<I', self._read(4))[0]

    def _reply(self) -> dict:
        r = json.loads(self._wait(self._p.stdout.readline) or b'{}')
        if not r:
            raise self._exited()
        if 'error' in r:
            raise RuntimeError(f'the voice pack: {r["error"]}')
        return r

    def synth(self, text: str, lang: str, voice: Voice | None = None) -> Iterator[np.ndarray]:
        """`voice`: a profile, or one with `state` (condition()'s id: a voice of the other person in a call)."""
        if voice is None:
            raise ValueError('the clone engine speaks only in an enrolled voice')
        st = getattr(voice, 'state', None)
        with self._lock:  # the whole stream: condition() from the other person's thread waits for its end
            rid = self._send({'cmd': 'synth', 'text': text, 'lang': lang,
                              **({'state': st} if st is not None else {'voice': str(voice.path)})})
            try:
                while n := self._u32():
                    if n == ERR:
                        r = json.loads(self._wait(self._p.stdout.readline) or b'{}')
                        if not r:  # it died or was killed (hung) right then: the clone's death, said now
                            raise self._exited()
                        self._errors += 1  # it lives on: this clause is lost, not the clone (L-43)
                        if self._errors >= self.MAX_ERRORS:
                            raise RuntimeError(f'the voice pack: {r.get("error")}')
                        log.warning('the voice pack could not say a clause, it is left out: %s', r.get('error'))
                        return
                    yield np.frombuffer(self._read(n), '<f4')
                self._errors = 0
            except GeneratorExit:  # the caller stopped (barge-in): the runner stops at its next chunk; the rest read away
                self._send({'cmd': 'stop', 'id': rid})
                while n := self._u32():
                    if n == ERR:
                        self._wait(self._p.stdout.readline)
                        break
                    self._read(n)
                raise

    def _ask(self, req: dict) -> dict:
        """One request and its reply, alone on the pipe."""
        with self._lock:
            self._send(req)
            return self._reply()

    def latents(self, voice: Voice, fresh: bool = False) -> Path:
        """Conditioning of a profile, computed by the runner and cached as `xtts.pt` next to the recording."""
        self._ask({'cmd': 'latents', 'voice': str(voice.path), 'fresh': fresh})
        return Path(voice.path) / 'xtts.pt'

    def similarity(self, voice: Voice, pcm: np.ndarray, sr: int) -> float:
        return float(self._ask({'cmd': 'similarity', 'voice': str(voice.path), 'pcm': _b64(pcm), 'sr': sr})['similarity'])

    def condition(self, pcm: np.ndarray, sr: int) -> str:
        """A voice from a sample, no profile folder (voices/live.py): the runner's id of its state, for synth() as a
        voice's `state`. It lives in the runner's memory (its last MAX_STATES): gone with this runner, and a synth with
        it then is that clause's error (left out, as any)."""
        return self._ask({'cmd': 'condition', 'pcm': _b64(pcm), 'sr': sr})['state']

    def forget_states(self) -> None:
        """Every voice condition() made goes from the runner's memory (voices/live.py: a new call on warm engines)."""
        self._ask({'cmd': 'forget'})

    def stats(self) -> dict:
        """What the runner sees now: VRAM free and the peak so far (bench/voice_pack.py)."""
        return self._ask({'cmd': 'info'})

    def enroll(self, pcm: np.ndarray, sample_rate: int, transcript: str | None = None, name: str = 'me',
               lang: str = 'ru', root: Path | None = None) -> Voice:
        v = profiles.save(pcm, sample_rate, name, lang, transcript, root)
        self.latents(v, fresh=True)
        return Voice(name=v.name, engine='xtts', path=v.path, meta=v.meta)

    def close(self) -> None:
        with contextlib.suppress(OSError):  # a dead runner: the pipe is broken already
            self._p.stdin.close()
        try:
            self._p.wait(10)
        except subprocess.TimeoutExpired:
            self._p.kill()


def main(argv=None) -> None:
    import argparse
    import time

    import soundfile as sf
    ap = argparse.ArgumentParser(prog='python -m peresmeshnik.local.clone', description='Say a text in an enrolled voice.')
    ap.add_argument('text')
    ap.add_argument('--voice', type=Path, required=True, help='voice profile folder')
    ap.add_argument('--lang', default='en')
    ap.add_argument('--out', type=Path, default=Path('clone.wav'))
    ap.add_argument('--model-dir', type=Path, required=True, help='XTTS-v2 folder')
    ap.add_argument('--device', help='in this process on this device (default: the voice pack where installed)')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(argv)
    eng = XttsClone(a.model_dir, a.device, a.threads) if a.device else make_clone(a.model_dir, a.threads)
    v = profiles.load(a.voice.name, a.voice.parent)
    t0, chunks = time.perf_counter(), []
    for c in eng.synth(a.text, a.lang, v):
        if not chunks:
            print(f'first audio {time.perf_counter() - t0:.2f} s', flush=True)
        chunks.append(c)
    wav = np.concatenate(chunks)
    dt = time.perf_counter() - t0
    sf.write(a.out, wav, eng.sample_rate)
    print(f'{len(wav) / eng.sample_rate:.2f} s of audio in {dt:.2f} s (RTF {dt * eng.sample_rate / len(wav):.2f}) -> {a.out}')


if __name__ == '__main__':
    main()

"""Windows: a real voice pack from the network, tiny (tests/data/pack-tiny.json, voice-runner/lock.py: the embeddable
CPython 3.12.10 from python.org, pip, numpy, soundfile, all by SHA-256), installed by local/voicepack.py like the
wizard does, in a folder outside the ANSI code page (a user profile like C:\\Users\\Иван Петров); then the runner on
that Python, with the stub engine (tests/voice_stub.py), says a phrase through VoicePackClone. Skipped off Windows
unless PERESMESHNIK_REQUIRE_WINDOWS=1 (CI: a skip there would hide a failure)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil
import pytest

from peresmeshnik.engines import Voice
from peresmeshnik.local import voicepack
from peresmeshnik.local.clone import RUNNER, VoicePackClone

TINY = Path(__file__).with_name('data') / 'pack-tiny.json'
STUB = Path(__file__).with_name('voice_stub.py')
pytestmark = pytest.mark.skipif(sys.platform != 'win32' and os.environ.get('PERESMESHNIK_REQUIRE_WINDOWS') != '1',
                                reason='Windows only (PERESMESHNIK_REQUIRE_WINDOWS=1: never skipped)')


def test_a_real_pack_runs_the_runner(tmp_path):
    where, stages = tmp_path / 'Данные Иван' / 'voice-pack', []  # cp1252 has no Cyrillic: the default text encoding fails
    voicepack.install(lambda stage, done, total, text='': stages.append((stage, text)), lock=TINY, where=where)
    assert voicepack.state(TINY, where) == 'ready', stages[-5:]
    pip = [t for stage, t in stages if stage == 'installing']
    assert not any('Logging error' in t for t in pip) and any('данные иван' in t.casefold() for t in pip), pip[:5]
    py = voicepack.python(where)
    out = subprocess.run([str(py), '-c', 'import sys, numpy, soundfile; print(soundfile.__libsndfile_version__); print(sys.prefix)'],
                         capture_output=True, encoding='utf-8', timeout=60, env={**os.environ, 'PYTHONUTF8': '1'})  # its prefix
    assert out.returncode == 0 and Path(out.stdout.splitlines()[-1]).samefile(where), out.stderr  # its packages, its DLL
    (tmp_path / 'me').mkdir()
    p = VoicePackClone(tmp_path / 'xtts-v2', cmd=[str(py), str(RUNNER), '--engine', str(STUB), '--device', 'cpu'],
                       log=where.with_name('logs') / 'voice-runner.log')
    try:
        assert Path(psutil.Process(p._p.pid).exe()).samefile(py) and p.info['device'] == 'cpu'
        x = np.concatenate(list(p.synth('hello', 'en', Voice('me', 'xtts', tmp_path / 'me'))))
        assert len(x) == 3 * 2400 and np.all(x == np.float32(0.05))
    finally:
        p.close()

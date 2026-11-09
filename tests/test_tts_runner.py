"""The Piper runner's stderr, no models: the app gives it a log file (a windowed app has no stderr to pass on), and
the reason it died comes back in the error; the runner itself survives a stderr that is a pipe nobody reads."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner_module():
    spec = importlib.util.spec_from_file_location('piper_runner', ROOT / 'piper-runner' / 'piper_runner.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # piper and onnxruntime are imported only in main()
    return m


def test_the_runner_survives_a_stderr_nobody_reads(monkeypatch):
    r, w = os.pipe()
    os.close(r)  # the reading end is gone: what a windowed parent without stderr hands on
    broken = os.fdopen(w, 'w')
    with pytest.raises(OSError):
        broken.write('x' * 65536)
        broken.flush()
    monkeypatch.setattr(sys, 'stderr', os.fdopen(os.dup(broken.fileno()), 'w'))
    runner_module().usable_stderr()
    sys.stderr.write('a warning\n')  # goes nowhere, does not raise
    sys.stderr.flush()
    monkeypatch.undo()
    try:
        broken.close()
    except OSError:
        pass


FAKE = '''import struct, sys
sys.stderr.write("a warning from the runner\\n"); sys.stderr.flush()
sys.stdout.buffer.write(struct.pack("<I", 22050)); sys.stdout.buffer.flush()
sys.stdin.readline()
raise ValueError("the voice broke")
'''


def test_its_stderr_goes_to_the_log_and_the_reason_comes_back(tmp_path, monkeypatch):
    from peresmeshnik import config
    from peresmeshnik.local import tts
    (tmp_path / 'fake.py').write_text(FAKE, encoding='utf-8')
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path / 'data')
    monkeypatch.setattr(tts, 'voice_path', lambda voice_id: tmp_path / 'voice.onnx')
    monkeypatch.setattr(tts, 'runner_cmd', lambda: [sys.executable, str(tmp_path / 'fake.py')])
    t = tts.PiperTts()
    try:
        assert t.sample_rate == 22050
        with pytest.raises(RuntimeError, match='piper-runner.log') as e:
            list(t.synth('Привет.', 'ru'))
        assert 'voice broke' not in str(e.value)  # the error may go into a report: Piper's own words stay in its log
    finally:
        t.close()
    log = (tmp_path / 'data' / 'logs' / 'piper-runner.log').read_text(encoding='utf-8')
    assert 'a warning from the runner' in log and 'ValueError: the voice broke' in log


def test_an_error_says_its_type_not_its_message(tmp_path):
    """An error's message can quote what was said: the runner's log keeps its type and place only (exit code 1, as the
    frozen Windows entry needs: no dialog)."""
    r = subprocess.run([sys.executable, str(ROOT / 'piper-runner' / 'piper_runner.py'), str(tmp_path / 'missing.onnx')],
                       capture_output=True, text=True, timeout=120)
    last = r.stderr.strip().splitlines()[-1]
    assert r.returncode == 1 and last.startswith('piper-runner: ') and 'Error at ' in last, r.stderr
    assert 'missing.onnx' not in r.stderr and 'No module named' not in r.stderr and 'Traceback' not in r.stderr, r.stderr

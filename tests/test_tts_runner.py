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


def test_onnxruntime_sends_no_telemetry(monkeypatch):
    """H-03: the app and the Piper runner turn onnxruntime's telemetry off before its first session: the variable on
    Linux and macOS (inherited by the runner), the call on Windows (TraceLogging does not read it)."""
    env = {k: v for k, v in os.environ.items() if k != 'ORT_DISABLE_TELEMETRY'}
    code = 'import os, peresmeshnik; print(os.environ.get("ORT_DISABLE_TELEMETRY"))'
    assert subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, cwd=ROOT,
                          check=True).stdout.strip() == '1'
    m = runner_module()
    assert os.environ.get('ORT_DISABLE_TELEMETRY') == '1'  # set by the runner's own import, before onnxruntime's
    import onnxruntime as rt
    calls = []
    monkeypatch.setattr(rt, 'disable_telemetry_events', lambda: calls.append(1))
    from peresmeshnik.local import vad
    vad.ort_options(1)
    m.no_telemetry()
    assert len(calls) == 2


HUNG = '''import struct, sys, time
sys.stdout.buffer.write(struct.pack("<I", 22050)); sys.stdout.buffer.flush()
sys.stdin.readline()
time.sleep(999)  # hung on the phrase: nothing more comes
'''


def test_a_runner_that_stops_answering_is_killed(tmp_path, monkeypatch):
    """M-08: a Piper runner that starts and then sends nothing (hung on a phrase, held by an antivirus) is killed after
    REPLY_S: the read ends with an error (crash.Respawn starts it again), the side is not held for ever; one that never
    starts gives up after START_S."""
    import time

    from peresmeshnik import config
    from peresmeshnik.local import tts
    (tmp_path / 'hung.py').write_text(HUNG, encoding='utf-8')
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path / 'data')
    monkeypatch.setattr(tts, 'voice_path', lambda voice_id: tmp_path / 'voice.onnx')
    monkeypatch.setattr(tts, 'REPLY_S', 1.0)
    monkeypatch.setattr(tts, 'START_S', 1.0)
    monkeypatch.setattr(tts, 'runner_cmd', lambda: [sys.executable, str(tmp_path / 'hung.py')])
    t = tts.PiperTts()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match='did not answer'):
        list(t.synth('Привет.', 'ru'))
    assert time.monotonic() - t0 < 5 and t._p.poll() is not None
    t.close()
    monkeypatch.setattr(tts, 'runner_cmd', lambda: [sys.executable, '-c', 'import time; time.sleep(999)'])
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match='did not answer'):
        tts.PiperTts()
    assert time.monotonic() - t0 < 5


def test_the_runner_starts_offline_and_goes_online_only_the_first_time(tmp_path, monkeypatch):
    """M-15: from source, `uv run` resolves Piper's packages from its cache (--offline): no pypi.org at each start, and
    no network needed once it was set up; only when that fails by itself (nothing cached yet) does it try online."""
    from peresmeshnik import config
    from peresmeshnik.local import tts
    monkeypatch.delenv('PERESMESHNIK_PIPER_CMD', raising=False)
    assert '--offline' in tts.runner_cmd()
    ok = tmp_path / 'ok.py'
    ok.write_text('import struct, sys\nsys.stdout.buffer.write(struct.pack("<I", 22050)); sys.stdout.buffer.flush()\n'
                  'sys.stdin.readline()\n', encoding='utf-8')
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path / 'data')
    monkeypatch.setattr(tts, 'voice_path', lambda voice_id: tmp_path / 'voice.onnx')
    starts = []
    real = subprocess.Popen

    def popen(cmd, **kw):  # "uv": offline fails (nothing cached), online works
        starts.append(cmd)
        return real([sys.executable, '-c', 'raise SystemExit(2)'] if '--offline' in cmd else [sys.executable, str(ok)], **kw)
    monkeypatch.setattr(tts.subprocess, 'Popen', popen)
    t = tts.PiperTts()
    t.close()
    assert t.sample_rate == 22050 and ['--offline' in c for c in starts] == [True, False]

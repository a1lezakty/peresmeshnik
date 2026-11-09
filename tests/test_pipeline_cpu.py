"""Working-mode CPU gate, next to the idle gate: `peresmeshnik run` (Silero VAD, Parakeet int8 with
partials, quickmt, Piper) on a 30 s English recording, fed as fast as it goes, in its own process
like the app. Fails when the pipeline is slower than real time or burns more CPU per second of
audio than the limit, or when the captions are wrong.

Needs the models (`peresmeshnik models fetch silero-vad parakeet-tdt-0.6b-v3 quickmt-en-ru
piper-ru_RU-dmitri-medium`, ~0.8 GB), so it runs only with PERESMESHNIK_WORK_GATE=1 (CI job
`work-gate`). Limits can be overridden: WORK_RTF_LIMIT, WORK_CPU_LIMIT.
"""
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

GATE = os.environ.get('PERESMESHNIK_WORK_GATE') == '1'
pytestmark = pytest.mark.skipif(not GATE and os.environ.get('GITHUB_JOB') != 'work-gate', reason='needs models: PERESMESHNIK_WORK_GATE=1')
RTF_LIMIT = float(os.environ.get('WORK_RTF_LIMIT', 1.0))
CPU_LIMIT = float(os.environ.get('WORK_CPU_LIMIT', 2.0))
DATA = Path(__file__).parent / 'data'


def cpu_model():  # runners of one label differ up to 2x: name the CPU next to the numbers
    try:
        if sys.platform == 'darwin':
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
        if sys.platform.startswith('linux'):
            return next(ln.split(':', 1)[1].strip() for ln in open('/proc/cpuinfo') if ln.startswith('model name'))
    except Exception:
        pass
    return platform.processor()


def words(s):
    return re.findall(r"[\w']+", s.lower())


def test_work_cpu():
    assert GATE, 'job work-gate without PERESMESHNIK_WORK_GATE=1: the gate would silently skip'
    r = subprocess.run([sys.executable, '-m', 'peresmeshnik', 'run', '--in', str(DATA / 'call-en.ogg'), '--from', 'en'],
                       capture_output=True, text=True, encoding='utf-8', timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads(r.stderr.strip().splitlines()[-1])
    lines = r.stdout.splitlines()
    original = ' '.join(ln[4:] for ln in lines if ln.startswith('en: '))
    translation = [ln[4:] for ln in lines if ln.startswith('ru: ')]
    ref = words((DATA / 'call-en.txt').read_text(encoding='utf-8'))
    hyp = set(words(original))
    s['word_recall'] = round(sum(w in hyp for w in ref) / len(ref), 2)
    s['cpu'], s['cores'] = cpu_model(), os.cpu_count()
    print('\nwork gate:', json.dumps(s))  # ASCII only: the Windows console of pytest is not UTF-8
    assert s['word_recall'] > 0.6, original
    assert translation and all(re.search('[а-яА-Я]', t) for t in translation), translation
    assert s['rtf'] < RTF_LIMIT, f"RTF {s['rtf']} >= {RTF_LIMIT}"
    assert s['cpu_s_per_audio_s'] < CPU_LIMIT, f"{s['cpu_s_per_audio_s']} CPU s per audio s >= {CPU_LIMIT}"

"""Shared helpers for bench scripts: bench set, timing, peak RAM, result records."""
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('BENCH_DATA', 'bench-data'))
SET = ROOT / 'set'
RESULTS = Path(__file__).parent / 'results'


def host():
    """Short host tag: results from different machines go to different files."""
    if os.environ.get('BENCH_HOST'):
        return os.environ['BENCH_HOST']
    m = platform.machine().lower()
    return f'{sys.platform}-{m}-{os.cpu_count()}c'


def manifest():
    return json.loads((SET / 'manifest.json').read_text(encoding='utf-8'))


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024 if sys.platform == 'darwin' else 1024))  # bytes on macOS, KiB on Linux


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


def record(family, model, **fields):
    """Append one result row to bench/results/<host>.jsonl and echo it."""
    row = {'family': family, 'model': model, 'host': host(), 'peak_rss_mb': peak_rss_mb(),
           'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f'{host()}.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False))
    return row

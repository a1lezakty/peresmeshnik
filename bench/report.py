"""Turn bench/results/*.jsonl into Markdown tables (pasted into docs/BENCH.md).

    python bench/report.py [family ...]
"""
import json
import sys
from pathlib import Path

COLS = {
    'asr': ['host', 'model', 'lang', 'wer', 'cer', 'utt_latency_ms_median', 'win2s_latency_ms', 'chunk_latency_ms', 'rtf',
            'cpu_s_per_audio_s', 'peak_rss_mb', 'load_s'],
    'mt': ['host', 'model', 'direction', 'chrf', 'bleu', 'latency_ms_median', 'latency_ms_p90', 'cpu_cores', 'peak_rss_mb',
           'server_peak_rss_mb', 'load_s'],
    'tts': ['host', 'model', 'lang', 'voice', 'ttfa_ms_median', 'rtf', 'cpu_cores', 'peak_rss_mb', 'cer', 'load_s'],
    'idle_loaded': ['host', 'model', 'threads', 'no_spin', 'omp_wait_policy', 'busy_cpu_s', 'busy_wall_s', 'idle_s', 'idle_cpu_s'],
}
SKIP = {'family', 'time', 'samples', 'sample', 'n'}


def rows():
    out = []
    for f in sorted((Path(__file__).parent / 'results').glob('*.jsonl')):
        out += [json.loads(line) for line in f.read_text(encoding='utf-8').splitlines() if line.strip()]
    latest = {}  # a rerun of the same case replaces the older row
    for r in out:
        key = tuple(r.get(k) for k in ('family', 'host', 'model', 'lang', 'direction', 'device', 'ref_lang', 'out_lang',
                                         'case', 'no_spin', 'omp_wait_policy'))
        latest[key] = r
    return list(latest.values())


def table(family, rs):
    if family == 'tts':  # attach intelligibility from the judge pass
        cer = {(r['host'], r['model'], r['lang']): r['cer'] for r in rs if r['family'] == 'tts_cer'}
        rs = [{**r, 'cer': cer.get((r['host'], r['model'], r['lang']))} for r in rs if r['family'] == 'tts']
    else:
        rs = [r for r in rs if r['family'] == family]
    if not rs:
        return ''
    cols = COLS.get(family) or [k for k in dict.fromkeys(k for r in rs for k in r) if k not in SKIP]
    cols = [c for c in cols if any(r.get(c) is not None for r in rs)]
    rs.sort(key=lambda r: [str(r.get(c, '')) for c in cols[:3]])
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join('—' if r.get(c) is None else str(r[c]) for c in cols) + ' |' for r in rs]
    return f'### {family}\n\n' + '\n'.join(lines) + '\n'


if __name__ == '__main__':
    rs = rows()
    fams = sys.argv[1:] or [f for f in dict.fromkeys(r['family'] for r in rs) if f != 'tts_cer']
    print('\n'.join(table(f, rs) for f in fams))

"""COMET (Unbabel/wmt22-comet-da, Apache-2.0) of the MT hypotheses bench/mt.py saved: one result row per file.

    python bench/mt_comet.py [--gpu] [--only REGEX]   # in its own venv (torch + unbabel-comet); $BENCH_DATA/mt-hyps/*.json

Not bench/comet.py: the script's folder comes first on sys.path and would hide the comet package.

A file is `<model>-<src>-<tgt>-<device>.json` with the sources, the hypotheses and the references (FLEURS test).
"""
import argparse
import json
import re

from common import ROOT, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--only', default='', help='score the files whose name matches (re.search)')
    a = ap.parse_args()
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))
    for f in sorted((ROOT / 'mt-hyps').glob('*.json')):
        m = re.match(r'(.+)-([a-z]{2})-([a-z]{2})-(cpu|cuda)$', f.stem)
        if not m or not re.search(a.only, f.stem):
            continue
        name, src, tgt, device = m.groups()
        d = json.loads(f.read_text(encoding='utf-8'))
        data = [{'src': s, 'mt': h, 'ref': r} for s, h, r in zip(d['src'], d['hyp'], d['ref'])]
        out = model.predict(data, batch_size=a.batch, gpus=1 if a.gpu else 0, progress_bar=False)
        record('mt_comet', name, direction=f'{src}-{tgt}', device=device, n=len(data), judge='wmt22-comet-da',
               comet=round(float(out.system_score), 4))


if __name__ == '__main__':
    main()

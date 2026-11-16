"""R4: a 'Russian speakers' direction for ZONOS3-Mini's 192-d voice vector: mean of the unit vectors of the FLEURS ru
clips minus that of the FLEURS en clips ($DIR_CLIPS under $BENCH_DATA, default the bench set), saved as an .npy of the encoder's own shape.

    python r4/zonos_dir.py OUT.npy        # needs $ZONOS_BIN (spk-encoder-cli) and $ZONOS_GGUF (spk-encoder.gguf)
"""
import os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('BENCH_DATA', 'bench-data'))
PATTERN = os.environ.get('DIR_CLIPS', 'set/{lang}/*.wav')  # e.g. fleurs/{lang}/test/*.wav: speakers apart from dev
MAX = int(os.environ.get('DIR_MAX', 40))
b, g = os.environ['ZONOS_BIN'], os.environ['ZONOS_GGUF']


def unit_mean(lang):
    vs = []
    with tempfile.TemporaryDirectory() as t:
        for w in sorted(ROOT.glob(PATTERN.format(lang=lang)))[:MAX]:
            o = Path(t) / (w.stem + '.npy')
            subprocess.run([f'{b}/spk-encoder-cli', f'{g}/spk-encoder.gguf', '--audio', str(w), str(o), '--cpu'],
                           check=True, capture_output=True)
            e = np.load(o)
            vs.append(e / np.linalg.norm(e))
    m = np.mean(vs, 0)
    print(lang, len(vs), 'clips, mean pairwise cos to the mean', round(float(np.mean([v.ravel() @ (m / np.linalg.norm(m)).ravel() for v in vs])), 3))
    return m


ru, en = unit_mean('ru'), unit_mean('en')
d = ru - en
print('direction norm', round(float(np.linalg.norm(d)), 3), 'cos(ru mean, en mean)',
      round(float((ru / np.linalg.norm(ru)).ravel() @ (en / np.linalg.norm(en)).ravel()), 3), 'shape', d.shape)
np.save(sys.argv[1], d.astype(np.float32))
